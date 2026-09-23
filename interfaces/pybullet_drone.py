"""
Concrete DroneInterface backed by the PyBullet rigid-body sim.

This is the "flight controller" layer: it turns high-level commands
(move_forward, turn_left, hover, ...) into per-step thrust/torque via a
cascade of PID loops, the same shape a real flight controller uses -
- an outer loop converts a desired horizontal velocity into a target tilt
  angle (to move forward, a quad pitches its nose down and lets gravity +
  thrust do the rest)
- an inner loop converts that target angle (+ current rotation rate) into
  a torque
- a separate altitude loop converts target altitude into total thrust
- yaw is rate-controlled directly (turn_left/right set a target yaw rate)

FlyBrain/the reflex controller never touches any of this - it only calls
the methods below.
"""

import math
import pybullet as p

from simulation.drone_sim import QuadcopterBody
from vision.camera import DroneCamera
from interfaces.drone_interface import DroneInterface

GRAVITY = 9.81

# --- Tunable parameters -----------------------------------------------
MASS = 1.0                 # kg
MIN_ALTITUDE = 0.5         # m - safety floor while flying (never flies into the ground)
NORMAL_ALTITUDE = 1.5      # m - default cruise altitude; also the takeoff target,
                            # and what relax_altitude() pulls back toward when
                            # nothing has a reason to climb/descend
MAX_ALTITUDE = 3.0         # m - safety ceiling
MAX_SPEED = 1.3            # m/s - safety cap on commanded horizontal speed (must
                            # stay above controllers/safety_layer.py's FAST_SPEED,
                            # or the "go fast in open space" upgrade gets clamped
                            # away uselessly)
MAX_YAW_RATE = 1.5         # rad/s
MAX_TILT = 0.30            # rad (~17 deg) - cap on how hard it'll lean over

# Proximity safety net (item 7): overrides whatever the controller (manual
# or autonomous) commanded if something is actually this close, using real
# physics distance rather than the vision estimate - a last line of defense
# independent of the reflex controller's own (imperfect) avoidance logic.
SAFETY_DISTANCE = 0.6      # m - forward motion gets capped to 0 past this
CRITICAL_DISTANCE = 0.3    # m - forces a backward retreat past this
RETREAT_SPEED = 0.25       # m/s - how hard it backs away when critical
ALTITUDE_STEP = 0.02       # m added to target altitude per move_up/move_down call
RETURN_TO_NORMAL_STEP = 0.01  # m per relax_altitude() call (see relax_altitude)

ALT_KP, ALT_KI, ALT_KD = 10.0, 1.0, 9.0         # altitude error (m) -> thrust (N)
VEL_TO_TILT_KP = 0.18                           # velocity error (m/s) -> target tilt (rad)
TILT_KP, TILT_KD = 0.09, 0.025                  # tilt error (rad) / rate -> torque (Nm)
YAW_KP = 0.22                                   # yaw rate error -> torque (Nm) - was
                                                 # 0.03, which took a full second to
                                                 # spin up; too slow for the heading to
                                                 # catch up during a real turn-then-move
                                                 # key combo, making "forward" feel like
                                                 # it was using the wrong direction

EMERGENCY_HOVER_STEPS = 180  # how long to just hover after a collision before auto-landing
# -----------------------------------------------------------------------

PHYSICS_DT = 1.0 / 240.0


class _PID:
    def __init__(self, kp, ki=0.0, kd=0.0):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.integral = 0.0
        self.prev_error = 0.0

    def update(self, error, dt, derivative=None):
        self.integral += error * dt
        d = derivative if derivative is not None else (error - self.prev_error) / dt
        self.prev_error = error
        return self.kp * error + self.ki * self.integral + self.kd * d

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0


class PyBulletDrone(DroneInterface):

    def __init__(self, start_pos=(0, 0, 0.05), ground_id=None, obstacle_ids=None):
        self.start_pos = start_pos
        self.body = QuadcopterBody(start_pos=start_pos, mass=MASS)
        self.camera = DroneCamera()

        self.ground_id = ground_id
        self.obstacle_ids = set(obstacle_ids or [])

        self._alt_pid = _PID(ALT_KP, ALT_KI, ALT_KD)
        self._roll_pid = _PID(TILT_KP, kd=TILT_KD)
        self._pitch_pid = _PID(TILT_KP, kd=TILT_KD)
        self._yaw_pid = _PID(YAW_KP)

        self._reset_flight_vars()

    def _reset_flight_vars(self):
        self.state = "idle"  # idle | taking_off | flying | emergency | landing | landed
        self.target_altitude = NORMAL_ALTITUDE
        self._landing_altitude = None
        self.target_vx = 0.0
        self.target_vy = 0.0
        self.target_yaw_rate = 0.0
        self.collided = False
        self.emergency = False
        self._emergency_timer = 0
        self.emergency_stop_count = 0
        self.min_obstacle_distance = None
        self.safety_override = False
        self._alt_pid.reset()
        self._roll_pid.reset()
        self._pitch_pid.reset()
        self._yaw_pid.reset()
        self._last_frame = None

    # --- High-level commands (called by the reflex/FlyBrain controller) ---

    def takeoff(self):
        if self.state in ("idle", "landed"):
            self.state = "taking_off"
            self.target_altitude = NORMAL_ALTITUDE

    def land(self):
        if self.state in ("flying", "emergency", "taking_off"):
            self.state = "landing"
            self._landing_altitude = self._current_altitude()

    def hover(self):
        self.target_vx = 0.0
        self.target_vy = 0.0
        self.target_yaw_rate = 0.0

    def move_forward(self, speed):
        self.target_vx = max(-MAX_SPEED, min(MAX_SPEED, speed))

    def move_backward(self, speed):
        self.target_vx = -max(-MAX_SPEED, min(MAX_SPEED, speed))

    def move_left(self, speed):
        self.target_vy = max(-MAX_SPEED, min(MAX_SPEED, speed))

    def move_right(self, speed):
        self.target_vy = -max(-MAX_SPEED, min(MAX_SPEED, speed))

    def move_up(self):
        """Nudge the target altitude up by one step. Not part of the
        abstract DroneInterface (real hardware may expose altitude control
        differently) - both manual and autonomous control call this
        directly on PyBulletDrone. Call every decision cycle while
        climbing is wanted; call relax_altitude() instead when it isn't,
        so it drifts back toward NORMAL_ALTITUDE."""
        self.target_altitude = min(MAX_ALTITUDE, self.target_altitude + ALTITUDE_STEP)

    def move_down(self):
        self.target_altitude = max(MIN_ALTITUDE, self.target_altitude - ALTITUDE_STEP)

    def relax_altitude(self):
        """Call once per decision cycle whenever nothing wants to climb or
        descend. Nudges target_altitude back toward NORMAL_ALTITUDE rather
        than leaving it wherever the last move_up/move_down left it -
        "normally return toward NORMAL_ALTITUDE when there is no reason to
        go higher/lower." Small step size so it never fights an active
        move_up/move_down happening on other cycles."""
        if self.target_altitude > NORMAL_ALTITUDE:
            self.target_altitude = max(NORMAL_ALTITUDE, self.target_altitude - RETURN_TO_NORMAL_STEP)
        elif self.target_altitude < NORMAL_ALTITUDE:
            self.target_altitude = min(NORMAL_ALTITUDE, self.target_altitude + RETURN_TO_NORMAL_STEP)

    def turn_left(self, rate):
        self.target_yaw_rate = max(-MAX_YAW_RATE, min(MAX_YAW_RATE, rate))

    def turn_right(self, rate):
        self.target_yaw_rate = -max(-MAX_YAW_RATE, min(MAX_YAW_RATE, rate))

    # --- Sensing ---

    def get_camera_frame(self):
        state = self.body.get_state()
        self._last_frame = self.camera.capture(state["position"], state["orientation"])
        return self._last_frame

    def get_state(self):
        raw = self.body.get_state()
        vx, vy, vz = raw["linear_velocity"]
        speed = math.sqrt(vx ** 2 + vy ** 2)

        # Body-frame actual forward/left velocity, for comparing directly
        # against target_vx/target_vy (also body-frame) in the debug
        # overlay - "target velocity" vs "actual velocity" should mean the
        # same axes, not a world-frame vs body-frame mismatch.
        _, _, yaw = p.getEulerFromQuaternion(raw["orientation"])
        cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
        vx_body = vx * cos_yaw + vy * sin_yaw
        vy_body = -vx * sin_yaw + vy * cos_yaw

        return {
            "position": raw["position"],
            "altitude": raw["position"][2],
            "target_altitude": self.target_altitude,
            "orientation": raw["orientation"],
            "yaw_degrees": math.degrees(yaw),
            "horizontal_speed": speed,
            "vertical_speed": vz,
            "actual_vx": vx_body,
            "actual_vy": vy_body,
            "target_vx": self.target_vx,
            "target_vy": self.target_vy,
            "yaw_rate": raw["angular_velocity"][2],
            "flight_state": self.state,
            "collided": self.collided,
            "emergency": self.emergency,
            "min_obstacle_distance": self.min_obstacle_distance,
            "safety_override": self.safety_override,
        }

    # --- Main control cycle: call every physics tick ---

    def step(self):
        self._run_state_machine()
        self._check_safety_and_collisions()

        state = self.body.get_state()
        position = state["position"]
        roll, pitch, yaw = p.getEulerFromQuaternion(state["orientation"])
        vx, vy, vz = state["linear_velocity"]

        # pybullet's angular velocity is in WORLD frame. Using it directly
        # as roll/pitch damping only happens to work near yaw=0 - once the
        # drone is actively yawing, world-frame wx/wy stop corresponding to
        # "how fast is it tilting nose-up/bank-left" and instead mix with
        # the yaw rotation, feeding wrong damping into the pitch/roll PIDs.
        # That mismatch was the actual cause of a real bug: sustained
        # emergency turning (high yaw rate + active pitch/roll) diverged
        # into an exponentially growing oscillation. Rotate into body
        # frame first, same as a real flight controller's gyro reads.
        rot = p.getMatrixFromQuaternion(state["orientation"])
        wx, wy, wz = state["angular_velocity"]
        roll_rate = rot[0] * wx + rot[3] * wy + rot[6] * wz
        pitch_rate = rot[1] * wx + rot[4] * wy + rot[7] * wz
        yaw_rate = rot[2] * wx + rot[5] * wy + rot[8] * wz

        if self.state in ("idle", "landed"):
            # Motors off - just let it sit there.
            p.stepSimulation()
            return

        # --- Altitude loop -> thrust ---
        effective_target_alt = self._effective_target_altitude()
        alt_error = effective_target_alt - position[2]
        thrust = MASS * GRAVITY + self._alt_pid.update(alt_error, PHYSICS_DT, derivative=-vz)
        thrust = max(0.0, thrust)

        # --- Proximity safety net: overrides whatever was commanded (manual
        # or autonomous) using real physics distance, independent of - and a
        # backstop for - the reflex controller's own (imperfect) avoidance.
        self.min_obstacle_distance = self._closest_obstacle_distance()
        self.safety_override = False
        target_vx, target_vy = self.target_vx, self.target_vy

        if self.min_obstacle_distance is not None:
            if self.min_obstacle_distance < CRITICAL_DISTANCE:
                self.safety_override = True
                target_vx, target_vy = -RETREAT_SPEED, 0.0
            elif self.min_obstacle_distance < SAFETY_DISTANCE:
                self.safety_override = True
                target_vx = min(target_vx, 0.0)  # can brake/back away/strafe, just not push forward

        # --- Velocity -> desired tilt (outer loop) ---
        # target_vx/target_vy are body-relative ("forward"/"left"), but the
        # measured vx/vy from the physics engine are world-frame. Rotate the
        # MEASURED velocity into body frame (by -yaw) rather than rotating
        # the target into world frame: pitch/roll are inherently body-frame
        # concepts (tilting the body's own axes), and at nonzero yaw a tilt
        # actually accelerates the drone along a mix of both world axes -
        # trying to hit a world-frame velocity target directly would need a
        # coupled pitch+roll formula. Comparing in body frame instead lets
        # the same simple, already-verified "pitch <- forward error, roll
        # <- left error" mapping work correctly at any heading.
        cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
        vx_body = vx * cos_yaw + vy * sin_yaw
        vy_body = -vx * sin_yaw + vy * cos_yaw

        vx_error = target_vx - vx_body
        vy_error = target_vy - vy_body
        desired_pitch = max(-MAX_TILT, min(MAX_TILT, VEL_TO_TILT_KP * vx_error))
        desired_roll = max(-MAX_TILT, min(MAX_TILT, -VEL_TO_TILT_KP * vy_error))

        # --- Attitude loop (inner) -> torques ---
        torque_pitch = self._pitch_pid.update(desired_pitch - pitch, PHYSICS_DT, derivative=-pitch_rate)
        torque_roll = self._roll_pid.update(desired_roll - roll, PHYSICS_DT, derivative=-roll_rate)
        torque_yaw = self._yaw_pid.update(self.target_yaw_rate - yaw_rate, PHYSICS_DT)

        self.body.apply_control(thrust, torque_roll, torque_pitch, torque_yaw)
        p.stepSimulation()

    def reset(self):
        self.body.reset(self.start_pos)
        self._reset_flight_vars()

    # --- Internals ---

    def _current_altitude(self):
        return self.body.get_state()["position"][2]

    def _effective_target_altitude(self):
        if self.state == "landing":
            # Ramp down toward the ground instead of snapping to 0.
            self._landing_altitude = max(0.0, self._landing_altitude - 0.4 * PHYSICS_DT)
            return self._landing_altitude
        return max(MIN_ALTITUDE, min(MAX_ALTITUDE, self.target_altitude))

    def _closest_obstacle_distance(self):
        if not self.obstacle_ids:
            return None
        distances = [
            self.body.closest_distance(oid, max_distance=2.0)
            for oid in self.obstacle_ids
        ]
        distances = [d for d in distances if d is not None]
        return min(distances) if distances else None

    def _run_state_machine(self):
        altitude = self._current_altitude()
        vz = self.body.get_state()["linear_velocity"][2]

        if self.state == "taking_off" and abs(altitude - NORMAL_ALTITUDE) < 0.05 and abs(vz) < 0.1:
            self.state = "flying"

        elif self.state == "landing" and altitude < 0.08 and abs(vz) < 0.1:
            self.state = "landed"
            self.target_vx = self.target_vy = self.target_yaw_rate = 0.0

        elif self.state == "emergency":
            self._emergency_timer += 1
            if self._emergency_timer > EMERGENCY_HOVER_STEPS:
                self.land()

    def _check_safety_and_collisions(self):
        contacts = self.body.get_contacts()
        hit_obstacle = any(
            c[2] in self.obstacle_ids or (c[2] != self.ground_id and c[2] != self.body.id)
            for c in contacts
        )
        hit_ground_while_flying = (
            self.ground_id is not None
            and self.state in ("taking_off", "flying")
            and any(c[2] == self.ground_id for c in contacts)
        )

        if (hit_obstacle or hit_ground_while_flying) and not self.emergency:
            self.collided = True
            self.emergency = True
            self.emergency_stop_count += 1
            self._emergency_timer = 0
            self.state = "emergency"
            self.hover()
        elif not (hit_obstacle or hit_ground_while_flying) and self.state != "emergency":
            self.emergency = False
