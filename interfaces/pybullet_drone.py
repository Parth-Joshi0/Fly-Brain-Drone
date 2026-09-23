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
HOVER_ALTITUDE = 1.2       # m - fixed altitude for v1 (FlyBrain doesn't control this yet)
MIN_ALTITUDE = 0.3         # m - safety floor while flying
MAX_ALTITUDE = 2.5         # m - safety ceiling
MAX_SPEED = 1.5            # m/s - safety cap on commanded horizontal speed
MAX_YAW_RATE = 1.5         # rad/s
MAX_TILT = 0.30            # rad (~17 deg) - cap on how hard it'll lean over

ALT_KP, ALT_KI, ALT_KD = 10.0, 1.0, 9.0         # altitude error (m) -> thrust (N)
VEL_TO_TILT_KP = 0.18                           # velocity error (m/s) -> target tilt (rad)
TILT_KP, TILT_KD = 0.09, 0.025                  # tilt error (rad) / rate -> torque (Nm)
YAW_KP = 0.03                                   # yaw rate error -> torque (Nm)

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
        self.target_altitude = HOVER_ALTITUDE
        self._landing_altitude = None
        self.target_vx = 0.0
        self.target_vy = 0.0
        self.target_yaw_rate = 0.0
        self.collided = False
        self.emergency = False
        self._emergency_timer = 0
        self.emergency_stop_count = 0
        self._alt_pid.reset()
        self._roll_pid.reset()
        self._pitch_pid.reset()
        self._yaw_pid.reset()
        self._last_frame = None

    # --- High-level commands (called by the reflex/FlyBrain controller) ---

    def takeoff(self):
        if self.state in ("idle", "landed"):
            self.state = "taking_off"
            self.target_altitude = HOVER_ALTITUDE

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
        return {
            "position": raw["position"],
            "altitude": raw["position"][2],
            "orientation": raw["orientation"],
            "horizontal_speed": speed,
            "vertical_speed": vz,
            "yaw_rate": raw["angular_velocity"][2],
            "flight_state": self.state,
            "collided": self.collided,
            "emergency": self.emergency,
        }

    # --- Main control cycle: call every physics tick ---

    def step(self):
        self._run_state_machine()
        self._check_safety_and_collisions()

        state = self.body.get_state()
        position = state["position"]
        roll, pitch, yaw = p.getEulerFromQuaternion(state["orientation"])
        vx, vy, vz = state["linear_velocity"]
        roll_rate, pitch_rate, yaw_rate = state["angular_velocity"]

        if self.state in ("idle", "landed"):
            # Motors off - just let it sit there.
            p.stepSimulation()
            return

        # --- Altitude loop -> thrust ---
        effective_target_alt = self._effective_target_altitude()
        alt_error = effective_target_alt - position[2]
        thrust = MASS * GRAVITY + self._alt_pid.update(alt_error, PHYSICS_DT, derivative=-vz)
        thrust = max(0.0, thrust)

        # --- Velocity -> desired tilt (outer loop) ---
        vx_error = self.target_vx - vx
        vy_error = self.target_vy - vy
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

    def _run_state_machine(self):
        altitude = self._current_altitude()
        vz = self.body.get_state()["linear_velocity"][2]

        if self.state == "taking_off" and abs(altitude - HOVER_ALTITUDE) < 0.05 and abs(vz) < 0.1:
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
