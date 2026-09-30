"""
Concrete DroneInterface backed by a real DJI Tello over djitellopy.

Unlike PyBulletDrone there is no physics/PID cascade here - the Tello's
own onboard flight controller already turns an RC velocity request into
motor output. This class's only job is bookkeeping (flight state, a
dead-reckoned position estimate) and translating the DroneInterface
methods into `send_rc_control(left_right, forward_backward, up_down,
yaw)` calls, each an int in -100..100.

No accurate world position is available (no motion-capture/GPS - the
Tello's own vgx/vgy telemetry is frequently unreliable/zero on
non-EDU units). `get_state()["position"]` is instead dead-reckoned by
integrating the last COMMANDED body-frame velocity over time, rotated by
the measured yaw. This is only meant to feed
controllers/flybrain_controller.py's ESCAPE displacement tracking (how
far into a dodge it's gotten, relative to where the dodge started) - it
is never used for absolute navigation here, and `bounds=None` skips the
boundary-containment logic that would otherwise need real position.
Because the Tello's flight controller tracks a commanded RC velocity
fairly directly (unlike the sim's mass/PID model, it doesn't coast much
once told to stop), integrating the COMMAND rather than a measured
velocity is a reasonable approximation for timing a ~1s dodge.

RC_SPEED_SCALE (m/s -> percent) is a rough assumption, not a measured
constant - Tello's RC percentage is "percent of the current max-speed
setting" (`set_speed`), which was never changed here (default ~1.5-2
m/s corresponds to 100%). VERIFY at low speed before trusting it for a
real dodge: send a small move_forward/move_left command first and watch
whether the resulting motion direction and rough speed look right, the
same way tello_neuron_test.py's docstring flagged the attitude sign
convention as something to confirm empirically rather than assume.
"""

import math
import time

from interfaces.drone_interface import DroneInterface

RC_SPEED_SCALE = 60.0   # percent per m/s - ASSUMED, see module docstring
MAX_RC_PERCENT = 100

# The Tello reports yaw increasing CLOCKWISE (turning right); this project's
# convention is positive = counter-clockwise (turning left), the same
# convention positive yaw_rate carries in every controller and in
# vision/optical_flow.derotate_flow. So the reported heading is negated once,
# here, where it enters the codebase - after which everything downstream
# (yaw_rate, the orientation quaternion, yaw_degrees, the dead-reckoned
# position) is in one convention.
#
# MEASURED, not assumed, which is the point of it being a named constant:
# Testing/tello_dng02_test.py --mode calibrate regresses horizontal optic flow
# against the reported yaw. On 2026-09-27 that fit came out at -86.6 with
# R^2 0.888, negative at every window size from 1 to 30 frames. A matching
# convention would have given a positive slope. Re-run that mode after any
# firmware change and it will say whether this constant is still right: with
# the constant applied, a CORRECT value now yields a POSITIVE slope.
#
# Why it mattered enough to chase down: derotate_flow subtracts
# pixels_per_radian * yaw_rate * dt to cancel the drone's own rotation, and
# LoomingDetector warps frames by this orientation to remove rotation before
# measuring expansion. With the sign wrong both ADD self-motion instead of
# removing it, so every turn inflates apparent looming.
#
# NOTE this covers yaw only. Pitch and roll enter the same quaternion and are
# still unverified - they sit near zero on a desk, so the props-off tests could
# not measure them.
TELLO_YAW_SIGN = -1.0

# Tello send_rc_control sign convention (per djitellopy/SDK docs):
#   left_right_velocity:      -100 = left,  +100 = right
#   forward_backward_velocity: -100 = back,  +100 = forward
#   up_down_velocity:          -100 = down,  +100 = up
#   yaw_velocity:              -100 = ccw (turn left), +100 = cw (turn right)


def _mps_to_rc(speed_mps):
    return max(-MAX_RC_PERCENT, min(MAX_RC_PERCENT, round(speed_mps * RC_SPEED_SCALE)))


def _rate_to_rc(rate_rad_s, rate_at_100=1.5):
    """rate_at_100: assumed yaw rate (rad/s) at 100% yaw RC - also unverified,
    see module docstring."""
    return max(-MAX_RC_PERCENT, min(MAX_RC_PERCENT, round(rate_rad_s / rate_at_100 * 100)))


class TelloDrone(DroneInterface):

    def __init__(self, tello, frame_read, proc_width=320, proc_height=240):
        """tello: a connected, streamed-on djitellopy.Tello. frame_read: its
        get_frame_read() result. Both constructed by the caller (see
        Testing/tello_escape_flight_test.py) so this class never owns the
        connect/streamon handshake or its failure modes - those need
        different retry/abort handling than anything else here."""
        self._tello = tello
        self._frame_read = frame_read
        self.proc_width, self.proc_height = proc_width, proc_height

        self.state = "idle"  # idle | taking_off | flying | landing | landed
        self.collided = False       # no collision sensor on the Tello - always False
        self.emergency = False
        self.min_obstacle_distance = None  # no forward distance sensor
        self.safety_override = False

        self.target_vx = 0.0   # m/s, body forward
        self.target_vy = 0.0   # m/s, body left
        self.target_yaw_rate = 0.0  # rad/s
        self.target_altitude_delta = 0.0  # -1/0/+1, see move_up/move_down/relax_altitude

        self._pos = [0.0, 0.0, 0.0]
        self._last_step_time = None
        self._prev_yaw_rad = None
        self._yaw_rate = 0.0
        self._last_frame = None

    # --- High-level commands ---

    def takeoff(self):
        if self.state in ("idle", "landed"):
            self.state = "taking_off"
            self._tello.takeoff()
            self.state = "flying"
            self._last_step_time = time.perf_counter()
            self._prev_yaw_rad = None

    def land(self):
        if self.state in ("flying", "taking_off"):
            self.state = "landing"
            self.hover()
            self._tello.land()
            self.state = "landed"

    def hover(self):
        self.target_vx = 0.0
        self.target_vy = 0.0
        self.target_yaw_rate = 0.0

    def move_forward(self, speed):
        self.target_vx = speed

    def move_backward(self, speed):
        self.target_vx = -speed

    def move_left(self, speed):
        self.target_vy = speed

    def move_right(self, speed):
        self.target_vy = -speed

    def move_up(self):
        self.target_altitude_delta = 1.0

    def move_down(self):
        self.target_altitude_delta = -1.0

    def relax_altitude(self):
        self.target_altitude_delta = 0.0

    def turn_left(self, rate):
        self.target_yaw_rate = rate

    def turn_right(self, rate):
        self.target_yaw_rate = -rate

    # --- Sensing ---

    def get_camera_frame(self):
        """Returns the latest BGR frame, resized to (proc_width,
        proc_height). Caches and returns the last good frame when the
        background decoder hasn't produced a new/real one yet (djitellopy's
        BackgroundFrameRead seeds .frame with a black placeholder and never
        returns None - see tello_neuron_test.py's frame_not_ready)."""
        import cv2
        import numpy as np

        frame = self._frame_read.frame
        placeholder = frame is None or frame.shape[:2] == (300, 400) or not frame.any()
        if placeholder and self._last_frame is not None:
            return self._last_frame
        if placeholder:
            return np.zeros((self.proc_height, self.proc_width, 3), dtype=np.uint8)

        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        small = cv2.resize(frame_bgr, (self.proc_width, self.proc_height), interpolation=cv2.INTER_AREA)
        self._last_frame = small
        return small

    def get_state(self):
        try:
            raw_state = self._tello.get_current_state() or {}
        except Exception:
            raw_state = {}

        def num(key, default=0.0):
            try:
                return float(raw_state[key])
            except (KeyError, TypeError, ValueError):
                return default

        pitch_deg, roll_deg = num("pitch"), num("roll")
        # Into this project's convention once, at the boundary - see TELLO_YAW_SIGN.
        yaw_deg = TELLO_YAW_SIGN * num("yaw")
        yaw_rad = math.radians(yaw_deg)

        now = time.perf_counter()
        dt = 0.0 if self._last_step_time is None else max(0.0, now - self._last_step_time)
        self._last_step_time = now

        if self._prev_yaw_rad is not None and dt > 0:
            d_yaw = (yaw_rad - self._prev_yaw_rad + math.pi) % (2 * math.pi) - math.pi
            self._yaw_rate = d_yaw / dt
        self._prev_yaw_rad = yaw_rad

        # Dead-reckon position from the COMMANDED body velocity - see
        # module docstring for why this (not measured vgx/vgy) is used.
        cos_yaw, sin_yaw = math.cos(yaw_rad), math.sin(yaw_rad)
        self._pos[0] += (self.target_vx * cos_yaw - self.target_vy * sin_yaw) * dt
        self._pos[1] += (self.target_vx * sin_yaw + self.target_vy * cos_yaw) * dt
        height_cm = raw_state.get("h") or raw_state.get("tof") or 0.0
        self._pos[2] = float(height_cm) / 100.0

        orientation = _euler_deg_to_quat(roll_deg, pitch_deg, yaw_deg)

        return {
            "position": tuple(self._pos),
            "altitude": self._pos[2],
            "target_altitude": self._pos[2],  # Tello holds altitude itself; nothing to report
            "orientation": orientation,
            "yaw_degrees": yaw_deg,
            "horizontal_speed": math.hypot(self.target_vx, self.target_vy),
            "vertical_speed": 0.0,
            "actual_vx": self.target_vx,
            "actual_vy": self.target_vy,
            "target_vx": self.target_vx,
            "target_vy": self.target_vy,
            "yaw_rate": self._yaw_rate,
            "flight_state": self.state,
            "collided": self.collided,
            "emergency": self.emergency,
            "min_obstacle_distance": self.min_obstacle_distance,
            "safety_override": self.safety_override,
        }

    # --- Main control cycle: call once per decision cycle ---

    def step(self):
        if self.state != "flying":
            return
        lr = _mps_to_rc(-self.target_vy)   # send_rc_control: +lr = right; move_right is -target_vy
        fb = _mps_to_rc(self.target_vx)
        ud = 0  # altitude changes not used by the hover/escape flight test; Tello holds altitude on its own
        # target_yaw_rate follows the rest of this codebase's convention
        # (positive = turn left / ccw, see turn_left/turn_right above and
        # interfaces/pybullet_drone.py); Tello's yaw_velocity is the
        # opposite sign (+100 = cw/right, per the module docstring), so flip it.
        yaw = _rate_to_rc(-self.target_yaw_rate)
        self._tello.send_rc_control(lr, fb, ud, yaw)

    def reset(self):
        if self.state == "flying":
            self.land()
        self._pos = [0.0, 0.0, 0.0]
        self._prev_yaw_rad = None
        self._yaw_rate = 0.0
        self.collided = False
        self.emergency = False
        self.takeoff()


def _euler_deg_to_quat(roll_deg, pitch_deg, yaw_deg):
    """Matches vision/optical_flow._quat_to_matrix and
    flybrain_controller._pitch_roll's convention: body x forward / y left /
    z up, R = Rz(yaw) @ Ry(pitch) @ Rx(roll). Same function as
    tello_neuron_test.py's euler_deg_to_quat - see its docstring for the
    caveat that the Tello's own pitch/roll sign convention needs empirical
    verification, which the desk test (props off) already exercised for
    the perception half; it matters more here since roll/pitch now
    actually change with real flight."""
    r, p_, y = (math.radians(roll_deg), math.radians(pitch_deg), math.radians(yaw_deg))
    cr, sr = math.cos(r / 2), math.sin(r / 2)
    cp, sp = math.cos(p_ / 2), math.sin(p_ / 2)
    cy, sy = math.cos(y / 2), math.sin(y / 2)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )
