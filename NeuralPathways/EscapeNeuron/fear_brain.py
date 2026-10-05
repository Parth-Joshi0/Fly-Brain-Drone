"""
Fear reflex for the banana-eating drone: the real looming -> escape
circuit (LC4/LPLC2 -> DNp01 Giant Fiber, DNp03, DNp06) running on the
same Tello camera picture that fly_tello.py uses for banana detection.

    camera frame -> LoomingDetector (image expansion, rotation removed)
                 -> FlyBrainController (Brian2 connectome subprocess)
                 -> did the Giant Fiber just fire (a NEW escape)?

Same pipeline as Drone/tests/tello_escape_flight_test.py, packaged so
fly_tello.py can ask one question per frame. The brain decides WHEN
the fly gets scared; feeding_behaviour.py decides HOW the drone reacts (a
clear back-off, then come back and keep eating). The brain's own dodge
command isn't used: on the real Tello it only pushed for ~0.17 s, far
too short to see.

With stabilize=True the same brain also runs the DNg02 flight-motor
population (StabilizerNeuron/) and becomes the drone's yaw stabilizer,
in every behaviour state:

    camera frame -> flow -> derotate by the COMMANDED turn (efference copy)
                 -> signed_hemifield_flow rotation -> DNg02 left/right
                 -> steer -> dng02_yaw_rc, added to the behaviour's yaw

The efference copy is what lets it stay on while the drone turns on
purpose (the 360 scan, centring on the banana): the image motion the
behaviour's own yaw command should cause is subtracted before DNg02
sees it, so only rotation nobody asked for - drift, a bump, wind -
gets corrected. See EFFERENCE_PIXELS_PER_RADIAN.

The brain needs Brian2, which lives in its own venv (.venv-brain) so it
can't disturb the banana detector's packages. FLYBRAIN_PYTHON is pointed
at it here unless the caller already set it.
"""

import os
import time
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

_BRAIN_PYTHON = REPO_ROOT / ".venv-brain" / "bin" / "python"

if _BRAIN_PYTHON.exists():
    os.environ.setdefault("FLYBRAIN_PYTHON", str(_BRAIN_PYTHON))

from NeuralPathways.flybrain_controller import FlyBrainController, ESCAPE_STATE_THRESHOLD, DNG02_YAW_AUTHORITY
from Drone.tello_drone import TelloDrone, RC_SPEED_SCALE, RC_YAW_RATE_AT_100, _rate_to_rc
from NeuralPathways.EscapeNeuron.optical_flow import (LoomingDetector, compute_flow, derotate_flow,
                                                      grid_flow_strengths, signed_hemifield_flow)


# Same values as Drone/tests/tello_escape_flight_test.py
PROC_WIDTH, PROC_HEIGHT = 320, 240

# The brain simulates BRAIN_STEP_S of neuron time per request, but our
# loop only manages ~11-16 pictures/s, so with one request per picture
# the neurons lived well under real time and a hand swipe was over
# before the Giant Fiber could build up (flight 15:47: escape peaked at
# 0.56-0.57, needs 0.6). So each picture gets as many brain steps as
# real time passed - but only up to 2: each step costs ~13 ms, and a
# slower camera loop hurts more (a fast hand jumps too far between
# pictures for the optical flow to follow). Tested on quick 0.35 s
# swipes: 4 steps at 11 pictures/s caught 0/4, 2 steps at 16/s
# (fly_tello.py runs the banana AI every 3rd picture) caught 4/4.
BRAIN_STEP_S = 0.02

MAX_BRAIN_STEPS = 2

# Fire once the escape level crosses ESCAPE_STATE_THRESHOLD; ready to
# fire again once it has dropped below this.
REARM_LEVEL = 0.3

# EFFERENCE COPY: flying forward / turning / going up-down makes the
# whole scene expand, which the looming neurons can't tell from a
# real threat - in flight 16:44 all 4 scares came from the drone's own
# movement (escape level up to 1.0, as strong as a real hand wave).
# Real flies ignore what their eyes see during their own deliberate
# movements; so do we: while any rc command is bigger than this, and
# for SELF_MOTION_HOLD after (the looming detector smooths over a few
# pictures), scares are ignored and the trigger disarmed.
SELF_MOTION_RC = 8

SELF_MOTION_HOLD = 0.6

# Ignore the bottom part of the picture for looming - feet and things
# on the floor pass through there; a hand waved at the drone is higher
IGNORE_BOTTOM_FRACTION = 1 / 3

# Tello's 82.6 deg spec is diagonal; LoomingDetector wants vertical
VERTICAL_FOV = 55.6

# Frames to fill the looming detector's history before trusting it
WARMUP_FRAMES = 15

# After takeoff the climb itself looks like a huge loom - ignore the
# brain's output this long after start() is called.
ARM_GRACE_SECONDS = 2.0


# --- DNg02 stabilizer (stabilize=True only) ---

# EFFERENCE COPY: px of horizontal flow per radian of the drone's own
# COMMANDED turn, subtracted before DNg02 sees the flow. The Tello's real
# figure isn't pinned down - 106-184 measured at flight yaw rates, a
# calibration lower bound of 208 (dng02_calibrate3.log) - and this is
# deliberately NOT the measured-rate derotation tello_optomotor_flight_test.py
# uses, because here a wrong value can't flip the loop's sign: it only
# multiplies the commanded part, so with the true figure k the drone
# settles at (this / k) x the turn rate the behaviour asked for (0.75-1.3x
# over the measured range) instead of fighting the turn. Uncommanded
# rotation always reaches DNg02 at full strength, with the right sign.
EFFERENCE_PIXELS_PER_RADIAN = 140.0

# rad/s of yaw at |steer| = 1. tello_optomotor_flight_test.py flew 0.6
# (opto_fly2.log: corr -0.77, no oscillation), but derotated by the MEASURED
# rate at 75 px/rad, which left only (k - 75) px/rad of a drift visible to
# DNg02. Without that measured-rate cancel all k px/rad reach it, ~2x the
# loop gain, so this is 0.6 x (140 - 75) / 140 to keep the flown loop gain.
# Unflown in this form - check a hover in the flight log before raising it.
DNG02_TELLO_YAW_GAIN = 0.3


class FearBrain:

    def __init__(self, tello, frame_read, stabilize=False):

        self.looming = LoomingDetector(
            width=PROC_WIDTH,
            height=PROC_HEIGHT,
            fov=VERTICAL_FOV
        )

        self.stabilize = stabilize

        # Builds the connectome network in a subprocess (a few seconds);
        # optomotor=True adds the DNg02 population to the same network
        self.brain = FlyBrainController(bounds=None, optomotor=stabilize)

        # Remember the last loom request decide() sent to the brain
        # process (and the escape level it answered with), so extra
        # brain steps can replay the same input - see MAX_BRAIN_STEPS.
        self._last_payload = None
        self._last_escape = 0.0

        send = self.brain._brain.request

        def remember(payload):
            t0 = time.perf_counter()
            result = send(payload)
            self.brain_ms += (time.perf_counter() - t0) * 1000
            if "loom_left" in payload:
                self._last_payload = payload
                self._last_escape = result.get("escape", 0.0)
            # Replayed steps advance DNg02 too - keep its latest answer
            if "dng02" in result:
                self.brain.dng02 = result["dng02"]
            return result

        self.brain._brain.request = remember

        # Only used for its telemetry (orientation, yaw rate) and its
        # dead-reckoned speed, which raises the brain's loom threshold
        # while flying forward. fly_tello.py sends the rc commands.
        self.drone = TelloDrone(tello, frame_read, PROC_WIDTH, PROC_HEIGHT)

        self.prev_gray = None
        self.last_time = None
        self.frames = 0
        self.start_time = None
        self.armed = False

        # For the HUD / flight log
        self.expansion = {"left": 0.0, "center": 0.0, "right": 0.0}
        self.escaping = False
        self.escape_direction = ""

        # Times the Giant Fiber has fired this flight
        self.scares = 0

        # For the flight log
        self.escape_level = 0.0
        self.wobble_floor = 0.0
        self.loom_in = (0.0, 0.0)
        self._ready = True

        # Efference copy (see SELF_MOTION_RC)
        self.last_self_motion = 0.0
        self.self_moving = False

        # DNg02 stabilizer: the behaviour's own yaw (rc) from the last
        # command - its efference copy, see EFFERENCE_PIXELS_PER_RADIAN -
        # and what DNg02 wants added to the next one
        self.intended_yaw_rc = 0
        self.dng02_rotation = 0.0
        self.dng02_yaw_rc = 0

        # ms spent in the brain subprocess for the latest picture -
        # measured, because the loop-rate tuning (MAX_BRAIN_STEPS,
        # the banana AI's background thread in ScaredEating/) was done on one particular laptop
        self.brain_ms = 0.0


    def start(self):
        """Call right after takeoff (or at the start of a dry run)."""

        self.start_time = time.time()
        self.armed = False


    def update(self, frame_bgr):
        """
        Feed one camera frame (BGR, any size). Returns True on the frame
        the brain STARTS a new escape (Giant Fiber fired), else False.
        """

        now = time.time()

        dt = 1.0 / 20.0 if self.last_time is None else max(1e-3, now - self.last_time)

        self.last_time = now


        small = cv2.resize(
            frame_bgr,
            (PROC_WIDTH, PROC_HEIGHT),
            interpolation=cv2.INTER_AREA
        )

        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        # Blank out the floor area (see IGNORE_BOTTOM_FRACTION) - a flat
        # grey patch has no motion, so nothing there can look like a loom
        cut = int(PROC_HEIGHT * (1 - IGNORE_BOTTOM_FRACTION))

        gray[cut:, :] = 128


        state = self.drone.get_state()

        self.expansion = self.looming.update(gray, state["orientation"], dt)

        flow = {"left": 0.0, "right": 0.0, "top": 0.0, "bottom": 0.0, "center": 0.0}

        rotation = translation = 0.0

        if self.prev_gray is not None:

            raw_flow = compute_flow(self.prev_gray, gray)

            flow = grid_flow_strengths(
                derotate_flow(raw_flow, state["yaw_rate"], dt)
            )

            if self.stabilize:

                # DNg02 sees the flow minus what our own commanded turn
                # should have caused (efference copy). rc yaw is
                # +right; derotate_flow wants this project's +left.
                commanded_rate = -self.intended_yaw_rc / 100 * RC_YAW_RATE_AT_100

                hemi = signed_hemifield_flow(
                    derotate_flow(raw_flow, commanded_rate, dt,
                                  pixels_per_radian=EFFERENCE_PIXELS_PER_RADIAN)
                )

                rotation, translation = hemi["rotation"], hemi["translation"]

        self.prev_gray = gray

        flow.update({f"expansion_{side}": v for side, v in self.expansion.items()})

        # Only read by decide() with optomotor on (zero drive otherwise)
        flow["rotation"], flow["translation"] = rotation, translation

        self.dng02_rotation = rotation

        self.dng02_yaw_rc = 0

        self.brain_ms = 0.0

        self.frames += 1


        # Arm once the detector has history and the takeoff has settled
        if not self.armed:

            settled = (
                self.start_time is not None
                and now - self.start_time >= ARM_GRACE_SECONDS
            )

            if self.frames >= WARMUP_FRAMES and settled:

                self.looming.reset()

                self.prev_gray = None

                self.armed = True

            self.escaping = False

            return False


        # One brain step through decide() (it also updates the wobble
        # floor from this frame's tilt)...
        self.brain.decide(flow, state)

        level = self._last_escape

        # ...plus extra steps with the same input, so the neurons live
        # as much time as actually passed
        steps = max(1, min(MAX_BRAIN_STEPS, round(dt / BRAIN_STEP_S)))

        for _ in range(steps - 1):

            self.brain._brain.request(self._last_payload)

            level = max(level, self._last_escape)


        self.escape_level = level

        self.wobble_floor = self.brain._rotation_floor

        self.loom_in = (
            self._last_payload["loom_left"],
            self._last_payload["loom_right"],
        )


        # DNg02 stabilizer - same term flybrain_controller's
        # dng02_yaw_rate() adds (steer opposes uncommanded rotation,
        # faded out as the escape level rises), at the Tello's own gain.
        # Runs whether or not we're moving on purpose: the efference
        # copy already took our own turn out of what it sees.
        if self.stabilize:

            yaw_rate = -DNG02_TELLO_YAW_GAIN * self.brain.dng02.get("steer", 0.0) * (1.0 - level)

            yaw_rate = max(-DNG02_YAW_AUTHORITY, min(DNG02_YAW_AUTHORITY, yaw_rate))

            # +left rad/s -> Tello's +right rc
            self.dng02_yaw_rc = _rate_to_rc(-yaw_rate)


        # Moving on purpose? Then what the eyes see is our own motion.
        self.self_moving = now - self.last_self_motion < SELF_MOTION_HOLD

        if self.self_moving:

            # Disarmed until the (self-made) escape level has died down
            self._ready = False

            self.escaping = False

            return False


        if level >= ESCAPE_STATE_THRESHOLD and self._ready:

            # Giant Fiber fired - react straight away
            self._ready = False

            self.escaping = True

            # Away from the side that loomed more
            self.escape_direction = "RIGHT" if self.loom_in[0] >= self.loom_in[1] else "LEFT"

            self.scares += 1

            return True


        if level < REARM_LEVEL:

            self._ready = True

            self.escaping = False

        return False


    def record_command(self, lr, fb, ud=0, yaw=0, intended_yaw=None):
        """Tell the brain what rc command was just sent: for dead
        reckoning, and as the efference copy (see SELF_MOTION_RC).
        intended_yaw: the behaviour's yaw before DNg02's correction was
        added - the turn DNg02 shouldn't fight (defaults to yaw)."""

        self.intended_yaw_rc = yaw if intended_yaw is None else intended_yaw

        self.drone.target_vx = fb / RC_SPEED_SCALE

        self.drone.target_vy = -lr / RC_SPEED_SCALE

        if max(abs(lr), abs(fb), abs(ud), abs(yaw)) > SELF_MOTION_RC:

            self.last_self_motion = time.time()


    def close(self):

        self.brain.close()
