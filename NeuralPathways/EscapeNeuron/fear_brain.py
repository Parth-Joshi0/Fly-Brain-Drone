"""
Fear reflex for the banana-eating drone: the real looming -> escape
circuit (LC4/LPLC2 -> DNp01 Giant Fiber, DNp03, DNp06) running on the
same Tello camera picture that tello_camera.py uses for banana detection.

    camera frame -> LoomingDetector (image expansion, rotation removed)
                 -> FlyBrainController (Brian2 connectome subprocess)
                 -> did the Giant Fiber just fire (a NEW escape)?

Same pipeline as Drone/tests/tello_escape_flight_test.py, packaged so
tello_camera.py can ask one question per frame. The brain decides WHEN
the fly gets scared; food_orbit.py decides HOW the drone reacts (a
clear back-off, then come back and keep eating). The brain's own dodge
command isn't used: on the real Tello it only pushed for ~0.17 s, far
too short to see.

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

from NeuralPathways.flybrain_controller import FlyBrainController, ESCAPE_STATE_THRESHOLD
from Drone.tello_drone import TelloDrone, RC_SPEED_SCALE
from NeuralPathways.EscapeNeuron.optical_flow import LoomingDetector, compute_flow, derotate_flow, grid_flow_strengths


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
# (tello_camera.py runs the banana AI every 3rd picture) caught 4/4.
BRAIN_STEP_S = 0.02

MAX_BRAIN_STEPS = 2

# Fire once the escape level crosses ESCAPE_STATE_THRESHOLD; ready to
# fire again once it has dropped below this.
REARM_LEVEL = 0.3

# Tello's 82.6 deg spec is diagonal; LoomingDetector wants vertical
VERTICAL_FOV = 55.6

# Frames to fill the looming detector's history before trusting it
WARMUP_FRAMES = 15

# After takeoff the climb itself looks like a huge loom - ignore the
# brain's output this long after start() is called.
ARM_GRACE_SECONDS = 2.0


class FearBrain:

    def __init__(self, tello, frame_read):

        self.looming = LoomingDetector(
            width=PROC_WIDTH,
            height=PROC_HEIGHT,
            fov=VERTICAL_FOV
        )

        # Builds the connectome network in a subprocess (a few seconds)
        self.brain = FlyBrainController(bounds=None)

        # Remember the last loom request decide() sent to the brain
        # process (and the escape level it answered with), so extra
        # brain steps can replay the same input - see MAX_BRAIN_STEPS.
        self._last_payload = None
        self._last_escape = 0.0

        send = self.brain._brain.request

        def remember(payload):
            result = send(payload)
            if "loom_left" in payload:
                self._last_payload = payload
                self._last_escape = result.get("escape", 0.0)
            return result

        self.brain._brain.request = remember

        # Only used for its telemetry (orientation, yaw rate) and its
        # dead-reckoned speed, which raises the brain's loom threshold
        # while flying forward. tello_camera.py sends the rc commands.
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


        state = self.drone.get_state()

        self.expansion = self.looming.update(gray, state["orientation"], dt)

        flow = {"left": 0.0, "right": 0.0, "top": 0.0, "bottom": 0.0, "center": 0.0}

        if self.prev_gray is not None:

            raw_flow = compute_flow(self.prev_gray, gray)

            flow = grid_flow_strengths(
                derotate_flow(raw_flow, state["yaw_rate"], dt)
            )

        self.prev_gray = gray

        flow.update({f"expansion_{side}": v for side, v in self.expansion.items()})

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


    def record_command(self, lr, fb):
        """Tell the dead reckoning what rc command was sent."""

        self.drone.target_vx = fb / RC_SPEED_SCALE

        self.drone.target_vy = -lr / RC_SPEED_SCALE


    def close(self):

        self.brain.close()
