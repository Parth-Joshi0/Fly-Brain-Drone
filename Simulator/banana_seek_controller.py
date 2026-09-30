"""
The real Tello's banana-seek flight (Drone/tello_camera.py) as a simulator
controller: BananaModel's detector finds the banana, NeuralPathways/
FoodNeuron/food_orbit.py's FoodOrbitBehaviour decides what to do about it,
and the fly brain (NeuralPathways/flybrain_controller.py) runs alongside as a
background fear reflex - same three pieces, same wiring as on the drone.

Same decide(flow, state) -> command dict / .state / .reset() surface as
ReflexController and FlyBrainController, so main.py swaps it in the same way,
plus one extra call: see(frame), each decision cycle before decide(), since
this is the only controller that looks at the picture itself rather than at
optic flow.

Two things differ from the Tello, on purpose:

  * The dodge itself. On the Tello the brain only decides WHEN to be scared
    and food_orbit's slow back-off (0.67 m/s for 1 s) is the whole reaction,
    because the brain's own dodge command only pushed for ~0.17 s there (see
    NeuralPathways/EscapeNeuron/fear_brain.py). In the sim the brain's dodge
    is the tuned, tested one (Simulator/Tests/test_escape_sim.py), and the
    click-to-spawn test obstacle closes at 1.5 m/s - food_orbit's back-off
    alone gets hit. So while the brain is in ESCAPE its dodge flies the
    drone; food_orbit is still told it was scared and still runs, so it
    comes out of the dodge in WAIT and does the "is it safe? come back and
    finish eating" part exactly as on the drone.

    Only when food_orbit takes the scare, though - it turns one down while
    already backed off (SCARED/WAIT) or done eating (DONE/LAND), and then the
    drone doesn't react at all, as on the Tello. That matters: DONE's slide
    to the right fired the Giant Fiber 1.5 s in on 4 of 4 runs (sideways
    motion past the banana card's edge reads as looming - the loom floor
    only discounts forward speed), and flying that dodge threw the drone
    2 m/s backwards right after it had finished eating.

  * Picture size. food_orbit's pixel deadzones were tuned on the Tello's
    960x720 frame; the sim camera is 320x240. Detections are scaled up to
    TELLO_FRAME before food_orbit sees them rather than retuning it -
    box-size ratios don't change, and the deadzones stay the same fraction
    of the picture.

FoodOrbitBehaviour speaks Tello RC percentages. They're converted to this
project's m/s / rad/s command dict with the same assumed scales
Drone/tello_drone.py uses in the other direction.
"""

import time

from Drone.tello_drone import RC_SPEED_SCALE
from NeuralPathways.FoodNeuron.food_orbit import FoodOrbitBehaviour

TELLO_FRAME = (960, 720)
YAW_RATE_AT_100_RC = 1.5    # rad/s - Drone/tello_drone._rate_to_rc's assumption

# Run the detector (~30-50 ms) every Nth decision cycle, as tello_camera.py
# does with the brain running (BANANA_EVERY_N_SCARED) - the banana barely
# moves between frames, and the looming detector needs the frame rate more.
DETECT_EVERY_N = 3


class BananaSeekController:

    def __init__(self, detector, brain=None, clock=time.time):
        """detector: a BananaModel.liveDetect.BananaDetector (or anything with
        its detect()). brain: a FlyBrainController, or None for no fear
        reflex. clock: seconds, for food_orbit's timers - pass sim time."""
        self.detector = detector
        self.brain = brain
        self.food = FoodOrbitBehaviour(clock=clock)
        self.detections = []
        self.scares = 0           # Giant Fiber firings food_orbit reacted to
        self.ignored_scares = 0   # ...and ones it turned down
        self.state = self.food.state
        self.escape_direction = "LEFT"
        self._frames = 0
        self._escaping = False
        self._dodging = False
        self._frame_size = (320, 240)

    def reset(self):
        self.food = FoodOrbitBehaviour(clock=self.food._clock)
        self.detections = []
        self._escaping = False
        self._dodging = False
        self.state = self.food.state
        if self.brain is not None:
            self.brain.reset()

    def close(self):
        if self.brain is not None:
            self.brain.close()

    def see(self, frame):
        """Feed this cycle's camera frame (BGR). Runs the detector every
        DETECT_EVERY_N calls; in between, the last detections stand."""
        h, w = frame.shape[:2]
        self._frame_size = (w, h)
        if self._frames % DETECT_EVERY_N == 0:
            self.detections = self.detector.detect(frame)
        self._frames += 1

    def decide(self, flow, state=None):
        brain_cmd = self.brain.decide(flow, state) if self.brain is not None else None
        escaping = self.brain is not None and self.brain.state == "ESCAPE"
        if escaping and not self._escaping:
            # A new Giant Fiber firing - same hand-off as tello_camera.py's
            # fear.update() -> behaviour.scare()
            self._dodging = self.food.scare()
            if self._dodging:
                self.scares += 1
            else:
                self.ignored_scares += 1
        self._escaping = escaping

        w, h = self._frame_size
        sx, sy = TELLO_FRAME[0] / w, TELLO_FRAME[1] / h
        rc = self.food.update([_ScaledDetection(d, sx, sy) for d in self.detections], *TELLO_FRAME)

        if escaping and self._dodging:
            self.state = "ESCAPE"
            self.escape_direction = self.brain.escape_direction
            return brain_cmd

        self.state = self.food.state
        cmd = {
            "forward_speed": rc.fb / RC_SPEED_SCALE,
            "strafe_speed": -rc.lr / RC_SPEED_SCALE,              # +lr = right; +strafe = left
            "yaw_rate": -rc.yaw / 100.0 * YAW_RATE_AT_100_RC,      # +yaw = clockwise; +yaw_rate = left
            "altitude_delta": float(rc.ud),
            "hover": False,
            "land": self.food.should_land,
            "reset": False,
            "pressed_direction": "(banana)",
        }
        if self.brain is not None:
            # DNg02 course stabilization on top (0.0 unless the brain was
            # built with optomotor=True). Not its thrust term: that holds a
            # cruising image speed, and this behaviour mostly hovers.
            cmd["yaw_rate"] += self.brain.dng02_yaw_rate()
        return cmd


class _ScaledDetection:
    """A Detection with its box in another frame size (the original is
    left alone - it's still drawn on the sim frame)."""

    def __init__(self, d, sx, sy):
        x1, y1, x2, y2 = d.box
        self.box = (x1 * sx, y1 * sy, x2 * sx, y2 * sy)
        self.label, self.det_conf, self.cls_conf = d.label, d.det_conf, d.cls_conf
