"""
The real Tello's banana-seek flight (Drone/fly_tello.py) as a simulator
controller: BananaModel's detector finds the banana, NeuralPathways/
FoodNeuron/feeding_behaviour.py's FeedingBehaviour decides what to do about it,
and the fly brain (NeuralPathways/flybrain_controller.py) runs alongside as a
background fear reflex - same three pieces, same wiring as on the drone.

Same decide(flow, state) -> command dict / .state / .reset() surface as
ReflexController and FlyBrainController, so main.py swaps it in the same way,
plus one extra call: see(frame), each decision cycle before decide(), since
this is the only controller that looks at the picture itself rather than at
optic flow.

Two things differ from the Tello, on purpose:

  * The dodge itself. On the Tello the brain only decides WHEN to be scared
    and feeding_behaviour's slow back-off (0.67 m/s for 1 s) is the whole reaction,
    because the brain's own dodge command only pushed for ~0.17 s there (see
    NeuralPathways/EscapeNeuron/fear_brain.py). In the sim the brain's dodge
    is the tuned, tested one (Simulator/Tests/test_escape_sim.py), and the
    click-to-spawn test obstacle closes at 1.5 m/s - feeding_behaviour's back-off
    alone gets hit. So while the brain is in ESCAPE its dodge flies the
    drone; feeding_behaviour is still told it was scared and still runs, so it
    comes out of the dodge in WAIT and does the "is it safe? come back and
    finish eating" part exactly as on the drone.

    Only when feeding_behaviour takes the scare, though - it turns one down while
    already backed off (SCARED/WAIT) or done eating (DONE/LAND), and then the
    drone doesn't react at all, as on the Tello. That matters: DONE's slide
    to the right fired the Giant Fiber 1.5 s in on 4 of 4 runs (sideways
    motion past the banana card's edge reads as looming - the loom floor
    only discounts forward speed), and flying that dodge threw the drone
    2 m/s backwards right after it had finished eating.

  * Picture size. feeding_behaviour's pixel deadzones were tuned on the Tello's
    960x720 frame; the sim camera is 320x240. Detections are scaled up to
    TELLO_FRAME before feeding_behaviour sees them rather than retuning it -
    box-size ratios don't change, and the deadzones stay the same fraction
    of the picture.

FeedingBehaviour speaks Tello RC percentages. They're converted to this
project's m/s / rad/s command dict with the same assumed scales
Drone/tello_drone.py uses in the other direction.
"""

import time

from Drone.tello_drone import RC_SPEED_SCALE, RC_YAW_RATE_AT_100
from NeuralPathways.FoodNeuron.feeding_behaviour import EAT_SIZE_RATIO, FeedingBehaviour

TELLO_FRAME = (960, 720)

# Run the detector (~30-50 ms) every Nth decision cycle, as fly_tello.py
# does with the brain running (BANANA_EVERY_N_SCARED) - the banana barely
# moves between frames, and the looming detector needs the frame rate more.
DETECT_EVERY_N = 3

# feeding_behaviour's forward/back speeds are the Tello's (approach tops out at
# 15 rc = 0.25 m/s), slowed for a real room. The sim flies them this many
# times faster. Forward/back only: the scan's 45 deg turns count heading
# from the yaw commands (feeding_behaviour.DEG_PER_RC_S), and DONE's sideways
# slide is already enough to fire the Giant Fiber.
FORWARD_SPEED_SCALE = 2.5
# ...fading back to the Tello's own speed as the banana grows from this
# fraction of the size feeding_behaviour is flying in to eat at, to that size.
# At the full speed-up its 10 rc minimum is still ~0.4 m/s, and coming
# back after a scare (aiming for the bigger pre-scare size) it coasted to
# ~0.75 m from the card and the Giant Fiber fired on arrival.
FAST_UNTIL_SIZE_FRACTION = 0.5

# Added to the brain's loom floor while flying to the banana and eating it.
# The card growing in view, and the pitch as the drone brakes in front of
# it, read up to ~1.0 over the floor and fired the Giant Fiber on most
# arrivals. The thrown test box goes 2-19 over, so it still gets through
# (~0.1 s later).
APPROACH_LOOM_FLOOR_OFFSET = 0.8


class BananaSeekController:

    def __init__(self, detector, brain=None, clock=time.time):
        """detector: a BananaModel.banana_detector.BananaDetector (or anything with
        its detect()). brain: a FlyBrainController, or None for no fear
        reflex. clock: seconds, for feeding_behaviour's timers - pass sim time."""
        self.detector = detector
        self.brain = brain
        self.food = FeedingBehaviour(clock=clock)
        self.detections = []
        self.scares = 0           # Giant Fiber firings feeding_behaviour reacted to
        self.ignored_scares = 0   # ...and ones it turned down
        self.state = self.food.state
        self.escape_direction = "LEFT"
        self._frames = 0
        self._escaping = False
        self._dodging = False
        self._frame_size = (320, 240)

    def reset(self):
        self.food = FeedingBehaviour(clock=self.food._clock)
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
        if self.brain is not None:
            closing_in = self.food.state in ("APPROACH", "FEED")
            self.brain.loom_floor_offset = APPROACH_LOOM_FLOOR_OFFSET if closing_in else 0.0
        brain_cmd = self.brain.decide(flow, state) if self.brain is not None else None
        escaping = self.brain is not None and self.brain.state == "ESCAPE"
        if escaping and not self._escaping:
            # A new Giant Fiber firing - same hand-off as fly_tello.py's
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
            "forward_speed": rc.fb / RC_SPEED_SCALE * self._speed_scale(),
            "strafe_speed": -rc.lr / RC_SPEED_SCALE,              # +lr = right; +strafe = left
            "yaw_rate": -rc.yaw / 100.0 * RC_YAW_RATE_AT_100,      # +yaw = clockwise; +yaw_rate = left
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

    def _speed_scale(self):
        """FORWARD_SPEED_SCALE far from the banana, 1.0 at eating size."""
        goal = max(EAT_SIZE_RATIO, self.food.size_before_scare)
        start = goal * FAST_UNTIL_SIZE_FRACTION
        closeness = (self.food.last_box_ratio - start) / (goal - start)
        return FORWARD_SPEED_SCALE - (FORWARD_SPEED_SCALE - 1.0) * min(1.0, max(0.0, closeness))


class _ScaledDetection:
    """A Detection with its box in another frame size (the original is
    left alone - it's still drawn on the sim frame)."""

    def __init__(self, d, sx, sy):
        x1, y1, x2, y2 = d.box
        self.box = (x1 * sx, y1 * sy, x2 * sx, y2 * sy)
        self.label, self.det_conf, self.cls_conf = d.label, d.det_conf, d.cls_conf
