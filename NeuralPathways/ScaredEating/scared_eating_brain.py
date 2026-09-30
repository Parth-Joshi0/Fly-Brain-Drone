"""
Scared-while-eating brain for the banana-eating Tello.

One call per camera picture - "here's what I see, how should I move?":

    brain = ScaredEatingBrain(detector, tello, frame_read, scared=True)
    brain.start()                      # right after takeoff
    cmd = brain.step(frame_bgr)        # every picture
    tello.send_rc_control(cmd.lr, cmd.fb, cmd.ud, cmd.yaw)

Inside, two parts are wired together:

    picture ──► FearBrain (EscapeNeuron/fear_brain.py)
    │            looming detector -> real fly connectome (LC4/LPLC2 ->
    │            DNp01 Giant Fiber) -> "scared!" - but only while the
    │            drone is holding still (efference copy)
    │                 │ scared
    │                 ▼
    └──► banana AI ──► FoodOrbitBehaviour (FoodNeuron/food_orbit.py)
                       search -> fly to banana -> eat; scared -> back
                       straight off -> wait until clear -> come back

Without scared=True it's just the eating behaviour (no fly brain, so
no Brian2 needed). Drone/tello_camera.py does the rest: camera, safety,
screen, keys, flight log.
"""

from NeuralPathways.FoodNeuron.food_orbit import FoodOrbitBehaviour


# With the fly brain running, the banana AI (~40-50 ms) only runs on
# every Nth picture, so the looming detector + fly brain get more
# pictures per second - at ~11/s a fast hand jumps too far between
# pictures to be noticed (quick swipes caught: 0/4 -> 4/4). The banana
# barely moves between pictures, so its last position is reused.
BANANA_EVERY_N_SCARED = 3


class ScaredEatingBrain:

    def __init__(self, detector, tello=None, frame_read=None, scared=False):

        self.detector = detector

        self.behaviour = FoodOrbitBehaviour()

        self.fear = None

        if scared:

            # Imported here so plain eating doesn't need Brian2 set up
            from NeuralPathways.EscapeNeuron.fear_brain import FearBrain

            self.fear = FearBrain(tello, frame_read)

        self.banana_every = BANANA_EVERY_N_SCARED if scared else 1

        self.picture_count = 0

        # Latest banana detections (may be 1-2 pictures old)
        self.detections = []


    def start(self):
        """Call right after takeoff (or at the start of a dry run): the
        fly brain ignores the first ~2 s, since the climb looks like a loom."""

        if self.fear is not None:

            self.fear.start()


    def step(self, frame_bgr):
        """
        One camera picture (BGR, not drawn on yet) in, one rc command out
        (food_orbit.RCCommand: lr, fb, ud, yaw).
        """

        # Looming first, on the clean picture - boxes drawn on it would
        # look like motion. Giant Fiber fired -> back away (scare() is
        # ignored once done eating / landing).
        if self.fear is not None and self.fear.update(frame_bgr):

            self.behaviour.scare()


        if self.picture_count % self.banana_every == 0:

            self.detections = self.detector.detect(frame_bgr)

        self.picture_count += 1


        h, w = frame_bgr.shape[:2]

        cmd = self.behaviour.update(self.detections, w, h)


        # Efference copy + dead reckoning for the fly brain
        if self.fear is not None:

            self.fear.record_command(cmd.lr, cmd.fb, cmd.ud, cmd.yaw)

        return cmd


    def close(self):

        if self.fear is not None:

            self.fear.close()
