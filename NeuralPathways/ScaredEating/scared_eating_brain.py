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
                       scan the room (360) -> fly to banana -> eat;
                       scared -> back straight off -> wait until
                       clear -> come back

While the drone holds still to look for the banana (scan pauses,
waiting after a scare), the banana AI uses a slower zoomed-in look
that sees about twice as far - see detect_zoomed().

Without scared=True it's just the eating behaviour (no fly brain, so
no Brian2 needed). Drone/tello_camera.py does the rest: camera, safety,
screen, keys, flight log.
"""

from BananaModel.liveDetect import Detection
from NeuralPathways.FoodNeuron.food_orbit import FoodOrbitBehaviour


# With the fly brain running, the banana AI (~40-50 ms) only runs on
# every Nth picture, so the looming detector + fly brain get more
# pictures per second - at ~11/s a fast hand jumps too far between
# pictures to be noticed (quick swipes caught: 0/4 -> 4/4). The banana
# barely moves between pictures, so its last position is reused.
BANANA_EVERY_N_SCARED = 3

# Zoomed-in look: as well as the whole picture, check 4 overlapping
# pieces this big (fraction of width/height), each shown to the banana
# AI at full size - so a small far-away banana is ~1.7x bigger to it.
# Tested on small bananas pasted into a room photo (24 each): filling
# 0.75% of the picture, found 3/24 -> 15/24; 1%: 7/24 -> 14/24; no
# wrong boxes. ~175 ms per picture, so only while holding still.
ZOOM_PIECE = 0.6

# Boxes from the pieces overlapping this much are the same banana
SAME_BANANA_OVERLAP = 0.3


def _overlap(a, b):
    """Intersection over union of two (x1, y1, x2, y2) boxes."""

    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def detect_zoomed(detector, frame_bgr):
    """Whole picture + 4 overlapping zoomed-in pieces; boxes in
    whole-picture pixels, duplicates (same banana seen twice) removed."""

    h, w = frame_bgr.shape[:2]

    ph, pw = int(h * ZOOM_PIECE), int(w * ZOOM_PIECE)

    found = list(detector.detect(frame_bgr))

    for y0 in (0, h - ph):

        for x0 in (0, w - pw):

            for d in detector.detect(frame_bgr[y0:y0 + ph, x0:x0 + pw]):

                x1, y1, x2, y2 = d.box

                found.append(Detection((x1 + x0, y1 + y0, x2 + x0, y2 + y0),
                                       d.det_conf, d.label, d.cls_conf))

    kept = []

    for d in sorted(found, key=lambda d: -d.det_conf):

        if all(_overlap(d.box, k.box) < SAME_BANANA_OVERLAP for k in kept):

            kept.append(d)

    return kept


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


    def step(self, frame_bgr, yaw_deg=None):
        """
        One camera picture (BGR, not drawn on yet) in, one rc command out
        (food_orbit.RCCommand: lr, fb, ud, yaw). yaw_deg: the drone's
        compass heading, if known - counts the scan's turns.
        """

        # Looming first, on the clean picture - boxes drawn on it would
        # look like motion. Giant Fiber fired -> back away (scare() is
        # ignored once done eating / landing).
        if self.fear is not None and self.fear.update(frame_bgr):

            self.behaviour.scare()


        if self.behaviour.wants_zoom:

            # Holding still to look for it: take the slow, careful look
            self.detections = detect_zoomed(self.detector, frame_bgr)

        elif self.picture_count % self.banana_every == 0:

            self.detections = self.detector.detect(frame_bgr)

        self.picture_count += 1


        h, w = frame_bgr.shape[:2]

        cmd = self.behaviour.update(self.detections, w, h, yaw_deg)


        # Efference copy + dead reckoning for the fly brain
        if self.fear is not None:

            self.fear.record_command(cmd.lr, cmd.fb, cmd.ud, cmd.yaw)

        return cmd


    def close(self):

        if self.fear is not None:

            self.fear.close()
