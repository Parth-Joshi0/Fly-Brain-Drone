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
    └──► banana AI ──► FeedingBehaviour (FoodNeuron/feeding_behaviour.py)
                       scan the room (360) -> fly to banana -> eat;
                       scared -> back straight off -> wait until
                       clear -> come back

While the drone holds still to look for the banana (scan pauses,
waiting after a scare), the banana AI uses a slower zoomed-in look
that sees about twice as far - see detect_zoomed().

stabilize=True also runs the fly brain's DNg02 flight-motor population
(StabilizerNeuron/) as a yaw stabilizer in every state: its correction
is added to whatever yaw the eating behaviour asks for. It only fights
rotation the behaviour didn't ask for - see fear_brain.py.

Without scared=True or stabilize=True it's just the eating behaviour
(no fly brain, so no Brian2 needed). Drone/fly_tello.py does the
rest: camera, safety, screen, keys, flight log.
"""

import threading
import time

from BananaModel.banana_detector import Detection
from NeuralPathways.FoodNeuron.feeding_behaviour import FeedingBehaviour, RCCommand, clamp


# The banana AI runs on its own thread (_BananaWorker), so the camera +
# fly brain loop never waits for it: the fly brain needs ~15+ pictures/s
# to catch a hand (quick swipes caught: 0/4 at ~11/s, 4/4 at ~16/s), and
# the sharp-eyes zoomed look takes ~0.4 s per picture - run inline, it
# dropped the loop to ~2 pictures/s (dry run 10-04 15:30). Meanwhile the
# latest banana result is reused; it barely moves between pictures.

# Results older than this are dropped rather than steered by
MAX_DETECTION_AGE = 0.8

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


class _BananaWorker:
    """Runs the banana AI on a background thread, one picture at a time:
    hand it the newest picture whenever it's free, read back the latest
    result whenever you like."""

    def __init__(self, detector, sharp_detector):

        self.detector = detector

        self.sharp_detector = sharp_detector

        self._lock = threading.Lock()

        self._wake = threading.Event()

        self._job = None

        self._busy = False

        self._stopping = False

        # Latest result: detections, a counter that goes up with each new
        # result, and when it finished
        self.result = []

        self.result_id = 0

        self.result_time = 0.0

        self._thread = threading.Thread(target=self._run, daemon=True)

        self._thread.start()


    def submit(self, frame_bgr, sharp):
        """Start on this picture if free (returns False if still busy)."""

        with self._lock:

            if self._busy:
                return False

            self._busy = True

            self._job = (frame_bgr, sharp)

        self._wake.set()

        return True


    def _run(self):

        while True:

            self._wake.wait()

            self._wake.clear()

            if self._stopping:
                return

            frame_bgr, sharp = self._job

            try:

                if sharp:
                    found = detect_zoomed(self.sharp_detector, frame_bgr)
                else:
                    found = self.detector.detect(frame_bgr)

            except Exception as err:

                print("Banana AI error:", err)

                found = []

            with self._lock:

                self.result = found

                self.result_id += 1

                self.result_time = time.time()

                self._busy = False


    def close(self):

        self._stopping = True

        self._wake.set()


class ScaredEatingBrain:

    def __init__(self, detector, tello=None, frame_read=None, scared=False,
                 stabilize=False, sharp_detector=None):

        self.detector = detector

        # Slower, more accurate banana AI for the zoomed-in look while
        # holding still (fly_tello.py passes YOLOv8 small); falls back
        # to the normal one
        self.sharp_detector = sharp_detector or detector

        self.behaviour = FeedingBehaviour()

        self.scared = scared

        self.stabilize = stabilize

        self.fear = None

        if scared or stabilize:

            # Imported here so plain eating doesn't need Brian2 set up
            from NeuralPathways.EscapeNeuron.fear_brain import FearBrain

            self.fear = FearBrain(tello, frame_read, stabilize=stabilize)

        self.banana = _BananaWorker(self.detector, self.sharp_detector)

        self._seen_result_id = 0

        # Latest banana detections (may be a few pictures old)
        self.detections = []


    def start(self):
        """Call right after takeoff (or at the start of a dry run): the
        fly brain ignores the first ~2 s, since the climb looks like a loom."""

        if self.fear is not None:

            self.fear.start()


    def step(self, frame_bgr, yaw_deg=None, flying=True):
        """
        One camera picture (BGR, not drawn on yet) in, one rc command out
        (feeding_behaviour.RCCommand: lr, fb, ud, yaw). yaw_deg: the drone's
        compass heading, if known - counts the scan's turns. flying:
        False in a dry run, so the fly brain knows the drone isn't
        actually doing the moves it's asked to (and doesn't ignore waves).
        """

        # Looming first, on the clean picture - boxes drawn on it would
        # look like motion. Giant Fiber fired -> back away (scare() is
        # ignored once done eating / landing). This also runs DNg02.
        if self.fear is not None and self.fear.update(frame_bgr) and self.scared:

            self.behaviour.scare()


        # Banana AI on its own thread: give it this picture if it's free
        # (a copy - fly_tello.py draws on the frame afterwards). Holding
        # still to look for it -> the slow, careful sharp-eyes look.
        self.banana.submit(frame_bgr.copy(), sharp=self.behaviour.wants_zoom)

        fresh = self.banana.result_id != self._seen_result_id

        self._seen_result_id = self.banana.result_id

        if time.time() - self.banana.result_time <= MAX_DETECTION_AGE:
            self.detections = self.banana.result
        else:
            self.detections = []


        h, w = frame_bgr.shape[:2]

        # Lets the behaviour tell when the coast is clear to come back
        # (nothing looming) - see feeding_behaviour.py's COME BACK FAST
        if self.fear is not None:
            self.behaviour.escape_level = self.fear.escape_level

        cmd = self.behaviour.update(self.detections, w, h, yaw_deg, fresh=fresh)

        intended_yaw = cmd.yaw


        # DNg02 stabilizer on top of whatever the behaviour wants
        if self.stabilize:

            cmd = RCCommand(
                lr=cmd.lr,
                fb=cmd.fb,
                ud=cmd.ud,
                yaw=int(clamp(cmd.yaw + self.fear.dng02_yaw_rc, -100, 100))
            )


        # Efference copy + dead reckoning for the fly brain. The whole
        # command counts as self-motion for the looming circuit (a DNg02
        # turn sweeps the scene too); only the behaviour's own yaw is
        # what DNg02 must not fight.
        if self.fear is not None:

            if flying:

                self.fear.record_command(cmd.lr, cmd.fb, cmd.ud, cmd.yaw, intended_yaw=intended_yaw)

            else:

                # Dry run: nothing actually moves
                self.fear.record_command(0, 0, 0, 0, intended_yaw=0)

        return cmd


    def close(self):

        self.banana.close()

        if self.fear is not None:

            self.fear.close()
