"""
Food-orbit behaviour - when the drone spots food it likes, it circles
around it while keeping the camera pointed at it (like a fly circling
fruit it is interested in).

The Tello camera faces forward, so "circling" means ORBITING the food:
strafe sideways at a steady speed while yawing to keep the food centred
in frame, and moving forward/back to keep its on-screen size (i.e. the
orbit radius) roughly constant.

This class never talks to the drone - it turns a list of detections
(food_detection.liveDetect.Detection) into an rc command, so the logic
can be tested without flying:

    behaviour = FoodOrbitBehaviour()
    cmd = behaviour.update(detections, frame_width, frame_height)
    tello.send_rc_control(cmd.lr, cmd.fb, cmd.ud, cmd.yaw)

States:

    SEARCH  hover, slowly yawing to look around
    ORBIT   liked food confirmed - circle it
    REST    orbit finished (or food lost) - hover still for a moment so
            it doesn't immediately re-trigger on the same food
"""

import time
from collections import namedtuple

RcCommand = namedtuple('RcCommand', ['lr', 'fb', 'ud', 'yaw'])
HOVER = RcCommand(0, 0, 0, 0)

SEARCH = 'SEARCH'
ORBIT = 'ORBIT'
REST = 'REST'

# --- Tune freely ---------------------------------------------------------
LIKED_LABELS = {'overripe'}
MIN_CLASSIFIER_CONF = 0.6   # ignore low-confidence ripeness predictions
CONFIRM_FRAMES = 3          # consecutive frames of liked food before orbiting
LOST_TIMEOUT_S = 1.5        # food out of sight this long -> give up orbit
ORBIT_DURATION_S = 20.0     # how long to circle before resting
REST_DURATION_S = 5.0       # hover still after an orbit

# rc values are Tello units, -100..100
SEARCH_YAW = 15             # slow look-around rotation while searching
ORBIT_LR = 20               # sideways speed while orbiting (+ = right)
YAW_FEEDFORWARD = -18       # yaw needed to counter the strafe (opposite sign to ORBIT_LR)
YAW_GAIN = 60               # yaw per unit of horizontal centring error
TARGET_HEIGHT_FRAC = 0.30   # desired food box height / frame height (sets orbit radius)
FB_GAIN = 80                # forward/back per unit of size error
MAX_RC = 35                 # hard cap on any rc value


def _clip(value, limit=MAX_RC):
    return int(max(-limit, min(limit, value)))


class FoodOrbitBehaviour:
    def __init__(self, liked_labels=LIKED_LABELS, clock=time.monotonic):
        self.liked_labels = set(liked_labels)
        self.clock = clock
        self.state = SEARCH
        self._confirm_count = 0
        self._state_started = self.clock()
        self._last_seen = None
        self._target_center = None

    def _set_state(self, state):
        self.state = state
        self._state_started = self.clock()
        self._confirm_count = 0
        if state != ORBIT:
            self._target_center = None

    def _liked(self, detections):
        return [d for d in detections
                if d.label in self.liked_labels and d.cls_conf >= MIN_CLASSIFIER_CONF]

    def _pick_target(self, liked):
        """Stick with the food we're already circling; otherwise take the most confident."""
        if self._target_center is not None:
            tx, ty = self._target_center
            return min(liked, key=lambda d: ((d.box[0] + d.box[2]) / 2 - tx) ** 2
                                            + ((d.box[1] + d.box[3]) / 2 - ty) ** 2)
        return max(liked, key=lambda d: d.det_conf * d.cls_conf)

    def update(self, detections, frame_width, frame_height):
        """Advance the state machine one frame and return the RcCommand to send."""
        now = self.clock()
        liked = self._liked(detections)

        if self.state == REST:
            if now - self._state_started >= REST_DURATION_S:
                self._set_state(SEARCH)
            return HOVER

        if self.state == SEARCH:
            if liked:
                self._confirm_count += 1
                if self._confirm_count >= CONFIRM_FRAMES:
                    self._set_state(ORBIT)
                    self._last_seen = now
                    return self._orbit_command(self._pick_target(liked), frame_width, frame_height)
                return HOVER  # hold still while confirming so the food stays in view
            self._confirm_count = 0
            return RcCommand(0, 0, 0, SEARCH_YAW)

        # ORBIT
        if now - self._state_started >= ORBIT_DURATION_S:
            self._set_state(REST)
            return HOVER
        if not liked:
            if now - self._last_seen >= LOST_TIMEOUT_S:
                self._set_state(REST)
            return HOVER  # pause the orbit and wait for the food to reappear
        self._last_seen = now
        return self._orbit_command(self._pick_target(liked), frame_width, frame_height)

    def _orbit_command(self, target, frame_width, frame_height):
        x1, y1, x2, y2 = target.box
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        self._target_center = (cx, cy)

        # -1 (far left) .. +1 (far right); positive yaw turns right
        x_err = (cx - frame_width / 2) / (frame_width / 2)
        yaw = YAW_FEEDFORWARD + YAW_GAIN * x_err

        # box too small -> food too far -> move forward
        size_err = TARGET_HEIGHT_FRAC - (y2 - y1) / frame_height
        fb = FB_GAIN * size_err

        return RcCommand(_clip(ORBIT_LR), _clip(fb), 0, _clip(yaw))
