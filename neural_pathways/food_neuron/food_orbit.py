"""
Fly-inspired food behaviour.

    SEARCH   turn a little, hover still, repeat - looking for a banana
    -> FEED  banana seen: stay where we are, keep it in the picture, eat
    -> DONE  full: slide a little right, hover
    -> LAND  5 s after eating (tello_camera.py lands when should_land)

Any banana is food. Hunger only goes down while the banana is
actually in view. The drone never flies closer - it eats from
where it first saw the banana, only making small moves to keep
the banana in the picture.
"""

import time
from dataclasses import dataclass


# ============================================================
# FOOD
# ============================================================

BANANA_LABELS = {
    "ripe",
    "overripe",
    "freshripe",
    "rotten",
    "unripe",
    "freshunripe",
}


# ============================================================
# DETECTION
# ============================================================

MIN_DETECTION_CONFIDENCE = 0.20
MIN_CLASSIFIER_CONFIDENCE = 0.10

# Banana out of sight this long while eating -> go back to searching.
# Shorter gaps (detection flicker) just pause eating.
TARGET_LOST_GRACE_TIME = 1.0


# ============================================================
# HUNGER
# ============================================================

# 100 = very hungry
# 0 = completely full

STARTING_HUNGER = 100.0

FULL_THRESHOLD = 10.0

# Hunger points lost per second while eating
# (100 -> 10 takes about 6 seconds)
FEED_RATE = 15.0


# ============================================================
# SEARCH
# ============================================================

# Turn a little, then hover still. Non-stop turning made the
# drone curve off to the right.
SEARCH_YAW_SPEED = 8

SEARCH_TURN_TIME = 1.0

SEARCH_PAUSE_TIME = 2.0


# ============================================================
# KEEP BANANA IN THE PICTURE
# ============================================================

# Only move once the banana drifts this far (pixels) from the
# middle of the picture - small wobbles are ignored.
CENTER_DEADZONE = 120

VERTICAL_DEADZONE = 100

# Slide left/right to bring it back to the middle
LR_GAIN = 0.06

MAX_LR = 20

# Move up/down to bring it back to the middle
UD_GAIN = 0.08

MAX_UD = 20

# If the drone drifts so close that the banana fills more than
# this much of the screen, back off a little.
TOO_CLOSE_RATIO = 0.25

BACK_OFF_SPEED = 10

COMMAND_SMOOTHING = 0.35


# ============================================================
# DONE EATING
# ============================================================

# After eating: slide a little right, hover, then land.

DONE_VEER_SPEED = 15

DONE_VEER_TIME = 1.5

LAND_AFTER_EATING = 5.0


# ============================================================
# RC COMMAND
# ============================================================

@dataclass
class RCCommand:
    lr: int = 0
    fb: int = 0
    ud: int = 0
    yaw: int = 0


def clamp(value, minimum, maximum):
    return max(
        minimum,
        min(maximum, value)
    )


def smooth(old_value, new_value, alpha):
    return (
        old_value * (1 - alpha)
        + new_value * alpha
    )


# ============================================================
# FOOD BRAIN
# ============================================================

class FoodOrbitBehaviour:

    def __init__(self):

        self.state = "SEARCH"

        # Hunger
        self.hunger = STARTING_HUNGER
        self.last_update_time = time.time()

        # Target
        self.current_target = None
        self.last_target_label = None
        self.last_box_ratio = 0.0
        self.last_target_time = 0.0

        # Timers
        self.search_start_time = time.time()
        self.done_start_time = None

        # tello_camera.py lands when this becomes True
        self.should_land = False

        # Smooth commands
        self.prev_lr = 0
        self.prev_fb = 0
        self.prev_ud = 0
        self.prev_yaw = 0


    # ========================================================
    # SMOOTH MOVEMENT
    # ========================================================

    def _smooth_command(
        self,
        lr,
        fb,
        ud,
        yaw
    ):

        self.prev_lr = smooth(self.prev_lr, lr, COMMAND_SMOOTHING)
        self.prev_fb = smooth(self.prev_fb, fb, COMMAND_SMOOTHING)
        self.prev_ud = smooth(self.prev_ud, ud, COMMAND_SMOOTHING)
        self.prev_yaw = smooth(self.prev_yaw, yaw, COMMAND_SMOOTHING)

        return RCCommand(
            lr=int(self.prev_lr),
            fb=int(self.prev_fb),
            ud=int(self.prev_ud),
            yaw=int(self.prev_yaw)
        )


    def _hover(self):

        return self._smooth_command(0, 0, 0, 0)


    # ========================================================
    # CHOOSE BANANA
    # ========================================================

    def _choose_target(
        self,
        detections
    ):

        valid = [
            d for d in detections
            if d.label.lower() in BANANA_LABELS
            and d.det_conf >= MIN_DETECTION_CONFIDENCE
            and d.cls_conf >= MIN_CLASSIFIER_CONFIDENCE
        ]

        if not valid:
            return None


        # Biggest, most confident banana
        def score(d):

            x1, y1, x2, y2 = d.box

            area = max(1, (x2 - x1) * (y2 - y1))

            return area * d.det_conf

        return max(valid, key=score)


    # ========================================================
    # BANANA SIZE
    # ========================================================

    def _box_ratio(
        self,
        target,
        frame_width,
        frame_height
    ):

        x1, y1, x2, y2 = target.box

        banana_area = max(1, (x2 - x1) * (y2 - y1))

        return banana_area / (frame_width * frame_height)


    # ========================================================
    # KEEP BANANA IN THE PICTURE
    # ========================================================

    def _keep_in_frame(
        self,
        target,
        frame_width,
        frame_height
    ):

        x1, y1, x2, y2 = target.box


        # Positive = banana right of middle -> slide right
        x_error = (x1 + x2) / 2.0 - frame_width / 2.0

        lr = 0

        if abs(x_error) > CENTER_DEADZONE:

            lr = clamp(x_error * LR_GAIN, -MAX_LR, MAX_LR)


        # Positive = banana below middle -> move down (negative ud)
        y_error = (y1 + y2) / 2.0 - frame_height / 2.0

        ud = 0

        if abs(y_error) > VERTICAL_DEADZONE:

            ud = clamp(-y_error * UD_GAIN, -MAX_UD, MAX_UD)


        # Never fly closer - only back off if we've drifted into it
        fb = 0

        if self.last_box_ratio > TOO_CLOSE_RATIO:

            fb = -BACK_OFF_SPEED


        return self._smooth_command(lr, fb, ud, 0)


    # ========================================================
    # SEARCH: TURN A LITTLE, THEN HOVER STILL
    # ========================================================

    def _search_command(
        self,
        now
    ):

        cycle = SEARCH_TURN_TIME + SEARCH_PAUSE_TIME

        t = (now - self.search_start_time) % cycle

        if t < SEARCH_TURN_TIME:

            return self._smooth_command(0, 0, 0, SEARCH_YAW_SPEED)

        return self._hover()


    # ========================================================
    # MAIN UPDATE
    # ========================================================

    def update(
        self,
        detections,
        frame_width,
        frame_height
    ):

        now = time.time()

        dt = now - self.last_update_time

        self.last_update_time = now


        target = self._choose_target(detections)

        self.current_target = target

        target_visible = target is not None


        if target_visible:

            self.last_target_time = now

            self.last_target_label = target.label.lower()

            self.last_box_ratio = self._box_ratio(
                target,
                frame_width,
                frame_height
            )


        # ====================================================
        # DONE EATING -> SLIDE RIGHT, HOVER, THEN LAND
        # ====================================================

        if self.state == "LAND":

            return RCCommand()


        if self.state == "DONE":

            elapsed = now - self.done_start_time

            if elapsed >= LAND_AFTER_EATING:

                self.state = "LAND"

                self.should_land = True

                return self._hover()


            if elapsed < DONE_VEER_TIME:

                return self._smooth_command(DONE_VEER_SPEED, 0, 0, 0)


            return self._hover()


        # ====================================================
        # FEED: KEEP BANANA IN THE PICTURE AND EAT
        # ====================================================

        if self.state == "FEED":

            if not target_visible:

                # Lost it for a while: go back to looking.
                if now - self.last_target_time > TARGET_LOST_GRACE_TIME:

                    self.state = "SEARCH"

                    return self._search_command(now)


                # Short blink: hold still, eating pauses.
                return self._hover()


            # Eat (only while the banana is in view)
            self.hunger = clamp(
                self.hunger - FEED_RATE * dt,
                0.0,
                100.0
            )


            if self.hunger <= FULL_THRESHOLD:

                self.state = "DONE"

                self.done_start_time = now

                return self._smooth_command(DONE_VEER_SPEED, 0, 0, 0)


            return self._keep_in_frame(
                target,
                frame_width,
                frame_height
            )


        # ====================================================
        # SEARCH
        # ====================================================

        if target_visible:

            # Found one - start eating right here.
            self.state = "FEED"

            return self._keep_in_frame(
                target,
                frame_width,
                frame_height
            )


        return self._search_command(now)
