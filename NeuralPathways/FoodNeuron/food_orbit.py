"""
Fly-inspired food behaviour.

    SEARCH     turn a little, hover still, repeat - looking for a banana
    -> APPROACH  banana seen: fly toward it, aimed by the camera,
                 slowing down as it gets close, until it looks big
                 enough to eat (lost it for a while -> SEARCH)
    -> FEED    keep it in the picture and eat
    -> DONE    full: slide a little right, hover
    -> LAND    5 s after eating (tello_camera.py lands when should_land)

    SCARED   (tello_camera.py --scared) the fly brain's Giant Fiber
             fired: back straight away for 1 s - eating pauses
    -> WAIT    hover and look for at least 2 s, and until the banana has
               been clearly in view for 1 s (something held in front of
               it blocks the view, so the drone stays back until it's
               gone). Further scares are ignored - it's already backed off.
    -> APPROACH  same as above. Can't see the banana (too far, or it
                 flickers)? Creep straight forward until it has made up
                 all the distance it backed off, looking for it on the
                 way. Only the brain makes it back away - never the
                 banana flickering out (flight 16:07: that pushed it out
                 of sight of the banana)
    -> FEED    carry on eating where hunger left off (or SEARCH if the
               banana isn't in view). Every new scare starts over.

The looming neurons only fire for things getting CLOSER, not for
something held still - so "is it safe to come back?" is answered by
the camera (can I see my food?), like a fly looking before it returns.

Any banana is food. Hunger only goes down while the banana is
actually in view. While eating the drone holds still, only making
small moves to keep the banana in the picture.
"""

import time
from collections import deque
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

# Flying to the banana and it flickers out: keep going (it's ahead)
# this long before giving up to search - far-away bananas flicker a lot
# and giving up after 1 s made the drone bounce between approaching
# and searching 7 times in one flight (16:33)
APPROACH_LOST_TIME = 2.5

# While eating, the banana flickering out is usually just the detector
# (flight 16:19: eating at ~4% size, seen 15/56 pictures) - hold still
# this long before giving up and turning to look for it.
FEED_LOST_HOLD_TIME = 3.0


# ============================================================
# HUNGER
# ============================================================

# 100 = very hungry
# 0 = completely full

STARTING_HUNGER = 100.0

FULL_THRESHOLD = 10.0

# Hunger points lost per second while eating
# (100 -> 10 takes 15 seconds of the banana being in view)
FEED_RATE = 6.0


# ============================================================
# SEARCH
# ============================================================

# Turn a little, then hover still. Non-stop turning made the
# drone curve off to the right.
SEARCH_YAW_SPEED = 10

SEARCH_TURN_TIME = 1.0

SEARCH_PAUSE_TIME = 1.0

# Just lost the banana and know which side it went: turn that way
# quickly and without pausing for this long first (flight 16:33: the
# slow stop-start search took up to 16 s to find it again)
QUICK_TURN_SPEED = 12

QUICK_TURN_TIME = 2.5


# ============================================================
# KEEP BANANA IN THE PICTURE
# ============================================================

# Only move once the banana drifts this far (pixels) from the
# middle of the picture - small wobbles are ignored.
CENTER_DEADZONE = 120

VERTICAL_DEADZONE = 100

# TURN left/right to face it, like a fly steering toward food.
# (Turning keeps the banana in view without moving the drone; the
# old sideways slide combined with turning made it swing.)
TURN_GAIN = 0.08

MAX_TURN = 20

# Approaching: steer more precisely than while eating
APPROACH_TURN_DEADZONE = 60

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
# SCARED -> BACK AWAY -> COME BACK
# ============================================================

# Scared: back straight away - clear, but short (2 s at 50% went
# much too far indoors)
BACK_AWAY_SPEED = 40

BACK_AWAY_TIME = 1.0

# Fly toward the banana until it fills at least this
# much of the screen (or as much as it did before the scare, if more).
# Speed shrinks as it gets close, so it doesn't fly past. Slow also
# means approaching things doesn't look like a new loom to the brain.
EAT_SIZE_RATIO = 0.07

APPROACH_GAIN = 600

APPROACH_MIN_SPEED = 10

# (25 was faster, but flying in that fast made the whole scene loom
# and scared the fly brain - flight 16:44)
APPROACH_MAX_SPEED = 15

# Give up approaching after this long (eat from wherever we got to)
APPROACH_MAX_TIME = 15.0

# WAIT: always stay back at least this long after backing off...
MIN_WAIT_TIME = 1.0

# ...and come back once the banana has been seen more than once
# recently: its sightings since we started waiting span at least
# CLEAR_SPAN, the latest within CLEAR_SPAN (one banana-AI result
# covers ~0.2 s of pictures, so this needs 2+ separate sightings).
# Asking for an unbroken look instead kept resetting on the real
# drone's flickery detection.
CLEAR_WINDOW = 1.0

CLEAR_SPAN = 0.4

# Banana never seen while waiting (maybe too far to spot): after this
# long, creep back anyway - if the object is still there, approaching
# it looms and the brain scares us off again.
WAIT_GIVE_UP_TIME = 5.0

# Creeping back blind: slowly, until we've made up the distance we
# backed off (tracked as speed x time, x a bit extra for the coasting
# after each back-off), or at most COME_BACK_MAX_TIME.
COME_BACK_SLOW_SPEED = 12

COME_BACK_EXTRA = 1.3

COME_BACK_MAX_TIME = 12.0


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

    def __init__(self, clock=time.time):

        # Seconds, for every timer below. Wall time on the real drone; the
        # simulator passes its own physics time, since it runs slower than
        # real time once the detector and the brain are in the loop.
        self._clock = clock

        self.state = "SEARCH"

        # Hunger
        self.hunger = STARTING_HUNGER
        self.last_update_time = clock()

        # Target
        self.current_target = None
        self.last_target_label = None
        self.last_box_ratio = 0.0
        self.last_target_time = 0.0

        # Timers
        self.search_start_time = clock()
        self.done_start_time = None

        # Scared -> back away -> come back
        self.scared_start_time = None
        self.wait_start_time = None
        self.return_start_time = None
        self.return_saw_banana = False
        self.approach_after_scare = False
        self.size_before_scare = 0.0

        # How far we've backed off since last eating (rc speed x seconds)
        self.backed_off = 0.0

        # Which side the banana was last seen on (+1 right, -1 left) -
        # when it's lost, turn that way to find it again
        self.last_target_side = 1
        self.search_dir = 1

        # Quick turn only when we've actually seen the banana before
        self.quick_turn = False

        # Last camera-guided approach speed, kept through flickers
        self.approach_speed = APPROACH_MIN_SPEED

        # When the banana was seen recently (for WAIT's "is it clear?")
        self.sightings = deque()

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
        frame_height,
        forward=0,
        center_deadzone=CENTER_DEADZONE
    ):

        x1, y1, x2, y2 = target.box


        # Positive = banana right of middle -> turn right (+yaw)
        x_error = (x1 + x2) / 2.0 - frame_width / 2.0

        yaw = 0

        if abs(x_error) > center_deadzone:

            yaw = clamp(x_error * TURN_GAIN, -MAX_TURN, MAX_TURN)


        # Positive = banana below middle -> move down (negative ud)
        y_error = (y1 + y2) / 2.0 - frame_height / 2.0

        ud = 0

        if abs(y_error) > VERTICAL_DEADZONE:

            ud = clamp(-y_error * UD_GAIN, -MAX_UD, MAX_UD)


        # Only fly closer when asked to (approach) - and always back
        # off if we've drifted into it
        fb = forward

        if self.last_box_ratio > TOO_CLOSE_RATIO:

            fb = -BACK_OFF_SPEED


        return self._smooth_command(0, fb, ud, yaw)


    # ========================================================
    # SEARCH: TURN A LITTLE, THEN HOVER STILL
    # ========================================================

    def _search_command(
        self,
        now
    ):

        since = now - self.search_start_time

        if self.quick_turn and since < QUICK_TURN_TIME:

            return self._smooth_command(0, 0, 0, QUICK_TURN_SPEED * self.search_dir)


        cycle = SEARCH_TURN_TIME + SEARCH_PAUSE_TIME

        t = since % cycle

        if t < SEARCH_TURN_TIME:

            return self._smooth_command(0, 0, 0, SEARCH_YAW_SPEED * self.search_dir)

        return self._hover()


    def _lost_banana(self, now):
        """Give up on the banana for now: look for it, turning first
        toward the side it was last seen on. No quick turn after a scare:
        we only backed straight off, so the banana should still be ahead
        (the quick turn looked like the drone flying off to one side)."""

        after_scare = self.state == "APPROACH" and self.approach_after_scare

        self.state = "SEARCH"

        self.search_dir = self.last_target_side

        self.search_start_time = now

        self.quick_turn = not after_scare

        return self._search_command(now)


    # ========================================================
    # SCARED BY THE FLY BRAIN (tello_camera.py --scared)
    # ========================================================

    def scare(self):
        """
        The brain's Giant Fiber just fired. Back away - unless we're
        already backing off / waiting (backing off twice carried the
        drone out of sight of the banana), or done eating / landing.
        Returns True if we reacted.
        """

        if self.state in ("SCARED", "WAIT", "DONE", "LAND"):
            return False

        # Remember how close we were, to get at least that close again
        if self.state in ("FEED", "APPROACH"):
            self.size_before_scare = self.last_box_ratio

        self.state = "SCARED"

        self.scared_start_time = self._clock()

        return True


    def _start_approach(self, now, saw_banana, after_scare):

        self.state = "APPROACH"

        self.return_start_time = now

        self.return_saw_banana = saw_banana

        self.approach_after_scare = after_scare

        if not after_scare:
            self.size_before_scare = 0.0


    def _direct(self, lr, fb, ud, yaw):
        """Send exactly this command, no smoothing (for the clear back-off)."""

        self.prev_lr, self.prev_fb, self.prev_ud, self.prev_yaw = lr, fb, ud, yaw

        return RCCommand(lr=lr, fb=fb, ud=ud, yaw=yaw)


    # ========================================================
    # MAIN UPDATE
    # ========================================================

    def update(
        self,
        detections,
        frame_width,
        frame_height
    ):

        now = self._clock()

        dt = now - self.last_update_time

        self.last_update_time = now


        target = self._choose_target(detections)

        self.current_target = target

        target_visible = target is not None


        if target_visible:

            self.sightings.append(now)

        while self.sightings and now - self.sightings[0] > CLEAR_WINDOW:

            self.sightings.popleft()


        if target_visible:

            self.last_target_time = now

            self.last_target_label = target.label.lower()

            x1, _, x2, _ = target.box

            self.last_target_side = 1 if (x1 + x2) / 2.0 >= frame_width / 2.0 else -1

            self.last_box_ratio = self._box_ratio(
                target,
                frame_width,
                frame_height
            )


        # ====================================================
        # SCARED: back straight away (eating pauses)
        # ====================================================

        if self.state == "SCARED":

            if now - self.scared_start_time < BACK_AWAY_TIME:

                self.backed_off += BACK_AWAY_SPEED * dt * COME_BACK_EXTRA

                return self._direct(0, -BACK_AWAY_SPEED, 0, 0)


            self.state = "WAIT"

            self.wait_start_time = now


        # ====================================================
        # WAIT: hover and look - is the coast clear?
        # ====================================================

        if self.state == "WAIT":

            recent = [t for t in self.sightings if t >= self.wait_start_time]

            banana_in_view = (
                len(recent) > 0
                and recent[-1] - recent[0] >= CLEAR_SPAN
                and now - recent[-1] <= CLEAR_SPAN
            )

            if banana_in_view:

                if now - self.wait_start_time >= MIN_WAIT_TIME:

                    # Food in view, nothing in the way - come back
                    self._start_approach(now, True, after_scare=True)

            else:

                if now - self.wait_start_time >= WAIT_GIVE_UP_TIME:

                    # Can't see it at all - creep back and let the
                    # brain scare us off if something's still there
                    self._start_approach(now, False, after_scare=True)


            if self.state == "WAIT":

                return self._hover()


        # ====================================================
        # APPROACH: FLY TO THE BANANA (first find, or coming back)
        # ====================================================

        if self.state == "APPROACH":

            elapsed = now - self.return_start_time


            # Blind creep (banana wasn't visible while waiting) - until
            # it shows up, then switch to the camera-guided approach
            if not self.return_saw_banana:

                if target_visible:

                    # Spotted it - camera-guided from here (its own timer)
                    self.return_saw_banana = True

                    self.return_start_time = now

                    elapsed = 0.0

                elif (
                    self.backed_off > 0
                    and elapsed < COME_BACK_MAX_TIME
                ):

                    self.backed_off -= COME_BACK_SLOW_SPEED * dt

                    return self._smooth_command(0, COME_BACK_SLOW_SPEED, 0, 0)

                else:

                    # Made up the distance and still can't see it
                    return self._lost_banana(now)


            # Lost the banana for a while
            if now - self.last_target_time > APPROACH_LOST_TIME:

                if self.approach_after_scare and self.backed_off > 0:

                    # Probably just too far to spot - creep back blind
                    self.return_saw_banana = False

                    self.return_start_time = now

                    return self._hover()


                return self._lost_banana(now)


            # Short flicker: keep going forward - the banana's straight
            # ahead. (Stopping on every flicker made far-away approaches
            # stand still most of the time.)
            if not target_visible:

                self.backed_off = max(0.0, self.backed_off - self.approach_speed * dt)

                return self._smooth_command(0, self.approach_speed, 0, 0)


            goal = max(EAT_SIZE_RATIO, self.size_before_scare)

            if self.last_box_ratio >= goal or elapsed > APPROACH_MAX_TIME:

                # Close enough - carry on eating
                self.state = "FEED"

                return self._keep_in_frame(
                    target,
                    frame_width,
                    frame_height
                )


            # Fly toward it, slowing down as it gets bigger
            forward = clamp(
                (goal - self.last_box_ratio) * APPROACH_GAIN,
                APPROACH_MIN_SPEED,
                APPROACH_MAX_SPEED
            )

            self.backed_off = max(0.0, self.backed_off - forward * dt)

            # Keep this speed through flickers (see above)
            self.approach_speed = forward

            return self._keep_in_frame(
                target,
                frame_width,
                frame_height,
                forward=forward,
                center_deadzone=APPROACH_TURN_DEADZONE
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

            # Back at the food - nothing left to make up
            self.backed_off = 0.0

            if not target_visible:

                # Lost it for a while: go and look for it.
                if now - self.last_target_time > FEED_LOST_HOLD_TIME:

                    return self._lost_banana(now)


                # Blink: hold still, eating pauses.
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

            # Found one - fly to it (from next frame)
            self._start_approach(now, True, after_scare=False)

            return self._keep_in_frame(
                target,
                frame_width,
                frame_height
            )


        return self._search_command(now)
