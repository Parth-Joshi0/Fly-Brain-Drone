"""
Fly-inspired food behaviour.

    SCAN       look around the room: pause and look (zoomed in, see
               wants_zoom), turn 45 deg, repeat - 8 turns = a full 360.
               Banana seen twice -> APPROACH. Two full 360s with no
               banana -> LAND.
    -> APPROACH  banana seen: fly toward it, aimed by the camera,
                 slowing down as it gets close, until it looks big
                 enough to eat (lost it for a while -> LOOK: stop and
                 look zoomed in for 1 s, then SCAN)
    -> FEED    keep it in the picture and eat
    -> DONE    full: slide a little right, hover
    -> LAND    5 s after eating (fly_tello.py lands when should_land)

    SCARED   (fly_tello.py --scared) the fly brain's Giant Fiber
             fired: back straight away (quick 0.7 s jump) - eating pauses
    -> WAIT    steady for 0.3 s, then come back as soon as the coast is
               clear: the fly brain sees nothing looming AND the banana
               is in view (something held in front of it blocks the
               view, so the drone stays back until it's gone). Further
               scares are ignored - it's already backed off.
    -> APPROACH  dash straight back (most of the distance it backed off),
                 then the careful approach. Banana blocked mid-dash ->
                 stop and WAIT again. Can't see the banana (too far, or it
                 flickers)? Creep straight forward until it has made up
                 all the distance it backed off, looking for it on the
                 way. Only the brain makes it back away - never the
                 banana flickering out (flight 16:07: that pushed it out
                 of sight of the banana)
    -> FEED    carry on eating where hunger left off (or SCAN if the
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
# SCAN THE ROOM
# ============================================================

# Pause and look, then turn SCAN_STEP_DEG, repeat. The camera sees
# ~70 deg across, so 45 deg steps overlap a little and miss nothing;
# 8 of them make a full 360 in ~22 s. The pauses matter: pictures
# taken while turning are blurred and the banana AI misses far-away
# bananas in them, and while paused it uses the slower zoomed-in look
# (see wants_zoom), which sees ~2x further.
SCAN_STEP_DEG = 45

SCAN_TURN_SPEED = 30

# Stop turning after this long even if the compass hasn't reached
# SCAN_STEP_DEG yet (or there's no compass - see DEG_PER_RC_S)
SCAN_TURN_TIMEOUT = 2.5

# Long enough to be properly still before looking: in flight 10-02
# 21:26 (0.7 s pause, turn command fading out gradually) only 9 of 394
# scan pictures were taken while still - the rest were blurred
SCAN_PAUSE_TIME = 1.2

# Spotted something? Stop and keep looking this long to confirm it
# (the sharp zoomed-in look takes ~0.4 s per picture)
SCAN_CONFIRM_TIME = 2.0

# A banana counts as found once seen in this many pictures within the
# last CLEAR_WINDOW, at least SCAN_CONFIRM_SPAN apart
SCAN_CONFIRM_SIGHTINGS = 2

SCAN_CONFIRM_SPAN = 0.15

# Full 360s with no banana before giving up and landing
MAX_SCANS = 2

# Without a compass (e.g. the simulator), heading is estimated from
# the yaw commands sent: roughly this many degrees per second per rc
# unit. The real drone passes its compass heading to update() instead.
DEG_PER_RC_S = 1.0

# Approaching and the banana's gone for APPROACH_LOST_TIME: stop and
# look (zoomed in) this long before going back to scanning
LOOK_TIME = 1.0


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

# Scared: back straight away - a quick jump like a fly's escape, but
# short (2 s at 50% went much too far indoors). 60% for 0.7 s covers
# about the same distance as the old 40% for 1 s, just snappier.
BACK_AWAY_SPEED = 60

BACK_AWAY_TIME = 0.7

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

# Banana sightings older than this are forgotten
CLEAR_WINDOW = 1.0


# ============================================================
# COME BACK FAST (after a scare, once the coast is clear)
# ============================================================

# After backing off, hold still this long before judging (lets the
# drone stop and the looming from its own jump die down)
SETTLE_TIME = 0.3

# Coast is clear when the fly brain's escape level (set each picture
# by ScaredEatingBrain from fear_brain.py) has stayed below this...
QUIET_ESCAPE_LEVEL = 0.3

# ...for this long, AND a fresh banana sighting is this recent
QUIET_TIME = 0.3

SEEN_RECENTLY = 0.5

# WAIT looks with the fast banana AI first - right after backing off the
# banana is still close and easy to see - and only switches to the slow
# sharp-eyes look if it hasn't found it after this long
SHARP_EYES_AFTER = 1.5

# Dash straight back at this speed for this fraction of the distance we
# backed off, steering toward the banana; then the careful approach
# (slows down near it, so no overshoot)
DASH_SPEED = 35

DASH_FRACTION = 0.85

# Coming back: carry on eating once the banana looks at least this big
# a fraction of its size before the scare - no need to creep back to
# exactly the same spot (the slow final creep was the longest part)
RETURN_GOAL_FRACTION = 0.8

# Banana out of sight this long mid-dash -> something moved in front:
# stop and WAIT. (Looming is ignored while moving - efference copy -
# so the brain can't catch a wave mid-dash; a blocked banana can.)
DASH_BLOCKED_TIME = 0.5

# Banana never seen while waiting (maybe too far to spot): after this
# long, creep back anyway - if the object is still there, approaching
# it looms and the brain scares us off again.
WAIT_GIVE_UP_TIME = 3.0

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

class FeedingBehaviour:

    def __init__(self, clock=time.time):

        # Seconds, for every timer below. Wall time on the real drone; the
        # simulator passes its own physics time, since it runs slower than
        # real time once the detector and the brain are in the loop.
        self._clock = clock

        self.state = "SCAN"

        # Hunger
        self.hunger = STARTING_HUNGER
        self.last_update_time = clock()

        # Target
        self.current_target = None
        self.last_target_label = None
        self.last_box_ratio = 0.0
        self.last_target_time = 0.0

        # Timers
        self.done_start_time = None
        self.look_start_time = None

        # Why should_land became True (for fly_tello.py's message)
        self.land_reason = ""

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

        # Heading (degrees, keeps counting past 360) - from the compass
        # if update() gets one, else estimated from our yaw commands
        self.heading = 0.0
        self._last_compass = None

        # Scan progress (see _start_scan)
        self._start_scan(clock())

        # Last camera-guided approach speed, kept through flickers
        self.approach_speed = APPROACH_MIN_SPEED

        # When the banana was seen recently (for WAIT's "is it clear?")
        self.sightings = deque()

        # Fly brain's escape level, set each picture by ScaredEatingBrain
        # (stays 0 without the fly brain, e.g. the simulator)
        self.escape_level = 0.0

        self.quiet_since = None

        # Come back fast: dash until backed_off drops to this
        self.dash_until = 0.0

        # fly_tello.py lands when this becomes True
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
    # SCAN THE ROOM: LOOK, TURN 45 DEG, REPEAT
    # ========================================================

    @property
    def wants_zoom(self):
        """True while holding still to look for the banana - the slower
        zoomed-in banana AI is worth it then (ScaredEatingBrain)."""

        return (
            (self.state == "SCAN" and self.scan_phase == "pause")
            or self.state == "LOOK"
            or (
                self.state == "WAIT"
                and self._clock() - self.wait_start_time >= SHARP_EYES_AFTER
            )
        )


    def _stop_turning(self):
        """Hover with the turn cut to zero at once (not faded out like
        _hover()), so scan pauses are actually still."""

        return self._direct(0, 0, 0, 0)


    def _start_scan(self, now):

        self.scan_phase = "pause"      # look first, then turn
        self.phase_start_time = now
        self.step_start_heading = self.heading
        self.scan_turned = 0.0
        self.scans_done = 0
        self.scan_spotted = False


    def _banana_confirmed(self):

        if len(self.sightings) < SCAN_CONFIRM_SIGHTINGS:
            return False

        return self.sightings[-1] - self.sightings[0] >= SCAN_CONFIRM_SPAN


    def _scan_command(self, now, target_visible):

        # Spotted something while turning: stop and take a proper look
        if target_visible and self.scan_phase == "turn":

            self.scan_turned += abs(self.heading - self.step_start_heading)

            self.scan_phase = "pause"

            self.phase_start_time = now

        if target_visible:

            self.scan_spotted = True


        if self.scan_phase == "turn":

            turned = abs(self.heading - self.step_start_heading)

            if (
                turned >= SCAN_STEP_DEG
                or now - self.phase_start_time >= SCAN_TURN_TIMEOUT
            ):

                self.scan_turned += turned

                self.scan_phase = "pause"

                self.phase_start_time = now

                self.scan_spotted = False

                return self._stop_turning()


            return self._smooth_command(0, 0, 0, SCAN_TURN_SPEED * self.search_dir)


        # Pause: hold still and look
        pause = SCAN_CONFIRM_TIME if self.scan_spotted else SCAN_PAUSE_TIME

        if now - self.phase_start_time < pause:

            return self._stop_turning()


        # Done looking here - full circle?
        if self.scan_turned >= 360 - SCAN_STEP_DEG / 2:

            self.scans_done += 1

            self.scan_turned = 0.0

            if self.scans_done >= MAX_SCANS:

                self.state = "LAND"

                self.should_land = True

                self.land_reason = f"no banana found after {MAX_SCANS} full 360s"

                return self._stop_turning()


        # Next turn
        self.scan_phase = "turn"

        self.phase_start_time = now

        self.step_start_heading = self.heading

        self.scan_spotted = False

        return self._smooth_command(0, 0, 0, SCAN_TURN_SPEED * self.search_dir)


    def _lost_banana(self, now):
        """Give up on the banana for now and scan for it: look first
        (after a scare we only backed straight off, so it's probably
        still ahead), then turn toward the side it was last seen on."""

        self.state = "SCAN"

        self.search_dir = self.last_target_side

        self._start_scan(now)

        return self._hover()


    # ========================================================
    # SCARED BY THE FLY BRAIN (fly_tello.py --scared)
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

        # Dash back most of the way we backed off (see DASH_FRACTION)
        self.dash_until = self.backed_off * (1 - DASH_FRACTION) if after_scare else self.backed_off

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
        frame_height,
        yaw_deg=None,
        fresh=True
    ):
        """yaw_deg: the drone's compass heading (Tello's yaw, degrees),
        if known - used to count the scan's turns. Without it, heading
        is estimated from the yaw commands (see DEG_PER_RC_S).

        fresh: False when `detections` is a reused earlier banana-AI
        result (it runs slower than the camera, see ScaredEatingBrain) -
        then it isn't counted again as a new sighting."""

        now = self._clock()

        dt = now - self.last_update_time

        self.last_update_time = now


        if yaw_deg is not None:

            if self._last_compass is not None:

                # -180..180 wraps around - take the short way
                self.heading += (yaw_deg - self._last_compass + 180) % 360 - 180

            self._last_compass = yaw_deg

        else:

            self.heading += self.prev_yaw * DEG_PER_RC_S * dt


        target = self._choose_target(detections)

        self.current_target = target

        target_visible = target is not None


        if target_visible and fresh:

            self.sightings.append(now)

        while self.sightings and now - self.sightings[0] > CLEAR_WINDOW:

            self.sightings.popleft()


        # Is the fly brain quiet (nothing looming)?
        if self.escape_level < QUIET_ESCAPE_LEVEL:

            if self.quiet_since is None:
                self.quiet_since = now

        else:

            self.quiet_since = None


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

            banana_in_view = len(recent) > 0 and now - recent[-1] <= SEEN_RECENTLY

            quiet = (
                self.quiet_since is not None
                and now - self.quiet_since >= QUIET_TIME
            )

            settled = now - self.wait_start_time >= SETTLE_TIME

            if banana_in_view:

                if settled and quiet:

                    # Food in view, nothing looming - come back fast
                    self._start_approach(now, True, after_scare=True)

            else:

                if now - self.wait_start_time >= WAIT_GIVE_UP_TIME and quiet:

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


                # Stop and take a proper (zoomed-in) look first
                self.state = "LOOK"

                self.look_start_time = now

                return self._hover()


            # Coming back fast after a scare: dash most of the way
            dashing = self.approach_after_scare and self.backed_off > self.dash_until

            if dashing and now - self.last_target_time > DASH_BLOCKED_TIME:

                # Banana blocked mid-dash - something's in front: stop
                self.state = "WAIT"

                self.wait_start_time = now

                return self._direct(0, 0, 0, 0)


            # Short flicker: keep going forward - the banana's straight
            # ahead. (Stopping on every flicker made far-away approaches
            # stand still most of the time.)
            if not target_visible:

                speed = DASH_SPEED if dashing else self.approach_speed

                self.backed_off = max(0.0, self.backed_off - speed * dt)

                return self._smooth_command(0, speed, 0, 0)


            goal = max(EAT_SIZE_RATIO, self.size_before_scare * RETURN_GOAL_FRACTION)

            if self.last_box_ratio >= goal or elapsed > APPROACH_MAX_TIME:

                # Close enough - carry on eating
                self.state = "FEED"

                return self._keep_in_frame(
                    target,
                    frame_width,
                    frame_height
                )


            # Fly toward it, slowing down as it gets bigger (dashing:
            # full DASH_SPEED until most of the way back)
            if dashing:

                forward = DASH_SPEED

            else:

                forward = clamp(
                    (goal - self.last_box_ratio) * APPROACH_GAIN,
                    APPROACH_MIN_SPEED,
                    APPROACH_MAX_SPEED
                )

            self.backed_off = max(0.0, self.backed_off - forward * dt)

            # Keep this speed through flickers (see above) - not the dash's
            if not dashing:
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

        # ====================================================
        # LOOK: lost it on the way - stop and look (zoomed in)
        # ====================================================

        if self.state == "LOOK":

            if self._banana_confirmed():

                self._start_approach(now, True, self.approach_after_scare)

                return self._hover()


            if now - self.look_start_time >= LOOK_TIME:

                return self._lost_banana(now)


            return self._hover()


        if self.state == "LAND":

            return RCCommand()


        if self.state == "DONE":

            elapsed = now - self.done_start_time

            if elapsed >= LAND_AFTER_EATING:

                self.state = "LAND"

                self.should_land = True

                self.land_reason = "finished eating"

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
        # SCAN THE ROOM
        # ====================================================

        if target_visible and self._banana_confirmed():

            # Found one - fly to it (from next frame)
            self._start_approach(now, True, after_scare=False)

            return self._keep_in_frame(
                target,
                frame_width,
                frame_height
            )


        return self._scan_command(now, target_visible)
