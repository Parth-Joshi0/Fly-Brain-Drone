"""
Autonomous mode's "3D Navigation Controller" - a small explicit state
machine, not a fresh from-scratch decision every frame:

    CRUISE          move forward, watch LEFT/CENTER/RIGHT optic flow
    AVOID_LEFT      committed left turn, hold until the obstacle clears
    AVOID_RIGHT     committed right turn, hold until the obstacle clears
    BOUNDARY_RETURN outside the course's soft flight area - steer back
                    toward its center, hold until well back inside

Pipeline position:

    Danger Reflex (controllers/safety_layer.py) > Obstacle Avoidance +
    Normal Navigation (this file)

This is deliberately where "which side is safer" gets DECIDED and
COMMITTED TO - re-picking a direction fresh every frame (which an
earlier version of this project did) flip-flops on noisy readings, so
once AVOID_LEFT/RIGHT starts it holds for a minimum number of cycles and
until the path is actually clear, not just momentarily quieter.
controllers/safety_layer.py still applies on top of whatever this
returns and can escalate to a stronger emergency reaction (stop/retreat)
if flow gets a lot closer than what triggers avoidance here - defense in
depth, not two different opinions about the same threshold.

Swap-in contract: any controller used by main.py just needs a
`decide(flow, state) -> command dict` method with the same keys as
ManualController.decide() returns. flybrain_controller.py implements the
same shape.
"""

import math

# --- Obstacle detection / avoidance - tune freely -----------------------
#
# AVOID_TURN_RATE and the two flow thresholds below are coupled in a way
# that isn't obvious - measured directly, not guessed: turning itself
# generates real optical flow (de-rotation only cancels the *mean* shift;
# a wide-FOV camera's rotational flow genuinely varies across the frame,
# so meaningful residual flow remains even mid-turn - see the note in
# vision/optical_flow.derotate_flow). At AVOID_TURN_RATE below, that
# residual floor measured ~0.6 with noise spikes higher, which is why
# AVOID_CLEAR_THRESHOLD sits above it - if it didn't, the AVOID state
# could never see "clear" and would never be able to exit, since its own
# turning would permanently look like an obstacle to itself. If you raise
# AVOID_TURN_RATE, re-measure the residual (rotate in place with nothing
# nearby, watch controllers/safety_layer's flow readings) and raise
# AVOID_CLEAR_THRESHOLD to comfortably clear it, or this state can get
# permanently stuck.
AVOID_TRIGGER_THRESHOLD = 1.2    # LEFT, CENTER, or RIGHT flow above this ->
                                  # start avoiding. Lower = starts farther
                                  # away from the obstacle (more cautious).
AVOID_CLEAR_THRESHOLD = 0.9      # flow must drop below this on ALL of
                                  # LEFT/CENTER/RIGHT before the obstacle
                                  # counts as cleared and CRUISE resumes.
AVOID_MIN_CYCLES = 30            # minimum decision cycles (~1s at the 30Hz
                                  # decision loop) committed to one avoidance
                                  # direction before it's allowed to
                                  # re-evaluate - this is what stops rapid
                                  # left-right-left-right flapping.
CRUISE_SPEED = 1.0               # m/s forward during CRUISE (SafetyLayer
                                  # may still speed this up in very open space)
AVOID_FORWARD_SPEED = 0.25       # m/s forward speed kept WHILE turning
                                  # around an obstacle - moves around it
                                  # rather than stopping dead in place, but
                                  # slow enough that a modest turn rate has
                                  # time to actually clear the obstacle
                                  # before reaching it.
AVOID_TURN_RATE = 0.15           # rad/s - how hard it turns during
                                  # AVOID_LEFT/AVOID_RIGHT. Kept moderate
                                  # specifically to limit the self-motion
                                  # flow problem above - see the note.

# --- Boundary containment - tune freely ----------------------------------
BOUNDARY_FORWARD_SPEED = 0.5     # m/s while steering back toward the course
BOUNDARY_TURN_RATE = 0.35        # rad/s cap on the boundary-return turn -
                                  # also kept moderate for the same
                                  # self-motion-flow reason as AVOID_TURN_RATE
BOUNDARY_TURN_GAIN = 1.2         # proportional gain: heading error (rad)
                                  # -> yaw rate. Higher = turns back more
                                  # sharply; lower = a gentler, wider arc.
BOUNDARY_RELEASE_MARGIN = 1.0    # m - how far back inside the bounds it has
                                  # to get before leaving BOUNDARY_RETURN
                                  # (hysteresis, so it doesn't immediately
                                  # re-trigger right at the edge)


class ReflexController:
    def __init__(self, bounds=None):
        """bounds: the dict simulation.environment.build_environment()
        returns as "bounds" - {"min_x", "max_x", "min_y", "max_y",
        "center_x", "center_y"}. None disables boundary containment."""
        self.bounds = bounds
        self.state = "CRUISE"
        self._avoid_direction = None  # "LEFT" or "RIGHT" while avoiding
        self._avoid_timer = 0

    def reset(self):
        self.state = "CRUISE"
        self._avoid_direction = None
        self._avoid_timer = 0

    def decide(self, flow, state=None):
        position = state["position"] if state else (0.0, 0.0, 0.0)

        # --- BOUNDARY_RETURN takes priority over obstacle avoidance -
        # going out of bounds is the one thing nothing else in the course
        # protects against (the perimeter walls are physical obstacles
        # the flow system avoids on its own; nothing stops the drone from
        # just flying past the goal into open space in X). ---
        if self.bounds is not None:
            if self.state == "BOUNDARY_RETURN":
                if self._well_inside_bounds(position):
                    self.state = "CRUISE"
                    self._avoid_direction = None
            elif self._outside_bounds(position):
                self.state = "BOUNDARY_RETURN"
                self._avoid_direction = None

        if self.state == "BOUNDARY_RETURN":
            return self._boundary_return_command(position, state)

        left, center, right = flow["left"], flow["center"], flow["right"]

        if self.state == "CRUISE":
            if max(left, center, right) > AVOID_TRIGGER_THRESHOLD:
                # More free space (lower flow) on one side -> go that way.
                # If they're close, this still deterministically picks one
                # and the state machine below commits to it - "choose a
                # direction and commit to it long enough to get around."
                self._avoid_direction = "LEFT" if left < right else "RIGHT"
                self.state = f"AVOID_{self._avoid_direction}"
                self._avoid_timer = AVOID_MIN_CYCLES
            else:
                return self._cruise_command()

        # AVOID_LEFT / AVOID_RIGHT: hold the turn until both the minimum
        # commitment time has elapsed AND the path has actually cleared -
        # not just one or the other, otherwise a lingering elevated
        # reading right as the timer expires would still cut the turn
        # short before it actually got around the thing.
        self._avoid_timer = max(0, self._avoid_timer - 1)
        cleared = (left < AVOID_CLEAR_THRESHOLD
                   and center < AVOID_CLEAR_THRESHOLD
                   and right < AVOID_CLEAR_THRESHOLD)
        if self._avoid_timer == 0 and cleared:
            self.state = "CRUISE"
            self._avoid_direction = None
            return self._cruise_command()

        return self._avoid_command()

    # --- command builders ---

    def _cruise_command(self):
        return self._command(CRUISE_SPEED, 0.0)

    def _avoid_command(self):
        rate = AVOID_TURN_RATE if self._avoid_direction == "LEFT" else -AVOID_TURN_RATE
        return self._command(AVOID_FORWARD_SPEED, rate)

    def _boundary_return_command(self, position, state):
        yaw = math.radians(state["yaw_degrees"]) if state else 0.0
        dx = self.bounds["center_x"] - position[0]
        dy = self.bounds["center_y"] - position[1]
        target_heading = math.atan2(dy, dx)

        # Signed angle from current heading to the target, wrapped to
        # [-pi, pi] - positive means the target is to the left (matches
        # this project's yaw convention: positive yaw = turn left).
        heading_error = (target_heading - yaw + math.pi) % (2 * math.pi) - math.pi
        yaw_rate = max(-BOUNDARY_TURN_RATE, min(BOUNDARY_TURN_RATE, heading_error * BOUNDARY_TURN_GAIN))

        return self._command(BOUNDARY_FORWARD_SPEED, yaw_rate)

    def _command(self, forward_speed, yaw_rate):
        return {
            "forward_speed": forward_speed,
            "strafe_speed": 0.0,
            "yaw_rate": yaw_rate,
            "altitude_delta": 0.0,
            "hover": False,
            "land": False,
            "reset": False,
            "pressed_direction": "(autonomous)",
        }

    # --- boundary helpers ---

    def _outside_bounds(self, position):
        x, y = position[0], position[1]
        b = self.bounds
        return x < b["min_x"] or x > b["max_x"] or y < b["min_y"] or y > b["max_y"]

    def _well_inside_bounds(self, position):
        x, y = position[0], position[1]
        b = self.bounds
        m = BOUNDARY_RELEASE_MARGIN
        return (b["min_x"] + m < x < b["max_x"] - m
                and b["min_y"] + m < y < b["max_y"] - m)
