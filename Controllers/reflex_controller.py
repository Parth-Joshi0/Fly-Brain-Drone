"""
Autonomous mode's "3D Navigation Controller" - a small explicit state
machine, not a fresh from-scratch decision every frame:

    CRUISE          move forward, watch LEFT/CENTER/RIGHT optic flow, bias
                     gently toward less-visited parts of the arena
    AVOID_LEFT      obstacle mainly on the LEFT - keep moving forward
                     while steering RIGHT around it
    AVOID_RIGHT     obstacle mainly on the RIGHT - keep moving forward
                     while steering LEFT around it
    WALL_ESCAPE     been avoiding the same side for too long (following a
                     wall instead of getting past it) - turn harder toward
                     the arena interior for a fixed stretch
    EMERGENCY_ESCAPE flow says something is extremely close - the only
                     state allowed to kill forward motion; briefly
                     reverses/turns, then hands back to CRUISE
    BOUNDARY_RETURN outside the course's soft flight area - steer back
                    toward its center, hold until well back inside

Pipeline position:

    Danger Reflex (safety_layer.py) > Obstacle Avoidance +
    Normal Navigation (this file)

This is deliberately where "which side is safer" gets DECIDED and
COMMITTED TO - re-picking a direction fresh every frame (which an
earlier version of this project did) flip-flops on noisy readings, so
once AVOID_LEFT/RIGHT starts it holds for a minimum number of cycles and
until the path is actually clear, not just momentarily quieter.
safety_layer.py still applies on top of whatever this
returns and can escalate to a stronger emergency reaction (stop/retreat)
if flow gets a lot closer than what triggers avoidance here - defense in
depth, not two different opinions about the same threshold.

IMPORTANT: only EMERGENCY_ESCAPE is allowed to drop forward_speed to (or
below) zero. Every other state - including every flavor of obstacle
avoidance - keeps moving forward while it steers, per an explicit "the
drone stopped beside a wall instead of continuing past it" bug report.
(The other, bigger half of that bug was actually in
Simulator/pybullet_drone.py's proximity safety net, which used to zero
forward motion for anything close in ANY direction, including a wall
merely alongside the drone - see FORWARD_CONE_DEGREES there.)

Swap-in contract: any controller used by main.py just needs a
`decide(flow, state) -> command dict` method with the same keys as
ManualController.decide() returns. flybrain_controller.py implements the
same shape.
"""

from Controllers.boundary_math import (
    heading_error_toward,
    heading_rate_toward,
    near_or_outside_bounds,
    outside_bounds,
    well_inside_bounds,
)

# --- Obstacle detection / avoidance - tune freely -----------------------
#
# AVOID_TURN_RATE and the two flow thresholds below are coupled in a way
# that isn't obvious - measured directly, not guessed: turning itself
# generates real optical flow (de-rotation only cancels the *mean* shift;
# a wide-FOV camera's rotational flow genuinely varies across the frame,
# so meaningful residual flow remains even mid-turn - see the note in
# NeuralPathways/EscapeNeuron/optical_flow.derotate_flow). At AVOID_TURN_RATE below, that
# residual floor measured ~0.6 with noise spikes higher, which is why
# AVOID_CLEAR_THRESHOLD sits above it - if it didn't, the AVOID state
# could never see "clear" and would never be able to exit, since its own
# turning would permanently look like an obstacle to itself. If you raise
# AVOID_TURN_RATE, re-measure the residual (rotate in place with nothing
# nearby, watch safety_layer's flow readings) and raise
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
                                  # re-evaluate - "turn commitment," stops
                                  # rapid left-right-left-right flapping.
CRUISE_SPEED = 1.0               # m/s forward during CRUISE (SafetyLayer
                                  # may still speed this up in very open space)
AVOID_TURN_RATE = 0.15           # rad/s - how hard it turns during
                                  # AVOID_LEFT/AVOID_RIGHT. Measured
                                  # directly (rotate in place with nothing
                                  # nearby, read the flow) rather than
                                  # picked to satisfy a "keep more forward
                                  # speed" wishlist: 0.15 -> residual
                                  # self-motion flow stays ~0.5-0.9; 0.20
                                  # -> it spikes to 3+ (a periodic aliasing
                                  # effect against the striped obstacle
                                  # texture, not a smooth degradation) and
                                  # falsely triggers EMERGENCY_ESCAPE from
                                  # the turn itself, with nothing actually
                                  # nearby - measured and confirmed this is
                                  # what caused an earlier attempt at a
                                  # faster turn rate to get stuck
                                  # oscillating in place near the first
                                  # obstacle instead of ever getting past
                                  # it. Forward speed during avoidance
                                  # below is chosen to suit THIS turn rate
                                  # (turning radius ~= forward_speed /
                                  # AVOID_TURN_RATE), not the other way
                                  # around.
SIDE_FORWARD_FRACTION = 0.28     # forward speed (x CRUISE_SPEED) while
                                  # going around a wall detected mainly on
                                  # one side - keeps real forward progress
                                  # while steering past it, never stops.
CENTER_FORWARD_FRACTION = 0.2    # forward speed (x CRUISE_SPEED) while an
                                  # obstacle is roughly dead ahead - still
                                  # moving, just more cautiously than a
                                  # side-wall case while it picks a side.

# --- Wall-escape - tune freely -------------------------------------------
# Following a wall (repeatedly re-entering AVOID_LEFT/RIGHT without ever
# getting a fully clear reading) is what "stuck beside a wall on the
# left" looks like from the state machine's point of view even once it
# no longer literally halts. WALL_FOLLOW_TIMEOUT_CYCLES bounds how long
# it's willing to keep nibbling around the same obstacle before forcing a
# harder turn back toward open arena interior instead.
WALL_FOLLOW_TIMEOUT_CYCLES = 150  # ~5s continuously avoiding -> escalate
WALL_ESCAPE_CYCLES = 45           # ~1.5s commitment to the escape turn
WALL_ESCAPE_TURN_RATE = 0.2       # stronger than AVOID_TURN_RATE, but still
                                   # under the residual self-motion-flow
                                   # danger zone measured above (0.25+
                                   # spikes past EMERGENCY_TRIGGER_THRESHOLD
                                   # from the turn alone, which would keep
                                   # re-interrupting this exact state)

# --- Emergency escape - tune freely --------------------------------------
# The only state allowed to actually kill forward motion - reserved for
# flow readings well past AVOID_TRIGGER_THRESHOLD, i.e. genuinely very
# close, not just "something's in view." Time-boxed (not flow-cleared)
# exit, so the self-motion-flow residual noted above can't strand it here.
EMERGENCY_TRIGGER_THRESHOLD = 1.8
EMERGENCY_ESCAPE_CYCLES = 20      # ~0.7s commitment
EMERGENCY_REVERSE_SPEED = 0.3     # m/s backward while escaping an obstacle
                                   # in the open (direction picked from flow)
EMERGENCY_INTERIOR_SPEED = 0.15   # m/s FORWARD, used instead of reversing
                                   # when near the boundary (see
                                   # NEAR_BOUNDARY_MARGIN below) - direction
                                   # there is already known-safe (heading
                                   # toward the arena center), so creeping
                                   # that way is more reliable than
                                   # reversing blind: found by testing that
                                   # reverse+hard-turn near a wall can
                                   # produce a brief resultant velocity that
                                   # still carries toward the wall before
                                   # the turn fully takes hold (momentum
                                   # during the transition), which caused a
                                   # real collision right at a corner.
EMERGENCY_TURN_RATE = 0.5
NEAR_BOUNDARY_MARGIN = 1.0        # m - within this of the soft boundary (or
                                   # already outside it), EMERGENCY_ESCAPE
                                   # steers toward the arena interior instead
                                   # of picking a side from flow alone. Found
                                   # by testing: right in a corner (a wall AND
                                   # an obstacle both close at once), L/R flow
                                   # can be noisy/near-symmetric and picking
                                   # by flow occasionally turned the escape
                                   # straight into the wall instead of away
                                   # from it - heading toward the known-safe
                                   # arena center is a much more reliable
                                   # signal right at an edge.

# --- Boundary containment - tune freely ----------------------------------
BOUNDARY_FORWARD_SPEED = 0.4     # m/s while steering back toward the course -
                                  # kept modest on purpose so it doesn't carry
                                  # much momentum into the turn-back
BOUNDARY_TURN_RATE = 0.2         # rad/s cap on the boundary-return turn -
                                  # kept under the residual self-motion-flow
                                  # danger zone (see AVOID_TURN_RATE's note)
                                  # so a return arc doesn't keep tripping
                                  # EMERGENCY_ESCAPE purely from its own turn
BOUNDARY_TURN_GAIN = 1.6         # proportional gain: heading error (rad)
                                  # -> yaw rate. Higher = turns back more
                                  # sharply; lower = a gentler, wider arc.
BOUNDARY_RELEASE_MARGIN = 1.0    # m - how far back inside the bounds it has
                                  # to get before leaving BOUNDARY_RETURN
                                  # (hysteresis, so it doesn't immediately
                                  # re-trigger right at the edge)

# --- Full-area exploration - tune freely ----------------------------------
# A coarse visit heatmap over the flight area. Every decision cycle
# increments the count for whichever cell the drone is currently in;
# CRUISE then applies a gentle heading bias toward the least-visited
# cell's center. This is only ever a bias layered on top of the normal
# CRUISE heading (yaw_rate stays small, capped well below AVOID_TURN_RATE)
# and only ever runs during CRUISE - obstacle avoidance, wall escape,
# emergency escape and boundary return all take priority and are
# untouched by this. It steers the drone there over time; nothing ever
# teleports.
EXPLORATION_GRID_SIZE = 4          # NxN cells covering the flight bounds
EXPLORATION_BIAS_GAIN = 0.5        # proportional gain, heading error (rad)
                                    # -> yaw rate bias
EXPLORATION_MAX_YAW_BIAS = 0.1     # rad/s cap - deliberately below
                                    # AVOID_TURN_RATE so exploration alone
                                    # can't itself generate enough residual
                                    # self-motion flow to look like an
                                    # obstacle and falsely trigger avoidance.

DECISION_HZ = 30.0  # matches main.py's DECISION_INTERVAL_STEPS at 240Hz physics


class ReflexController:
    def __init__(self, bounds=None):
        """bounds: the dict Simulator.environment.build_environment()
        returns as "bounds" - {"min_x", "max_x", "min_y", "max_y",
        "center_x", "center_y"}. None disables boundary containment AND
        exploration (both need a finite area to work within)."""
        self.bounds = bounds
        self.state = "CRUISE"

        self._avoid_direction = None     # "LEFT" or "RIGHT" turn direction
        self._avoid_forward_fraction = SIDE_FORWARD_FRACTION
        self._avoid_timer = 0
        self._avoid_streak = 0           # consecutive cycles spent avoiding,
                                          # drives WALL_ESCAPE triggering

        self._wall_escape_timer = 0
        self._emergency_timer = 0
        self._emergency_direction = None
        self._emergency_toward_interior = False

        # --- debug/telemetry (read by main.py's overlay) ---
        self.turn_command = "NONE"
        self.current_cell = (0, 0)
        self.least_visited_cell = (0, 0)
        self.wall_time_seconds = 0.0

        self._visit_counts = None
        if bounds is not None:
            n = EXPLORATION_GRID_SIZE
            self._visit_counts = [[0] * n for _ in range(n)]

    def reset(self):
        self.state = "CRUISE"
        self._avoid_direction = None
        self._avoid_forward_fraction = SIDE_FORWARD_FRACTION
        self._avoid_timer = 0
        self._avoid_streak = 0
        self._wall_escape_timer = 0
        self._emergency_timer = 0
        self._emergency_direction = None
        self._emergency_toward_interior = False
        self.turn_command = "NONE"
        self.wall_time_seconds = 0.0
        if self._visit_counts is not None:
            n = EXPLORATION_GRID_SIZE
            self._visit_counts = [[0] * n for _ in range(n)]

    def decide(self, flow, state=None):
        position = state["position"] if state else (0.0, 0.0, 0.0)
        yaw_degrees = state["yaw_degrees"] if state else 0.0
        left, center, right = flow["left"], flow["center"], flow["right"]
        worst = max(left, center, right)

        self._update_exploration_grid(position)
        self.wall_time_seconds = self._avoid_streak / DECISION_HZ

        # --- Priority 1: EMERGENCY_ESCAPE - extremely close, overrides
        # everything else. The only state allowed to stop/reverse. ---
        if self.state == "EMERGENCY_ESCAPE" and self._emergency_timer > 0:
            return self._emergency_escape_command()
        if worst > EMERGENCY_TRIGGER_THRESHOLD:
            self._enter_emergency_escape(position, yaw_degrees, left, right)
            return self._emergency_escape_command()

        # --- Priority 2: BOUNDARY_RETURN - never leave the walled area. ---
        if self.bounds is not None:
            if self.state == "BOUNDARY_RETURN":
                if well_inside_bounds(position, self.bounds, BOUNDARY_RELEASE_MARGIN):
                    self._exit_to_cruise()
            elif outside_bounds(position, self.bounds):
                self._enter_boundary_return()
        if self.state == "BOUNDARY_RETURN":
            return self._boundary_return_command(position, yaw_degrees)

        # --- Priority 3a: WALL_ESCAPE - been avoiding the same obstacle/
        # wall for too long without a clear reading; turn harder toward
        # the interior instead of continuing to nibble around it. ---
        if self.state == "WALL_ESCAPE" and self._wall_escape_timer > 0:
            return self._wall_escape_command(position, yaw_degrees)
        if self._avoid_streak >= WALL_FOLLOW_TIMEOUT_CYCLES:
            self._enter_wall_escape()
            return self._wall_escape_command(position, yaw_degrees)

        # --- Priority 3b: obstacle avoidance - keep moving, steer around.
        # Hold the turn until both the minimum commitment time has
        # elapsed AND the path has actually cleared - not just one or the
        # other, otherwise a lingering elevated reading right as the timer
        # expires would still cut the turn short before it actually got
        # around the thing. ---
        if self.state in ("AVOID_LEFT", "AVOID_RIGHT"):
            self._avoid_timer = max(0, self._avoid_timer - 1)
            cleared = (left < AVOID_CLEAR_THRESHOLD and center < AVOID_CLEAR_THRESHOLD
                       and right < AVOID_CLEAR_THRESHOLD)
            if not (self._avoid_timer == 0 and cleared):
                self._avoid_streak += 1
                return self._avoid_command()
            self._exit_to_cruise()

        if worst > AVOID_TRIGGER_THRESHOLD:
            self._enter_avoid(left, center, right)
            self._avoid_streak += 1
            return self._avoid_command()

        # --- Priority 4 (lowest): CRUISE, with a gentle bias toward
        # less-visited parts of the arena. ---
        self._avoid_streak = 0
        return self._cruise_command(position, yaw_degrees)

    # --- state entry helpers ---

    def _enter_avoid(self, left, center, right):
        # Turn toward whichever side has more free space (lower flow). If
        # they're close, this still deterministically picks one and the
        # state machine commits to it - "choose a direction and commit to
        # it long enough to get around." The STATE name reflects which
        # side the obstacle is actually on (not which way it's turning):
        # obstacle-on-the-left means turning right, so that's AVOID_LEFT.
        turn_left = left < right
        self._avoid_direction = "LEFT" if turn_left else "RIGHT"
        self.state = "AVOID_RIGHT" if turn_left else "AVOID_LEFT"
        self._avoid_timer = AVOID_MIN_CYCLES
        # Roughly centered obstacle (dead ahead, not clearly to one side) ->
        # a bit more cautious while it commits to a side; clearly one-sided
        # (a wall along the flank) -> keep more forward speed, it's just
        # passing it.
        self._avoid_forward_fraction = (
            CENTER_FORWARD_FRACTION if center >= left and center >= right
            else SIDE_FORWARD_FRACTION
        )

    def _enter_boundary_return(self):
        self.state = "BOUNDARY_RETURN"
        self._avoid_direction = None
        self._avoid_streak = 0

    def _enter_wall_escape(self):
        self.state = "WALL_ESCAPE"
        self._wall_escape_timer = WALL_ESCAPE_CYCLES
        self._avoid_streak = 0

    def _enter_emergency_escape(self, position, yaw_degrees, left, right):
        self.state = "EMERGENCY_ESCAPE"
        self._emergency_timer = EMERGENCY_ESCAPE_CYCLES
        near_boundary = self.bounds is not None and near_or_outside_bounds(position, self.bounds, NEAR_BOUNDARY_MARGIN)
        self._emergency_toward_interior = near_boundary
        if near_boundary:
            target = (self.bounds["center_x"], self.bounds["center_y"])
            heading_error = heading_error_toward(position, yaw_degrees, target)
            self._emergency_direction = "LEFT" if heading_error > 0 else "RIGHT"
        else:
            self._emergency_direction = "LEFT" if left < right else "RIGHT"
        self._avoid_streak = 0

    def _exit_to_cruise(self):
        self.state = "CRUISE"
        self._avoid_direction = None
        self._avoid_streak = 0
        self.turn_command = "NONE"

    # --- command builders ---

    def _cruise_command(self, position, yaw_degrees):
        yaw_rate = 0.0
        if self.bounds is not None and self._visit_counts is not None:
            target = self._least_visited_cell_center()
            yaw_rate = heading_rate_toward(
                position, yaw_degrees, target, EXPLORATION_MAX_YAW_BIAS, EXPLORATION_BIAS_GAIN
            )
        self.turn_command = "NONE" if abs(yaw_rate) < 1e-3 else ("LEFT" if yaw_rate > 0 else "RIGHT")
        return self._command(CRUISE_SPEED, yaw_rate)

    def _avoid_command(self):
        rate = AVOID_TURN_RATE if self._avoid_direction == "LEFT" else -AVOID_TURN_RATE
        speed = CRUISE_SPEED * self._avoid_forward_fraction
        self.turn_command = self._avoid_direction
        return self._command(speed, rate)

    def _wall_escape_command(self, position, yaw_degrees):
        self._wall_escape_timer -= 1
        if self._wall_escape_timer <= 0:
            self._exit_to_cruise()
        target = (self.bounds["center_x"], self.bounds["center_y"])
        yaw_rate = heading_rate_toward(
            position, yaw_degrees, target, WALL_ESCAPE_TURN_RATE, BOUNDARY_TURN_GAIN
        )
        self.turn_command = "LEFT" if yaw_rate > 0 else "RIGHT" if yaw_rate < 0 else "NONE"
        return self._command(CRUISE_SPEED * SIDE_FORWARD_FRACTION, yaw_rate)

    def _emergency_escape_command(self):
        rate = EMERGENCY_TURN_RATE if self._emergency_direction == "LEFT" else -EMERGENCY_TURN_RATE
        self.turn_command = f"EMERGENCY_{self._emergency_direction}"
        self._emergency_timer -= 1
        if self._emergency_timer <= 0:
            self._exit_to_cruise()
        speed = EMERGENCY_INTERIOR_SPEED if self._emergency_toward_interior else -EMERGENCY_REVERSE_SPEED
        return self._command(speed, rate)

    def _boundary_return_command(self, position, yaw_degrees):
        target = (self.bounds["center_x"], self.bounds["center_y"])
        yaw_rate = heading_rate_toward(
            position, yaw_degrees, target, BOUNDARY_TURN_RATE, BOUNDARY_TURN_GAIN
        )
        self.turn_command = "LEFT" if yaw_rate > 0 else "RIGHT" if yaw_rate < 0 else "NONE"
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

    # --- exploration helpers ---

    def _update_exploration_grid(self, position):
        if self._visit_counts is None:
            return
        col, row = self._cell_index(position)
        self.current_cell = (col, row)
        self._visit_counts[row][col] += 1

    def _cell_index(self, position):
        n = EXPLORATION_GRID_SIZE
        b = self.bounds
        fx = (position[0] - b["min_x"]) / (b["max_x"] - b["min_x"])
        fy = (position[1] - b["min_y"]) / (b["max_y"] - b["min_y"])
        col = min(n - 1, max(0, int(fx * n)))
        row = min(n - 1, max(0, int(fy * n)))
        return col, row

    def _cell_center(self, col, row):
        n = EXPLORATION_GRID_SIZE
        b = self.bounds
        cell_w = (b["max_x"] - b["min_x"]) / n
        cell_h = (b["max_y"] - b["min_y"]) / n
        x = b["min_x"] + (col + 0.5) * cell_w
        y = b["min_y"] + (row + 0.5) * cell_h
        return x, y

    def _least_visited_cell_center(self):
        n = EXPLORATION_GRID_SIZE
        best_cell, best_count = (0, 0), None
        for row in range(n):
            for col in range(n):
                count = self._visit_counts[row][col]
                if best_count is None or count < best_count:
                    best_count = count
                    best_cell = (col, row)
        self.least_visited_cell = best_cell
        return self._cell_center(*best_cell)
