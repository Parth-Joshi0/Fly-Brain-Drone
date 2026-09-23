"""
Obstacle safety/reflex layer - sits between WHATEVER produced a command
(manual keyboard input, or the autonomous navigation controller's intent)
and the drone interface, and can override that command when something is
too close. This is what makes "hold/request forward into a wall"
impossible: the raw command is only ever a *request*, this layer has the
final say.

    Keyboard command       \
                             >-- SafetyLayer.apply() --> Drone Interface
    Autonomous exploration /

Priority system (each tier can override everything below it):

    STUCK ESCAPE          (not making progress - try something different)
    DANGER REFLEX          (CRITICAL: very close - stop, retreat, turn hard)
    OBSTACLE AVOIDANCE     (DANGER: close - stop advancing, steer away)
    (WARNING: getting close - slow down, gentle nudge)
    NORMAL NAVIGATION      (CLEAR: whatever was requested, speed-staged)

Uses the optical-flow 3x3 grid (vision/optical_flow.grid_flow_strengths) -
not the physics-distance safety net in interfaces/pybullet_drone.py,
which is a separate, independent backstop (defense in depth: one uses
vision, the other ground-truth distance; either can catch what the other
misses).
"""

import math
from collections import deque

# --- Speed staging (item 2) - tune freely ------------------------------
FAST_SPEED = 1.0        # m/s - wide open space, nothing worth slowing for
NORMAL_SPEED = 0.6      # m/s - default cruise
WARNING_SPEED = 0.3     # m/s - something's getting closer
AVOIDANCE_SPEED = 0.15  # m/s - actively avoiding (used as the retreat speed too)

# --- Flow thresholds (tiers) - re-measure/retune if you change the
# camera FOV/resolution or the course layout. Same calibration basis as
# before: normal cruising flow tops out ~0.6-0.8, a real obstacle a meter
# or so away pushes flow above 1.0, and it climbs past 1.5-2.4+ up close.
CLEAR_THRESHOLD = 0.35     # below this everywhere -> safe to go FAST_SPEED
WARNING_THRESHOLD = 0.6    # -> slow to WARNING_SPEED, gentle nudge
DANGER_THRESHOLD = 1.4     # -> stop advancing, steer toward the safest side
CRITICAL_THRESHOLD = 2.2   # -> stop, retreat, turn/move hard

TURN_RATE = 0.8
EMERGENCY_TURN_RATE = 1.0

# --- Stuck detection (item 6) -------------------------------------------
STUCK_WINDOW = 90            # decision cycles (~3s at the 30Hz decision loop)
STUCK_RADIUS = 0.35          # m - net movement below this over the window = not making progress
STUCK_ESCAPE_CYCLES = 60     # how long to commit to one escape maneuver (~2s)


class SafetyLayer:
    def __init__(self, warning_threshold=WARNING_THRESHOLD, danger_threshold=DANGER_THRESHOLD,
                 critical_threshold=CRITICAL_THRESHOLD, clear_threshold=CLEAR_THRESHOLD,
                 fast_speed=FAST_SPEED, normal_speed=NORMAL_SPEED,
                 warning_speed=WARNING_SPEED, avoidance_speed=AVOIDANCE_SPEED,
                 turn_rate=TURN_RATE, emergency_turn_rate=EMERGENCY_TURN_RATE,
                 smoothing=0.3, stuck_window=STUCK_WINDOW, stuck_radius=STUCK_RADIUS,
                 stuck_escape_cycles=STUCK_ESCAPE_CYCLES):
        self.warning_threshold = warning_threshold
        self.danger_threshold = danger_threshold
        self.critical_threshold = critical_threshold
        self.clear_threshold = clear_threshold
        self.fast_speed = fast_speed
        self.normal_speed = normal_speed
        self.warning_speed = warning_speed
        self.avoidance_speed = avoidance_speed
        self.turn_rate = turn_rate
        self.emergency_turn_rate = emergency_turn_rate
        self.smoothing = smoothing
        self.stuck_radius = stuck_radius
        self.stuck_escape_cycles = stuck_escape_cycles

        # A single frame's flow is noisy, especially while an obstacle is
        # still far away and the real signal is weak - deciding "which
        # direction is safest" from one raw frame can pick the wrong one.
        # Smoothing first, then always recomputing the preferred direction
        # from the smoothed values (never locking onto one early guess),
        # fixed a real bug where the layer would occasionally steer
        # *toward* a distant obstacle because the very first noisy frame
        # happened to suggest that direction.
        self._smoothed = {"left": 0.0, "right": 0.0, "top": 0.0, "bottom": 0.0, "center": 0.0}

        self._position_history = deque(maxlen=stuck_window)
        # Each of these moves diagonally (lateral/vertical + a little
        # forward), not in place - a pure yaw-in-place or climb-in-place
        # doesn't relocate the drone at all, so escaping never actually
        # got it past whatever it was stuck next to; it would cycle
        # through motions and still be in the same spot afterward.
        self._escape_sequence = ["left", "right", "up", "down"]
        self._escape_index = 0
        self._escape_action = None
        self._escape_timer = 0

    def reset(self):
        self._smoothed = {k: 0.0 for k in self._smoothed}
        self._position_history.clear()
        self._escape_action = None
        self._escape_timer = 0

    def apply(self, cmd, flow, position):
        """cmd: the raw command dict (from ManualController or the
        autonomous navigation controller). flow: the dict returned by
        vision.optical_flow.grid_flow_strengths (must have at least
        left/right/top/bottom/center). position: the drone's (x, y, z).

        Returns (final_cmd, info). info is for the debug overlay:
        {"level": "CLEAR"|"WARNING"|"DANGER", "active": bool,
         "direction": "FORWARD"|"LEFT"|"RIGHT"|"UP"|"DOWN"|"BACK"|"ROTATE",
         "stuck": bool}. Never touches hover/land/reset - only
        forward_speed, strafe_speed, yaw_rate, altitude_delta, and only
        when it would make things safer."""

        a = self.smoothing
        for key in self._smoothed:
            self._smoothed[key] = a * flow[key] + (1 - a) * self._smoothed[key]
        left, right = self._smoothed["left"], self._smoothed["right"]
        top, bottom = self._smoothed["top"], self._smoothed["bottom"]
        center = self._smoothed["center"]

        # TOP and BOTTOM deliberately do NOT contribute to the trigger
        # decision below, only to which escape direction looks safest
        # once something else already triggered it. Two reasons, both
        # measured directly: (1) ground-proximity ("ventral") flow is
        # present on essentially every cruise cycle just from flying
        # forward at normal altitude - up to ~1.0 in completely open
        # space with zero obstacles anywhere - so treating BOTTOM as a
        # trigger source made WARNING trip constantly on routine flight.
        # (2) open sky above is trivially ~0 flow (nothing there to
        # reflect), so once anything triggered WARNING, "UP" always won
        # the safest-direction pick regardless of the actual threat - the
        # drone just climbed to the ceiling and stayed there. Floor
        # avoidance is still handled by the hard MIN_ALTITUDE clamp in
        # interfaces/pybullet_drone.py independent of any of this.
        worst = max(left, right, center)

        final = dict(cmd)
        info = {"level": "CLEAR", "active": False, "direction": "FORWARD", "stuck": False}

        # --- Priority 1: stuck escape - overrides everything else ---
        # Deliberate hover (the pilot/controller explicitly asked to hold
        # position) is exempt - "stuck" means "trying to make progress and
        # failing," not "chose to stay still." Position-based only (not
        # gated on flow being elevated): a real deadlock can happen with
        # low flow too - e.g. the independent physics-distance safety net
        # in interfaces/pybullet_drone.py capping forward motion near an
        # obstacle before this layer's own flow-based trigger ever fires.
        if not cmd.get("hover"):
            self._position_history.append((position[0], position[1]))
            if self._escape_timer > 0 or self._is_stuck():
                return self._run_escape(final, info)

        # --- Priority 4 (lowest): nothing nearby - speed-stage and pass through ---
        if worst < self.warning_threshold:
            final["forward_speed"] = self.fast_speed if worst < self.clear_threshold else self.normal_speed
            return final, info

        # Something's within WARNING range or worse - figure out the
        # safest of the four avoidance directions (now including top/
        # bottom as real candidates, even though they didn't trigger
        # this). High left flow -> prefer right; high right flow ->
        # prefer left; same idea for top/bottom. Recomputed fresh every
        # call from the smoothed values (not locked to an early guess).
        options = {"LEFT": left, "RIGHT": right, "UP": top, "DOWN": bottom}
        safest = min(options, key=options.get)

        # --- Priority 2: danger reflex (critical - stop, retreat, turn hard) ---
        if worst >= self.critical_threshold:
            info["level"] = "DANGER"
            info["active"] = True
            info["direction"] = "BACK"
            final["forward_speed"] = -self.avoidance_speed
            self._steer_toward(final, safest, self.emergency_turn_rate)
            return final, info

        # --- Priority 3: obstacle avoidance (danger - stop advancing, steer away) ---
        if center > self.danger_threshold or worst >= self.danger_threshold:
            info["level"] = "DANGER"
            info["active"] = True
            info["direction"] = safest
            final["forward_speed"] = min(cmd["forward_speed"], 0.0)
            self._steer_toward(final, safest, self.turn_rate)
            return final, info

        # --- WARNING: slow down, gentle nudge toward the safer side ---
        info["level"] = "WARNING"
        info["active"] = True
        info["direction"] = safest
        final["forward_speed"] = self.warning_speed
        self._steer_toward(final, safest, self.turn_rate * 0.5)
        return final, info

    def _steer_toward(self, final, direction, rate):
        if direction == "LEFT":
            final["yaw_rate"] = rate
        elif direction == "RIGHT":
            final["yaw_rate"] = -rate
        elif direction == "UP":
            final["altitude_delta"] = 1.0
        elif direction == "DOWN":
            final["altitude_delta"] = -1.0

    def _is_stuck(self):
        if len(self._position_history) < self._position_history.maxlen:
            return False
        oldest = self._position_history[0]
        newest = self._position_history[-1]
        return math.hypot(newest[0] - oldest[0], newest[1] - oldest[1]) < self.stuck_radius

    def _run_escape(self, final, info):
        if self._escape_timer <= 0:
            self._escape_action = self._escape_sequence[self._escape_index % len(self._escape_sequence)]
            self._escape_index += 1
            self._escape_timer = self.stuck_escape_cycles
            self._position_history.clear()

        info["level"] = "DANGER"
        info["active"] = True
        info["stuck"] = True
        info["direction"] = self._escape_action.upper()

        # A little forward speed on left/right/up escape actions (not just
        # "back") is what actually gets it past whatever it's stuck next
        # to - pure sideways/vertical motion with zero forward just
        # relocates it to a different spot beside the same obstacle.
        # "down" is the exception: descending while still horizontally
        # near an obstacle's footprint risks clipping its top on the way
        # down, and the physics-distance safety net in
        # interfaces/pybullet_drone.py only ever restricts *horizontal*
        # forward motion near something close, never vertical descent -
        # so this specific combination isn't backstopped the way the
        # others are. Keep "down" pure vertical.
        final["forward_speed"] = 0.0 if self._escape_action == "down" else self.avoidance_speed
        final["strafe_speed"] = 0.0
        final["yaw_rate"] = 0.0
        final["altitude_delta"] = 0.0

        if self._escape_action == "left":
            final["strafe_speed"] = self.avoidance_speed
        elif self._escape_action == "right":
            final["strafe_speed"] = -self.avoidance_speed
        elif self._escape_action == "up":
            final["altitude_delta"] = 1.0
        elif self._escape_action == "down":
            final["altitude_delta"] = -1.0

        self._escape_timer -= 1
        return final, info
