"""
Obstacle safety/reflex layer - sits between WHATEVER produced a command
(manual keyboard input, or the autonomous controller's intent) and the
drone interface, and can override that command when something is too
close. This is what makes "hold the forward key into a wall" impossible:
the raw command is only ever a *request*, this layer has the final say.

    Keyboard command  \
                        >-- SafetyLayer.apply() --> Drone Interface
    Autonomous intent /

Uses the existing optical-flow LEFT/CENTER/RIGHT regions - not the
physics-distance safety net in interfaces/pybullet_drone.py, which is a
separate, independent backstop (defense in depth: one uses vision, the
other uses ground-truth distance, and either can catch what the other
misses).
"""


class SafetyLayer:
    def __init__(self, warning_threshold=0.6, danger_threshold=1.4, critical_threshold=2.2,
                 turn_rate=0.8, emergency_turn_rate=1.0, retreat_speed=0.25, smoothing=0.3):
        """
        Two+ thresholds, each a stronger response than the last. Calibrated
        against measured flow values in this environment/camera: normal
        cruising flow (just distant walls) tops out around 0.6-0.8, a real
        obstacle a meter or so away pushes flow above 1.0, and it climbs
        past 1.5-2.4+ as it gets close. Re-measure and retune if you
        change the camera FOV/resolution or the course layout.

        warning_threshold: flow above this -> reduce forward speed, but
            manual steering still goes through untouched.
        danger_threshold: CENTER flow above this -> forward motion is
            capped at 0 regardless of what was requested, and it steers
            toward whichever side has less flow.
        critical_threshold: flow above this on ANY side -> forces a
            backward retreat and a harder turn, full override.
        smoothing: EMA weight (0-1) applied to each region's flow before
            any decision is made. A single frame's flow is noisy,
            especially while an obstacle is still far away and the real
            signal is weak - deciding "which side is safer" from one raw
            frame can pick the wrong side. Smoothing first, then always
            recomputing the preferred side from the smoothed values (never
            locking onto one early guess), fixed a real bug where the
            layer would occasionally steer *toward* a distant obstacle
            because the very first noisy frame happened to suggest that
            direction.
        """
        self.warning_threshold = warning_threshold
        self.danger_threshold = danger_threshold
        self.critical_threshold = critical_threshold
        self.turn_rate = turn_rate
        self.emergency_turn_rate = emergency_turn_rate
        self.retreat_speed = retreat_speed
        self.smoothing = smoothing

        self._smooth_left = 0.0
        self._smooth_center = 0.0
        self._smooth_right = 0.0

    def reset(self):
        self._smooth_left = 0.0
        self._smooth_center = 0.0
        self._smooth_right = 0.0

    def apply(self, cmd, left_flow, center_flow, right_flow):
        """cmd: the raw command dict (from ManualController or an
        autonomous controller). Returns (final_cmd, info) - info is for
        the debug overlay: {"level": "CLEAR"|"WARNING"|"DANGER", "active": bool}.
        Never touches hover/land/reset/altitude_delta - only forward_speed
        and yaw_rate, and only when it would make things worse."""
        a = self.smoothing
        self._smooth_left = a * left_flow + (1 - a) * self._smooth_left
        self._smooth_center = a * center_flow + (1 - a) * self._smooth_center
        self._smooth_right = a * right_flow + (1 - a) * self._smooth_right
        left, center, right = self._smooth_left, self._smooth_center, self._smooth_right

        worst = max(left, center, right)
        final = dict(cmd)
        info = {"level": "CLEAR", "active": False}

        if worst < self.warning_threshold:
            return final, info

        # High left flow -> prefer right; high right flow -> prefer left.
        # Recomputed every call from the smoothed values (not locked to a
        # single early decision), so it can correct itself as the
        # picture clarifies while still being stable frame-to-frame.
        safer_direction = 1 if left < right else -1

        if worst >= self.critical_threshold:
            info["level"] = "DANGER"
            info["active"] = True
            final["forward_speed"] = min(cmd["forward_speed"], -self.retreat_speed)
            final["yaw_rate"] = safer_direction * self.emergency_turn_rate

        elif center > self.danger_threshold:
            info["level"] = "DANGER"
            info["active"] = True
            final["forward_speed"] = min(cmd["forward_speed"], 0.0)
            final["yaw_rate"] = safer_direction * self.turn_rate

        else:
            # WARNING tier: slow down, but keep the pilot's own steering -
            # they may already be turning away on their own.
            info["level"] = "WARNING"
            if cmd["forward_speed"] > 0:
                info["active"] = True
                final["forward_speed"] = cmd["forward_speed"] * 0.5
            if cmd["yaw_rate"] == 0:
                final["yaw_rate"] = safer_direction * self.turn_rate * 0.5

        return final, info
