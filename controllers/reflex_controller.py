"""
Simple rule-based obstacle-avoidance controller ("autonomous mode"). This
exists to prove the whole pipeline (camera -> optical flow -> controller ->
drone interface) works before the real FlyBrain neural/reflex network
replaces it.

Swap-in contract: any controller used by main.py just needs a
`decide(left_flow, center_flow, right_flow, state) -> command dict` method
with the same keys as ManualController.decide() returns. flybrain_controller.py
implements the same shape.
"""


class ReflexController:
    def __init__(self, turn_threshold=0.9, stop_threshold=1.4, emergency_threshold=2.2,
                 forward_speed=0.4, turn_rate=0.8, emergency_turn_rate=1.0,
                 retreat_speed=0.25):
        """
        Thresholds were calibrated by measuring actual flow values in this
        environment/camera: normal cruising flow (just the distant side
        walls) tops out around 0.6-0.8, a real obstacle a meter or so away
        pushes flow above 1.0, and it climbs past 1.5-2.4+ as it gets
        close. If you change the camera FOV/resolution or the course
        layout, re-measure and retune these.

        Three response tiers, each stronger than the last:
          - below turn_threshold: cruise straight ahead.
          - above turn_threshold: something's off to one side - slow down
            and lean away from it (still moving forward, cautiously).
          - above stop_threshold (checked on center flow specifically):
            it's close and roughly dead ahead - stop advancing, just turn.
          - above emergency_threshold (any region): it's very close - stop,
            back away, and turn harder.

        forward_speed is deliberately slow (item 5: keep it slow while
        avoidance is unreliable) - the hard MAX_SPEED cap in
        interfaces/pybullet_drone.py backs this up regardless.
        """
        self.turn_threshold = turn_threshold
        self.stop_threshold = stop_threshold
        self.emergency_threshold = emergency_threshold
        self.forward_speed = forward_speed
        self.turn_rate = turn_rate
        self.emergency_turn_rate = emergency_turn_rate
        self.retreat_speed = retreat_speed

        # +1 = committed to turning left, -1 = committed right, 0 = none.
        # A dead-center obstacle makes left_flow and right_flow nearly
        # equal, so re-deciding "whichever side is smaller" fresh every
        # cycle flip-flops on noise instead of actually turning away. Once
        # a turn starts, keep turning the same way until the obstacle
        # actually clears.
        self._committed_direction = 0

    def reset(self):
        self._committed_direction = 0

    def decide(self, left_flow, center_flow, right_flow, state=None):
        worst = max(left_flow, center_flow, right_flow)

        if worst < self.turn_threshold:
            self._committed_direction = 0
            return self._command(self.forward_speed, 0.0)

        if self._committed_direction == 0:
            # If the right side is more blocked, turn left (away from it);
            # if the left side is more blocked, turn right.
            self._committed_direction = 1 if left_flow < right_flow else -1

        if worst >= self.emergency_threshold:
            # Very close - stop advancing, back away, turn hard.
            return self._command(-self.retreat_speed,
                                  self._committed_direction * self.emergency_turn_rate)

        if center_flow > self.stop_threshold:
            # Close and roughly dead ahead - stop advancing, keep turning.
            return self._command(0.0, self._committed_direction * self.turn_rate)

        # Moderate: something's off to one side - slow down and lean away
        # *before* it becomes a close call.
        return self._command(self.forward_speed * 0.5, self._committed_direction * self.turn_rate)

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
