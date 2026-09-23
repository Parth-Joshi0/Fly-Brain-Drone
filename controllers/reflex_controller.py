"""
Simple rule-based obstacle-avoidance controller. This exists to prove the
whole pipeline (camera -> optical flow -> controller -> drone interface)
works before the real FlyBrain neural/reflex network replaces it.

Swap-in contract: any controller used by main.py just needs a
`decide(left_flow, center_flow, right_flow, state) -> {"forward_speed", "yaw_rate"}`
method. flybrain_controller.py implements the same shape.
"""


class ReflexController:
    def __init__(self, turn_threshold=0.9, stop_threshold=1.4,
                 forward_speed=0.8, turn_rate=0.8):
        """
        turn_threshold/stop_threshold were calibrated by measuring actual
        flow values in this environment/camera: normal cruising flow (just
        the distant side walls) tops out around 0.6-0.8, while a real
        obstacle a meter or so away pushes flow above 1.0 and climbs past
        1.5-2.4 as it gets close. If you change the camera FOV/resolution
        or the course layout, re-measure and retune these.

        turn_threshold: side flow above this (and bigger than the other
            side) means "something's over there, lean away from it."
        stop_threshold: center flow above this means "something's dead
            ahead and close - stop advancing, just turn."
        forward_speed: cruising speed in m/s when the path looks clear.
        turn_rate: yaw rate in rad/s used for avoidance turns.
        """
        self.turn_threshold = turn_threshold
        self.stop_threshold = stop_threshold
        self.forward_speed = forward_speed
        self.turn_rate = turn_rate

        # +1 = committed to turning left, -1 = committed right, 0 = none.
        # A dead-center obstacle makes left_flow and right_flow nearly
        # equal, so re-deciding "whichever side is smaller" fresh every
        # cycle flip-flops on noise instead of actually turning away.
        # Once a turn starts, keep turning the same way until the
        # obstacle actually clears.
        self._committed_direction = 0

    def decide(self, left_flow, center_flow, right_flow, state=None):
        obstacle_ahead = (
            center_flow > self.stop_threshold
            or left_flow > self.turn_threshold
            or right_flow > self.turn_threshold
        )

        if not obstacle_ahead:
            self._committed_direction = 0
            return {"forward_speed": self.forward_speed, "yaw_rate": 0.0}

        if self._committed_direction == 0:
            if left_flow < right_flow:
                self._committed_direction = 1  # turn left, away from the right
            else:
                self._committed_direction = -1  # turn right, away from the left

        forward = 0.0 if center_flow > self.stop_threshold else self.forward_speed * 0.5
        return {"forward_speed": forward, "yaw_rate": self._committed_direction * self.turn_rate}
