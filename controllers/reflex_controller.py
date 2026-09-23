"""
Autonomous mode's "intent" generator ("autonomous mode"). Deliberately
naive - it always requests a forward cruise. All actual obstacle
avoidance now lives in controllers/safety_layer.py, applied identically
whether the raw intent came from here or from the keyboard, so there's
one source of avoidance behavior instead of two that could disagree.

Swap-in contract: any controller used by main.py just needs a
`decide(left_flow, center_flow, right_flow, state) -> command dict` method
with the same keys as ManualController.decide() returns. flybrain_controller.py
implements the same shape - when the real FlyBrain network replaces this,
it can make a much smarter request (its own turns, speed changes, etc.)
and SafetyLayer still applies on top as the final backstop.
"""


class ReflexController:
    def __init__(self, forward_speed=0.4):
        self.forward_speed = forward_speed

    def reset(self):
        pass

    def decide(self, left_flow, center_flow, right_flow, state=None):
        return {
            "forward_speed": self.forward_speed,
            "strafe_speed": 0.0,
            "yaw_rate": 0.0,
            "altitude_delta": 0.0,
            "hover": False,
            "land": False,
            "reset": False,
            "pressed_direction": "(autonomous)",
        }
