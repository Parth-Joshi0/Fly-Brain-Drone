"""
Autonomous mode's "Normal Navigation" stage - the bottom of the priority
pipeline:

    Danger Reflex > Obstacle Avoidance > Normal Navigation

Deliberately naive: it only signals "I want to explore forward." All
speed staging (NORMAL/FAST/WARNING/AVOIDANCE) and every actual avoidance
decision (left/right/up/down/back) lives in controllers/safety_layer.py,
which has final override authority regardless of what this returns - so
there's one place avoidance behavior is decided, not two that could
disagree.

Swap-in contract: any controller used by main.py just needs a
`decide(flow, state) -> command dict` method with the same keys as
ManualController.decide() returns. flybrain_controller.py implements the
same shape - when the real FlyBrain network replaces this, it can make a
much smarter navigation request (its own turns, speed preferences, a
real exploration strategy) and SafetyLayer still applies on top as the
final backstop.
"""


class ReflexController:
    def reset(self):
        pass

    def decide(self, flow, state=None):
        return {
            # Magnitude doesn't matter - SafetyLayer sets the real speed
            # every cycle regardless (NORMAL/FAST when clear, WARNING/
            # AVOIDANCE when not). This only needs to be > 0 to mean
            # "explore," matching the priority system: normal navigation
            # never gets to insist on a specific speed.
            "forward_speed": 1.0,
            "strafe_speed": 0.0,
            "yaw_rate": 0.0,
            "altitude_delta": 0.0,
            "hover": False,
            "land": False,
            "reset": False,
            "pressed_direction": "(autonomous)",
        }
