"""
Placeholder for the real bio-inspired FlyBrain neural/reflex network.

Not implemented yet - use ReflexController for now. When the FlyBrain
model is ready, implement `decide()` with the exact same signature/return
shape as ReflexController.decide() and swap the controller class in
main.py. Nothing in vision/, interfaces/, or evaluation/ needs to change.
"""


class FlyBrainController:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "FlyBrainController isn't built yet - use ReflexController for now."
        )

    def decide(self, left_flow, center_flow, right_flow, state=None):
        """Must return the same command dict shape as ReflexController and
        ManualController: {"forward_speed", "strafe_speed", "yaw_rate",
        "altitude_delta", "hover", "land", "reset"}."""
        raise NotImplementedError
