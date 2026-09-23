"""
Placeholder for the real bio-inspired FlyBrain neural/reflex network.

Not implemented yet - use ReflexController for now. When the FlyBrain
model is ready, implement `decide()` with the exact same signature/return
shape as ReflexController.decide() and swap the controller class in
main.py. Nothing in vision/, interfaces/, evaluation/, or
controllers/safety_layer.py needs to change - main.py runs every
controller's output through SafetyLayer.apply() before it reaches the
drone, so FlyBrain gets the same obstacle-avoidance backstop for free.
"""


class FlyBrainController:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "FlyBrainController isn't built yet - use ReflexController for now."
        )

    def decide(self, flow, state=None):
        """flow: the dict from vision.optical_flow.grid_flow_strengths
        (9 grid cells + left/right/top/bottom/center aggregates). state:
        drone.get_state(). Must return the same command dict shape as
        ReflexController and ManualController: {"forward_speed",
        "strafe_speed", "yaw_rate", "altitude_delta", "hover", "land",
        "reset"}. SafetyLayer.apply() still runs on the result and has
        final override authority - this only needs to propose where to
        go, not guarantee it's safe."""
        raise NotImplementedError
