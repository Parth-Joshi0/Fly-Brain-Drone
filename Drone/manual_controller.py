"""
Manual-flight controller. Reads the normalized input dict any
SimulatorInterface.poll_input() produces (see Simulator/
simulator_interface.py) - never a platform-specific keyboard/gamepad API
directly, so it works unchanged whether that input came from the PyBullet
GUI or a real controller flying the real drone.

Keys (PyBullet backend):
    Up/Down       forward / backward
    Left/Right    strafe left / right
    W/S           altitude up / down
    Q/E           yaw left / right
    Space         hover (zeroes forward/strafe/yaw, holds altitude)
    L             land
    R             reset
"""

MANUAL_SPEED = 0.4      # m/s - deliberately slow, see item 5 (keep it slow
                          # while testing obstacle avoidance)
MANUAL_YAW_RATE = 0.6    # rad/s


class ManualController:
    def decide(self, input_state):
        """input_state: a dict shaped like simulator_interface.EMPTY_INPUT."""

        forward_speed = 0.0
        strafe_speed = 0.0
        yaw_rate = 0.0
        altitude_delta = 0.0
        pressed_direction = []

        if input_state["forward"]:
            forward_speed = MANUAL_SPEED
            pressed_direction.append("FORWARD")
        if input_state["backward"]:
            forward_speed = -MANUAL_SPEED
            pressed_direction.append("BACKWARD")
        if input_state["strafe_left"]:
            strafe_speed = MANUAL_SPEED
            pressed_direction.append("LEFT")
        if input_state["strafe_right"]:
            strafe_speed = -MANUAL_SPEED
            pressed_direction.append("RIGHT")
        if input_state["up"]:
            altitude_delta = 1.0
            pressed_direction.append("UP")
        if input_state["down"]:
            altitude_delta = -1.0
            pressed_direction.append("DOWN")
        if input_state["yaw_left"]:
            yaw_rate = MANUAL_YAW_RATE
            pressed_direction.append("YAW_LEFT")
        if input_state["yaw_right"]:
            yaw_rate = -MANUAL_YAW_RATE
            pressed_direction.append("YAW_RIGHT")

        hover = input_state["hover"]
        if hover:
            forward_speed = strafe_speed = yaw_rate = 0.0
            pressed_direction = ["HOVER"]

        return {
            "forward_speed": forward_speed,
            "strafe_speed": strafe_speed,
            "yaw_rate": yaw_rate,
            "altitude_delta": altitude_delta,
            "hover": hover,
            "land": input_state["land_pressed"],
            "reset": input_state["reset_pressed"],
            "pressed_direction": "+".join(pressed_direction) if pressed_direction else "-",
        }
