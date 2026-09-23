"""
Keyboard controller for manually flying the drone in the PyBullet GUI -
useful for testing without the reflex/FlyBrain controller in the loop at
all. Deliberately simulator-specific (it reads raw PyBullet keyboard
state), unlike the drone interface it drives.

Keys:
    Up/Down       forward / backward
    Left/Right    strafe left / right
    W/S           altitude up / down
    Q/E           yaw left / right
    Space         hover (zeroes forward/strafe/yaw, holds altitude)
    L             land
    R             reset
"""

import pybullet as p

MANUAL_SPEED = 0.4      # m/s - deliberately slow, see item 5 (keep it slow
                          # while testing obstacle avoidance)
MANUAL_YAW_RATE = 0.6    # rad/s


class ManualController:
    def decide(self, keys):
        """keys: the dict returned by p.getKeyboardEvents()."""

        def down(code):
            return code in keys and keys[code] & p.KEY_IS_DOWN

        def pressed(code):
            return code in keys and keys[code] & p.KEY_WAS_TRIGGERED

        forward_speed = 0.0
        strafe_speed = 0.0
        yaw_rate = 0.0
        altitude_delta = 0.0
        pressed_direction = []

        if down(p.B3G_UP_ARROW):
            forward_speed = MANUAL_SPEED
            pressed_direction.append("FORWARD")
        if down(p.B3G_DOWN_ARROW):
            forward_speed = -MANUAL_SPEED
            pressed_direction.append("BACKWARD")
        if down(p.B3G_LEFT_ARROW):
            strafe_speed = MANUAL_SPEED
            pressed_direction.append("LEFT")
        if down(p.B3G_RIGHT_ARROW):
            strafe_speed = -MANUAL_SPEED
            pressed_direction.append("RIGHT")
        if down(ord('w')):
            altitude_delta = 1.0
            pressed_direction.append("UP")
        if down(ord('s')):
            altitude_delta = -1.0
            pressed_direction.append("DOWN")
        if down(ord('q')):
            yaw_rate = MANUAL_YAW_RATE
            pressed_direction.append("YAW_LEFT")
        if down(ord('e')):
            yaw_rate = -MANUAL_YAW_RATE
            pressed_direction.append("YAW_RIGHT")

        hover = down(ord(' '))
        if hover:
            forward_speed = strafe_speed = yaw_rate = 0.0
            pressed_direction = ["HOVER"]

        return {
            "forward_speed": forward_speed,
            "strafe_speed": strafe_speed,
            "yaw_rate": yaw_rate,
            "altitude_delta": altitude_delta,
            "hover": hover,
            "land": pressed(ord('l')),
            "reset": pressed(ord('r')),
            "pressed_direction": "+".join(pressed_direction) if pressed_direction else "-",
        }
