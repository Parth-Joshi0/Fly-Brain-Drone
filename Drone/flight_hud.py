"""
fly_tello.py's on-screen overlay: what the drone is doing and why, drawn
onto the camera picture before it's shown.
"""

import cv2

YELLOW = (0, 255, 255)
RED = (0, 0, 255)
ORANGE = (0, 165, 255)
WHITE = (255, 255, 255)


def _line(frame, text, y, color=YELLOW, scale=0.65, thickness=2):
    cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


def draw_hud(frame, flying, behaviour, fear, fps, sent):
    """`fear` is None without --scared/--stabilize; `sent` is the
    (lr, fb, ud, yaw) RC command actually sent this picture."""
    h = frame.shape[0]
    scared = behaviour.state == "SCARED"

    _line(frame, f"MODE: {'FLYING' if flying else 'DRY RUN'}", 30)
    _line(frame, f"STATE: {behaviour.state}", 60)
    _line(frame, f"HUNGER: {int(behaviour.hunger)}%", 90)
    target_label = behaviour.last_target_label
    _line(frame, f"TARGET: {'NONE' if target_label is None else target_label}", 120)
    # How much of the screen the banana fills
    _line(frame, f"SIZE: {behaviour.last_box_ratio:.1%}", 150)

    if fear is not None:
        _line(frame,
              f"BRAIN: {fear.brain.state if fear.armed else 'ARMING'}  "
              f"LOOM L/C/R: {fear.expansion['left']:.1f}/"
              f"{fear.expansion['center']:.1f}/"
              f"{fear.expansion['right']:.1f}  "
              f"ESCAPE: {fear.escape_level:.2f}/0.60  "
              f"SCARES: {fear.scares}"
              + ("  (moving - ignoring)" if fear.self_moving else ""),
              180, color=RED if scared else YELLOW)

        if fear.stabilize:
            d = fear.brain.dng02
            _line(frame,
                  f"DNg02 L={d.get('n_left', 0):2d} R={d.get('n_right', 0):2d}  "
                  f"steer={d.get('steer', 0.0):+.2f}  "
                  f"rot={fear.dng02_rotation:+.2f}px  "
                  f"yaw {int(fear.intended_yaw_rc):+d} + {fear.dng02_yaw_rc:+d}",
                  210)

        _line(frame, f"LOOP: {fps:.0f}/s  BRAIN: {fear.brain_ms:.0f} ms", 240)

        if scared:
            _line(frame, "SCARED! DNp01 fired - BACKING AWAY", h // 2,
                  color=RED, scale=1.2, thickness=3)
        elif behaviour.state == "WAIT":
            _line(frame, "WAITING - is it safe? (need to see the banana)", h // 2,
                  color=ORANGE, scale=0.9)

    # Movement commands (what was actually sent)
    _line(frame, f"LR:{sent[0]}  FB:{sent[1]}  UD:{sent[2]}  YAW:{sent[3]}", h - 20,
          color=WHITE, scale=0.5, thickness=1)
