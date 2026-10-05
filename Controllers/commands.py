"""
The command dict every controller's decide() returns, and the one function
that turns it into DroneInterface calls.

Shared by main.py, Simulator/Evaluation/run_trials.py, the sim tests and the
real-Tello flight tests. Deliberately free of pybullet/cv2 imports so the
Tello scripts can use it from an env that only has djitellopy/opencv/numpy.
"""

# States where the navigation FSM is already actively steering away from
# something - SafetyLayer won't layer its own (possibly disagreeing) turn
# decision on top of any of these, see already_avoiding in its apply().
AVOIDING_STATES = ("AVOID_LEFT", "AVOID_RIGHT", "BOUNDARY_RETURN", "WALL_ESCAPE", "EMERGENCY_ESCAPE", "ESCAPE")

EMPTY_CMD = {"forward_speed": 0.0, "strafe_speed": 0.0, "yaw_rate": 0.0,
             "altitude_delta": 0.0, "hover": False, "land": False, "reset": False,
             "pressed_direction": "-"}


def apply_command(drone, cmd):
    """Dispatches one FINAL command dict (already passed through the
    safety layer) to the drone interface. Returns True if a reset
    happened, so the caller can clear stale vision/controller state."""

    if cmd["reset"]:
        drone.reset()
        drone.takeoff()
        return True

    if cmd["land"]:
        drone.land()
        return False

    if drone.state != "flying":
        # Still taking off / landing / recovering from an emergency - the
        # low-level state machine and safety net own the drone right now,
        # not the controller.
        return False

    if cmd["hover"]:
        drone.hover()
    else:
        drone.move_forward(cmd["forward_speed"])
        if cmd["strafe_speed"] > 0:
            drone.move_left(cmd["strafe_speed"])
        elif cmd["strafe_speed"] < 0:
            drone.move_right(-cmd["strafe_speed"])
        else:
            drone.move_left(0)
        if cmd["yaw_rate"] > 0:
            drone.turn_left(cmd["yaw_rate"])
        elif cmd["yaw_rate"] < 0:
            drone.turn_right(-cmd["yaw_rate"])
        else:
            drone.turn_left(0)

    if cmd["altitude_delta"] > 0:
        drone.move_up()
    elif cmd["altitude_delta"] < 0:
        drone.move_down()
    else:
        drone.relax_altitude()  # drift back toward NORMAL_ALTITUDE when idle

    return False
