"""
Interactive, visual run of the FlyBrain drone sim:

    Virtual Camera -> Optic Flow -> 3D Navigation Controller -> Safety
    Override -> Drone Interface -> PyBullet Drone

Starts in AUTONOMOUS mode: takes off, hovers briefly, then explores on
its own - no keyboard input needed. Every command (autonomous or manual)
passes through the SafetyLayer before reaching the drone, so it can
override even a continuous "go forward" request if something's too
close. Opens a PyBullet GUI window plus two debug windows (camera feed
with a telemetry overlay, and a color-coded optical-flow visualization).

Keys (work in BOTH modes):
    Space     emergency hover
    L         emergency land
    R         reset (re-takes off after resetting)
    M         switch AUTONOMOUS <-> MANUAL

Manual-only movement (only read while in MANUAL mode):
    Up/Down       move forward / backward (relative to current heading)
    Left/Right    strafe left / right (relative to current heading)
    W/S           altitude up / down
    Q/E           yaw left / right

Run:
    python main.py
"""

import math
import time

import cv2
import pybullet as p
import pybullet_data

from simulation.environment import build_environment
from interfaces.pybullet_drone import PyBulletDrone, PHYSICS_DT, GRAVITY
from controllers.reflex_controller import ReflexController
from controllers.manual_controller import ManualController
from controllers.safety_layer import SafetyLayer
from vision.optical_flow import compute_flow, derotate_flow, grid_flow_strengths, FlowVisualizer

DECISION_INTERVAL_STEPS = 8   # 240Hz physics / 8 = 30Hz decision loop, in the
                               # ~20-30 FPS range requested for the camera
HOVER_BEFORE_EXPLORE_CYCLES = 45  # ~1.5s at 30Hz: "take off, hover briefly,
                                   # THEN begin exploring" (item 1)

# --- "Easy to watch" debug camera - follows the drone at a fixed angle/
# distance rather than a tight close-up, so nearby obstacles and the
# direction of travel stay visible. ---
DEBUG_CAMERA_DISTANCE = 7.0
DEBUG_CAMERA_YAW = 50
DEBUG_CAMERA_PITCH = -35
HEADING_LINE_LENGTH = 1.5  # m - length of the drawn forward-direction line

_ACTION_FOR_STATE = {
    "CRUISE": "FORWARD",
    "AVOID_LEFT": "TURN LEFT",
    "AVOID_RIGHT": "TURN RIGHT",
    "BOUNDARY_RETURN": "RETURN TO COURSE",
}

EMPTY_CMD = {"forward_speed": 0.0, "strafe_speed": 0.0, "yaw_rate": 0.0,
             "altitude_delta": 0.0, "hover": False, "land": False, "reset": False,
             "pressed_direction": "-"}


def emergency_keys(keys):
    """Space/L/R work in both manual and autonomous mode - these are the
    keys item 1 says to keep regardless of mode (emergency hover,
    emergency land, reset)."""
    def down(code):
        return code in keys and keys[code] & p.KEY_IS_DOWN

    def pressed(code):
        return code in keys and keys[code] & p.KEY_WAS_TRIGGERED

    return {"hover": down(ord(' ')), "land": pressed(ord('l')), "reset": pressed(ord('r'))}


def action_label(mode, nav_state, safety_info):
    """The single, human-readable "what is it doing right now" string -
    an emergency override (SafetyLayer) always takes priority to display
    over the navigation controller's own state, since it means the
    controller's plan just got overruled."""
    if safety_info["stuck"]:
        return f"STUCK -> {safety_info['direction']}"
    if safety_info["level"] == "DANGER":
        return f"EMERGENCY {safety_info['direction']}"
    if mode != "autonomous":
        return "MANUAL"
    return _ACTION_FOR_STATE.get(nav_state, nav_state)


def draw_debug_overlay(frame, mode, nav_state, state, final_cmd, safety_info, flow):
    dist = state["min_obstacle_distance"]
    dist_str = f"{dist:.2f}m" if dist is not None else "n/a"
    action = action_label(mode, nav_state, safety_info)
    lines = [
        f"MODE: {mode.upper()}  (M to switch)   STATE: {nav_state}",
        f"ACTION: {action}",
        f"LEFT FLOW: {flow['left']:.2f}   CENTER FLOW: {flow['center']:.2f}   RIGHT FLOW: {flow['right']:.2f}",
        f"cmd: fwd={final_cmd['forward_speed']:.2f}m/s  yaw={final_cmd['yaw_rate']:.2f}rad/s",
        f"alt: {state['altitude']:.2f}m -> {state['target_altitude']:.2f}m   yaw: {state['yaw_degrees']:.0f}deg",
        f"pos: ({state['position'][0]:.1f}, {state['position'][1]:.1f})   "
        f"forward_vel: {state['actual_vx']:.2f}m/s",
        f"top/bottom flow: T={flow['top']:.2f} B={flow['bottom']:.2f}   physics dist: {dist_str}",
        f"safety override: {'YES' if safety_info['active'] else 'NO'} ({safety_info['level']})"
        f"{'  [STUCK]' if safety_info['stuck'] else ''}   collided: {state['collided']}",
    ]
    out = frame.copy()
    for i, line in enumerate(lines):
        color = (0, 0, 255) if safety_info["active"] else (0, 255, 0)
        cv2.putText(out, line, (6, 16 + i * 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, color, 1, cv2.LINE_AA)
    return out


def update_debug_camera(position):
    """Keeps the PyBullet 3D viewport centered on the drone at a fixed
    distance/angle - "easy to watch" without the user needing to manually
    pan/zoom, and without zooming in so tight that nearby obstacles fall
    out of view."""
    p.resetDebugVisualizerCamera(
        cameraDistance=DEBUG_CAMERA_DISTANCE,
        cameraYaw=DEBUG_CAMERA_YAW,
        cameraPitch=DEBUG_CAMERA_PITCH,
        cameraTargetPosition=position,
    )


def draw_heading_line(position, yaw_degrees, line_id):
    """A short line in the PyBullet 3D view showing the drone's current
    forward direction. Reuses the same debug-item id every call
    (replaceItemUniqueId) so it updates in place instead of accumulating
    thousands of lines."""
    yaw = math.radians(yaw_degrees)
    end = [
        position[0] + HEADING_LINE_LENGTH * math.cos(yaw),
        position[1] + HEADING_LINE_LENGTH * math.sin(yaw),
        position[2],
    ]
    if line_id is None:
        return p.addUserDebugLine(position, end, lineColorRGB=[1, 1, 0], lineWidth=3)
    return p.addUserDebugLine(position, end, lineColorRGB=[1, 1, 0], lineWidth=3,
                               replaceItemUniqueId=line_id)


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


def main():
    p.connect(p.GUI)
    time.sleep(0.5)  # let the renderer finish initializing before loading meshes
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0, 0, -GRAVITY)

    env = build_environment()
    drone = PyBulletDrone(
        start_pos=(0, 0, 0.05),
        ground_id=env["plane"],
        obstacle_ids=set(env["obstacles"]),
    )
    manual = ManualController()
    autonomous = ReflexController(bounds=env["bounds"])
    safety = SafetyLayer()
    flow_viz = FlowVisualizer(drone.camera.width, drone.camera.height)

    mode = "autonomous"  # fully autonomous by default (item 1) - press M for manual
    drone.takeoff()
    prev_gray = None
    flow = {"left": 0.0, "right": 0.0, "top": 0.0, "bottom": 0.0, "center": 0.0}
    final_cmd = dict(EMPTY_CMD)
    safety_info = {"level": "CLEAR", "active": False, "direction": "FORWARD", "stuck": False}
    flying_cycle_count = 0  # counts decision cycles spent in "flying" state,
                             # for the "hover briefly before exploring" grace period
    heading_line_id = None

    step_count = 0
    while True:
        drone.step()

        if step_count % DECISION_INTERVAL_STEPS == 0:
            frame = drone.get_camera_frame()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if prev_gray is not None:
                raw_flow = compute_flow(prev_gray, gray)
                yaw_rate = drone.get_state()["yaw_rate"]
                derotated = derotate_flow(raw_flow, yaw_rate, DECISION_INTERVAL_STEPS * PHYSICS_DT)
                flow = grid_flow_strengths(derotated)
                cv2.imshow("Optical Flow", flow_viz.render(derotated))
            prev_gray = gray

            keys = p.getKeyboardEvents()
            if ord('m') in keys and keys[ord('m')] & p.KEY_WAS_TRIGGERED:
                mode = "manual" if mode == "autonomous" else "autonomous"
                print(f"--- switched to {mode.upper()} mode ---", flush=True)

            state_now = drone.get_state()
            if state_now["flight_state"] == "flying":
                flying_cycle_count += 1
            else:
                flying_cycle_count = 0

            # Item 1: take off, hover briefly, THEN start exploring. During
            # that deliberate hover, skip the safety layer entirely rather
            # than feeding it "not moving" position samples - otherwise
            # its stuck-detector's window fills with the *intentional*
            # hover before real navigation ever gets a turn, and it
            # immediately (and permanently) thinks it's stuck the moment
            # exploring starts.
            exploring = mode == "autonomous" and flying_cycle_count > HOVER_BEFORE_EXPLORE_CYCLES

            if mode == "manual":
                raw_cmd = manual.decide(keys)
            elif exploring:
                raw_cmd = autonomous.decide(flow, state_now)
                raw_cmd.update(emergency_keys(keys))  # Space/L/R still work
            else:
                raw_cmd = dict(EMPTY_CMD)
                raw_cmd.update(emergency_keys(keys))

            if mode == "manual" or exploring:
                # 2. Obstacle safety/reflex layer - final override
                # authority (this is what makes "hold/request forward
                # into a wall" impossible even while flying itself).
                # already_avoiding tells it the FSM is already turning
                # away from something, so it won't add a second,
                # possibly-disagreeing turn decision on top.
                already_avoiding = autonomous.state in ("AVOID_LEFT", "AVOID_RIGHT", "BOUNDARY_RETURN")
                final_cmd, safety_info = safety.apply(raw_cmd, flow, state_now["position"], already_avoiding)
            else:
                final_cmd = raw_cmd
                safety_info = {"level": "CLEAR", "active": False, "direction": "FORWARD", "stuck": False}

            # 3. Send the final (possibly overridden) command to the drone
            if apply_command(drone, final_cmd):
                prev_gray = None
                flying_cycle_count = 0
                autonomous.reset()
                safety.reset()

            state = drone.get_state()
            update_debug_camera(state["position"])
            heading_line_id = draw_heading_line(state["position"], state["yaw_degrees"], heading_line_id)

            cv2.imshow("Drone Camera", draw_debug_overlay(
                frame, mode, autonomous.state, state, final_cmd, safety_info, flow))
            cv2.waitKey(1)

            if step_count % (DECISION_INTERVAL_STEPS * 15) == 0:
                action = action_label(mode, autonomous.state, safety_info)
                print(
                    f"[{mode:>10}][STATE={autonomous.state:>15}][ACTION={action:>18}] "
                    f"alt={state['altitude']:.2f}m yaw={state['yaw_degrees']:.0f}deg "
                    f"speed={state['horizontal_speed']:.2f}m/s "
                    f"pos=({state['position'][0]:.1f},{state['position'][1]:.1f}) "
                    f"collided={state['collided']} avoidance={safety_info['active']} stuck={safety_info['stuck']} "
                    f"LEFT={flow['left']:.2f} CENTER={flow['center']:.2f} RIGHT={flow['right']:.2f}",
                    flush=True,
                )

            if state["position"][0] >= env["goal_x"]:
                print("Course completed!", flush=True)
                drone.land()

        step_count += 1
        time.sleep(PHYSICS_DT)


if __name__ == "__main__":
    main()
