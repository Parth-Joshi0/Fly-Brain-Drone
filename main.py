"""
Interactive, visual run of the FlyBrain drone sim:

    Keyboard/Autonomous intent -> SafetyLayer (optic flow) -> Drone Interface -> PyBullet

Starts in MANUAL mode (keyboard control) - nothing drives the drone until
you tell it to. Press M to switch to AUTONOMOUS mode. In BOTH modes, every
command passes through the optic-flow SafetyLayer before reaching the
drone - it can override your keyboard input if something is too close,
even while you're holding a movement key. Opens a PyBullet GUI window
plus two debug windows (camera feed with a telemetry overlay, and a
color-coded optical-flow visualization).

Manual controls:
    Up/Down       move forward / backward (relative to the drone's CURRENT heading)
    Left/Right    strafe left / right (relative to the drone's CURRENT heading)
    W/S           altitude up / down
    Q/E           yaw left / right
    Space         hover
    L             land
    R             reset (re-takes off after resetting)
    M             switch MANUAL <-> AUTONOMOUS

Run:
    python main.py
"""

import time

import cv2
import pybullet as p
import pybullet_data

from simulation.environment import build_environment
from interfaces.pybullet_drone import PyBulletDrone, PHYSICS_DT, GRAVITY
from controllers.reflex_controller import ReflexController
from controllers.manual_controller import ManualController
from controllers.safety_layer import SafetyLayer
from vision.optical_flow import compute_flow, derotate_flow, region_flow_strengths, FlowVisualizer

DECISION_INTERVAL_STEPS = 8  # 240Hz physics / 8 = 30Hz decision loop, in the
                              # ~20-30 FPS range requested for the camera


def draw_debug_overlay(frame, mode, state, raw_cmd, final_cmd, safety_info, left, center, right):
    dist = state["min_obstacle_distance"]
    dist_str = f"{dist:.2f}m" if dist is not None else "n/a"
    lines = [
        f"MODE: {mode.upper()}  (M to switch)   state: {state['flight_state']}",
        f"keyboard cmd: {raw_cmd.get('pressed_direction', '(autonomous)')}"
        f"  (fwd={raw_cmd['forward_speed']:.2f} strafe={raw_cmd['strafe_speed']:.2f} yaw={raw_cmd['yaw_rate']:.2f})",
        f"actual cmd sent: fwd={final_cmd['forward_speed']:.2f} strafe={final_cmd['strafe_speed']:.2f} "
        f"yaw={final_cmd['yaw_rate']:.2f}",
        f"yaw: {state['yaw_degrees']:.0f}deg   alt: {state['altitude']:.2f}m -> {state['target_altitude']:.2f}m",
        f"flow  LEFT={left:.2f}  CENTER={center:.2f}  RIGHT={right:.2f}",
        f"safety: {safety_info['level']}   avoidance active: {'YES' if safety_info['active'] else 'NO'}",
        f"physics dist: {dist_str}  collided: {state['collided']}",
    ]
    out = frame.copy()
    for i, line in enumerate(lines):
        color = (0, 0, 255) if safety_info["active"] else (0, 255, 0)
        cv2.putText(out, line, (6, 16 + i * 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, color, 1, cv2.LINE_AA)
    return out


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
    autonomous = ReflexController()
    safety = SafetyLayer()
    flow_viz = FlowVisualizer(drone.camera.width, drone.camera.height)

    mode = "manual"  # starts in manual - nothing drives the drone until
                      # you press a key or switch to autonomous with M
    drone.takeoff()  # takes off and hovers; stays put with no keys held
    prev_gray = None
    left = center = right = 0.0
    empty_cmd = {"forward_speed": 0.0, "strafe_speed": 0.0, "yaw_rate": 0.0,
                 "altitude_delta": 0.0, "hover": False, "land": False, "reset": False,
                 "pressed_direction": "-"}
    raw_cmd = final_cmd = dict(empty_cmd)
    safety_info = {"level": "CLEAR", "active": False}

    step_count = 0
    while True:
        drone.step()

        if step_count % DECISION_INTERVAL_STEPS == 0:
            frame = drone.get_camera_frame()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if prev_gray is not None:
                flow = compute_flow(prev_gray, gray)
                yaw_rate = drone.get_state()["yaw_rate"]
                flow = derotate_flow(flow, yaw_rate, DECISION_INTERVAL_STEPS * PHYSICS_DT)
                left, center, right = region_flow_strengths(flow)
                cv2.imshow("Optical Flow", flow_viz.render(flow))
            prev_gray = gray

            keys = p.getKeyboardEvents()
            if ord('m') in keys and keys[ord('m')] & p.KEY_WAS_TRIGGERED:
                mode = "autonomous" if mode == "manual" else "manual"
                print(f"--- switched to {mode.upper()} mode ---", flush=True)

            # 1. Get raw intent (keyboard command or autonomous request)
            if mode == "manual":
                raw_cmd = manual.decide(keys)
            else:
                raw_cmd = autonomous.decide(left, center, right)

            # 2. Obstacle safety/reflex layer - can override the raw
            # intent regardless of its source. This runs every decision
            # cycle in BOTH modes, so holding a movement key toward a wall
            # cannot fly the drone into it.
            final_cmd, safety_info = safety.apply(raw_cmd, left, center, right)

            # 3. Send the final (possibly overridden) command to the drone
            if apply_command(drone, final_cmd):
                prev_gray = None
                autonomous.reset()
                safety.reset()

            state = drone.get_state()
            cv2.imshow(
                "Drone Camera",
                draw_debug_overlay(frame, mode, state, raw_cmd, final_cmd, safety_info, left, center, right),
            )
            cv2.waitKey(1)

            if step_count % (DECISION_INTERVAL_STEPS * 15) == 0:
                print(
                    f"[{mode:>10}][{state['flight_state']:>10}][{safety_info['level']:>7}] "
                    f"alt={state['altitude']:.2f}m yaw={state['yaw_degrees']:.0f}deg "
                    f"speed={state['horizontal_speed']:.2f}m/s "
                    f"pos=({state['position'][0]:.1f},{state['position'][1]:.1f}) "
                    f"collided={state['collided']} avoidance={safety_info['active']} "
                    f"flow L/C/R={left:.2f}/{center:.2f}/{right:.2f}",
                    flush=True,
                )

            if state["position"][0] >= env["goal_x"]:
                print("Course completed!", flush=True)
                drone.land()

        step_count += 1
        time.sleep(PHYSICS_DT)


if __name__ == "__main__":
    main()
