"""
Interactive, visual run of the FlyBrain drone sim:

    Camera -> Optical Flow -> Reflex Controller -> Drone Interface -> PyBullet

Opens a PyBullet GUI window plus two debug windows (camera feed with a
telemetry overlay, and a color-coded optical-flow visualization).

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
from vision.optical_flow import compute_flow, derotate_flow, region_flow_strengths, FlowVisualizer

DECISION_INTERVAL_STEPS = 8  # 240Hz physics / 8 = 30Hz decision loop, in the
                              # ~20-30 FPS range requested for the camera


def draw_debug_overlay(frame, state, decision, left, center, right):
    lines = [
        f"state: {state['flight_state']}  alt: {state['altitude']:.2f}m",
        f"speed: {state['horizontal_speed']:.2f}m/s  vz: {state['vertical_speed']:.2f}m/s",
        f"collided: {state['collided']}  emergency: {state['emergency']}",
        f"flow L/C/R: {left:.3f} / {center:.3f} / {right:.3f}",
        f"cmd: fwd={decision['forward_speed']:.2f} yaw={decision['yaw_rate']:.2f}",
    ]
    out = frame.copy()
    for i, line in enumerate(lines):
        cv2.putText(out, line, (6, 16 + i * 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (0, 255, 0), 1, cv2.LINE_AA)
    return out


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
    controller = ReflexController()
    flow_viz = FlowVisualizer(drone.camera.width, drone.camera.height)

    drone.takeoff()
    prev_gray = None
    last_decision = {"forward_speed": 0.0, "yaw_rate": 0.0}
    left = center = right = 0.0

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
                last_decision = controller.decide(left, center, right)

                if drone.state == "flying":
                    drone.move_forward(last_decision["forward_speed"])
                    if last_decision["yaw_rate"] > 0:
                        drone.turn_left(last_decision["yaw_rate"])
                    elif last_decision["yaw_rate"] < 0:
                        drone.turn_right(-last_decision["yaw_rate"])
                    else:
                        drone.turn_left(0)

                cv2.imshow("Optical Flow", flow_viz.render(flow))

            prev_gray = gray

            state = drone.get_state()
            cv2.imshow("Drone Camera", draw_debug_overlay(frame, state, last_decision, left, center, right))
            cv2.waitKey(1)

            if step_count % (DECISION_INTERVAL_STEPS * 15) == 0:
                print(
                    f"[{state['flight_state']:>10}] alt={state['altitude']:.2f}m "
                    f"speed={state['horizontal_speed']:.2f}m/s "
                    f"pos=({state['position'][0]:.1f},{state['position'][1]:.1f}) "
                    f"collided={state['collided']} emergency={state['emergency']} "
                    f"flow L/C/R={left:.3f}/{center:.3f}/{right:.3f}",
                    flush=True,
                )

            if state["position"][0] >= env["goal_x"]:
                print("Course completed!")
                drone.land()

        step_count += 1
        time.sleep(PHYSICS_DT)


if __name__ == "__main__":
    main()
