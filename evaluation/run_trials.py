"""
Runs many trials headlessly (no GUI window, so this is fast) and prints an
aggregate summary. Each trial gets a slightly randomized start position so
trials actually differ - otherwise every run would be identical.

Usage:
    python -m evaluation.run_trials --trials 20
"""

import argparse
import random

import cv2
import pybullet as p
import pybullet_data

from simulation.environment import build_environment
from interfaces.pybullet_drone import PyBulletDrone, PHYSICS_DT, GRAVITY
from controllers.reflex_controller import ReflexController
from controllers.safety_layer import SafetyLayer
from vision.optical_flow import compute_flow, derotate_flow, region_flow_strengths
from evaluation.metrics import TrialMetrics, print_summary

DECISION_INTERVAL_STEPS = 8  # 240Hz physics / 8 = 30Hz decision loop
MAX_TRIAL_SECONDS = 25


def run_trial():
    p.resetSimulation()
    p.setGravity(0, 0, -GRAVITY)
    env = build_environment()

    start_y = random.uniform(-0.6, 0.6)
    drone = PyBulletDrone(
        start_pos=(0, start_y, 0.05),
        ground_id=env["plane"],
        obstacle_ids=set(env["obstacles"]),
    )
    controller = ReflexController()
    safety = SafetyLayer()
    metrics = TrialMetrics(goal_x=env["goal_x"])
    metrics.start(drone.get_state()["position"])

    drone.takeoff()
    prev_gray = None

    physics_steps = int(MAX_TRIAL_SECONDS / PHYSICS_DT)

    for i in range(physics_steps):
        drone.step()

        if i % DECISION_INTERVAL_STEPS == 0:
            frame = drone.get_camera_frame()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            is_avoidance = False

            if prev_gray is not None:
                flow = compute_flow(prev_gray, gray)
                yaw_rate = drone.get_state()["yaw_rate"]
                flow = derotate_flow(flow, yaw_rate, DECISION_INTERVAL_STEPS * PHYSICS_DT)
                left, center, right = region_flow_strengths(flow)
                raw_cmd = controller.decide(left, center, right)
                cmd, safety_info = safety.apply(raw_cmd, left, center, right)

                if drone.state == "flying":
                    drone.move_forward(cmd["forward_speed"])
                    if cmd["yaw_rate"] > 0:
                        drone.turn_left(cmd["yaw_rate"])
                    elif cmd["yaw_rate"] < 0:
                        drone.turn_right(-cmd["yaw_rate"])
                    else:
                        drone.turn_left(0)
                    is_avoidance = safety_info["active"]
            prev_gray = gray

            state = drone.get_state()
            distances = [
                drone.body.closest_distance(oid, max_distance=5.0)
                for oid in env["obstacles"]
            ]
            distances = [d for d in distances if d is not None]
            min_dist = min(distances) if distances else None

            metrics.update(
                position=state["position"],
                speed=state["horizontal_speed"],
                obstacle_distance=min_dist,
                emergency_stop_count=drone.emergency_stop_count,
                collided=state["collided"],
                is_avoidance_command=is_avoidance,
            )

            if metrics.completed or drone.state == "landed":
                break

    return metrics.finalize(decision_dt=DECISION_INTERVAL_STEPS * PHYSICS_DT)


def main():
    parser = argparse.ArgumentParser(description="Run FlyBrain drone-sim evaluation trials.")
    parser.add_argument("--trials", type=int, default=20)
    args = parser.parse_args()

    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    results = []
    for i in range(args.trials):
        result = run_trial()
        print(f"Trial {i + 1}/{args.trials}: {result.summary_line()}")
        results.append(result)

    p.disconnect()
    print_summary(results)


if __name__ == "__main__":
    main()
