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
from vision.optical_flow import compute_flow, derotate_flow, grid_flow_strengths
from evaluation.metrics import TrialMetrics, print_summary
from main import apply_command, HOVER_BEFORE_EXPLORE_CYCLES, EMPTY_CMD

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
    flow = {"left": 0.0, "right": 0.0, "top": 0.0, "bottom": 0.0, "center": 0.0}
    flying_cycle_count = 0

    physics_steps = int(MAX_TRIAL_SECONDS / PHYSICS_DT)

    for i in range(physics_steps):
        drone.step()

        if i % DECISION_INTERVAL_STEPS == 0:
            frame = drone.get_camera_frame()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            is_avoidance = False

            state_now = drone.get_state()
            if state_now["flight_state"] == "flying":
                flying_cycle_count += 1
            else:
                flying_cycle_count = 0

            if prev_gray is not None:
                raw_flow = compute_flow(prev_gray, gray)
                yaw_rate = state_now["yaw_rate"]
                derotated = derotate_flow(raw_flow, yaw_rate, DECISION_INTERVAL_STEPS * PHYSICS_DT)
                flow = grid_flow_strengths(derotated)

                # Skip the safety layer during the pre-exploration hover -
                # see the matching comment in main.py for why (its stuck-
                # detector would otherwise prime on the intentional hover).
                exploring = flying_cycle_count > HOVER_BEFORE_EXPLORE_CYCLES
                if exploring:
                    raw_cmd = controller.decide(flow, state_now)
                    cmd, safety_info = safety.apply(raw_cmd, flow, state_now["position"])
                else:
                    cmd = dict(EMPTY_CMD)
                    safety_info = {"level": "CLEAR", "active": False, "direction": "FORWARD", "stuck": False}
                apply_command(drone, cmd)
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
