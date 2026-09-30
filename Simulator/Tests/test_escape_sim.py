"""Headless end-to-end test of the FlyBrain escape: replicates main.py's
decision loop (camera -> optic flow + LoomingDetector -> FlyBrainController
-> SafetyLayer -> drone) in a PyBullet DIRECT session - no GUI, no keyboard.

Scenarios:
    hover   NEURON_TEST_MODE behavior (hover unless ESCAPE); the click-test
            obstacle is fired straight at the drone. Pass = no collision.
    fly     brain flies the drone; same obstacle fired at it head-on.
    hover2  like hover, then a second obstacle once the first dodge is over.
            Pass = both dodged, drone still flying at normal altitude.
    cruise  brain flies the course for 20s, no test obstacle. Pass = no
            escape unless something is actually close.

Run under an env with pybullet + opencv (the brain subprocess finds its own
brian2 env, see NeuralPathways/flybrain_controller.py):
    python Simulator/Tests/test_escape_sim.py hover|fly|cruise [-v] [--magnitude] [CONST=value ...]
CONST=value overrides NeuralPathways/flybrain_controller.py constants, e.g.
LOOM_EXPANSION_FLOOR=1.2. -v prints every decision cycle. --magnitude feeds
the brain plain flow magnitude instead of LoomingDetector's expansion, as a
baseline.
"""
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import cv2
import pybullet as p

GUI = "--gui" in sys.argv
if not GUI:
    _real_connect = p.connect
    p.connect = lambda *_a, **_k: _real_connect(p.DIRECT)

import NeuralPathways.flybrain_controller as fbc
import main as M
from safety_layer import SafetyLayer
from Simulator.pybullet_simulator import PyBulletSimulator
from NeuralPathways.EscapeNeuron.optical_flow import LoomingDetector, compute_flow, derotate_flow, grid_flow_strengths

args = [a for a in sys.argv[1:] if a not in ("-v", "--magnitude", "--gui")]
VERBOSE = "-v" in sys.argv
# Baseline for comparison: feed the brain plain flow magnitude (the pre-
# LoomingDetector input) with the constants it was tuned with.
MAGNITUDE = "--magnitude" in sys.argv
if MAGNITUDE:
    fbc.LOOM_EXPANSION_FLOOR, fbc.LOOM_EXPANSION_FLOOR_PER_MPS = 0.3, 0.4
    fbc.LOOM_EXPANSION_FLOOR_PER_RAD, fbc.LOOM_ROTATION_FLOOR_DECAY = 75.0, 0.0
MODE = args[0] if args else "hover"
for kv in args[1:]:
    k, v = kv.split("=")
    setattr(fbc, k, type(getattr(fbc, k))(float(v)))

TEST_MODE = MODE in ("hover", "hover2")
SPAWN = MODE != "cruise"
SECOND_SPAWN_T = 100 if MODE == "hover2" else None
RUN_CYCLES = {"hover2": 200, "cruise": 600}.get(MODE, 100)

sim = PyBulletSimulator()
env = sim.connect()
drone = sim.create_drone(start_pos=(0, 0, 0.05))
brain = fbc.FlyBrainController(bounds=env["bounds"])
brain._log_spikes = lambda *a: None
safety = SafetyLayer()
looming = LoomingDetector()
drone.takeoff()

_floor = brain._loom_floor
last_floor = [0.0]
def _record_floor(state):
    last_floor[0] = _floor(state)
    return last_floor[0]
brain._loom_floor = _record_floor

prev_gray = None
flow = {"left": 0.0, "right": 0.0, "top": 0.0, "bottom": 0.0, "center": 0.0,
        "expansion_left": 0.0, "expansion_center": 0.0, "expansion_right": 0.0}
flying = 0
start_cycle = start_pos = obs_id = prev_state = None
escape_events, states = [], {}
min_gaps = []   # closest approach per spawned obstacle (pybullet reuses body ids)

step = cycle = 0
while start_cycle is None or cycle < start_cycle + RUN_CYCLES:
    drone.step()
    if step % M.DECISION_INTERVAL_STEPS == 0:
        dt = M.DECISION_INTERVAL_STEPS * sim.physics_dt
        frame = drone.get_camera_frame()
        cap = drone.get_state()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        expansion = looming.update(gray, cap["orientation"], dt)
        if prev_gray is not None:
            flow = grid_flow_strengths(derotate_flow(compute_flow(prev_gray, gray), cap["yaw_rate"], dt))
        if MAGNITUDE:
            expansion = {k: flow[k] for k in ("left", "center", "right")}
        flow.update({f"expansion_{k}": v for k, v in expansion.items()})
        prev_gray = gray

        s = drone.get_state()
        flying = flying + 1 if s["flight_state"] == "flying" else 0
        exploring = flying > M.HOVER_BEFORE_EXPLORE_CYCLES
        if exploring:
            raw_cmd = brain.decide(flow, s)
            if TEST_MODE and brain.state != "ESCAPE":
                raw_cmd = dict(M.EMPTY_CMD)
                raw_cmd["hover"] = True
            avoiding = brain.state in ("AVOID_LEFT", "AVOID_RIGHT", "BOUNDARY_RETURN", "ESCAPE")
            final, sinfo = safety.apply(raw_cmd, flow, s["position"], avoiding)
        else:
            final, sinfo = dict(M.EMPTY_CMD), {"level": "CLEAR"}
        M.apply_command(drone, final)

        if start_cycle is None and flying == M.HOVER_BEFORE_EXPLORE_CYCLES + 10:
            start_cycle, start_pos = cycle, s["position"]
            if SPAWN:
                sim._spawn_test_obstacle(s["position"], s["yaw_degrees"])
                obs_id = sim._test_obstacles[0]["id"]
                min_gaps.append(float("inf"))
        if start_cycle is not None and cycle - start_cycle == SECOND_SPAWN_T:
            sim._spawn_test_obstacle(s["position"], s["yaw_degrees"])
            obs_id = sim._test_obstacles[-1]["id"]
            min_gaps.append(float("inf"))
        sim.update_test_obstacles(s["position"], s["yaw_degrees"], dt)

        if start_cycle is not None:
            t = cycle - start_cycle
            states[brain.state] = states.get(brain.state, 0) + 1
            gap = None
            if obs_id is not None and any(o["id"] == obs_id for o in sim._test_obstacles):
                cp = p.getClosestPoints(drone.body.id, obs_id, 10.0)
                if cp:
                    gap = min(c[8] for c in cp)
                    min_gaps[-1] = min(min_gaps[-1], gap)
            exp_str = f"{flow['expansion_left']:.2f}/{flow['expansion_center']:.2f}/{flow['expansion_right']:.2f}"
            if brain.state == "ESCAPE" and prev_state != "ESCAPE":
                escape_events.append((t, None if gap is None else round(gap, 2),
                                      None if s["min_obstacle_distance"] is None else round(s["min_obstacle_distance"], 2),
                                      exp_str, "LEFT" if brain._escape_dir > 0 else "RIGHT"))
            if VERBOSE or t % 5 == 0 or brain.state != prev_state:
                print(f"t={t:3d} gap={'%.2f' % gap if gap is not None else '  - '} "
                      f"expansion L/C/R={exp_str} floor={last_floor[0]:.2f} state={brain.state:11s} "
                      f"cmd fwd={final['forward_speed']:+.2f} strafe={final['strafe_speed']:+.2f} "
                      f"yaw={final['yaw_rate']:+.2f} safety={sinfo['level']:6s} "
                      f"pos=({s['position'][0]:+.2f},{s['position'][1]:+.2f}) alt={s['altitude']:.2f} "
                      f"{s['flight_state']} collided={s['collided']}")
            prev_state = brain.state
        cycle += 1
    step += 1

s = drone.get_state()
print(f"\n===== SUMMARY [{MODE}] =====")
print(f"state counts: {states}")
print(f"escapes (t, gap_to_test_obstacle_m, nearest_course_obstacle_m, expansion L/C/R, dodge): {escape_events}")
if SPAWN:
    print("closest approach to each test obstacle: " + ", ".join(f"{g:.2f}m" for g in min_gaps))
print(f"moved {math.dist(start_pos[:2], s['position'][:2]):.2f}m; collided={s['collided']} "
      f"final altitude={s['altitude']:.2f}m flight_state={s['flight_state']} brain state={brain.state}")
brain.close()
