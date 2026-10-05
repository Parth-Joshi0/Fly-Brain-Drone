"""Headless end-to-end test of banana seek-and-eat in the simulator: the
BananaModel detector on the sim camera -> food_orbit.py's behaviour, with the
fly brain (escape + DNg02) running alongside - main.py's USE_BANANA loop, in a
PyBullet DIRECT session. No GUI, no keyboard.

Scenarios:
    eat     nothing thrown at it. Pass = SEARCH/APPROACH -> FEED -> DONE ->
            lands, no collision, and flying up to the banana didn't fire
            the Giant Fiber. Firings at other times are printed, not failed
            - see "Giant Fiber firings" in the summary.
    scare   the click-test box is thrown at it a few seconds into FEED.
            Pass = the brain fires ESCAPE, the box misses, food_orbit goes
            SCARED -> WAIT -> back to eating, and it still finishes and lands.

Run under an env with pybullet + opencv + torch + ultralytics (the brain
subprocess finds its own brian2 env, see NeuralPathways/flybrain_controller.py):
    python Simulator/Tests/test_banana_sim.py eat|scare [-v] [--gui] [--no-brain] [--no-optomotor]
-v prints every decision cycle instead of every 15th. --no-brain runs
food_orbit alone (no fear reflex, so only `eat` makes sense); --no-optomotor
leaves the DNg02 half out of the brain.
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

import main as M
from BananaModel.liveDetect import BananaDetector
from Simulator.banana_seek_controller import BananaSeekController
from Simulator.pybullet_simulator import PyBulletSimulator
from NeuralPathways.EscapeNeuron.optical_flow import (LoomingDetector, compute_flow, derotate_flow,
                                                      grid_flow_strengths, signed_hemifield_flow)

FLAGS = ("-v", "--gui", "--no-brain", "--no-optomotor")
args = [a for a in sys.argv[1:] if a not in FLAGS]
MODE = args[0] if args else "eat"
VERBOSE = "-v" in sys.argv
USE_BRAIN = "--no-brain" not in sys.argv
OPTOMOTOR = "--no-optomotor" not in sys.argv

SCARE_AFTER_FEED_S = 3.0
MAX_SIM_S = 90.0

sim = PyBulletSimulator(banana_position=M.BANANA_POSITION)
env = sim.connect()
drone = sim.create_drone(start_pos=(0, 0, 0.05))
step = 0
clock = lambda: step * sim.physics_dt
brain = None
if USE_BRAIN:
    from NeuralPathways.flybrain_controller import FlyBrainController
    brain = FlyBrainController(bounds=env["bounds"], optomotor=OPTOMOTOR)
    brain._log_spikes = lambda *a: None
ctl = BananaSeekController(BananaDetector(), brain=brain, clock=clock)
looming = LoomingDetector()
drone.takeoff()

prev_gray = None
flow = {"left": 0.0, "right": 0.0, "top": 0.0, "bottom": 0.0, "center": 0.0,
        "expansion_left": 0.0, "expansion_center": 0.0, "expansion_right": 0.0}
flying = cycle = 0
prev_state = prev_brain_state = None
firings = []   # (t, food_orbit state when the brain's Giant Fiber fired)
feed_start = obstacle_spawned = None
states_seen, escapes, min_gap = [], [], float("inf")
obs_id = None
dng02_yaw_max = 0.0

while clock() < MAX_SIM_S:
    drone.step()
    if step % M.DECISION_INTERVAL_STEPS == 0:
        dt = M.DECISION_INTERVAL_STEPS * sim.physics_dt
        frame = drone.get_camera_frame()
        cap = drone.get_state()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        expansion = looming.update(gray, cap["orientation"], dt)
        if prev_gray is not None:
            derotated = derotate_flow(compute_flow(prev_gray, gray), cap["yaw_rate"], dt)
            flow = grid_flow_strengths(derotated)
            hemi = signed_hemifield_flow(derotated)
            flow["rotation"], flow["translation"] = hemi["rotation"], hemi["translation"]
        flow.update({f"expansion_{k}": v for k, v in expansion.items()})
        prev_gray = gray

        s = drone.get_state()
        flying = flying + 1 if s["flight_state"] == "flying" else 0
        food_state = ctl.food.state   # before decide(): a scare changes it
        if flying > M.HOVER_BEFORE_EXPLORE_CYCLES:
            ctl.see(frame)
            final = ctl.decide(flow, s)
            if brain is not None:
                dng02_yaw_max = max(dng02_yaw_max, abs(brain.dng02_yaw_rate()))
        else:
            final = dict(M.EMPTY_CMD)
        M.apply_command(drone, final)

        state = ctl.state
        if state != prev_state:
            states_seen.append((round(clock(), 1), state))
        if state == "FEED" and feed_start is None:
            feed_start = clock()
        if state == "ESCAPE" and prev_state != "ESCAPE":
            escapes.append(round(clock(), 1))
        brain_state = brain.state if brain else None
        if brain_state == "ESCAPE" and prev_brain_state != "ESCAPE":
            firings.append((round(clock(), 1), food_state))
        prev_brain_state = brain_state

        if (MODE == "scare" and obstacle_spawned is None and feed_start is not None
                and clock() - feed_start >= SCARE_AFTER_FEED_S):
            sim._spawn_test_obstacle(s["position"], s["yaw_degrees"])
            obs_id = sim._test_obstacles[-1]["id"]
            obstacle_spawned = clock()
        sim.update_test_obstacles(s["position"], s["yaw_degrees"], dt)
        if obs_id is not None and any(o["id"] == obs_id for o in sim._test_obstacles):
            cp = p.getClosestPoints(drone.body.id, obs_id, 10.0)
            if cp:
                min_gap = min(min_gap, min(c[8] for c in cp))

        if VERBOSE or cycle % 15 == 0 or state != prev_state:
            food = ctl.food
            det = max((d.det_conf for d in ctl.detections), default=0.0)
            print(f"t={clock():5.1f} {state:8s} hunger={food.hunger:5.1f} bananas={len(ctl.detections)} "
                  f"det={det:.2f} size={food.last_box_ratio:.3f} "
                  f"cmd fwd={final['forward_speed']:+.2f} strafe={final['strafe_speed']:+.2f} "
                  f"yaw={final['yaw_rate']:+.2f} up={final['altitude_delta']:+.0f} "
                  f"pos=({s['position'][0]:+.2f},{s['position'][1]:+.2f}) alt={s['altitude']:.2f} "
                  f"vx={s['actual_vx']:+.2f} exp=L{flow['expansion_left']:.2f}/C{flow['expansion_center']:.2f}"
                  f"/R{flow['expansion_right']:.2f} "
                  f"escape={brain.escape_level if brain else 0:.2f} {s['flight_state']}", flush=True)
        prev_state = state
        cycle += 1
        if s["flight_state"] == "landed" or s["collided"]:
            break
    step += 1

s = drone.get_state()
names = [st for _, st in states_seen]
print(f"\n===== SUMMARY [{MODE}] brain={USE_BRAIN} optomotor={OPTOMOTOR and USE_BRAIN} =====")
print(f"states: {states_seen}")
print(f"Giant Fiber firings (t, food state): {firings}")
print(f"dodges flown at t={escapes}; scares food_orbit took: {ctl.scares}, turned down: {ctl.ignored_scares}")
if obstacle_spawned is not None:
    print(f"box thrown at t={obstacle_spawned:.1f}; closest approach {min_gap:.2f}m")
print(f"largest DNg02 yaw correction: {dng02_yaw_max:.3f} rad/s")
print(f"final: flight_state={s['flight_state']} collided={s['collided']} hunger={ctl.food.hunger:.0f} "
      f"sim time {clock():.1f}s")

checks = {
    "ate (reached FEED)": "FEED" in names,
    "got full (reached DONE)": "DONE" in names,
    "landed": s["flight_state"] in ("landing", "landed"),
    "no collision": not s["collided"],
}
if MODE == "eat":
    checks["flying to the banana didn't fire the Giant Fiber"] = not any(st == "APPROACH" for _, st in firings)
else:
    checks["brain fired ESCAPE at the box"] = bool(escapes) and escapes[0] >= obstacle_spawned
    checks["box missed"] = min_gap > 0.0
    after = names[names.index("ESCAPE"):] if "ESCAPE" in names else []
    checks["came back to eat after the scare"] = "WAIT" in after and "FEED" in after
for name, ok in checks.items():
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
ctl.close()
sys.exit(0 if all(checks.values()) else 1)
