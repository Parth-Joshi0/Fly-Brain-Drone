"""Calibration check for NeuralPathways/EscapeNeuron/optical_flow.LoomingDetector against
ground-truth time-to-contact. Flies scripted maneuvers (no brain, no
SafetyLayer), records frames + orientations, then runs the real
LoomingDetector over them:
    - obstacle approaches: prints expansion next to 2/TTC (what a perfect
      detector would read) as the obstacle closes in
    - maneuvers with nothing approaching (yaw, strafe, hard start/stop,
      open flight): prints the background noise distribution
Use it to re-tune LOOM_EXPANSION_* in NeuralPathways/flybrain_controller.py
after changing the camera, the detector, or the drone's dynamics.

Run under an env with pybullet + opencv:
    python NeuralPathways/EscapeNeuron/Tools/calibrate_looming.py [scenario ...]
"""
import math
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

DECISION_STEPS = 8   # main.py's DECISION_INTERVAL_STEPS
DT = DECISION_STEPS / 240
TEST_OBSTACLE_SPEED = 1.5
BOX_FACE_X = 3.6 - 0.15     # course box face minus drone half-size, drone at y=0
WALL_FACE_X = 7.9 - 0.15    # doorway wall face, drone at y~1.6

# name: (cycles, control(t) -> (forward, strafe, yaw_rate, spawn_test_obstacle))
SCENARIOS = {
    "hover_obstacle": (90, lambda t: (0, 0, 0, t == 5)),
    "fly_obstacle": (110, lambda t: (1.0, 0, 0, t == 45)),
    "fly_box": (130, lambda t: (1.0, 0, 0, False)),
    "fly_open": (150, lambda t: (0, 1.0, 0, False) if t < 60 else (1.6, 0, 0, False)),
    "yaw_hover": (90, lambda t: (0, 0, 0.8 if 10 <= t < 70 else 0, False)),
    "strafe_dodge": (70, lambda t: (0, 2.0 if 10 <= t < 40 else 0, 0, False)),
    "start_stop": (90, lambda t: (1.6 if 10 <= t < 45 else 0, 0, 0, False)),
}


def time_to_contact(name, t, vx, x, gap, collided):
    if collided:
        return None
    if name == "hover_obstacle":
        return gap / TEST_OBSTACLE_SPEED if gap is not None else math.inf
    if name == "fly_obstacle":
        return gap / max(TEST_OBSTACLE_SPEED + vx, 0.1) if gap is not None else math.inf
    if name == "fly_box":
        return (BOX_FACE_X - x) / vx if vx > 0.05 else math.inf
    if name == "fly_open":
        return (WALL_FACE_X - x) / vx if vx > 0.05 else math.inf
    return math.inf


def record(name, out_path):
    import cv2
    import pybullet as p
    real_connect = p.connect
    p.connect = lambda *_a, **_k: real_connect(p.DIRECT)
    from Simulator.pybullet_simulator import PyBulletSimulator

    cycles, control = SCENARIOS[name]
    sim = PyBulletSimulator()
    sim.connect()
    drone = sim.create_drone(start_pos=(0, 0, 0.05))
    drone.takeoff()
    while drone.state != "flying":
        drone.step()
    for _ in range(240):
        drone.step()

    frames, quats, rows = [], [], []
    obstacle_id = None
    t = step = 0
    while t < cycles:
        forward, strafe, yaw_rate, spawn = control(t)
        drone.move_forward(forward)
        drone.move_left(strafe)
        drone.turn_left(yaw_rate)
        drone.step()
        if step % DECISION_STEPS == 0:
            frame = drone.get_camera_frame()
            s = drone.get_state()
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            quats.append(s["orientation"])
            gap = None
            if obstacle_id is not None and sim._test_obstacles:
                points = p.getClosestPoints(drone.body.id, obstacle_id, 20.0)
                gap = min(c[8] for c in points) if points else None
            rows.append((s["actual_vx"], s["position"][0], np.nan if gap is None else gap, float(s["collided"])))
            if spawn:
                sim._spawn_test_obstacle(s["position"], s["yaw_degrees"])
                obstacle_id = sim._test_obstacles[0]["id"]
            sim.update_test_obstacles(s["position"], s["yaw_degrees"], DT)
            t += 1
        step += 1
    np.savez_compressed(out_path, frames=np.stack(frames), quats=np.array(quats), rows=np.array(rows))


def analyze(name, path):
    from NeuralPathways.EscapeNeuron.optical_flow import LoomingDetector
    data = np.load(path)
    detector = LoomingDetector()
    samples = []
    for k, (gray, q, (vx, x, gap, collided)) in enumerate(zip(data["frames"], data["quats"], data["rows"])):
        e = detector.update(gray, q, DT)
        ttc = time_to_contact(name, k, vx, x, None if np.isnan(gap) else gap, collided)
        if ttc is not None:
            samples.append((ttc, max(e.values())))

    if name in ("hover_obstacle", "fly_obstacle", "fly_box"):
        print(f"\n{name}: max column expansion vs ideal 2/TTC")
        for ttc, value in samples:
            if ttc < 2.2 and round(ttc * 100) % 10 < 4:
                print(f"  TTC {ttc:4.2f}s  expansion {value:5.2f}  (ideal {2 / ttc:5.2f})")
    else:
        v = np.array([value for ttc, value in samples if ttc > 2.5])
        print(f"{name:13s} background expansion: median {np.median(v):.2f}  p90 {np.percentile(v, 90):.2f}  "
              f"p98 {np.percentile(v, 98):.2f}  max {v.max():.2f}")


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--record":
        record(sys.argv[2], sys.argv[3])
        sys.exit()

    names = sys.argv[1:] or list(SCENARIOS)
    out_dir = Path(tempfile.mkdtemp(prefix="looming_calibration_"))
    procs = {n: subprocess.Popen([sys.executable, __file__, "--record", n, str(out_dir / f"{n}.npz")],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
             for n in names}
    for n, proc in procs.items():
        if proc.wait() != 0:
            sys.exit(f"recording {n} failed")
    for n in names:
        analyze(n, out_dir / f"{n}.npz")
