"""Headless step response of PyBulletDrone: hover, command 2 m/s for 1s,
then 0 for 1s, printing velocity/displacement/tilt every 0.1s. Shows how
long the drone physically needs to get out of the way (e.g. for the
Giant Fiber dodge).

Run under an env with pybullet + opencv:
    python Simulator/Tools/drone_step_response.py strafe|forward [CONST=value ...]
Optional CONST=value overrides Simulator/pybullet_drone.py constants,
e.g. TILT_KP=0.2 TILT_KD=0.04.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import pybullet as p

_real_connect = p.connect
p.connect = lambda *_a, **_k: _real_connect(p.DIRECT)

import Simulator.pybullet_drone as D
from Simulator.pybullet_simulator import PyBulletSimulator

axis = sys.argv[1] if len(sys.argv) > 1 else "strafe"
for kv in sys.argv[2:]:
    k, v = kv.split("=")
    setattr(D, k, float(v))

sim = PyBulletSimulator()
sim.connect()
drone = sim.create_drone(start_pos=(0, 0, 0.05))
drone.takeoff()
while drone.state != "flying":
    drone.step()
for _ in range(240):
    drone.step()

p0 = drone.get_state()["position"]
for i in range(480):
    speed = 2.0 if i < 240 else 0.0
    if axis == "strafe":
        drone.move_left(speed)
    else:
        drone.move_forward(speed)
    drone.step()
    if i % 24 == 23:
        s = drone.get_state()
        roll, pitch, _ = p.getEulerFromQuaternion(s["orientation"])
        print(f"t={(i + 1) / 240:.1f}s vx={s['actual_vx']:+.2f} vy={s['actual_vy']:+.2f} "
              f"disp=({s['position'][0] - p0[0]:+.2f},{s['position'][1] - p0[1]:+.2f}) "
              f"roll={math.degrees(roll):+.1f} pitch={math.degrees(pitch):+.1f}")
