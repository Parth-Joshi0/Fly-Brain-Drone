"""Runs the real main.py - GUI, key handling, NEURON_TEST_MODE and all -
with the left-click that spawns a test obstacle faked at chosen decision
cycles, then prints how each box went. Catches problems in main.py's own
loop that Simulator/Tests/test_escape_sim.py (which re-implements that loop) can't.

Opens the PyBullet GUI and camera windows. Run under an env with
pybullet + opencv:
    python Simulator/Tests/test_main_gui.py [click_cycle,click_cycle,...]
Default clicks: 200,330,460 (takeoff + the pre-explore hover take ~140
decision cycles; a box sent before then is never seen by the brain).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import NeuralPathways.flybrain_controller as fbc
import Simulator.pybullet_simulator as ps
import main

CLICKS = sorted(int(c) for c in (sys.argv[1] if len(sys.argv) > 1 else "200,330,460").split(","))
RUN_AFTER_LAST_CLICK = 130   # decision cycles - enough for the last box to arrive

cycle = {"n": 0}
boxes = []   # per click: {"cycle", "escaped", "collided"}
sim_ref = {}

_update = ps.PyBulletSimulator.update_test_obstacles
def update_test_obstacles(self, position, yaw_degrees, dt):
    sim_ref["sim"] = self
    cycle["n"] += 1
    if boxes and self._drone.collided:
        boxes[-1]["collided"] = True
    if cycle["n"] > CLICKS[-1] + RUN_AFTER_LAST_CLICK:
        raise KeyboardInterrupt
    return _update(self, position, yaw_degrees, dt)
ps.PyBulletSimulator.update_test_obstacles = update_test_obstacles

def fake_click(self):
    if cycle["n"] in CLICKS:
        boxes.append({"cycle": cycle["n"], "escaped": None, "collided": False})
        return True
    return False
ps.PyBulletSimulator._left_click_triggered = fake_click

_decide = fbc.FlyBrainController.decide
def decide(self, flow, state=None):
    cmd = _decide(self, flow, state)
    if boxes and self.state == "ESCAPE" and boxes[-1]["escaped"] is None:
        boxes[-1]["escaped"] = self.escape_direction
    return cmd
fbc.FlyBrainController.decide = decide

main.main()

print("\n===== SUMMARY [main.py GUI, NEURON_TEST_MODE=%s] =====" % main.NEURON_TEST_MODE)
for i, b in enumerate(boxes, 1):
    print(f"box {i} (cycle {b['cycle']}): escape={b['escaped'] or 'NONE'} collided={b['collided']}")
drone = sim_ref["sim"]._drone
print(f"final: flight_state={drone.state} collided={drone.collided}")
