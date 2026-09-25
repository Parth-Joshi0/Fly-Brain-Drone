"""Drives the LC4/LPLC2 -> DNp01/03/06 circuit (fly_brain_controller.py)
directly with fixed looming inputs and reports spikes per output neuron,
escape/yaw/forward. No simulator involved.

Run under the brian2 env, from anywhere:
    conda run -n brian2 python Testing/test_brain_circuit.py
"""
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fly_brain_controller as fbc

ESCAPE_STATE_THRESHOLD = 0.6  # controllers/flybrain_controller.py
STEPS = 50                    # x 20ms = 1s of sustained stimulus

circuit = fbc._load_circuit()
i_pre, i_post, w = fbc._load_synapses(circuit["root_to_local"])
n_in = circuit["n_left"] + circuit["n_right"]
names = [f"{n['cell_type']}_{n['side']}" for n in circuit["outputs"]]
print(f"synapses: {len(w)}  inputs: {n_in} (L={circuit['n_left']} R={circuit['n_right']})")
print("summed input synapse weight per output neuron, by source:")
for k, name in enumerate(names):
    m = i_post == n_in + k
    by_src = defaultdict(float)
    for pre, ww in zip(i_pre[m], w[m]):
        src = circuit["inputs"][pre] if pre < n_in else None
        by_src[f"{src['cell_type']}_{src['side']}" if src else names[pre - n_in]] += ww
    print(f"  {name:12s} " + ", ".join(f"{s}:{v:.0f}" for s, v in sorted(by_src.items())))

t0 = time.time()
c = fbc.FlyBrainController()
print(f"\nnetwork build: {time.time() - t0:.1f}s")
print(f"\n{STEPS} steps ({STEPS * 20}ms) of sustained looming per row:")
for label, left, right in [("none", 0, 0), ("both 0.1", .1, .1), ("both 0.25", .25, .25),
                           ("both 0.5", .5, .5), ("both 1.0", 1, 1),
                           ("left 1.0", 1, 0), ("right 1.0", 0, 1)]:
    c.reset()
    totals = defaultdict(int)
    peak_escape = 0.0
    yaws = []
    t = time.time()
    for _ in range(STEPS):
        r = c.step(left, right)
        for k, v in r["spike_counts"].items():
            totals[k] += v
        peak_escape = max(peak_escape, r["escape"])
        yaws.append(r["yaw"])
    ms_per_step = (time.time() - t) / STEPS * 1000
    escaped = "YES" if peak_escape >= ESCAPE_STATE_THRESHOLD else "no"
    print(f"  {label:10s} spikes={dict(totals)}\n"
          f"             peak_escape={peak_escape:.2f} ESCAPE={escaped} "
          f"mean_yaw={np.mean(yaws):+.2f} forward={r['forward']:.2f} ({ms_per_step:.1f}ms/step)")
