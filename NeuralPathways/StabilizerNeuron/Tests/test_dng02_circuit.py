"""Characterizes the DNg02 flight-motor half of fly_brain_controller.py:
does the population recruit in a graded way, does the left/right difference
respond to an asymmetric request, does any of it disturb the escape circuit,
and does it all still fit the 30Hz control budget. No simulator, no drone.

Run under the brian2 env, from anywhere:
    conda run -n brian2 python NeuralPathways/StabilizerNeuron/Tests/test_dng02_circuit.py

This is the gate for everything downstream. Nothing should reach a real
drone until this passes, because the two things it measures - the sign of the
left/right difference, and whether optic-flow drive can disturb the Giant
Fiber - are the two ways the flight test could go wrong in a way that is
hard to see from the air.

It also PRINTS the calibration constants it measures (DNG02_RECRUIT_SAT and
the two recruitment curves) as paste-ready lines, so those stay regenerable
instead of turning into magic numbers nobody can re-derive.
"""
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import fly_brain_controller as fbc

STEPS = 50              # x 20ms = 1s of sustained stimulus per row
SETTLE = 15             # windows discarded while the EMAs and membranes settle
BUDGET_MS = 33.0        # the 30Hz decision loop's slice, same as test_brain_circuit.py
LEVELS = fbc.DNG02_DRIVE_LEVELS

failures = []


def check(ok, message):
    print(f"    {'PASS' if ok else 'FAIL'}  {message}")
    if not ok:
        failures.append(message)


def sweep(c, steps=STEPS, settle=SETTLE, **kw):
    """Runs the circuit at a fixed request and returns the post-settle means."""
    c.reset()
    n_l, n_r, thrust, steer, ms = [], [], [], [], []
    dn_spikes = defaultdict(int)
    recruited = defaultdict(int)
    for i in range(steps):
        t = time.perf_counter()
        r = c.step(**kw)
        ms.append((time.perf_counter() - t) * 1000)
        if i < settle:
            continue
        d = r["dng02"]
        n_l.append(d["n_left"]); n_r.append(d["n_right"])
        thrust.append(d["thrust"]); steer.append(d["steer"])
        for k, v in r["spike_counts"].items():
            dn_spikes[k] += v
        for k, v in d["counts"].items():
            recruited[k] += v
    return {
        "n_left": float(np.mean(n_l)), "n_right": float(np.mean(n_r)),
        "thrust": float(np.mean(thrust)), "steer": float(np.mean(steer)),
        "ms": float(np.mean(ms)), "ms_p95": float(np.percentile(ms, 95)),
        "dn_spikes": dict(dn_spikes), "recruited": dict(recruited),
    }


# ----------------------------------------------------------------- 1. wiring
circuit = fbc._load_circuit(with_dng02=True)
i_pre, i_post, w = fbc._load_synapses(circuit["root_to_local"])
n_in, dng02_at, drive_at = circuit["n_in"], circuit["dng02_at"], circuit["drive_at"]
dng02, drivers = circuit["dng02"], circuit["drivers"]
dn_names = [f"{n['cell_type']}_{n['side']}" for n in circuit["outputs"]]

print(f"merged network: {circuit['n_total']} neurons, {len(w)} synapses")
print(f"  looming inputs {n_in}  |  escape DNs {len(dn_names)}  |  "
      f"DNg02 {len(dng02)} (L={sum(1 for n in dng02 if n['side'] == 'left')} "
      f"R={sum(1 for n in dng02 if n['side'] == 'right')})  |  drive pool {len(drivers)}")
n_exc = sum(1 for d in drivers if d["sign"] > 0)
print(f"  drive pool: {n_exc} excitatory, {len(drivers) - n_exc} inhibitory")

print("\nrecruitment ladder (excitatory weight from the drive pool):")
for i, n in enumerate(dng02):
    bar = "#" * int(round(18 * n["exc_weight"] / max(dng02[0]["exc_weight"], 1)))
    print(f"  {i:>2} {n['label']:>16}  exc {n['exc_weight']:>7.1f}  "
          f"inh {n['inh_weight']:>8.1f}  {bar}")

# The interaction that actually matters: synapses running from the drive pool
# back into the escape circuit. Reported per escape DN and split by sign,
# because "how many" says much less here than "which direction".
print("\ndrive pool -> escape DNs (the cross-talk path):")
drive_local = set(range(drive_at, circuit["n_total"]))
for k, name in enumerate(dn_names):
    m = (i_post == n_in + k) & np.isin(i_pre, list(drive_local))
    if not m.any():
        print(f"  {name:>12}: none")
        continue
    pos, neg = w[m][w[m] > 0].sum(), w[m][w[m] < 0].sum()
    print(f"  {name:>12}: {m.sum():>3} synapses, excitatory weight {pos:+.0f}, "
          f"inhibitory {neg:+.0f}")
print("  (no EXCITATORY path onto DNp01 is what keeps optic flow from tripping")
print("   the Giant Fiber - asserted against live spikes below, not just here)")

t0 = time.time()
c = fbc.FlyBrainController(with_dng02=True)
print(f"\nnetwork build: {time.time() - t0:.1f}s")

# ------------------------------------------------- 2. symmetric drive sweep
print(f"\n=== symmetric drive sweep (loom=0, {STEPS} windows each, "
      f"first {SETTLE} discarded) ===")
print(f"{'request':>8} {'nL':>6} {'nR':>6} {'total':>6} {'thrust':>7} {'steer':>7} "
      f"{'ms/step':>8}")
sym = []
for level in LEVELS:
    # An exactly-zero request deliberately holds the pool silent, so nudge the
    # first level just off zero to measure the engaged-but-neutral resting state.
    r = sweep(c, drive_common=level if level > 0 else 1e-9)
    sym.append(r)
    print(f"{level:>8.2f} {r['n_left']:>6.2f} {r['n_right']:>6.2f} "
          f"{r['n_left'] + r['n_right']:>6.2f} {r['thrust']:>7.3f} {r['steer']:>+7.3f} "
          f"{r['ms']:>8.2f}")

idle = sweep(c)  # all requests exactly 0.0 -> pool held silent
print(f"{'idle':>8} {idle['n_left']:>6.2f} {idle['n_right']:>6.2f} "
      f"{idle['n_left'] + idle['n_right']:>6.2f} {idle['thrust']:>7.3f} "
      f"{idle['steer']:>+7.3f} {idle['ms']:>8.2f}   <- drive pool held at 0Hz")

def as_tuple(key):
    return "(" + ", ".join(f"{r[key]:.1f}" for r in sym) + ")"


print("\n  paste into fly_brain_controller.py:")
print(f"    DNG02_RECRUIT_SAT = {sym[-1]['n_left'] + sym[-1]['n_right']:.1f}")
print(f"    DNG02_DRIVE_LEVELS = {LEVELS}")
print(f"    DNG02_LEFT_RECRUITMENT = {as_tuple('n_left')}")
print(f"    DNG02_RIGHT_RECRUITMENT = {as_tuple('n_right')}")

print("\n  which cells recruit at which level (population code, "
      "ladder order left to right):")
for level, r in zip(LEVELS, sym):
    marks = "".join("#" if r["recruited"].get(n["label"], 0) else "." for n in dng02)
    print(f"    {level:>4.2f}  {marks}")

# ----------------------------------------------- 3. asymmetric (steering)
print("\n=== asymmetric drive (the steering channel) ===")
print(f"{'dL':>6} {'dR':>6} {'nL':>6} {'nR':>6} {'steer':>8}")
asym = {}
for dl, dr in [(1.0, 0.0), (0.0, 1.0), (0.8, 0.4), (0.4, 0.8), (0.6, 0.6)]:
    r = sweep(c, drive_common=0.5, drive_left=dl - 0.5, drive_right=dr - 0.5)
    asym[(dl, dr)] = r
    print(f"{dl:>6.2f} {dr:>6.2f} {r['n_left']:>6.2f} {r['n_right']:>6.2f} "
          f"{r['steer']:>+8.3f}")

# ------------------------------------------------- 4. loom x drive crosstalk
# Each channel alone is not enough: the drive pool inhibits DNp01, so the
# question is whether flight-motor drive shifts the looming threshold, which
# only a 2-D sweep can answer.
print("\n=== loom x drive: peak escape, and whether the threshold moves ===")
print(f"{'loom':>6} " + " ".join(f"{d:>7.2f}" for d in (0.0, 0.5, 1.0)) + "    <- drive")
esc = {}
for loom in (0.0, 0.2, 0.3, 0.4, 0.6, 1.0):
    row = []
    for drive in (0.0, 0.5, 1.0):
        c.reset()
        peak = 0.0
        for i in range(STEPS):
            r = c.step(loom_left=loom, loom_right=loom, drive_common=drive)
            if i >= SETTLE:
                peak = max(peak, r["escape"])
        esc[(loom, drive)] = peak
        row.append(peak)
    print(f"{loom:>6.2f} " + " ".join(f"{v:>7.3f}" for v in row))


def threshold(drive, thr=0.6):
    """Lowest swept loom whose peak escape reaches the ESCAPE threshold."""
    for loom in (0.0, 0.2, 0.3, 0.4, 0.6, 1.0):
        if esc[(loom, drive)] >= thr:
            return loom
    return None


thr0, thr1 = threshold(0.0), threshold(1.0)
print(f"\n  ESCAPE threshold at drive=0.0: loom {thr0}   at drive=1.0: loom {thr1}")

# ------------------------------------------------------------ 5. assertions
print("\n=== assertions ===")
worst_ms = max(r["ms_p95"] for r in sym)
check(worst_ms < BUDGET_MS,
      f"step time p95 {worst_ms:.1f}ms < {BUDGET_MS}ms budget")
check(idle["n_left"] == 0 and idle["n_right"] == 0,
      "population silent when the DNg02 path is idle "
      f"(nL={idle['n_left']:.2f} nR={idle['n_right']:.2f})")
totals = [r["n_left"] + r["n_right"] for r in sym]
check(all(b >= a - 0.3 for a, b in zip(totals, totals[1:])),
      f"recruitment monotone in drive: {[round(v, 1) for v in totals]}")
check(totals[-1] >= 4.0,
      f"recruitment reaches a usable range ({totals[-1]:.1f} cells at full drive)")
check(len({round(v, 1) for v in totals}) >= 4,
      f"recruitment is graded, not all-or-nothing "
      f"({len({round(v, 1) for v in totals})} distinct levels over {len(LEVELS)} steps)")
check(asym[(0.0, 1.0)]["steer"] > 0.1,
      f"right-biased request gives steer > 0 ({asym[(0.0, 1.0)]['steer']:+.3f}) "
      "= 'the fly yaws right'")
check(asym[(1.0, 0.0)]["steer"] < -0.1,
      f"left-biased request gives steer < 0 ({asym[(1.0, 0.0)]['steer']:+.3f})")
check(abs(asym[(0.6, 0.6)]["steer"]) < 0.15,
      f"symmetric request leaves no standing bias "
      f"(steer {asym[(0.6, 0.6)]['steer']:+.3f})")
worst_sym = max(abs(r["steer"]) for r in sym)
check(worst_sym < 0.25,
      f"symmetric sweep stays near zero steer at every level (worst {worst_sym:.3f})")
dnp01_spikes = sum(v for r in sym for k, v in r["dn_spikes"].items() if k.startswith("DNp01"))
check(dnp01_spikes == 0,
      f"drive alone never spikes DNp01 - optic flow cannot trip the Giant Fiber "
      f"({dnp01_spikes} spikes across the whole sweep)")
check(thr0 == thr1,
      f"flight-motor drive does not move the ESCAPE threshold "
      f"(loom {thr0} at drive 0 vs {thr1} at drive 1)")

print(f"\n{'ALL CHECKS PASSED' if not failures else f'{len(failures)} CHECK(S) FAILED:'}")
for f in failures:
    print(f"  - {f}")
sys.exit(1 if failures else 0)
