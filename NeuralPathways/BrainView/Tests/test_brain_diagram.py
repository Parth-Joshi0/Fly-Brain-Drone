"""
Headless check of the live brain diagram - no drone, no simulator, no brian2.

Drives brain_diagram.attach() on a FAKE brain that sends what
fly_brain_controller.py does (handshake neuron_ids, per-step "spiked") through
quiet -> left-eye loom -> reset, and checks that attach() is transparent, that
the left LC4/LPLC2 somata glow (and the right ones don't) and the neuron list
agrees, that a reset clears both, and what a repaint costs.

    python NeuralPathways/BrainView/Tests/test_brain_diagram.py
    python NeuralPathways/BrainView/Tests/test_brain_diagram.py --out some/dir   # save PNG snapshots
    python NeuralPathways/BrainView/Tests/test_brain_diagram.py --show           # watch it animate
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from NeuralPathways.BrainView import brain_diagram as bd  # noqa: E402

with open(ROOT / "NeuralPathways" / "looming_circuit_neurons.json") as f:
    LOOMING = json.load(f)
# fly_brain_controller.py's local order: inputs left then right, then the DNs.
CELLS = sorted(LOOMING["input_neurons"], key=lambda n: n["side"]) + LOOMING["output_neurons"]


class FakeBrain:
    def __init__(self):
        self.info = {"neuron_ids": [n["root_id"] for n in CELLS]}

    def request(self, payload):
        if payload.get("reset"):
            return {"yaw": 0.0, "forward": 1.0, "escape": 0.0}
        loom = {"left": payload["loom_left"], "right": payload["loom_right"]}
        spiked = [i for i, n in enumerate(CELLS) if loom[n["side"]] > 0.3]
        return {"yaw": 0.0, "forward": 1.0, "escape": max(loom.values()), "spiked": spiked}


class FakeController:
    def __init__(self):
        self.state = "CRUISE"
        self._brain = FakeBrain()

    def decide(self, flow, state=None):
        r = self._brain.request({"loom_left": flow[0], "loom_right": flow[1]})
        self.state = "ESCAPE" if r["escape"] > 0.6 else "CRUISE"
        return {"forward_speed": r["forward"], "yaw_rate": r["yaw"]}

    def reset(self):
        self._brain.request({"reset": True})


SEQUENCE = [("quiet", (0.0, 0.0))] * 20 + [("loom_left", (0.8, 0.0))] * 20


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", help="save PNG snapshots of each phase here")
    parser.add_argument("--show", action="store_true", help="animate the sequence in a window")
    args = parser.parse_args()
    failures = []

    def check(ok, what):
        print(f"  [{'PASS' if ok else 'FAIL'}] {what}")
        if not ok:
            failures.append(what)

    plain, ctrl = FakeController(), FakeController()
    view = bd.attach(ctrl, max_fps=0, display=args.show)
    check(all(plain.decide(f) == ctrl.decide(f) for _, f in SEQUENCE),
          "decide() returns the identical command with the diagram attached")
    check(plain._brain.request({"loom_left": 0.5, "loom_right": 0.0})
          == ctrl._brain.request({"loom_left": 0.5, "loom_right": 0.0}),
          "_brain.request() returns the identical result with the diagram attached")

    ctrl.reset()
    frames, ms = {}, []
    for phase, f in SEQUENCE:
        start = time.perf_counter()
        ctrl.decide(f)
        ms.append((time.perf_counter() - start) * 1000)
        frames[phase] = view.last_frame
        if args.show:
            cv2.waitKey(33)

    def glow(img, side):
        pts = [view.points[view.row_of[n["root_id"]]] for n in LOOMING["input_neurons"] if n["side"] == side]
        return float(np.mean([img[y - 1:y + 2, x - 1:x + 2].mean() for x, y in pts]))

    quiet, loom = frames["quiet"], frames["loom_left"]
    check(glow(loom, "left") > glow(quiet, "left") + 20, "left LC4/LPLC2 somata glow during a left loom")
    check(abs(glow(loom, "right") - glow(quiet, "right")) < 3, "right LC4/LPLC2 somata stay dark")
    lc4 = 2 * view.type_names.index("LC4")
    check(view.counts[lc4] > 0 and view.counts[lc4 + 1] == 0,
          "neuron list: left LC4 cells firing, right LC4 none")
    ctrl.reset()
    check(not view.heat.any() and not view.peak.any(), "a reset request clears the glow and the bars")
    mean = statistics.fmean(ms)
    print(f"      decide()+repaint: mean {mean:.2f} ms")
    check(mean < 5.0, "repaint is cheap (mean < 5 ms)")

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        for phase, img in frames.items():
            cv2.imwrite(str(out / f"brain_{phase}.png"), img)
        print(f"snapshots written to {out}")
    if args.show:
        cv2.waitKey(0)

    print("\nall checks passed" if not failures else f"\nFAILED: {len(failures)} check(s)")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
