"""
Headless check of the live brain diagram - no drone, no simulator, no brian2.

Drives brain_diagram.attach() on a FAKE brain (same surface as
flybrain_controller.FlyBrainController: decide(), .state, ._brain.request,
the handshake's constants/dng02_labels/info) through a synthetic sequence -
quiet, something looming on the left, a Giant Fiber escape, a DNg02 steering
request, a reset - and checks:

  * the atlas covers every simulated circuit neuron
  * attach() is transparent: decide()'s command and the brain's result come
    back identical to an unattached controller fed the same inputs
  * the right somata on the brain map glow, and only then - both with a
    brain that reports every spike ("spiked" + handshake neuron_ids, what
    fly_brain_controller.py sends) and with one that only reports DN/DNg02
    counts (the sampled fallback)
  * a reset clears the glow and the history
  * the optomotor-off and food-only layouts draw
  * render time, against the 33 ms decision budget

    python NeuralPathways/BrainView/Tests/test_brain_diagram.py
    python NeuralPathways/BrainView/Tests/test_brain_diagram.py --out some/dir   # save PNG snapshots
    python NeuralPathways/BrainView/Tests/test_brain_diagram.py --show           # watch it animate

The spikes here are SYNTHETIC - this checks the drawing and the hooks, not the
circuit. NeuralPathways/Tests/test_brain_circuit.py checks the circuit.
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

CIRCUITS = bd.load_circuits()
LADDER = CIRCUITS["dng02_ladder"]
with open(bd.LOOMING_IDS_PATH) as f:
    LOOMING = json.load(f)
with open(bd.DNG02_IDS_PATH) as f:
    DNG02 = json.load(f)


def local_layout(with_dng02):
    """Root ids in fly_brain_controller.py's local index order (see its
    _load_circuit): inputs left then right, the 6 DNs, then DNg02 in ladder
    order and the drive pool."""
    inputs = sorted(LOOMING["input_neurons"], key=lambda n: n["side"])
    cells = inputs + LOOMING["output_neurons"]
    if with_dng02:
        cells += sorted(DNG02["output_neurons"], key=lambda n: -n["exc_weight"]) + DNG02["drive_neurons"]
    return cells


class FakeBrain:
    """Deterministic stand-in for _FlyBrainProcess. With report_spikes it also
    sends what fly_brain_controller.py does: neuron_ids in the handshake and
    every spike's local index in "spiked"."""

    def __init__(self, with_dng02, report_spikes=False):
        self.constants = dict(bd.DEFAULT_CONSTANTS)
        self.dng02_labels = [label for label, _ in LADDER] if with_dng02 else []
        self.dng02_sides = [side for _, side in LADDER] if with_dng02 else []
        self.report_spikes = report_spikes
        self.cells = local_layout(with_dng02)
        self.info = {"with_dng02": with_dng02}
        if report_spikes:
            self.info["neuron_ids"] = [n["root_id"] for n in self.cells]
        self.step = 0

    def request(self, payload):
        if payload.get("reset"):
            return {"yaw": 0.0, "forward": 1.0, "escape": 0.0}
        self.step += 1
        ll, lr = payload.get("loom_left", 0.0), payload.get("loom_right", 0.0)
        spikes = {
            "DNp01_left": int(ll > 0.6) * 2, "DNp01_right": int(lr > 0.6) * 2,
            "DNp03_left": int(ll > 0.3), "DNp03_right": int(lr > 0.3),
            "DNp06_left": int(ll > 0.2) * 2, "DNp06_right": int(lr > 0.2) * 2,
        }
        req_l = payload.get("drive_common", 0.0) + payload.get("drive_left", 0.0)
        req_r = payload.get("drive_common", 0.0) + payload.get("drive_right", 0.0)
        counts, n = {}, {"left": 0, "right": 0}
        for label, side in LADDER if self.dng02_labels else []:
            req = req_l if side == "left" else req_r
            fire = n[side] < int(max(0.0, req) * 8)
            counts[label] = int(fire)
            n[side] += int(fire)
        result = {
            "yaw": max(-1.0, min(1.0, lr - ll)), "forward": 1.0 - max(ll, lr), "escape": max(ll, lr),
            "spike_counts": spikes,
            "dng02": {"n_left": n["left"], "n_right": n["right"],
                      "thrust": (n["left"] + n["right"]) / 13.1,
                      "steer": (n["right"] - n["left"]) / 8.0, "counts": counts},
        }
        if self.report_spikes:
            result["spiked"] = self._spiked(ll, lr, spikes, counts, req_l != 0.0 or req_r != 0.0)
        return result

    def _spiked(self, ll, lr, spikes, counts, driving):
        dn_key = {f"{c['cell_type']}_{c['side']}": c["root_id"] for c in LOOMING["output_neurons"]}
        dng_label = CIRCUITS["dng02_ids"]
        fired = {dn_key[k]: v for k, v in spikes.items()}
        fired.update({dng_label[k]: v for k, v in counts.items()})
        out = []
        for i, c in enumerate(self.cells):
            rid = c["root_id"]
            if rid in fired:
                out += [i] * fired[rid]
            elif c.get("cell_type") in ("LC4", "LPLC2"):
                loom = ll if c["side"] == "left" else lr
                if (i * 7 + self.step) % 10 < loom * 10:     # ~loom of the cells, rotating
                    out.append(i)
            elif "sign" in c and driving and (i + self.step) % 3 == 0:
                out.append(i)
        return out


class FakeController:
    """Same contract as flybrain_controller.FlyBrainController, minus brian2."""

    def __init__(self, optomotor=False, report_spikes=False):
        self.optomotor = optomotor
        self.state = "CRUISE"
        self.escape_direction = "LEFT"
        self._brain = FakeBrain(optomotor, report_spikes)

    def decide(self, flow, state=None):
        payload = {"loom_left": min(1.0, max(flow["expansion_left"], flow["expansion_center"]) / 3),
                   "loom_right": min(1.0, max(flow["expansion_right"], flow["expansion_center"]) / 3)}
        if self.optomotor:
            payload.update(drive_common=flow.get("drive", 0.0),
                           drive_left=-flow.get("turn", 0.0), drive_right=flow.get("turn", 0.0))
        r = self._brain.request(payload)
        self.state = "ESCAPE" if r["escape"] > 0.6 else ("AVOID_RIGHT" if r["yaw"] < -0.05 else "CRUISE")
        return {"forward_speed": r["forward"], "strafe_speed": 2.0 if self.state == "ESCAPE" else 0.0,
                "yaw_rate": 0.9 * r["yaw"], "altitude_delta": 0.0, "hover": False,
                "land": False, "reset": False, "pressed_direction": "(fake)"}

    def reset(self):
        self._brain.request({"reset": True})


def flow(left=0.0, center=0.0, right=0.0, **extra):
    f = {"expansion_left": left, "expansion_center": center, "expansion_right": right}
    f.update(extra)
    return f


SEQUENCE = (
    [("quiet", flow())] * 20
    + [("loom_left", flow(left=1.2 + 0.05 * i)) for i in range(20)]
    + [("escape", flow(left=2.8, center=2.4))] * 10
    + [("steer", flow(drive=0.5, turn=0.6))] * 20
)


def brightness(img, boxes):
    return float(np.mean([img[y0:y1, x0:x1].astype(float).mean() for x0, y0, x1, y1 in boxes]))


def ids_of(cells, **match):
    return [c["root_id"] for c in cells if all(c.get(k) == v for k, v in match.items())]


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

    out = Path(args.out) if args.out else None
    if out:
        out.mkdir(parents=True, exist_ok=True)

    # --- 0. atlas ---
    print("0. atlas")
    atlas = bd.load_atlas()
    want = set(CIRCUITS["population"])
    check(want <= set(atlas.row_of), f"brain_atlas.npz places all {len(want)} simulated circuit neurons")
    inside = all(((atlas.px[v][:, 0] >= 0) & (atlas.px[v][:, 0] < atlas.shape(v)[1])
                  & (atlas.px[v][:, 1] >= 0) & (atlas.px[v][:, 1] < atlas.shape(v)[0])).all()
                 for v in atlas.views)
    check(inside, "every circuit soma lands inside its map")

    # --- 1. transparency: attached vs plain controller, same inputs ---
    print("1. attach() is transparent")
    plain = FakeController(optomotor=True, report_spikes=True)
    watched = FakeController(optomotor=True, report_spikes=True)
    bd.attach(watched, window="test-optomotor", max_fps=0, display=args.show)
    same_cmd = all(plain.decide(f) == watched.decide(f) for _, f in SEQUENCE)
    check(same_cmd, "decide() returns the identical command with the diagram attached")
    payload = {"loom_left": 0.7, "loom_right": 0.1, "drive_common": 0.3}
    check(plain._brain.request(payload) == watched._brain.request(payload),
          "_brain.request() returns the identical result with the diagram attached")

    # --- 2. the right somata light up, in both spike modes ---
    lc_left = ids_of(LOOMING["input_neurons"], side="left")
    gf_left = ids_of(LOOMING["output_neurons"], cell_type="DNp01", side="left")
    gf_right = ids_of(LOOMING["output_neurons"], cell_type="DNp01", side="right")
    dng02 = ids_of(DNG02["output_neurons"])
    region = atlas.region_names.index
    render_ms = []
    snapshots = {}
    for mode, report in (("recorded", True), ("sampled", False)):
        print(f"2. activity shows up where it should ({mode} spikes)")
        ctrl = FakeController(optomotor=True, report_spikes=report)
        view = bd.attach(ctrl, window=f"test-phases-{mode}", max_fps=0, display=args.show)
        diagram, act = view.diagram, view.activity
        frames, regions = {}, {}
        for phase, f in SEQUENCE:
            start = time.perf_counter()
            ctrl.decide(f)
            render_ms.append((time.perf_counter() - start) * 1000)
            frames[phase] = view.last_frame
            regions[phase] = act.region_counts.copy()
            if args.show:
                cv2.waitKey(33)
        snapshots[mode] = frames

        def boxes(ids):
            return [diagram.soma_box(rid, "frontal") for rid in ids]

        quiet = frames["quiet"]
        check(act.spike_source == mode, f"diagram reports the {mode} spike path")
        check(brightness(frames["loom_left"], boxes(lc_left)) > brightness(quiet, boxes(lc_left)) + 10,
              "left LC4/LPLC2 somata glow while something looms on the left")
        check(brightness(frames["escape"], boxes(gf_left)) > brightness(quiet, boxes(gf_left)) + 20,
              "left Giant Fiber (DNp01) soma glows during the escape phase")
        check(abs(brightness(frames["loom_left"], boxes(gf_right)) - brightness(quiet, boxes(gf_right))) < 5,
              "right Giant Fiber soma stays dark while only the left eye sees looming")
        check(brightness(frames["steer"], boxes(dng02)) > brightness(quiet, boxes(dng02)) + 5,
              "DNg02 somata glow during the steering request")
        check(regions["loom_left"][region("optic lobe L")] > 0
              and regions["loom_left"][region("optic lobe R")] == 0,
              "region bars: optic lobe L counts spikes during left loom, optic lobe R none")
        check(len(act.spike_history) == min(len(SEQUENCE), bd.HISTORY_TICKS),
              "spikes/tick history keeps one bar per brain step")

        if report:
            # --- 3. reset ---
            print("3. reset")
            ctrl.reset()
            check(len(act.spike_history) == 0 and not any(h.any() for h in act.heat.values()),
                  "a reset request clears the glow and the history")

    # --- 4. other layouts ---
    print("4. other layouts")
    escape_only = FakeController(optomotor=False, report_spikes=True)
    view_off = bd.attach(escape_only, window="test-escape-only", max_fps=0, display=args.show)
    escape_only.decide(flow(left=2.0))
    check(view_off.activity.with_dng02 is False and view_off.last_frame is not None,
          "optomotor-off controller renders with the DNg02 rows marked not built")

    class FakeFood:
        def __init__(self):
            self.state, self.hunger, self.current_target, self.last_target_label = "SEARCH", 100.0, None, None

        def update(self, detections, frame_width, frame_height):
            if detections:
                self.state, self.current_target, self.last_target_label = "FEED", detections[0], "ripe"
                self.hunger -= 5
            return "rc"

    food = FakeFood()
    view_food = bd.attach_food(food, window="test-food", max_fps=0, display=args.show)
    cmd = food.update(["banana"], 960, 720)
    check(cmd == "rc", "attach_food() passes update()'s command through unchanged")
    check(view_food.activity.feeding_running() and view_food.activity.food["state"] == "FEED",
          "food pathway shows as running, in FEED")
    fx0, fy0, fx1, fy1 = view_food.diagram.FOOD_BOX
    blank = view_food.diagram._background
    check(brightness(view_food.last_frame, [(fx0, fy0, fx1, fy1)])
          > brightness(blank, [(fx0, fy0, fx1, fy1)]) + 5,
          "feeding box lights up")

    # --- 5. timing ---
    print("5. timing")
    # Checked on the mean: p95 swings with whatever else the machine is doing,
    # the mean doesn't.
    mean = statistics.fmean(render_ms)
    p95 = float(np.percentile(render_ms, 95))
    print(f"      decide()+render: mean {mean:.1f} ms, p95 {p95:.1f} ms "
          f"(fake brain ~0 ms, so this is the diagram's own cost)")
    check(mean < 12.0, "diagram costs well under the 33 ms decision budget (mean < 12 ms)")

    if out:
        for mode, frames in snapshots.items():
            for phase, img in frames.items():
                cv2.imwrite(str(out / f"brain_{mode}_{phase}.png"), img)
        cv2.imwrite(str(out / "brain_escape_only.png"), view_off.last_frame)
        cv2.imwrite(str(out / "brain_food.png"), view_food.last_frame)
        print(f"snapshots written to {out}")

    if args.show:
        print("press any key in a diagram window to close")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s)")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
