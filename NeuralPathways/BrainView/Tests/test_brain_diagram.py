"""
Headless check of the live brain diagram - no drone, no simulator, no brian2.

Drives brain_diagram.attach() on a FAKE brain (same surface as
flybrain_controller.FlyBrainController: decide(), .state, ._brain.request,
the handshake's constants/dng02_labels) through a synthetic sequence - quiet,
something looming on the left, a Giant Fiber escape, a DNg02 steering request,
a reset - and checks:

  * attach() is transparent: decide()'s command and the brain's result come
    back identical to an unattached controller fed the same inputs
  * the parts of the picture that should light up do, and only then
  * a reset clears the raster
  * the optomotor-off and food-only layouts draw
  * render time, against the 33 ms decision budget

    python NeuralPathways/BrainView/Tests/test_brain_diagram.py
    python NeuralPathways/BrainView/Tests/test_brain_diagram.py --out some/dir   # save PNG snapshots
    python NeuralPathways/BrainView/Tests/test_brain_diagram.py --show           # watch it animate

The spike counts here are SYNTHETIC - this checks the drawing and the hooks,
not the circuit. NeuralPathways/Tests/test_brain_circuit.py checks the circuit.
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from NeuralPathways.BrainView import brain_diagram as bd  # noqa: E402

LADDER = bd.load_circuits()["dng02_ladder"]


class FakeBrain:
    """Deterministic stand-in for _FlyBrainProcess."""

    def __init__(self, with_dng02):
        self.constants = dict(bd.DEFAULT_CONSTANTS)
        self.dng02_labels = [label for label, _ in LADDER] if with_dng02 else []
        self.dng02_sides = [side for _, side in LADDER] if with_dng02 else []
        self.info = {"with_dng02": with_dng02}

    def request(self, payload):
        if payload.get("reset"):
            return {"yaw": 0.0, "forward": 1.0, "escape": 0.0}
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
        return {
            "yaw": max(-1.0, min(1.0, lr - ll)), "forward": 1.0 - max(ll, lr), "escape": max(ll, lr),
            "spike_counts": spikes,
            "dng02": {"n_left": n["left"], "n_right": n["right"],
                      "thrust": (n["left"] + n["right"]) / 13.1,
                      "steer": (n["right"] - n["left"]) / 8.0, "counts": counts},
        }


class FakeController:
    """Same contract as flybrain_controller.FlyBrainController, minus brian2."""

    def __init__(self, optomotor=False):
        self.optomotor = optomotor
        self.state = "CRUISE"
        self.escape_direction = "LEFT"
        self._brain = FakeBrain(optomotor)

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


def region_brightness(img, x0, y0, x1, y1):
    return float(img[y0:y1, x0:x1].astype(float).mean())


def dn_node_region(diagram, cell_type, side):
    x, y = diagram._dn_pos(cell_type, side)
    r = diagram.DN_R
    return x - r, y - r, x + r, y + r


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

    # --- 1. transparency: attached vs plain controller, same inputs ---
    print("1. attach() is transparent")
    plain, watched = FakeController(optomotor=True), FakeController(optomotor=True)
    view = bd.attach(watched, window="test-optomotor", max_fps=0, display=args.show)
    same_cmd = all(plain.decide(f) == watched.decide(f) for _, f in SEQUENCE)
    check(same_cmd, "decide() returns the identical command with the diagram attached")
    payload = {"loom_left": 0.7, "loom_right": 0.1, "drive_common": 0.3}
    check(plain._brain.request(payload) == watched._brain.request(payload),
          "_brain.request() returns the identical result with the diagram attached")

    # --- 2. the right parts light up ---
    print("2. activity shows up where it should")
    ctrl = FakeController(optomotor=True)
    view = bd.attach(ctrl, window="test-phases", max_fps=0, display=args.show)
    diagram = view.diagram
    frames, render_ms = {}, []
    for phase, f in SEQUENCE:
        start = time.perf_counter()
        ctrl.decide(f)
        render_ms.append((time.perf_counter() - start) * 1000)
        frames[phase] = view.last_frame
        if args.show:
            cv2.waitKey(33)

    gf_left = dn_node_region(diagram, "DNp01", "left")
    gf_right = dn_node_region(diagram, "DNp01", "right")
    quiet, escape = frames["quiet"], frames["escape"]
    check(region_brightness(escape, *gf_left) > region_brightness(quiet, *gf_left) + 20,
          "left Giant Fiber (DNp01) node lights up during the escape phase")
    check(abs(region_brightness(frames["loom_left"], *gf_right) - region_brightness(quiet, *gf_right)) < 5,
          "right Giant Fiber node stays dark while only the left eye sees looming")
    lobe_l = (140, 130, 240, 300)
    check(region_brightness(frames["loom_left"], *lobe_l) > region_brightness(quiet, *lobe_l) + 5,
          "left optic lobe (LC4/LPLC2) brightens while something looms on the left")
    dng = (300, diagram.DNG02_TOP, 700, diagram.DNG02_TOP + 20)
    check(region_brightness(frames["steer"], *dng) > region_brightness(quiet, *dng) + 3,
          "DNg02 cells light up during the steering request")
    check(len(view.activity.history) == min(len(SEQUENCE), bd.HISTORY_CYCLES),
          "raster keeps one column per brain step")

    # --- 3. reset ---
    print("3. reset")
    ctrl.reset()
    check(len(view.activity.history) == 0 and not view.activity.dn_glow,
          "a reset request clears the raster and the DN glow")

    # --- 4. other layouts ---
    print("4. other layouts")
    escape_only = FakeController(optomotor=False)
    view_off = bd.attach(escape_only, window="test-escape-only", max_fps=0, display=args.show)
    escape_only.decide(flow(left=2.0))
    check(view_off.activity.with_dng02 is False and view_off.last_frame is not None,
          "optomotor-off controller renders with the DNg02 half marked not built")

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
    check(region_brightness(view_food.last_frame, fx0, fy0, fx1, fy1)
          > region_brightness(blank, fx0, fy0, fx1, fy1) + 5,
          "feeding box lights up")

    # --- 5. timing ---
    print("5. timing")
    # Checked on the mean: p95 swings with whatever else the machine is doing
    # (measured 6.7-13.4 ms across back-to-back runs), the mean doesn't.
    mean = statistics.fmean(render_ms)
    p95 = float(np.percentile(render_ms, 95))
    print(f"      decide()+render: mean {mean:.1f} ms, p95 {p95:.1f} ms "
          f"(fake brain ~0 ms, so this is the diagram's own cost)")
    check(mean < 12.0, "diagram costs well under the 33 ms decision budget (mean < 12 ms)")

    if out:
        for phase, img in frames.items():
            cv2.imwrite(str(out / f"brain_{phase}.png"), img)
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
