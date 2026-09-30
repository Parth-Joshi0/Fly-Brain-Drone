"""
Live diagram of which parts of the fly brain are working - a picture of the
circuit, lit up by what it is actually doing, instead of a line of text
saying which neurons fired.

    compound eye -> optic lobe (LC4 / LPLC2) -> DNp01 / DNp03 / DNp06
                                                    |
                         DNg02 drive pool -> DNg02 x24
                                                    |
                                  neck -> motor command

Everything drawn comes from the circuit itself rather than being restated
here: cell counts and sides from looming_circuit_neurons.json and
StabilizerNeuron/dng02_circuit_neurons.json, the DNg02 ladder order and the
network's constants from the brain subprocess's own "ready" handshake, and the
per-cycle activity from the exact request/response pair that crosses
flybrain_controller.py's subprocess boundary.

What each part shows, and how directly it was measured:

    DNp01/03/06, DNg02   REAL spikes - the per-cell spike counts
                         fly_brain_controller.py's step() returns for the last
                         20ms window. Nothing is inferred.
    LC4 / LPLC2 dots     the Poisson drive each input cell is receiving
                         (loom x MAX_POI_RATE). The network does not report
                         per-input spikes, so the flicker is SAMPLED at that
                         same rate - statistically what the model does, not a
                         recording of which cell fired.
    DNg02 drive pool     each driver's rate, computed with the same rule
                         fly_brain_controller.py's step() uses (rest rate +/-
                         gain x its own left/right weight split, flipped for
                         inhibitory cells). Again a rate, not recorded spikes.
    motor bars / command what the circuit asked for (escape/yaw/forward,
                         thrust/steer) and the command FlyBrainController
                         returned. main.py's NEURON_TEST_MODE and SafetyLayer
                         can still override that command downstream.

Hooking in without touching the controller: attach() wraps the instance's
_brain.request (the same trick Drone/Tests/tello_neuron_test.py and
tello_dng02_test.py already use to see both sides of the subprocess call) and
its decide(). Both wrappers pass their arguments and results through
unchanged, so what the drone does is identical with or without the diagram.

The window is drawn with cv2.imshow() but never calls cv2.waitKey() itself -
that would swallow the host script's own keys (l = land on the real drone).
It repaints on the host loop's waitKey(1), which main.py and every Drone/
script already call once per cycle. HighGUI key events are shared between
windows, so those keys still work while this window has focus.
"""

import json
import math
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
PATHWAYS = HERE.parent
LOOMING_IDS_PATH = PATHWAYS / "looming_circuit_neurons.json"
DNG02_IDS_PATH = PATHWAYS / "StabilizerNeuron" / "dng02_circuit_neurons.json"

WINDOW_NAME = "Fly Brain Activity"

# The values fly_brain_controller.py ships with. Only used when no live brain
# has been attached (Tests/test_brain_diagram.py's synthetic run) - with a real
# brain the constants come from its "ready" handshake instead, so the diagram
# can't drift from what the network is actually configured with.
DEFAULT_CONSTANTS = {
    "MAX_POI_RATE": 30.0,
    "MAX_DRIVE_RATE": 150.0,
    "STEP_DT_MS": 20.0,
    "DRIVE_REST_FRACTION": 0.5,
    "DRIVE_GAIN": 0.5,
}

# Same thresholds flybrain_controller.py uses to pick .state - duplicated so
# this module can be imported (and tested) without Simulator/ on the path.
# Only used to draw threshold ticks on the motor bars.
ESCAPE_STATE_THRESHOLD = 0.6

CANVAS_W, CANVAS_H = 1000, 720
# A redraw costs ~6 ms, and on the real drone the brain step alone already
# uses ~20 of the 33 ms decision budget - so by default only every other
# decision cycle is drawn. Activity is still RECORDED every cycle (the raster
# never skips a step); only the repaint is rate-limited.
DEFAULT_MAX_FPS = 15.0
HISTORY_CYCLES = 129           # raster width: ~4.3s at the 30Hz decision loop
GLOW_DECAY = 0.55              # per brain step - a single 20ms spike stays
                               # visible for a few frames instead of one
FEEDING_ACTIVE_SECONDS = 1.0   # food pathway counts as running this long after
                               # its last update()


def _bgr(r, g, b):
    return (b, g, r)


BG = _bgr(16, 17, 22)
PANEL = _bgr(26, 28, 35)
OUTLINE = _bgr(70, 74, 88)
DIM = _bgr(52, 55, 66)
TEXT = _bgr(225, 228, 235)
MUTED = _bgr(135, 140, 155)

EYE = _bgr(255, 200, 70)
INPUT = _bgr(255, 190, 60)        # LC4 / LPLC2
DNP01 = _bgr(255, 75, 75)         # Giant Fiber - escape
DNP03 = _bgr(255, 150, 40)        # brake
DNP06 = _bgr(70, 200, 255)        # evasive turn
DNG02 = _bgr(90, 230, 130)
DRIVE_EXC = _bgr(90, 230, 130)
DRIVE_INH = _bgr(190, 120, 255)
FOOD = _bgr(255, 225, 60)

DN_COLORS = {"DNp01": DNP01, "DNp03": DNP03, "DNp06": DNP06}
DN_ROLES = {"DNp01": "escape jump", "DNp03": "brake", "DNp06": "turn away"}
DN_ROWS = ("DNp01", "DNp03", "DNp06")

STATE_COLORS = {
    "CRUISE": _bgr(150, 210, 150),
    "AVOID_LEFT": DNP06,
    "AVOID_RIGHT": DNP06,
    "ESCAPE": DNP01,
    "BOUNDARY_RETURN": DNP03,
    "OPTOMOTOR": DNG02,
    "SEARCH": FOOD,
    "APPROACH": FOOD,
    "FEED": FOOD,
    "DONE": FOOD,
    "SCARED": DNP01,      # food_orbit.py's reaction to the Giant Fiber (--scared)
    "WAIT": DNP03,
    "LAND": MUTED,
}

FONT = cv2.FONT_HERSHEY_SIMPLEX


# ---------------------------------------------------------------- circuit data

def load_circuits():
    """The two circuit JSONs, reduced to what the diagram draws. Needs neither
    brian2 nor pandas - just the files fly_brain_controller.py builds from."""
    with open(LOOMING_IDS_PATH) as f:
        looming = json.load(f)
    with open(DNG02_IDS_PATH) as f:
        dng02 = json.load(f)

    inputs = {}
    for n in looming["input_neurons"]:
        key = (n["cell_type"], n["side"])
        inputs[key] = inputs.get(key, 0) + 1

    return {
        "inputs": inputs,                                     # {(type, side): count}
        "outputs": [(n["cell_type"], n["side"]) for n in looming["output_neurons"]],
        "dng02_ladder": dng02_ladder(dng02["output_neurons"]),
        "drivers": dng02["drive_neurons"],
    }


def dng02_ladder(cells):
    """(label, side) in recruitment order, labelled exactly the way
    fly_brain_controller.py's _load_circuit() does it. Only used when the
    brain didn't report its own order (DNg02 not built, or no brain at all) -
    otherwise the handshake's dng02_labels/dng02_sides are used verbatim."""
    seen = {}
    ladder = []
    for n in sorted(cells, key=lambda n: -n["exc_weight"]):
        key = (n["cell_type"], n["side"])
        seen[key] = seen.get(key, 0) + 1
        ladder.append((f"{n['cell_type']}_{n['side']}_{seen[key]}", n["side"]))
    return ladder


def driver_rates(drivers, drive_common, drive_left, drive_right, constants):
    """Each drive-pool cell's firing rate as a 0..1 fraction of
    MAX_DRIVE_RATE - the same rule fly_brain_controller.py's step() applies,
    so the dots show what the network is being fed this cycle."""
    if drive_common == 0.0 and drive_left == 0.0 and drive_right == 0.0:
        return np.zeros(len(drivers))
    req_left = min(1.0, max(-1.0, drive_common + drive_left))
    req_right = min(1.0, max(-1.0, drive_common + drive_right))
    w_l = np.array([d["w_left"] for d in drivers], dtype=float)
    w_r = np.array([d["w_right"] for d in drivers], dtype=float)
    sign = np.array([d["sign"] for d in drivers], dtype=float)
    total = np.maximum(w_l + w_r, 1e-9)
    req = (w_l / total) * req_left + (w_r / total) * req_right
    return np.clip(constants["DRIVE_REST_FRACTION"] + constants["DRIVE_GAIN"] * sign * req, 0.0, 1.0)


# ---------------------------------------------------------------- activity model

class BrainActivity:
    """Everything the diagram needs, fed one brain step / decision at a time.
    Kept separate from drawing so it can be checked without a window."""

    def __init__(self, dng02_ladder):
        self.dng02_ladder = list(dng02_ladder)
        self.brain_attached = False
        self.with_dng02 = False
        self.loom = {"left": 0.0, "right": 0.0}
        self.drive = {"drive_common": 0.0, "drive_left": 0.0, "drive_right": 0.0}
        self.escape = 0.0
        self.yaw = 0.0
        self.forward = 1.0
        self.dng02 = {"n_left": 0, "n_right": 0, "thrust": 0.0, "steer": 0.0, "counts": {}}
        self.spike_counts = {}
        self.dn_glow = {}
        self.dng02_glow = np.zeros(len(self.dng02_ladder))
        self.brain_ms = 0.0
        self.brain_steps = 0

        self.expansion = {"left": 0.0, "center": 0.0, "right": 0.0}
        self.state = "-"
        self.escape_direction = ""
        self.cmd = None

        self.food = None          # dict once a FoodOrbitBehaviour reports in
        self.food_time = 0.0

        # Raster: one column per brain step.
        self.history = deque(maxlen=HISTORY_CYCLES)

    def set_ladder(self, ladder):
        self.dng02_ladder = list(ladder)
        self.dng02_glow = np.zeros(len(self.dng02_ladder))

    def record_brain(self, payload, result, ms):
        if payload.get("reset"):
            self.record_reset()
            return
        self.brain_steps += 1
        self.brain_ms = ms
        self.loom = {"left": float(payload.get("loom_left", 0.0)),
                     "right": float(payload.get("loom_right", 0.0))}
        self.drive = {k: float(payload.get(k, 0.0))
                      for k in ("drive_common", "drive_left", "drive_right")}
        self.escape = float(result.get("escape", 0.0))
        self.yaw = float(result.get("yaw", 0.0))
        self.forward = float(result.get("forward", 1.0))
        self.spike_counts = dict(result.get("spike_counts", {}))
        self.dng02 = result.get("dng02", self.dng02)

        for name, n in self.spike_counts.items():
            self.dn_glow[name] = max(self.dn_glow.get(name, 0.0) * GLOW_DECAY, min(1.0, n / 2.0))
        for name in self.dn_glow:
            if name not in self.spike_counts:
                self.dn_glow[name] *= GLOW_DECAY

        counts = self.dng02.get("counts", {})
        fired = np.array([min(1.0, counts.get(label, 0)) for label, _ in self.dng02_ladder])
        self.dng02_glow = np.maximum(self.dng02_glow * GLOW_DECAY, fired)

        self.history.append(self._history_column())

    def record_reset(self):
        self.loom = {"left": 0.0, "right": 0.0}
        self.drive = {k: 0.0 for k in self.drive}
        self.escape, self.yaw, self.forward = 0.0, 0.0, 1.0
        self.spike_counts = {}
        self.dn_glow = {}
        self.dng02 = {"n_left": 0, "n_right": 0, "thrust": 0.0, "steer": 0.0, "counts": {}}
        self.dng02_glow[:] = 0.0
        self.history.clear()

    def record_decision(self, flow, controller, cmd):
        self.expansion = {side: float(flow.get(f"expansion_{side}", 0.0))
                          for side in ("left", "center", "right")}
        self.state = getattr(controller, "state", "-")
        self.escape_direction = getattr(controller, "escape_direction", "")
        self.cmd = cmd

    def record_food(self, behaviour, cmd=None):
        self.food = {
            "state": behaviour.state,
            "hunger": float(behaviour.hunger),
            "visible": behaviour.current_target is not None,
            "label": behaviour.last_target_label or "",
            # food_orbit.py's RCCommand - in Drone/tello_camera.py this, not
            # the brain's own command, is what actually reaches the Tello.
            "rc": tuple(getattr(cmd, k, 0) for k in ("lr", "fb", "ud", "yaw")) if cmd is not None else None,
        }
        self.food_time = time.monotonic()

    def food_drives(self):
        """True while a FoodOrbitBehaviour is flying the drone (tello_camera.py).
        There the brain only decides WHEN to get scared - its own motor
        command is discarded (see NeuralPathways/EscapeNeuron/fear_brain.py)."""
        return self.feeding_running()

    # --- what's "on" right now (header chips) ---

    def escape_active(self):
        return (max(self.loom.values()) > 0.0
                or any(g > 0.15 for g in self.dn_glow.values()))

    def stabilizer_active(self):
        return self.with_dng02 and any(v != 0.0 for v in self.drive.values())

    def feeding_running(self):
        return self.food is not None and time.monotonic() - self.food_time < FEEDING_ACTIVE_SECONDS

    def _history_column(self):
        c = self.spike_counts
        n_left_cells = sum(1 for _, side in self.dng02_ladder if side == "left") or 1
        n_right_cells = sum(1 for _, side in self.dng02_ladder if side == "right") or 1
        return [
            self.loom["left"],
            self.loom["right"],
            min(1.0, c.get("DNp01_left", 0) / 2.0),
            min(1.0, c.get("DNp01_right", 0) / 2.0),
            min(1.0, c.get("DNp03_left", 0) / 2.0),
            min(1.0, c.get("DNp03_right", 0) / 2.0),
            min(1.0, c.get("DNp06_left", 0) / 2.0),
            min(1.0, c.get("DNp06_right", 0) / 2.0),
            # Scaled so ~half the side recruited reads as full brightness -
            # the population never recruits much past that (see
            # DNG02_RECRUIT_SAT in fly_brain_controller.py).
            min(1.0, self.dng02.get("n_left", 0) / (n_left_cells / 2)),
            min(1.0, self.dng02.get("n_right", 0) / (n_right_cells / 2)),
        ]


# ---------------------------------------------------------------- drawing helpers

def _mix(dim, bright, amount):
    a = max(0.0, min(1.0, amount))
    return tuple(int(d + (b - d) * a) for d, b in zip(dim, bright))


def _text(img, text, x, y, scale=0.4, color=TEXT, thickness=1, align="left"):
    if align != "left":
        (w, _), _ = cv2.getTextSize(text, FONT, scale, thickness)
        x = x - w // 2 if align == "center" else x - w
    cv2.putText(img, text, (int(x), int(y)), FONT, scale, color, thickness, cv2.LINE_AA)


def _grid(n, cols, x0, y0, spacing):
    return [(x0 + (i % cols) * spacing, y0 + (i // cols) * spacing) for i in range(n)]


def _mirror(x):
    return CANVAS_W - x


class _DotLayer:
    """Hundreds of small same-sized dots (neurons, ommatidia), each with its
    own colour every frame. One cv2.circle() call per dot per frame was the
    single biggest cost of a redraw, which matters on the real drone where
    the brain step already uses ~20 of the 33 ms budget - so the circles are
    rasterised ONCE into a pixel -> dot index map, and a frame is a single
    fancy-indexed assignment."""

    def __init__(self, points, radius, thickness=-1):
        label = np.full((CANVAS_H, CANVAS_W), -1.0, dtype=np.float32)
        for i, (x, y) in enumerate(points):
            cv2.circle(label, (int(x), int(y)), radius, float(i), thickness)
        self.pixels = np.flatnonzero(label >= 0)
        self.dot = label.ravel()[self.pixels].astype(np.intp)
        self.n = len(points)

    def paint(self, img, colors):
        img.reshape(-1, 3)[self.pixels] = np.asarray(colors, dtype=np.uint8)[self.dot]


def _mix_many(dim, bright, amount):
    """_mix() for an array of amounts -> (N, 3) colours."""
    a = np.clip(np.asarray(amount, dtype=float), 0.0, 1.0)[:, None]
    dim = np.asarray(dim, dtype=float)
    bright = np.asarray(bright, dtype=float)
    return dim + (bright - dim) * a


def _bar(img, x, y, w, h, value, color, signed=False, tick=None):
    cv2.rectangle(img, (x, y), (x + w, y + h), DIM, -1)
    if signed:
        mid = x + w // 2
        end = int(mid + max(-1.0, min(1.0, value)) * (w // 2))
        cv2.rectangle(img, (min(mid, end), y), (max(mid, end), y + h), color, -1)
        cv2.line(img, (mid, y - 2), (mid, y + h + 2), MUTED, 1)
    else:
        end = int(x + max(0.0, min(1.0, value)) * w)
        cv2.rectangle(img, (x, y), (end, y + h), color, -1)
    if tick is not None:
        tx = int(x + tick * w)
        cv2.line(img, (tx, y - 3), (tx, y + h + 3), TEXT, 1)


# ---------------------------------------------------------------- the diagram

class BrainDiagram:
    """Frontal view of the fly brain: compound eyes on the outside, optic
    lobes next, central brain in the middle, descending neurons leaving
    through the neck at the bottom. Static parts are drawn once and cached;
    render() only paints what changes."""

    # Geometry for the LEFT half; the right half is mirrored.
    EYE = ((46, 255), (32, 115))
    LOBE = ((190, 255), (105, 160))
    CENTRAL = ((500, 255), (200, 205))
    DN_X = 420
    DN_Y = {"DNp01": 150, "DNp03": 212, "DNp06": 274}
    DN_R = 17
    LOBE_EXIT = (262, 232)
    BUNDLE_JUNCTION = (340, 214)
    FOOD_BOX = (390, 72, 610, 122)
    DRIVE_TOP = 328
    DNG02_TOP = 370

    def __init__(self, circuits):
        self.circuits = circuits
        self.rng = np.random.default_rng()
        self._layout()
        self._background = self._draw_background()

    def _layout(self):
        inputs = self.circuits["inputs"]
        self.input_dots = {}
        for side in ("left", "right"):
            cx = self.LOBE[0][0] if side == "left" else _mirror(self.LOBE[0][0])
            lplc2 = _grid(inputs.get(("LPLC2", side), 0), 10, cx - 40, 140, 9)
            lc4 = _grid(inputs.get(("LC4", side), 0), 10, cx - 40, 262, 9)
            self.input_dots[side] = _DotLayer(lplc2 + lc4, 3)

        self.eye_dots = {}
        (ex, ey), (ax, ay) = self.EYE
        pts = []
        for row, y in enumerate(range(ey - ay + 12, ey + ay - 8, 11)):
            offset = 5 if row % 2 else 0
            for x in range(ex - ax + 4 + offset, ex + ax, 11):
                if ((x - ex) / ax) ** 2 + ((y - ey) / ay) ** 2 < 0.82:
                    pts.append((x, y))
        self.eye_dots["left"] = _DotLayer(pts, 4, thickness=1)
        self.eye_dots["right"] = _DotLayer([(_mirror(x), y) for x, y in pts], 4, thickness=1)

        drivers = self.circuits["drivers"]
        self.driver_pos = []
        per_side = {"left": 0, "right": 0}
        for d in drivers:
            side = "left" if d["side"] == "left" else "right"
            i = per_side[side]
            per_side[side] += 1
            x = 488 - (i % 23) * 7 if side == "left" else 512 + (i % 23) * 7
            self.driver_pos.append((x, self.DRIVE_TOP + (i // 23) * 8))
        self.driver_dots = _DotLayer(self.driver_pos, 2)
        self.driver_sign = np.array([d["sign"] for d in drivers])
        self.driver_counts = {"exc": sum(1 for d in drivers if d["sign"] > 0),
                              "inh": sum(1 for d in drivers if d["sign"] < 0)}
        self.set_ladder(self.circuits["dng02_ladder"])

    def set_ladder(self, ladder):
        """DNg02 cell rectangles in ladder order, strongest input nearest the
        midline on both sides, so recruitment visibly grows outward."""
        self.ladder = list(ladder)
        self.dng02_rects = []
        per_side = {"left": 0, "right": 0}
        for _, side in self.ladder:
            i = per_side[side]
            per_side[side] += 1
            if side == "left":
                x1 = 488 - i * 13
                rect = (x1 - 10, self.DNG02_TOP, x1, self.DNG02_TOP + 20)
            else:
                x0 = 512 + i * 13
                rect = (x0, self.DNG02_TOP, x0 + 10, self.DNG02_TOP + 20)
            self.dng02_rects.append(rect)
        self.dng02_per_side = per_side

    def _dn_pos(self, cell_type, side):
        x = self.DN_X if side == "left" else _mirror(self.DN_X)
        return x, self.DN_Y[cell_type]

    def _draw_background(self):
        img = np.full((CANVAS_H, CANVAS_W, 3), BG, dtype=np.uint8)

        # Brain outline: optic lobes, central brain, neck.
        for side in ("left", "right"):
            (lx, ly), axes = self.LOBE
            lx = lx if side == "left" else _mirror(lx)
            cv2.ellipse(img, (lx, ly), axes, 0, 0, 360, PANEL, -1, cv2.LINE_AA)
            cv2.ellipse(img, (lx, ly), axes, 0, 0, 360, OUTLINE, 1, cv2.LINE_AA)
        (cx, cy), axes = self.CENTRAL
        cv2.ellipse(img, (cx, cy), axes, 0, 0, 360, PANEL, -1, cv2.LINE_AA)
        cv2.ellipse(img, (cx, cy), axes, 0, 0, 360, OUTLINE, 1, cv2.LINE_AA)
        neck = np.array([(470, 455), (530, 455), (520, 486), (480, 486)], dtype=np.int32)
        cv2.fillConvexPoly(img, neck, PANEL, cv2.LINE_AA)
        cv2.line(img, (470, 457), (480, 486), OUTLINE, 1, cv2.LINE_AA)
        cv2.line(img, (530, 457), (520, 486), OUTLINE, 1, cv2.LINE_AA)
        _text(img, "neck", 500, 480, 0.33, MUTED, align="center")

        for side, label in (("left", "LEFT"), ("right", "RIGHT")):
            (ex, ey), _ = self.EYE
            ex = ex if side == "left" else _mirror(ex)
            _text(img, f"{label} eye", ex, 392, 0.36, MUTED, align="center")
            lx = self.LOBE[0][0] if side == "left" else _mirror(self.LOBE[0][0])
            _text(img, f"LPLC2 ({self.circuits['inputs'].get(('LPLC2', side), 0)})",
                  lx, 130, 0.38, INPUT, align="center")
            _text(img, f"LC4 ({self.circuits['inputs'].get(('LC4', side), 0)})",
                  lx, 252, 0.38, INPUT, align="center")
            _text(img, f"{label.lower()} optic lobe", lx, 396, 0.36, MUTED, align="center")

        # DN row labels between the two columns.
        for cell_type in DN_ROWS:
            y = self.DN_Y[cell_type]
            _text(img, cell_type, 500, y - 2, 0.42, DN_COLORS[cell_type], align="center")
            _text(img, DN_ROLES[cell_type], 500, y + 13, 0.33, MUTED, align="center")

        _text(img, f"DNg02 drive pool  {self.driver_counts['exc']} exc / "
                   f"{self.driver_counts['inh']} inh", 500, self.DRIVE_TOP - 9, 0.36, MUTED, align="center")

        # Panels.
        cv2.rectangle(img, (12, 496), (488, 710), PANEL, -1)
        cv2.rectangle(img, (512, 496), (988, 710), PANEL, -1)
        _text(img, "BRAIN -> MOTOR  (descending output)", 22, 514, 0.42, TEXT)
        _text(img, "ACTIVITY  last ~4 s", 522, 514, 0.42, TEXT)
        return img

    # --- render ---

    def render(self, act, constants):
        img = self._background.copy()
        self._draw_header(img, act)
        self._draw_eyes(img, act)
        self._draw_optic_lobes(img, act, constants)
        self._draw_descending(img, act)
        self._draw_stabilizer(img, act, constants)
        self._draw_food(img, act)
        self._draw_motor(img, act)
        self._draw_raster(img, act)
        return img

    def _draw_header(self, img, act):
        _text(img, "FLY BRAIN  live activity", 14, 24, 0.62, TEXT, 1)
        sub = (f"brain step {act.brain_ms:.1f} ms   step #{act.brain_steps}"
               if act.brain_attached else "connectome not running in this script")
        _text(img, sub, 14, 41, 0.36, MUTED)

        chips = (
            ("ESCAPE", DNP01, act.brain_attached, act.escape_active()),
            ("STABILIZER", DNG02, act.with_dng02, act.stabilizer_active()),
            ("FEEDING", FOOD, act.food is not None, act.feeding_running()),
        )
        x = 330
        for name, color, running, active in chips:
            box = (x, 12, x + 118, 36)
            if active:
                cv2.rectangle(img, box[:2], box[2:], color, -1)
                _text(img, name, x + 59, 29, 0.42, BG, 1, align="center")
            else:
                cv2.rectangle(img, box[:2], box[2:], _mix(DIM, color, 0.6) if running else DIM, 1)
                _text(img, name if running else f"{name} off", x + 59, 29, 0.38,
                      MUTED if running else DIM, align="center")
            x += 128

        # Whatever is actually flying the drone gets the state readout.
        state = act.food["state"] if act.food_drives() else act.state
        label = state
        if state == "ESCAPE" and act.escape_direction:
            label = f"ESCAPE {act.escape_direction}"
        _text(img, label, 986, 29, 0.55, STATE_COLORS.get(state, TEXT), 1, align="right")
        _text(img, "state", 986, 43, 0.32, MUTED, align="right")

    def _draw_eyes(self, img, act):
        # What each side's loom input is built from: flybrain_controller.py
        # feeds max(side, center) expansion to each eye's loom.
        for side in ("left", "right"):
            exp = max(act.expansion[side], act.expansion["center"])
            level = min(1.0, exp / 3.0)
            (ex, ey), axes = self.EYE
            ex = ex if side == "left" else _mirror(ex)
            cv2.ellipse(img, (ex, ey), axes, 0, 0, 360, _mix(PANEL, EYE, 0.15 + 0.5 * level), -1, cv2.LINE_AA)
            cv2.ellipse(img, (ex, ey), axes, 0, 0, 360, _mix(OUTLINE, EYE, level), 1, cv2.LINE_AA)
            layer = self.eye_dots[side]
            layer.paint(img, np.tile(_mix(DIM, EYE, 0.3 + 0.7 * level), (layer.n, 1)))
            _text(img, f"exp {exp:.2f}/s", ex, 408, 0.34, TEXT if level > 0.05 else MUTED, align="center")

    def _draw_optic_lobes(self, img, act, constants):
        dt = constants["STEP_DT_MS"] / 1000.0
        for side in ("left", "right"):
            loom = act.loom[side] if act.brain_attached else 0.0
            rate_hz = loom * constants["MAX_POI_RATE"]
            dots = self.input_dots[side]
            # Sampled at the model's own Poisson rate - see module docstring.
            fired = self.rng.random(dots.n) < (1.0 - math.exp(-rate_hz * dt))
            dots.paint(img, _mix_many(DIM, INPUT, np.where(fired, 1.0, 0.15 + 0.55 * loom)))

            lx = self.LOBE[0][0] if side == "left" else _mirror(self.LOBE[0][0])
            _text(img, f"loom {loom:.2f}", lx, 330, 0.42, INPUT if loom > 0 else MUTED, align="center")
            _text(img, f"{rate_hz:.1f} Hz / cell", lx, 347, 0.34, MUTED, align="center")

            # LC4/LPLC2 -> DN bundle. Every direct synapse in this circuit is
            # ipsilateral (fly_brain_controller.py's wiring check), so each
            # lobe only feeds its own side's DNs.
            mirror = (lambda p: p) if side == "left" else (lambda p: (_mirror(p[0]), p[1]))
            exit_pt, junction = mirror(self.LOBE_EXIT), mirror(self.BUNDLE_JUNCTION)
            width = 1 + int(round(4 * loom))
            color = _mix(DIM, INPUT, 0.25 + 0.75 * loom)
            cv2.line(img, exit_pt, junction, color, width, cv2.LINE_AA)
            for cell_type in DN_ROWS:
                cv2.line(img, junction, self._dn_pos(cell_type, side), color, max(1, width - 1), cv2.LINE_AA)

    def _draw_descending(self, img, act):
        for cell_type in DN_ROWS:
            for side in ("left", "right"):
                name = f"{cell_type}_{side}"
                glow = act.dn_glow.get(name, 0.0)
                n = act.spike_counts.get(name, 0)
                x, y = self._dn_pos(cell_type, side)
                color = DN_COLORS[cell_type]
                if glow > 0.05:
                    cv2.circle(img, (x, y), self.DN_R + 4 + int(6 * glow), _mix(PANEL, color, 0.5 * glow), -1, cv2.LINE_AA)
                cv2.circle(img, (x, y), self.DN_R, _mix(PANEL, color, 0.12 + 0.88 * glow), -1, cv2.LINE_AA)
                cv2.circle(img, (x, y), self.DN_R, color, 2 if n else 1, cv2.LINE_AA)
                _text(img, "L" if side == "left" else "R", x, y + 5, 0.4,
                      BG if glow > 0.5 else TEXT, 1, align="center")
                if n:
                    tx = x - self.DN_R - 6 if side == "left" else x + self.DN_R + 6
                    _text(img, f"{n}", tx, y + 5, 0.45, color, 1,
                          align="right" if side == "left" else "left")

        # Everything descending leaves through the neck: light it in the colour
        # of whichever descending cell is most active right now.
        glows = [(act.dn_glow.get(f"{t}_{s}", 0.0), DN_COLORS[t])
                 for t in DN_ROWS for s in ("left", "right")]
        glows.append((float(act.dng02_glow.max()) if len(act.dng02_glow) else 0.0, DNG02))
        glow, color = max(glows, key=lambda g: g[0])
        if glow > 0.05:
            neck = np.array([(472, 460), (528, 460), (519, 485), (481, 485)], dtype=np.int32)
            cv2.fillConvexPoly(img, neck, _mix(PANEL, color, 0.7 * glow), cv2.LINE_AA)
            _text(img, "neck", 500, 480, 0.33, TEXT, align="center")

    def _draw_stabilizer(self, img, act, constants):
        rates = (driver_rates(self.circuits["drivers"], act.drive["drive_common"],
                              act.drive["drive_left"], act.drive["drive_right"], constants)
                 if act.with_dng02 else np.zeros(len(self.driver_pos)))
        dt = constants["STEP_DT_MS"] / 1000.0
        fired = self.rng.random(len(rates)) < (1.0 - np.exp(-rates * constants["MAX_DRIVE_RATE"] * dt))
        amount = np.where(fired, 1.0, 0.15 + 0.6 * rates)
        colors = np.where((self.driver_sign > 0)[:, None],
                          _mix_many(DIM, DRIVE_EXC, amount), _mix_many(DIM, DRIVE_INH, amount))
        self.driver_dots.paint(img, colors)

        for i, (x0, y0, x1, y1) in enumerate(self.dng02_rects):
            glow = act.dng02_glow[i] if i < len(act.dng02_glow) else 0.0
            cv2.rectangle(img, (x0, y0), (x1, y1), _mix(PANEL, DNG02, 0.12 + 0.88 * glow), -1)
            cv2.rectangle(img, (x0, y0), (x1, y1), _mix(OUTLINE, DNG02, 0.4 if act.with_dng02 else 0.0), 1)

        n_l, n_r = act.dng02.get("n_left", 0), act.dng02.get("n_right", 0)
        if act.with_dng02:
            _text(img, f"DNg02   L {n_l}/{self.dng02_per_side['left']}   "
                       f"R {n_r}/{self.dng02_per_side['right']}   recruited",
                  500, self.DNG02_TOP + 36, 0.38, DNG02 if n_l + n_r else MUTED, align="center")
        else:
            why = "not built (optomotor off)" if act.brain_attached else "connectome not running"
            _text(img, f"DNg02 x24  -  {why}", 500, self.DNG02_TOP + 36, 0.38, MUTED, align="center")

    def _draw_food(self, img, act):
        x0, y0, x1, y1 = self.FOOD_BOX
        running = act.feeding_running()
        cv2.rectangle(img, (x0, y0), (x1, y1), _mix(PANEL, FOOD, 0.12 if running else 0.0), -1)
        cv2.rectangle(img, (x0, y0), (x1, y1), FOOD if running else OUTLINE, 1)
        if act.food is None:
            _text(img, "FoodNeuron (feeding) - idle", 500, y0 + 29, 0.36, MUTED, align="center")
            return
        food = act.food
        state_color = STATE_COLORS.get(food["state"], FOOD) if running else MUTED
        _text(img, "FEEDING", x0 + 8, y0 + 18, 0.42, FOOD if running else MUTED)
        _text(img, food["state"], x0 + 82, y0 + 18, 0.42, state_color)
        seen = food["label"] if food["visible"] else "no banana"
        _text(img, seen, x1 - 8, y0 + 18, 0.33, TEXT if food["visible"] else MUTED, align="right")
        _text(img, "hunger", x0 + 8, y0 + 40, 0.33, MUTED)
        _bar(img, x0 + 58, y0 + 31, 120, 10, food["hunger"] / 100.0, FOOD if running else DIM)
        _text(img, f"{food['hunger']:.0f}", x0 + 186, y0 + 40, 0.36, TEXT)

    def _draw_motor(self, img, act):
        # Signed bars are drawn screen-left = turn left. The two circuits use
        # opposite signs for that (DNp06 yaw > 0 = left, DNg02 steer > 0 =
        # right - see flybrain_controller.py), so each row says which one it is
        # and the number is shown as a magnitude plus L/R rather than a sign.
        rows = (
            ("ESCAPE", "DNp01", act.escape, DNP01, None, ESCAPE_STATE_THRESHOLD),
            ("TURN", "DNp06", act.yaw, DNP06, +1, None),
            ("FORWARD", "DNp03+06", act.forward, DNP03, None, None),
            ("THRUST", "DNg02", act.dng02.get("thrust", 0.0), DNG02, None, None),
            ("STEER", "DNg02", act.dng02.get("steer", 0.0), DNG02, -1, None),
        )
        live = act.brain_attached
        for i, (name, source, value, color, left_sign, tick) in enumerate(rows):
            y = 530 + i * 26
            if name in ("THRUST", "STEER") and not act.with_dng02:
                value, color = 0.0, DIM
            if not live:
                value, color = 0.0, DIM
            _text(img, name, 22, y + 11, 0.42, TEXT if live else MUTED)
            _text(img, source, 110, y + 11, 0.32, MUTED)
            if left_sign is None:
                _bar(img, 200, y, 220, 13, value, color, tick=tick)
                _text(img, f"{value:.2f}", 478, y + 11, 0.38, TEXT, align="right")
            else:
                toward_left = value * left_sign > 0
                _bar(img, 200, y, 220, 13, -abs(value) if toward_left else abs(value), color, signed=True)
                _text(img, "L", 192, y + 11, 0.3, MUTED, align="right")
                _text(img, "R", 426, y + 11, 0.3, MUTED)
                side = "" if abs(value) < 0.005 else (" L" if toward_left else " R")
                _text(img, f"{abs(value):.2f}{side}", 478, y + 11, 0.38, TEXT, align="right")

        if act.food_drives() and act.food["rc"] is not None:
            # Drone/tello_camera.py: food_orbit.py's RC command is what gets
            # sent; the brain only decides when to get scared.
            lr, fb, ud, yaw = act.food["rc"]
            _text(img, "sent to the Tello  (FoodNeuron RC, -100..100)", 22, 666, 0.34, MUTED)
            _text(img, f"LR {lr:+d}   FB {fb:+d}   UD {ud:+d}   YAW {yaw:+d}", 22, 685, 0.38, TEXT)
            if act.brain_attached:
                _text(img, "(brain's own dodge command not used here)", 22, 702, 0.32, MUTED)
            # Tello RC: lr > 0 = right, fb > 0 = forward, yaw > 0 = clockwise.
            self._draw_glyph(img, right=lr * 0.4, forward=fb * 0.4, turn_left=-yaw / 50.0)
            return

        _text(img, "brain command", 22, 666, 0.34, MUTED)
        _text(img, "(before main.py's test-mode hover / SafetyLayer)", 22, 702, 0.32, MUTED)
        cmd = act.cmd
        if cmd is None:
            _text(img, "-", 22, 685, 0.4, MUTED)
            return
        yaw = cmd["yaw_rate"]
        _text(img, f"fwd {cmd['forward_speed']:+.2f} m/s   strafe {cmd['strafe_speed']:+.2f} m/s   "
                   f"yaw {yaw:+.2f} rad/s", 22, 685, 0.38, TEXT)
        # This project's convention: strafe > 0 = left, yaw_rate > 0 = turn left.
        self._draw_glyph(img, right=-cmd["strafe_speed"] * 7, forward=cmd["forward_speed"] * 7,
                         turn_left=yaw)

    @staticmethod
    def _draw_glyph(img, right, forward, turn_left):
        """Top-down drone: arrow = commanded velocity in pixels (up =
        forward), arc = commanded turn (roughly rad/s, positive = left)."""
        gx, gy = 452, 676
        cv2.circle(img, (gx, gy), 16, OUTLINE, 1, cv2.LINE_AA)
        if math.hypot(right, forward) > 4:     # shorter just draws a blob
            cv2.arrowedLine(img, (gx, gy), (int(gx + right), int(gy - forward)), TEXT, 2,
                            cv2.LINE_AA, tipLength=0.3)
        if abs(turn_left) > 0.02:
            # 270 deg is straight up in image coordinates; decreasing sweeps
            # toward the left.
            sweep = int(min(150, abs(turn_left) * 150))
            end = 270 - sweep if turn_left > 0 else 270 + sweep
            cv2.ellipse(img, (gx, gy), (22, 22), 0, 270, end, DNP06, 2, cv2.LINE_AA)

    RASTER_ROWS = (
        ("loom L", INPUT), ("loom R", INPUT),
        ("DNp01 L", DNP01), ("DNp01 R", DNP01),
        ("DNp03 L", DNP03), ("DNp03 R", DNP03),
        ("DNp06 L", DNP06), ("DNp06 R", DNP06),
        ("DNg02 L", DNG02), ("DNg02 R", DNG02),
    )

    def _draw_raster(self, img, act):
        x0, y0, row_h, width = 596, 526, 15, 387
        n_rows = len(self.RASTER_ROWS)
        data = np.zeros((n_rows, HISTORY_CYCLES))
        if act.history:
            cols = np.array(act.history).T
            data[:, HISTORY_CYCLES - cols.shape[1]:] = cols
        colors = np.array([c for _, c in self.RASTER_ROWS], dtype=float)[:, None, :]
        base = np.array(DIM, dtype=float)[None, None, :] * 0.6
        strip = (base + (colors - base) * data[:, :, None]).astype(np.uint8)
        strip = cv2.resize(strip, (width, n_rows * row_h), interpolation=cv2.INTER_NEAREST)
        img[y0:y0 + n_rows * row_h, x0:x0 + width] = strip
        for i, (label, color) in enumerate(self.RASTER_ROWS):
            y = y0 + i * row_h
            _text(img, label, 522, y + 11, 0.34, color)
            cv2.line(img, (x0, y), (x0 + width, y), PANEL, 1)
        _text(img, "older", x0, y0 + n_rows * row_h + 14, 0.32, MUTED)
        _text(img, "now", x0 + width, y0 + n_rows * row_h + 14, 0.32, MUTED, align="right")
        _text(img, "bright = spiking / driven this 20 ms step", x0 + width // 2,
              y0 + n_rows * row_h + 14, 0.32, MUTED, align="center")


# ---------------------------------------------------------------- window + hooks

class BrainView:
    """One window, fed by whichever pathways are running in this process."""

    def __init__(self, window=WINDOW_NAME, max_fps=DEFAULT_MAX_FPS, display=True):
        self.window = window
        self.display = display
        self.min_interval = 1.0 / max_fps if max_fps else 0.0
        self.circuits = load_circuits()
        self.constants = dict(DEFAULT_CONSTANTS)
        self.activity = BrainActivity(self.circuits["dng02_ladder"])
        self.diagram = BrainDiagram(self.circuits)
        self._last_shown = 0.0
        self._window_open = False
        self.last_frame = None

    def bind_brain(self, controller):
        """Pulls the network's real configuration out of the brain
        subprocess's handshake (see flybrain_controller.py's _FlyBrainProcess)."""
        brain = controller._brain
        self.constants.update({k: v for k, v in getattr(brain, "constants", {}).items()
                               if k in DEFAULT_CONSTANTS})
        labels = getattr(brain, "dng02_labels", [])
        sides = getattr(brain, "dng02_sides", [])
        if labels and len(labels) == len(sides):
            ladder = list(zip(labels, sides))
            self.activity.set_ladder(ladder)
            self.diagram.set_ladder(ladder)
        self.activity.brain_attached = True
        self.activity.with_dng02 = bool(getattr(controller, "optomotor", False) or labels)

    def show(self, force=False):
        now = time.monotonic()
        if not force and now - self._last_shown < self.min_interval:
            return
        self._last_shown = now
        self.last_frame = self.diagram.render(self.activity, self.constants)
        if not self.display:
            return
        if not self._window_open:
            cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window, CANVAS_W, CANVAS_H)
            self._window_open = True
        cv2.imshow(self.window, self.last_frame)


_views = {}


def _view_for(window, **kwargs):
    if window not in _views:
        _views[window] = BrainView(window, **kwargs)
    return _views[window]


def attach(controller, *, window=WINDOW_NAME, max_fps=DEFAULT_MAX_FPS, display=True):
    """Shows `controller`'s (a NeuralPathways.flybrain_controller
    FlyBrainController) activity live. Wraps two of its instance methods;
    neither changes arguments, results or timing beyond the drawing itself.
    Returns the BrainView."""
    view = _view_for(window, max_fps=max_fps, display=display)
    view.bind_brain(controller)

    inner_request = controller._brain.request

    def request(payload):
        start = time.perf_counter()
        result = inner_request(payload)
        view.activity.record_brain(payload, result, (time.perf_counter() - start) * 1000)
        return result

    controller._brain.request = request

    inner_decide = controller.decide

    def decide(flow, state=None):
        cmd = inner_decide(flow, state)
        view.activity.record_decision(flow, controller, cmd)
        # When a FoodOrbitBehaviour is flying the drone (tello_camera.py), its
        # update() runs after the brain in the same frame and does the redraw,
        # so the picture pairs this brain step with the command actually sent.
        if not view.activity.food_drives():
            view.show()
        return cmd

    controller.decide = decide
    return view


def attach_food(behaviour, *, window=WINDOW_NAME, max_fps=DEFAULT_MAX_FPS, display=True):
    """Same as attach(), for a NeuralPathways.FoodNeuron.food_orbit
    FoodOrbitBehaviour. Shares the window with attach() if both run in the
    same process."""
    view = _view_for(window, max_fps=max_fps, display=display)
    inner_update = behaviour.update

    def update(detections, frame_width, frame_height):
        cmd = inner_update(detections, frame_width, frame_height)
        view.activity.record_food(behaviour, cmd)
        view.show()
        return cmd

    behaviour.update = update
    return view
