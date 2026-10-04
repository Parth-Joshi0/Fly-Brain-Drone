"""
Live picture of the fly brain at work: the whole brain drawn as a cloud of
its real neurons, with the cells the drone's circuit is simulating glowing
where they actually sit each time they spike.

Modelled on the brain view HUD in blendi-remade/fly-brain-minecraft
(src/client/java/com/fruitfly/client/hud/BrainViewHud.java):

    LEFT COLUMN - the brain view
      header         state, spikes this tick, real-time factor
      frontal map    every FlyWire neuron's soma (139k) projected head-on,
      dorsal map     and from above; each pixel coloured by the region most of
                     its somata belong to, brightness ~ log(how many). Spikes
                     splat into a heat buffer that fades over ~0.3 s, so
                     activity reads as glowing, fading points.
      region bars    spikes this tick per region, with a peak-hold tick
      spikes/tick    the last 60 brain steps

    RIGHT COLUMN - the circuit readout (like that repo's NeuroscopeHud)
      chips          which pathways are running in this process
      MOTOR          what the circuit asked for (escape/turn/forward/thrust/steer)
      POPULATIONS    firing rate of each simulated cell population
      FEEDING        FoodNeuron's state, when one is running
      BRAIN COMMAND  the command FlyBrainController returned
      KEY            what each population does

The projection itself is baked offline by build_brain_atlas.py into
brain_atlas.npz (Schlegel et al. 2024's FlyWire annotation table - soma
positions and super-classes). FlyWire is brain-only, so unlike the reference
there is no nerve cord: the second map is the dorsal view of the brain.

Only the 418 neurons fly_brain_controller.py simulates (LC4/LPLC2, DNp01/03/06,
DNg02 and its drive pool - see looming_circuit_neurons.json and
StabilizerNeuron/dng02_circuit_neurons.json) can ever light up. The rest of the
cloud is anatomy, drawn so you can see WHERE in the brain the circuit lives;
the faint yellow dots mark the simulated cells.

How directly each part was measured:

    map glow, region bars,   REAL spikes when the brain reports "spiked" (every
    spikes/tick, Hz          spike of every simulated cell in the last 20 ms -
                             fly_brain_controller.py's step() result, mapped to
                             positions through the handshake's neuron_ids).
                             With an older brain that doesn't, DN and DNg02
                             spikes still come from its real per-cell counts,
                             but LC4/LPLC2 and drive-pool spikes are SAMPLED at
                             the Poisson rate each cell is being driven at -
                             statistically what the model does, not a
                             recording. The footnote says which mode is live.
    motor bars / command     what the circuit asked for (escape/yaw/forward,
                             thrust/steer) and the command FlyBrainController
                             returned. main.py's NEURON_TEST_MODE and
                             SafetyLayer can still override it downstream.

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
ATLAS_PATH = HERE / "brain_atlas.npz"

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

CANVAS_W, CANVAS_H = 960, 664
# A redraw costs a few ms, and on the real drone the brain step alone already
# uses ~20 of the 33 ms decision budget - so by default only every other
# decision cycle is drawn. Activity is still RECORDED every cycle (every spike
# is splatted into the heat map and counted); only the repaint is rate-limited.
DEFAULT_MAX_FPS = 15.0

HISTORY_TICKS = 60             # spikes/tick chart width, as in the reference
HEAT_TAU_S = 0.30              # glow fade time constant (reference: 0.30 s)
PEAK_DECAY = 0.97              # per brain step, region bars' peak-hold tick
POP_HZ_SMOOTHING = 0.3         # EMA on the population Hz readout - one 20 ms
                               # window of a 2-cell population is a 0/25/50 Hz
                               # staircase otherwise
POP_PEAK_TICKS = 20            # population peak-hold = max of this many steps
POP_BAR_HZ = 100.0             # full-scale population bar, as in the reference
DN_ACTIVE_TICKS = 5            # ESCAPE chip stays lit this many steps after a DN spike
FEEDING_ACTIVE_SECONDS = 1.0   # food pathway counts as running this long after
                               # its last update()

# Stamp each spike leaves in the heat map: 1.0 on the soma's pixel, SPLAT_RING
# on the 3x3 around it, SPLAT_HALO on the 5x5. Bigger than the reference's
# 1 px + 4 neighbours: it has 141k cells that can fire, this circuit 418, so a
# single-pixel hit would be lost in the cloud.
SPLAT_RING, SPLAT_HALO = 0.55, 0.25
_KERNEL_3 = np.ones((3, 3), np.uint8)
_KERNEL_5 = np.ones((5, 5), np.uint8)


def _bgr(r, g, b):
    return (b, g, r)


def _hex(rgb):
    return _bgr((rgb >> 16) & 255, (rgb >> 8) & 255, rgb & 255)


# fly-brain-minecraft's HudStyle palette.
BG = _bgr(20, 22, 28)
PANEL = _bgr(11, 15, 20)
INSET = _bgr(0, 0, 0)
BORDER = _bgr(80, 84, 90)
TEXT = _bgr(232, 232, 232)
MUTED = _bgr(138, 147, 156)
DIM = _bgr(60, 66, 74)
BAR_BG = _bgr(32, 38, 45)
ACCENT = _hex(0x9CDCFE)
RED = _hex(0xFF4B4B)
ORANGE = _hex(0xFFA040)
YELLOW = _hex(0xFFE066)
GREEN = _hex(0x55D66E)
CYAN = _hex(0x40C4FF)
TEAL = _hex(0x3FD2C7)
PURPLE = _hex(0xB98CFF)
RASTER_LOW = _hex(0x2F5C8F)
HOT = np.array(_bgr(255, 250, 200), dtype=np.float32)   # what a fresh spike glows
# The reference's 0.30 + 0.65 x density, scaled down a little: it can light
# any of 141k cells, here a few hundred spikes have to stand out against the
# cloud.
MAP_BRIGHT_FLOOR, MAP_BRIGHT_GAIN = 0.24, 0.56
HEAT_ALPHA_GAIN = 1.4          # heat 0.7 and up is fully opaque

# The reference's region colours, keyed by build_brain_atlas.py's names.
REGION_COLORS = {
    "optic lobe L": _hex(0x3FB8B0),
    "optic lobe R": _hex(0x6FE0D8),
    "central brain": _hex(0x9DB0E6),
    "descending": _hex(0x5B9BFF),
    "ascending": _hex(0xC89A6A),
    "motor": _hex(0xFFA040),
    "sensory": _hex(0x55D66E),
    "other": _hex(0x8A8A8A),
}

DNP01, DNP03, DNP06, DNG02, FOOD = RED, ORANGE, CYAN, GREEN, YELLOW
INPUT = TEAL                      # LC4 / LPLC2 - optic-lobe colour family

# Population rows on the right, in display order (left column, then right).
POPULATIONS = (
    ("LC4 L", INPUT), ("LC4 R", INPUT), ("LPLC2 L", INPUT), ("LPLC2 R", INPUT),
    ("DNp01 L", DNP01), ("DNp01 R", DNP01), ("DNp03 L", DNP03),
    ("DNp03 R", DNP03), ("DNp06 L", DNP06), ("DNp06 R", DNP06),
    ("DNg02 L", DNG02), ("DNg02 R", DNG02), ("drive exc", DNG02), ("drive inh", PURPLE),
)
POP_INDEX = {name: i for i, (name, _) in enumerate(POPULATIONS)}

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

def _short_side(side):
    return "L" if side == "left" else "R"


def load_circuits():
    """The two circuit JSONs, reduced to what the diagram needs. Needs neither
    brian2 nor pandas - just the files fly_brain_controller.py builds from."""
    with open(LOOMING_IDS_PATH) as f:
        looming = json.load(f)
    with open(DNG02_IDS_PATH) as f:
        dng02 = json.load(f)

    # root id -> population row (see POPULATIONS).
    population = {}
    for n in looming["input_neurons"] + looming["output_neurons"]:
        population[n["root_id"]] = f"{n['cell_type']} {_short_side(n['side'])}"
    for n in dng02["output_neurons"]:
        population[n["root_id"]] = f"DNg02 {_short_side(n['side'])}"
    for n in dng02["drive_neurons"]:
        population[n["root_id"]] = "drive exc" if n["sign"] > 0 else "drive inh"

    return {
        "inputs": [(n["root_id"], n["side"]) for n in looming["input_neurons"]],
        # "DNp01_left" (fly_brain_controller.py's spike_counts key) -> root id
        "dn_ids": {f"{n['cell_type']}_{n['side']}": n["root_id"] for n in looming["output_neurons"]},
        "dng02_ladder": dng02_ladder(dng02["output_neurons"]),
        "dng02_ids": dict(zip((label for label, _ in dng02_ladder(dng02["output_neurons"])),
                              (n["root_id"] for n in _ladder_order(dng02["output_neurons"])))),
        "drivers": dng02["drive_neurons"],
        "population": population,
    }


def _ladder_order(cells):
    return sorted(cells, key=lambda n: -n["exc_weight"])


def dng02_ladder(cells):
    """(label, side) in recruitment order, labelled exactly the way
    fly_brain_controller.py's _load_circuit() does it - the same labels its
    dng02.counts uses."""
    seen = {}
    ladder = []
    for n in _ladder_order(cells):
        key = (n["cell_type"], n["side"])
        seen[key] = seen.get(key, 0) + 1
        ladder.append((f"{n['cell_type']}_{n['side']}_{seen[key]}", n["side"]))
    return ladder


def driver_rates(drivers, drive_common, drive_left, drive_right, constants):
    """Each drive-pool cell's firing rate as a 0..1 fraction of
    MAX_DRIVE_RATE - the same rule fly_brain_controller.py's step() applies.
    Only used to sample drive-pool spikes for a brain that doesn't report its
    own."""
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


class BrainAtlas:
    """brain_atlas.npz (see build_brain_atlas.py): per view, somata per pixel
    and each pixel's dominant region, plus the pixel every simulated circuit
    neuron sits at. Circuit neurons are addressed by their row in
    circuit_ids."""

    def __init__(self, path=ATLAS_PATH):
        with np.load(path) as a:
            self.region_names = [str(r) for r in a["region_names"]]
            self.views = [str(v) for v in a["views"]]
            self.circuit_ids = a["circuit_ids"].astype(np.int64)
            self.circuit_region = a["circuit_region"].astype(np.intp)
            self.count = {v: a[f"count_{v}"] for v in self.views}
            self.region = {v: a[f"region_{v}"] for v in self.views}
            self.px = {v: a[f"circuit_px_{v}"].astype(np.intp) for v in self.views}
        self.row_of = {int(rid): i for i, rid in enumerate(self.circuit_ids)}

    def shape(self, view):
        return self.count[view].shape


_atlas = None


def load_atlas():
    global _atlas
    if _atlas is None:
        _atlas = BrainAtlas()
    return _atlas


# ---------------------------------------------------------------- activity model

class BrainActivity:
    """Everything the diagram needs, fed one brain step / decision at a time.
    Kept separate from drawing so it can be checked without a window."""

    def __init__(self, circuits, atlas, constants=None):
        self.circuits = circuits
        self.atlas = atlas
        self.constants = constants if constants is not None else dict(DEFAULT_CONSTANTS)
        self.rng = np.random.default_rng()
        self.brain_attached = False
        self.with_dng02 = False
        self.loom = {"left": 0.0, "right": 0.0}
        self.drive = {"drive_common": 0.0, "drive_left": 0.0, "drive_right": 0.0}
        self.escape = 0.0
        self.yaw = 0.0
        self.forward = 1.0
        self.dng02 = {"n_left": 0, "n_right": 0, "thrust": 0.0, "steer": 0.0, "counts": {}}
        self.spike_counts = {}
        self.brain_ms = 0.0
        self.brain_steps = 0

        self.expansion = {"left": 0.0, "center": 0.0, "right": 0.0}
        self.state = "-"
        self.escape_direction = ""
        self.cmd = None

        self.food = None          # dict once a FoodOrbitBehaviour reports in
        self.food_time = 0.0

        # brain-local neuron index -> atlas circuit row, from the handshake's
        # neuron_ids. None until a brain that reports it is bound.
        self.neuron_rows = None
        self.spike_source = None  # "recorded" / "sampled" once a step arrives

        # atlas row -> population row; per-population cell counts.
        pop = circuits["population"]
        self.row_pop = np.array([POP_INDEX.get(pop.get(int(rid)), -1) for rid in atlas.circuit_ids])
        self.pop_cells = np.maximum(np.bincount(self.row_pop[self.row_pop >= 0],
                                                minlength=len(POPULATIONS)), 1)
        self._input_rows = {side: np.array([atlas.row_of[rid] for rid, s in circuits["inputs"] if s == side])
                            for side in ("left", "right")}
        self._driver_rows = np.array([atlas.row_of[d["root_id"]] for d in circuits["drivers"]])

        self.heat = {v: np.zeros(atlas.shape(v), dtype=np.float32) for v in atlas.views}
        self._heat_time = time.monotonic()
        self.n_regions = len(atlas.region_names)
        self._clear_counts()

    def _clear_counts(self):
        self.spikes = 0
        self.region_counts = np.zeros(self.n_regions)
        self.region_peak = np.zeros(self.n_regions)
        self.spike_history = deque(maxlen=HISTORY_TICKS)
        self.pop_hz = np.zeros(len(POPULATIONS))
        self.pop_recent = deque(maxlen=POP_PEAK_TICKS)
        self._dn_quiet = DN_ACTIVE_TICKS

    def set_neuron_ids(self, ids):
        self.neuron_rows = np.array([self.atlas.row_of.get(int(r), -1) for r in ids], dtype=np.intp)

    # --- spikes in -> atlas rows ---

    def _recorded_rows(self, spiked):
        idx = np.asarray(spiked, dtype=np.intp)
        idx = idx[(idx >= 0) & (idx < len(self.neuron_rows))]
        rows = self.neuron_rows[idx]
        return rows[rows >= 0]

    def _sampled_rows(self, result):
        """For a brain that doesn't report "spiked": real per-cell counts for
        the DNs and DNg02, Poisson samples for the cells it only gives a rate
        for. See the module docstring."""
        row_of = self.atlas.row_of
        rows, n = [], []
        for key, count in self.spike_counts.items():
            rid = self.circuits["dn_ids"].get(key)
            if rid is not None and count:
                rows.append(row_of[rid])
                n.append(count)
        for label, count in self.dng02.get("counts", {}).items():
            rid = self.circuits["dng02_ids"].get(label)
            if rid is not None and count:
                rows.append(row_of[rid])
                n.append(count)
        sampled = [np.repeat(np.array(rows, dtype=np.intp), n)]

        dt = self.constants["STEP_DT_MS"] / 1000.0
        for side, cells in self._input_rows.items():
            lam = self.loom[side] * self.constants["MAX_POI_RATE"] * dt
            if lam > 0:
                sampled.append(np.repeat(cells, self.rng.poisson(lam, len(cells))))
        if self.with_dng02:
            rates = driver_rates(self.circuits["drivers"], self.drive["drive_common"],
                                 self.drive["drive_left"], self.drive["drive_right"], self.constants)
            if rates.any():
                lam = rates * self.constants["MAX_DRIVE_RATE"] * dt
                sampled.append(np.repeat(self._driver_rows, self.rng.poisson(lam)))
        return np.concatenate(sampled)

    # --- recording ---

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

        spiked = result.get("spiked")
        if spiked is not None and self.neuron_rows is not None:
            rows = self._recorded_rows(spiked)
            self.spikes = len(spiked)
            self.spike_source = "recorded"
        else:
            rows = self._sampled_rows(result)
            self.spikes = len(rows)
            self.spike_source = "sampled"

        self._splat(rows)
        self.region_counts = np.bincount(self.atlas.circuit_region[rows], minlength=self.n_regions)
        self.region_peak = np.maximum(self.region_peak * PEAK_DECAY, self.region_counts)
        self.spike_history.append(self.spikes)

        pops = self.row_pop[rows]
        hz = np.bincount(pops[pops >= 0], minlength=len(POPULATIONS)) / (
            self.pop_cells * self.constants["STEP_DT_MS"] / 1000.0)
        self.pop_hz = POP_HZ_SMOOTHING * hz + (1 - POP_HZ_SMOOTHING) * self.pop_hz
        self.pop_recent.append(self.pop_hz.copy())
        self._dn_quiet = 0 if any(self.spike_counts.values()) else self._dn_quiet + 1

    def record_reset(self):
        self.loom = {"left": 0.0, "right": 0.0}
        self.drive = {k: 0.0 for k in self.drive}
        self.escape, self.yaw, self.forward = 0.0, 0.0, 1.0
        self.spike_counts = {}
        self.dng02 = {"n_left": 0, "n_right": 0, "thrust": 0.0, "steer": 0.0, "counts": {}}
        for heat in self.heat.values():
            heat[:] = 0.0
        self._clear_counts()

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

    # --- heat map ---

    def decay_heat(self, now=None):
        """Fades the glow by the wall time since the last call - the same
        exp(-dt/tau) the reference applies per frame."""
        now = time.monotonic() if now is None else now
        dt = min(0.25, now - self._heat_time)
        self._heat_time = now
        if dt <= 0:
            return
        k = math.exp(-dt / HEAT_TAU_S)
        for heat in self.heat.values():
            heat *= k
            heat[heat < 0.01] = 0.0

    def _splat(self, rows):
        # Decay first, so a spike recorded between two repaints starts fading
        # from when it happened rather than from the next repaint.
        self.decay_heat()
        if not len(rows):
            return
        for view, heat in self.heat.items():
            px = self.atlas.px[view][rows]
            hit = np.zeros(heat.shape, dtype=np.float32)
            hit[px[:, 1], px[:, 0]] = 1.0
            # max(hit, ring x 3x3 dilation, halo x 5x5 dilation) is exactly
            # the stamp, for every spike at once.
            np.maximum(heat, hit, out=heat)
            np.maximum(heat, SPLAT_RING * cv2.dilate(hit, _KERNEL_3), out=heat)
            np.maximum(heat, SPLAT_HALO * cv2.dilate(hit, _KERNEL_5), out=heat)

    # --- what's "on" right now (header chips) ---

    def food_drives(self):
        """True while a FoodOrbitBehaviour is flying the drone (tello_camera.py).
        There the brain only decides WHEN to get scared - its own motor
        command is discarded (see NeuralPathways/EscapeNeuron/fear_brain.py)."""
        return self.feeding_running()

    def escape_active(self):
        return max(self.loom.values()) > 0.0 or self._dn_quiet < DN_ACTIVE_TICKS

    def stabilizer_active(self):
        return self.with_dng02 and any(v != 0.0 for v in self.drive.values())

    def feeding_running(self):
        return self.food is not None and time.monotonic() - self.food_time < FEEDING_ACTIVE_SECONDS

    def pop_peak(self):
        return np.max(self.pop_recent, axis=0) if self.pop_recent else np.zeros(len(POPULATIONS))


# ---------------------------------------------------------------- drawing helpers

def _mix(dim, bright, amount):
    a = max(0.0, min(1.0, amount))
    return tuple(int(d + (b - d) * a) for d, b in zip(dim, bright))


def _text(img, text, x, y, scale=0.36, color=TEXT, thickness=1, align="left"):
    if align != "left":
        (w, _), _ = cv2.getTextSize(text, FONT, scale, thickness)
        x = x - w // 2 if align == "center" else x - w
    cv2.putText(img, text, (int(x), int(y)), FONT, scale, color, thickness, cv2.LINE_AA)


def _text_width(text, scale=0.36, thickness=1):
    return cv2.getTextSize(text, FONT, scale, thickness)[0][0]


def _bar(img, x, y, w, h, value, color, signed=False, tick=None, peak=None):
    cv2.rectangle(img, (x, y), (x + w, y + h), BAR_BG, -1)
    if signed:
        mid = x + w // 2
        end = int(mid + max(-1.0, min(1.0, value)) * (w // 2))
        if end != mid:
            cv2.rectangle(img, (min(mid, end), y), (max(mid, end), y + h), color, -1)
        cv2.line(img, (mid, y - 2), (mid, y + h + 2), MUTED, 1)
    else:
        end = int(x + max(0.0, min(1.0, value)) * w)
        if end > x:
            cv2.rectangle(img, (x, y), (end, y + h), color, -1)
    if peak is not None and peak > 0:
        px = int(x + max(0.0, min(1.0, peak)) * (w - 1))
        cv2.line(img, (px, y), (px, y + h), _mix(BAR_BG, TEXT, 0.6), 1)
    if tick is not None:
        tx = int(x + tick * w)
        cv2.line(img, (tx, y - 3), (tx, y + h + 3), TEXT, 1)


def _panel(img, x0, y0, x1, y1):
    cv2.rectangle(img, (x0, y0), (x1, y1), PANEL, -1)
    cv2.rectangle(img, (x0, y0), (x1, y1), BORDER, 1)


# ---------------------------------------------------------------- the diagram

class BrainDiagram:
    """Left: the brain view (point-cloud maps, region bars, spikes/tick).
    Right: the circuit readout. Static parts - panels, the baked maps, labels
    - are drawn once and cached; render() only paints what changes."""

    LEFT = (8, 8, 468, CANVAS_H - 8)
    RIGHT = (476, 8, CANVAS_W - 8, CANVAS_H - 8)
    MAP_X = 18
    MAP_TOP = 52
    MAP_GAP = 6
    RX = 486                     # right column content x
    RX1 = CANVAS_W - 18          # right column content right edge
    FOOD_BOX = (486, 300, CANVAS_W - 18, 350)

    def __init__(self, circuits, atlas):
        self.circuits = circuits
        self.atlas = atlas
        self.map_origin = {}
        y = self.MAP_TOP
        for view in atlas.views:
            self.map_origin[view] = (self.MAP_X, y)
            y += atlas.shape(view)[0] + self.MAP_GAP
        self.maps_bottom = y - self.MAP_GAP
        self.regions_top = self.maps_bottom + 22
        self.history_top = self.regions_top + 8 * 12 + 26
        self._hot_base = {}
        self._background = self._draw_background()

    def soma_box(self, root_id, view, r=2):
        """Screen rectangle around one circuit neuron's soma (for tests)."""
        x, y = self.atlas.px[view][self.atlas.row_of[root_id]]
        ox, oy = self.map_origin[view]
        return ox + x - r, oy + y - r, ox + x + r + 1, oy + y + r + 1

    # --- static ---

    def _bake_map(self, view):
        """The reference's background formula: region colour x (floor + gain x
        log-density), see MAP_BRIGHT_FLOOR. Normalised to the 99.5th percentile rather than the max:
        the ~2,000 ascending neurons the table pins to the neck cut all share
        one plane and would otherwise set the scale for everything else."""
        count = self.atlas.count[view].astype(np.float32)
        region = self.atlas.region[view]
        lit = count > 0
        ref = max(2.0, float(np.percentile(count[lit], 99.5)))
        bright = MAP_BRIGHT_FLOOR + MAP_BRIGHT_GAIN * np.clip(np.log1p(count) / math.log1p(ref), 0.0, 1.0)
        palette = np.array([REGION_COLORS.get(name, REGION_COLORS["other"])
                            for name in self.atlas.region_names], dtype=np.float32)
        img = np.empty(count.shape + (3,), dtype=np.float32)
        img[:] = INSET
        img[lit] = palette[region[lit]] * bright[lit, None]
        # Hot pixels start from the region hue (reference: rgb * 0.6 + offset);
        # empty pixels a splat spills onto use "other", as there.
        base = np.empty_like(img)
        base[:] = palette[self.atlas.region_names.index("other")]
        base[lit] = palette[region[lit]]
        self._hot_base[view] = base * 0.6 + np.array(_bgr(100, 100, 60), dtype=np.float32)
        # Faint markers where the simulated cells sit.
        px = self.atlas.px[view]
        img[px[:, 1], px[:, 0]] = img[px[:, 1], px[:, 0]] * 0.4 + np.array(YELLOW, dtype=np.float32) * 0.6
        return img.astype(np.uint8)

    def _draw_background(self):
        img = np.full((CANVAS_H, CANVAS_W, 3), BG, dtype=np.uint8)
        _panel(img, *self.LEFT)
        _panel(img, *self.RIGHT)

        for view in self.atlas.views:
            ox, oy = self.map_origin[view]
            tile = self._bake_map(view)
            h, w = tile.shape[:2]
            img[oy:oy + h, ox:ox + w] = tile
            _text(img, "L", ox + 3, oy + 12, 0.36, MUTED)
            _text(img, "R", ox + w - 3, oy + 12, 0.36, MUTED, align="right")
            _text(img, view, ox + w - 3, oy + h - 5, 0.34, MUTED, align="right")
            if view == "dorsal":
                _text(img, "anterior", ox + w // 2, oy + 11, 0.32, MUTED, align="center")
            else:
                _text(img, "brain", ox + 3, oy + h - 5, 0.34, MUTED)

        x = self.MAP_X
        _text(img, "SPIKES THIS TICK BY REGION (simulated cells)", x, self.regions_top - 6, 0.34, MUTED)
        for i, name in enumerate(self.atlas.region_names):
            _text(img, name, x, self.regions_top + i * 12 + 8, 0.33, REGION_COLORS.get(name, MUTED))

        y = CANVAS_H - 34
        _text(img, f"map: {self._n_somata()} FlyWire somata, coloured by region", x, y, 0.32, MUTED)

        rx = self.RX
        _text(img, "MOTOR  what the circuit asked for", rx, 58, 0.36, MUTED)
        _text(img, f"POPULATIONS  Hz  (bar = {POP_BAR_HZ:.0f} Hz)", rx, 182, 0.36, MUTED)
        _text(img, "BRAIN COMMAND", rx, 374, 0.36, MUTED)
        _text(img, "KEY POPULATIONS", rx, 470, 0.36, MUTED)
        key = (
            ("LC4 / LPLC2", INPUT, "looming detectors, both optic lobes"),
            ("DNp01", DNP01, "Giant Fiber - escape"),
            ("DNp03", DNP03, "brake"),
            ("DNp06", DNP06, "evasive turn away from the loom"),
            ("DNg02", DNG02, "wingbeat amplitude - thrust / steer"),
            ("drive", PURPLE, "central inputs to DNg02 (exc / inh)"),
            ("yellow dots", YELLOW, "where the simulated cells sit"),
        )
        for i, (name, color, what) in enumerate(key):
            yy = 488 + i * 16
            _text(img, name, rx, yy, 0.34, color)
            _text(img, what, rx + 96, yy, 0.34, MUTED)
        return img

    def _n_somata(self):
        total = int(self.atlas.count[self.atlas.views[0]].sum())
        return f"{total / 1000:.0f}k"

    # --- render ---

    def render(self, act, constants):
        img = self._background.copy()
        act.decay_heat()
        self._draw_header(img, act, constants)
        for view in self.atlas.views:
            self._draw_heat(img, act, view)
        self._draw_regions(img, act)
        self._draw_history(img, act)
        self._draw_chips(img, act)
        self._draw_motor(img, act)
        self._draw_populations(img, act)
        self._draw_food(img, act)
        self._draw_command(img, act)
        return img

    def _draw_header(self, img, act, constants):
        x, x1 = self.MAP_X, self.LEFT[2] - 10
        state = act.food["state"] if act.food_drives() else act.state
        label = state
        if state == "ESCAPE" and act.escape_direction:
            label = f"ESCAPE {act.escape_direction}"
        color = STATE_COLORS.get(state, TEXT)
        cv2.rectangle(img, (x, 17), (x + 7, 24), color, -1)
        cv2.rectangle(img, (x, 17), (x + 7, 24), _mix(BG, TEXT, 0.5), 1)
        _text(img, "FLY BRAIN", x + 13, 25, 0.42, ACCENT)
        _text(img, "FlyWire v630", x + 13 + _text_width("FLY BRAIN", 0.42) + 8, 25, 0.34, MUTED)
        if act.brain_attached:
            _text(img, f"step #{act.brain_steps}", x1, 25, 0.34, MUTED, align="right")

        if not act.brain_attached:
            _text(img, label if label != "-" else "", x, 42, 0.4, color)
            _text(img, "connectome not running in this script", x1, 42, 0.34, MUTED, align="right")
            return
        _text(img, label, x, 42, 0.4, color)
        spk = f"{act.spikes} spk/tick"
        _text(img, spk, x + _text_width(label, 0.4) + 10, 42, 0.38, TEXT)
        if act.brain_ms > 0:
            # Simulated time per wall-clock time for the last step (brain
            # subprocess round trip included).
            rt = constants["STEP_DT_MS"] / act.brain_ms
            _text(img, f"RT {rt:.2f}x", x1, 42, 0.38, RED if rt < 0.9 else GREEN, align="right")

    def _draw_heat(self, img, act, view):
        heat = act.heat[view]
        lit = (heat > 0.02).view(np.uint8)
        # Only the box around what is glowing - during a one-sided loom
        # that's one optic lobe, not the whole map.
        bx, by, bw, bh = cv2.boundingRect(lit)
        if bw == 0:
            return
        ox, oy = self.map_origin[view]
        heat = heat[by:by + bh, bx:bx + bw]
        mask = lit[by:by + bh, bx:bx + bw].view(bool)
        roi = img[oy + by:oy + by + bh, ox + bx:ox + bx + bw]
        v = np.minimum(heat[mask], 1.0)[:, None]
        alpha = np.minimum(v * HEAT_ALPHA_GAIN, 1.0)
        base = self._hot_base[view][by:by + bh, bx:bx + bw][mask]
        # Hot pixels tend to white-yellow, cooler ones keep the region hue.
        color = np.minimum(base + (HOT - base) * (v * v), 255.0)
        roi[mask] = (roi[mask] * (1.0 - alpha) + color * alpha).astype(np.uint8)

    def _draw_regions(self, img, act):
        x, x1 = self.MAP_X, self.LEFT[2] - 10
        bx = x + 100
        bw = x1 - 34 - bx
        peak_all = max(1.0, float(act.region_peak.max()))
        for i in range(act.n_regions):
            y = self.regions_top + i * 12
            name = self.atlas.region_names[i]
            n = int(act.region_counts[i])
            _bar(img, bx, y + 1, bw, 7, n / peak_all, REGION_COLORS.get(name, MUTED),
                 peak=act.region_peak[i] / peak_all)
            _text(img, str(n), x1, y + 8, 0.33, TEXT if n else MUTED, align="right")

    def _draw_history(self, img, act):
        x, x1 = self.MAP_X, self.LEFT[2] - 10
        y = self.history_top
        hist = list(act.spike_history)
        top = max([20] + hist)
        _text(img, f"spikes/tick  last {HISTORY_TICKS}  max {top}", x, y - 6, 0.34, MUTED)
        ch = 36
        cv2.rectangle(img, (x, y), (x1, y + ch), INSET, -1)
        if not hist:
            _text(img, "no history", x + 4, y + 14, 0.32, MUTED)
        step = (x1 - x) / HISTORY_TICKS
        bar_w = max(1, int(step) - 1)
        offset = HISTORY_TICKS - len(hist)
        for i, n in enumerate(hist):
            if n <= 0:
                continue
            f = n / top
            bh = max(1, int(round(f * (ch - 2))))
            bx = int(x + (offset + i) * step)
            cv2.rectangle(img, (bx, y + ch - 1 - bh), (bx + bar_w - 1, y + ch - 1),
                          _mix(RASTER_LOW, YELLOW, f), -1)

        if act.spike_source == "recorded":
            note = "glow: recorded spikes of the simulated cells"
        elif act.spike_source == "sampled":
            note = "glow: DN/DNg02 recorded, LC4/LPLC2/drive sampled from rates"
        else:
            note = "glow: spikes of the simulated cells"
        _text(img, note, x, CANVAS_H - 18, 0.32, MUTED)

    def _draw_chips(self, img, act):
        chips = (
            ("ESCAPE", DNP01, act.brain_attached, act.escape_active()),
            ("STABILIZER", DNG02, act.with_dng02, act.stabilizer_active()),
            ("FEEDING", FOOD, act.food is not None, act.feeding_running()),
        )
        w = (self.RX1 - self.RX - 2 * 8) // 3
        x = self.RX
        for name, color, running, active in chips:
            box = (x, 16, x + w, 38)
            if active:
                cv2.rectangle(img, box[:2], box[2:], color, -1)
                _text(img, name, x + w // 2, 31, 0.4, PANEL, 1, align="center")
            else:
                cv2.rectangle(img, box[:2], box[2:], _mix(DIM, color, 0.6) if running else DIM, 1)
                _text(img, name if running else f"{name} off", x + w // 2, 31, 0.36,
                      MUTED if running else DIM, align="center")
            x += w + 8

    def _draw_motor(self, img, act):
        # Signed bars are drawn screen-left = turn left. The two circuits use
        # opposite signs for that (DNp06 yaw > 0 = left, DNg02 steer > 0 =
        # right - see flybrain_controller.py), so each row says which one it is
        # and the number is shown as a magnitude plus L/R rather than a sign.
        rows = (
            ("escape", "DNp01", act.escape, DNP01, None, ESCAPE_STATE_THRESHOLD),
            ("turn", "DNp06", act.yaw, DNP06, +1, None),
            ("forward", "DNp03+06", act.forward, DNP03, None, None),
            ("thrust", "DNg02", act.dng02.get("thrust", 0.0), DNG02, None, None),
            ("steer", "DNg02", act.dng02.get("steer", 0.0), DNG02, -1, None),
        )
        live = act.brain_attached
        rx, rx1 = self.RX, self.RX1
        bx, bw = rx + 148, rx1 - 60 - (rx + 148)
        for i, (name, source, value, color, left_sign, tick) in enumerate(rows):
            y = 66 + i * 20
            if name in ("thrust", "steer") and not act.with_dng02:
                value, color = 0.0, DIM
            if not live:
                value, color = 0.0, DIM
            _text(img, name, rx, y + 10, 0.38, TEXT if live else MUTED)
            _text(img, source, rx + 66, y + 10, 0.32, MUTED)
            if left_sign is None:
                _bar(img, bx, y, bw, 11, value, color, tick=tick)
                _text(img, f"{value:.2f}", rx1, y + 10, 0.36, TEXT, align="right")
            else:
                toward_left = value * left_sign > 0
                _bar(img, bx, y, bw, 11, -abs(value) if toward_left else abs(value), color, signed=True)
                _text(img, "L", bx - 6, y + 10, 0.3, MUTED, align="right")
                _text(img, "R", bx + bw + 4, y + 10, 0.3, MUTED)
                side = "" if abs(value) < 0.005 else (" L" if toward_left else " R")
                _text(img, f"{abs(value):.2f}{side}", rx1, y + 10, 0.36, TEXT, align="right")

    def _draw_populations(self, img, act):
        half = (len(POPULATIONS) + 1) // 2
        col_w = (self.RX1 - self.RX) // 2
        peak = act.pop_peak()
        for i, (name, color) in enumerate(POPULATIONS):
            cx = self.RX + (i // half) * col_w
            y = 190 + (i % half) * 15
            built = act.brain_attached and (act.with_dng02 or not name.startswith(("DNg02", "drive")))
            hz = act.pop_hz[i] if built else 0.0
            _text(img, name, cx, y + 9, 0.34, color if built else DIM)
            _text(img, f"{hz:.0f}" if built else "-", cx + 104, y + 9, 0.34,
                  TEXT if hz >= 0.5 else MUTED, align="right")
            _bar(img, cx + 110, y + 2, col_w - 122, 7, hz / POP_BAR_HZ, color,
                 peak=peak[i] / POP_BAR_HZ if built else None)

    def _draw_food(self, img, act):
        x0, y0, x1, y1 = self.FOOD_BOX
        running = act.feeding_running()
        cv2.rectangle(img, (x0, y0), (x1, y1), _mix(PANEL, FOOD, 0.12 if running else 0.0), -1)
        cv2.rectangle(img, (x0, y0), (x1, y1), FOOD if running else BORDER, 1)
        if act.food is None:
            _text(img, "FoodNeuron (feeding) - idle", (x0 + x1) // 2, y0 + 29, 0.36, MUTED, align="center")
            return
        food = act.food
        state_color = STATE_COLORS.get(food["state"], FOOD) if running else MUTED
        _text(img, "FEEDING", x0 + 8, y0 + 18, 0.4, FOOD if running else MUTED)
        _text(img, food["state"], x0 + 82, y0 + 18, 0.4, state_color)
        seen = food["label"] if food["visible"] else "no banana"
        _text(img, seen, x1 - 8, y0 + 18, 0.33, TEXT if food["visible"] else MUTED, align="right")
        _text(img, "hunger", x0 + 8, y0 + 40, 0.33, MUTED)
        _bar(img, x0 + 58, y0 + 31, 160, 10, food["hunger"] / 100.0, FOOD if running else DIM)
        _text(img, f"{food['hunger']:.0f}", x0 + 226, y0 + 40, 0.36, TEXT)

    def _draw_command(self, img, act):
        rx = self.RX
        glyph = (self.RX1 - 26, 410)
        if act.food_drives() and act.food["rc"] is not None:
            # Drone/tello_camera.py: food_orbit.py's RC command is what gets
            # sent; the brain only decides when to get scared.
            lr, fb, ud, yaw = act.food["rc"]
            _text(img, "sent to the Tello  (FoodNeuron RC, -100..100)", rx, 392, 0.33, MUTED)
            _text(img, f"LR {lr:+d}   FB {fb:+d}   UD {ud:+d}   YAW {yaw:+d}", rx, 412, 0.38, TEXT)
            if act.brain_attached:
                _text(img, "(brain's own dodge command not used here)", rx, 430, 0.32, MUTED)
            # Tello RC: lr > 0 = right, fb > 0 = forward, yaw > 0 = clockwise.
            self._draw_glyph(img, *glyph, right=lr * 0.4, forward=fb * 0.4, turn_left=-yaw / 50.0)
            return

        _text(img, "before main.py's test-mode hover / SafetyLayer", rx, 430, 0.32, MUTED)
        cmd = act.cmd
        if cmd is None:
            _text(img, "-", rx, 404, 0.4, MUTED)
            return
        yaw = cmd["yaw_rate"]
        _text(img, f"fwd {cmd['forward_speed']:+.2f} m/s   strafe {cmd['strafe_speed']:+.2f} m/s",
              rx, 394, 0.38, TEXT)
        _text(img, f"yaw {yaw:+.2f} rad/s", rx, 412, 0.38, TEXT)
        # This project's convention: strafe > 0 = left, yaw_rate > 0 = turn left.
        self._draw_glyph(img, *glyph, right=-cmd["strafe_speed"] * 7, forward=cmd["forward_speed"] * 7,
                         turn_left=yaw)

    @staticmethod
    def _draw_glyph(img, gx, gy, right, forward, turn_left):
        """Top-down drone: arrow = commanded velocity in pixels (up =
        forward), arc = commanded turn (roughly rad/s, positive = left)."""
        cv2.circle(img, (gx, gy), 16, BORDER, 1, cv2.LINE_AA)
        if math.hypot(right, forward) > 4:     # shorter just draws a blob
            cv2.arrowedLine(img, (gx, gy), (int(gx + right), int(gy - forward)), TEXT, 2,
                            cv2.LINE_AA, tipLength=0.3)
        if abs(turn_left) > 0.02:
            # 270 deg is straight up in image coordinates; decreasing sweeps
            # toward the left.
            sweep = int(min(150, abs(turn_left) * 150))
            end = 270 - sweep if turn_left > 0 else 270 + sweep
            cv2.ellipse(img, (gx, gy), (22, 22), 0, 270, end, DNP06, 2, cv2.LINE_AA)


# ---------------------------------------------------------------- window + hooks

class BrainView:
    """One window, fed by whichever pathways are running in this process."""

    def __init__(self, window=WINDOW_NAME, max_fps=DEFAULT_MAX_FPS, display=True):
        self.window = window
        self.display = display
        self.min_interval = 1.0 / max_fps if max_fps else 0.0
        self.circuits = load_circuits()
        self.atlas = load_atlas()
        self.constants = dict(DEFAULT_CONSTANTS)
        self.activity = BrainActivity(self.circuits, self.atlas, self.constants)
        self.diagram = BrainDiagram(self.circuits, self.atlas)
        self._last_shown = 0.0
        self._window_open = False
        self.last_frame = None

    def bind_brain(self, controller):
        """Pulls the network's real configuration out of the brain
        subprocess's handshake (see flybrain_controller.py's _FlyBrainProcess):
        its constants, and which root id each of its "spiked" indices is."""
        brain = controller._brain
        self.constants.update({k: v for k, v in getattr(brain, "constants", {}).items()
                               if k in DEFAULT_CONSTANTS})
        ids = (getattr(brain, "info", None) or {}).get("neuron_ids")
        if ids:
            self.activity.set_neuron_ids(ids)
        labels = getattr(brain, "dng02_labels", [])
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
