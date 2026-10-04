"""
Live picture of the fly brain at work: the whole brain drawn as a cloud of its
real neurons (frontal view, baked by build_brain_atlas.py), with the 418 cells
fly_brain_controller.py simulates glowing where they actually sit each time
they spike, fading over ~0.3 s. Below it: which neurons are firing - for each
simulated cell type and side, how many of its cells spiked this tick (with a
peak-hold tick) - and a key to the cell types and colours. Modelled on the
brain view HUD in
blendi-remade/fly-brain-minecraft, cut down to stay cheap next to the
simulator.

The spikes are real: every spike of the last 20 ms window comes back in the
brain's step() result as "spiked" (local indices), and the ready handshake's
neuron_ids says which root id each index is.

attach() wraps the controller's _brain.request and decide() and passes their
arguments and results through unchanged. The window never calls
cv2.waitKey() itself (that would swallow the host script's keys); it repaints
on the host loop's own waitKey(1).
"""

import math
import time
from pathlib import Path

import cv2
import numpy as np

ATLAS_PATH = Path(__file__).resolve().parent / "brain_atlas.npz"
WINDOW_NAME = "Fly Brain Activity"
DEFAULT_MAX_FPS = 10.0
HEADER_H = 20
HEAT_TAU_S = 0.30
MARKER = np.array((102, 224, 255), dtype=float)   # BGR, same as the baked markers
HOT = np.array((200, 250, 255), dtype=float)      # what a fresh spike glows
PEAK_DECAY = 0.97                                 # per brain step, the bars' peak-hold tick
ROW_H = 13
TEXT, MUTED, BAR_BG = (232, 232, 232), (156, 147, 138), (45, 38, 32)
FONT = cv2.FONT_HERSHEY_SIMPLEX
# (name, region it sits in - for its colour, what it does)
KEY = (
    ("LC4 / LPLC2", "optic lobe L", "looming detectors, both eyes"),
    ("DNp01", "descending", "Giant Fiber - escape"),
    ("DNp03 / DNp06", "descending", "brake / turn away from the loom"),
    ("DNg02", "descending", "wingbeat amplitude - thrust, steer"),
    ("drive pool", "central brain", "central inputs to DNg02"),
)


def _text(img, text, x, y, color=TEXT, scale=0.36, align_right=False):
    if align_right:
        x -= cv2.getTextSize(text, FONT, scale, 1)[0][0]
    cv2.putText(img, text, (int(x), int(y)), FONT, scale, color, 1, cv2.LINE_AA)


class BrainView:
    def __init__(self, window=WINDOW_NAME, max_fps=DEFAULT_MAX_FPS, display=True):
        self.window, self.display = window, display
        self.min_interval = 1.0 / max_fps if max_fps else 0.0
        with np.load(ATLAS_PATH) as a:
            bg = a["background"]
            self.row_of = {int(rid): i for i, rid in enumerate(a["circuit_ids"])}
            self.points = [(int(x), int(y) + HEADER_H) for x, y in a["circuit_px"]]
            region = a["circuit_region"]
            self.cell_type = a["circuit_type"].astype(np.intp)    # type * 2 + (0 L / 1 R)
            self.type_names = [str(n) for n in a["type_names"]]
            names = [str(n) for n in a["region_names"]]
            colors = [tuple(int(c) for c in rgb) for rgb in a["region_colors"]]
        n_groups = 2 * len(self.type_names)
        self.group_size = np.maximum(np.bincount(self.cell_type, minlength=n_groups), 1)
        # Each type + side drawn in the colour of the region its cells sit in.
        self.group_color = [colors[region[np.flatnonzero(self.cell_type == g)[0]]] for g in range(n_groups)]
        self.counts = np.zeros(n_groups)   # cells of each type + side that spiked this tick
        self.peak = np.zeros(n_groups)
        self.background = self._draw_static(bg, names, colors)
        self.heat = np.zeros(len(self.points))
        self.rows = np.zeros(0, dtype=np.intp)   # brain-local index -> atlas row (-1 = unknown)
        self.state, self.spikes = "-", 0
        self._last_decay = self._last_shown = time.monotonic()
        self._window_open = False
        self.last_frame = None

    def _draw_static(self, bg, names, colors):
        """Header strip, map, neuron list labels and legend - drawn once;
        show() only adds the glow, the bar fills and the numbers."""
        w = bg.shape[1]
        self.bars_y = HEADER_H + bg.shape[0] + 38
        key_y = self.bars_y + len(self.type_names) * ROW_H + 22
        img = np.full((key_y + (len(KEY) + 2) * ROW_H + 4, w, 3), 11, np.uint8)
        img[HEADER_H:HEADER_H + bg.shape[0]] = bg
        # Two columns, left-side cells then right-side cells: bar, then "n/N".
        self.col_x, self.bar_w = (90, 268), 100
        _text(img, "NEURONS FIRING THIS TICK  (cells spiking / cells simulated)", 6, self.bars_y - 22, MUTED)
        for x, side in zip(self.col_x, ("left", "right")):
            _text(img, side, x, self.bars_y - 6, MUTED)
        for t, name in enumerate(self.type_names):
            y = self.bars_y + t * ROW_H
            _text(img, name, 6, y + 8, self.group_color[2 * t])
            for x in self.col_x:
                cv2.rectangle(img, (x, y + 1), (x + self.bar_w, y + 8), BAR_BG, -1)
        _text(img, "KEY", 6, key_y - 8, MUTED)
        for i, (name, region, what) in enumerate(KEY):
            y = key_y + i * ROW_H + 8
            _text(img, name, 6, y, colors[names.index(region)])
            _text(img, what, 110, y, MUTED)
        y = key_y + len(KEY) * ROW_H + 8
        cv2.circle(img, (9, y - 3), 2, tuple(int(c) for c in MARKER), -1)
        _text(img, "simulated cell", 16, y, MUTED)
        cv2.circle(img, (119, y - 3), 2, tuple(int(c) for c in HOT), -1)
        _text(img, "spiking now  (fades over 0.3 s)", 126, y, MUTED)
        x = 6
        for r in (names.index("sensory"), names.index("motor"), names.index("other")):
            _text(img, names[r], x, y + ROW_H, colors[r])
            x += 70
        _text(img, "also shown in the cloud", x, y + ROW_H, MUTED)
        return img

    def bind(self, controller):
        ids = (getattr(controller._brain, "info", None) or {}).get("neuron_ids", [])
        self.rows = np.array([self.row_of.get(int(r), -1) for r in ids], dtype=np.intp)

    def record(self, payload, result):
        if payload.get("reset"):
            self.heat[:] = 0.0
            self.counts[:] = 0.0
            self.peak[:] = 0.0
            self.spikes = 0
            return
        spiked = np.asarray(result.get("spiked", []), dtype=np.intp)
        spiked = spiked[(spiked >= 0) & (spiked < len(self.rows))]
        rows = self.rows[spiked]
        self._decay()
        rows = rows[rows >= 0]
        self.heat[rows] = 1.0
        self.spikes = len(spiked)
        # Cells that fired, not spikes: a cell that fired twice counts once.
        self.counts = np.bincount(self.cell_type[np.unique(rows)], minlength=len(self.counts))
        self.peak = np.maximum(self.peak * PEAK_DECAY, self.counts)

    def _decay(self):
        now = time.monotonic()
        self.heat *= math.exp(-min(0.25, now - self._last_decay) / HEAT_TAU_S)
        self._last_decay = now

    def show(self, force=False):
        now = time.monotonic()
        if not force and now - self._last_shown < self.min_interval:
            return
        self._last_shown = now
        self._decay()
        img = self.background.copy()
        for i in np.flatnonzero(self.heat > 0.05):
            v = min(1.0, self.heat[i])
            cv2.circle(img, self.points[i], 2, tuple(int(c) for c in MARKER + (HOT - MARKER) * v), -1)
        _text(img, f"{self.state}   {self.spikes} spk/tick", 6, 14, scale=0.4)
        # Bars are the fraction of that type's cells firing, so a 1-cell DN
        # and the 89 LPLC2 cells read on the same scale.
        for g in range(len(self.counts)):
            x, y = self.col_x[g % 2], self.bars_y + (g // 2) * ROW_H
            n, size = int(self.counts[g]), self.group_size[g]
            if n:
                cv2.rectangle(img, (x, y + 1), (x + int(n / size * self.bar_w), y + 8), self.group_color[g], -1)
            if self.peak[g] > 0:
                px = x + int(self.peak[g] / size * (self.bar_w - 1))
                cv2.line(img, (px, y + 1), (px, y + 8), (200, 200, 200), 1)
            _text(img, f"{n}/{size}", x + self.bar_w + 46, y + 8, TEXT if n else MUTED, align_right=True)
        self.last_frame = img
        if self.display:
            if not self._window_open:
                cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(self.window, int(img.shape[1] * 1.5), int(img.shape[0] * 1.5))
                self._window_open = True
            cv2.imshow(self.window, img)


def attach(controller, *, window=WINDOW_NAME, max_fps=DEFAULT_MAX_FPS, display=True):
    """Shows `controller`'s (a NeuralPathways.flybrain_controller
    FlyBrainController) spikes live. Returns the BrainView."""
    view = BrainView(window, max_fps, display)
    view.bind(controller)
    inner_request, inner_decide = controller._brain.request, controller.decide

    def request(payload):
        result = inner_request(payload)
        view.record(payload, result)
        return result

    def decide(flow, state=None):
        cmd = inner_decide(flow, state)
        view.state = getattr(controller, "state", "-")
        view.show()
        return cmd

    controller._brain.request, controller.decide = request, decide
    return view
