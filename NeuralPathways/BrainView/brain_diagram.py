"""
Live picture of the fly brain at work: the whole brain drawn as a cloud of its
real neurons (frontal view, baked by build_brain_atlas.py), with the 418 cells
fly_brain_controller.py simulates glowing where they actually sit each time
they spike, fading over ~0.3 s. Modelled on the brain view HUD in
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


class BrainView:
    def __init__(self, window=WINDOW_NAME, max_fps=DEFAULT_MAX_FPS, display=True):
        self.window, self.display = window, display
        self.min_interval = 1.0 / max_fps if max_fps else 0.0
        with np.load(ATLAS_PATH) as a:
            bg = a["background"]
            self.row_of = {int(rid): i for i, rid in enumerate(a["circuit_ids"])}
            self.points = [(int(x), int(y) + HEADER_H) for x, y in a["circuit_px"]]
        self.background = np.vstack([np.full((HEADER_H, bg.shape[1], 3), 11, np.uint8), bg])
        self.heat = np.zeros(len(self.points))
        self.rows = np.zeros(0, dtype=np.intp)   # brain-local index -> atlas row (-1 = unknown)
        self.state, self.spikes = "-", 0
        self._last_decay = self._last_shown = time.monotonic()
        self._window_open = False
        self.last_frame = None

    def bind(self, controller):
        ids = (getattr(controller._brain, "info", None) or {}).get("neuron_ids", [])
        self.rows = np.array([self.row_of.get(int(r), -1) for r in ids], dtype=np.intp)

    def record(self, payload, result):
        if payload.get("reset"):
            self.heat[:] = 0.0
            self.spikes = 0
            return
        spiked = np.asarray(result.get("spiked", []), dtype=np.intp)
        spiked = spiked[(spiked >= 0) & (spiked < len(self.rows))]
        rows = self.rows[spiked]
        self._decay()
        self.heat[rows[rows >= 0]] = 1.0
        self.spikes = len(spiked)

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
        cv2.putText(img, f"{self.state}   {self.spikes} spk/tick", (6, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (232, 232, 232), 1, cv2.LINE_AA)
        self.last_frame = img
        if self.display:
            if not self._window_open:
                cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(self.window, img.shape[1] * 2, img.shape[0] * 2)
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
