"""
Dense (Farneback) optical flow between consecutive camera frames, split
into a 3x3 grid (TOP/CENTER/BOTTOM x LEFT/CENTER/RIGHT). These are the
"eyes" of the autonomous navigation controller and safety layer - and
later, the FlyBrain network.
"""

from collections import deque
import math

import cv2
import numpy as np


def compute_flow(prev_gray, gray):
    """Farneback flow, prev_gray -> gray. Both must be grayscale uint8."""
    return cv2.calcOpticalFlowFarneback(
        prev_gray, gray, None,
        pyr_scale=0.5, levels=2, winsize=13,
        iterations=1, poly_n=5, poly_sigma=1.2, flags=0,
    )


# Empirically calibrated for the default DroneCamera (320px wide, 75 deg
# vertical FOV) by measuring actual Farneback output while yawing.
_PIXELS_PER_RADIAN = 75.0


def derotate_flow(flow, yaw_rate, dt, pixels_per_radian=_PIXELS_PER_RADIAN):
    """Cancel the flow caused by the drone's own yaw rotation, leaving only
    flow caused by actually getting closer to something.

    Why this matters: turning the camera sweeps the whole scene across the
    frame, which looks exactly like "something's rushing past" even with
    nothing nearby - and a naive reflex controller reacts to that by
    turning more, which sweeps the scene even faster, and so on. Real
    flies cancel this out using their halteres (a gyroscope) to subtract
    expected self-motion from what their eyes report; this does the same
    thing with the simulator's own known yaw rate.

    Important limitation, found by measuring rather than assumed: this
    only cancels the *mean* horizontal shift. Per-pixel flow during a
    turn has real, substantial variance around that mean (a wide-FOV
    camera's rotational flow genuinely isn't uniform across the frame -
    it's a perspective-projection effect, not sensor noise, and a
    position-dependent correction model didn't reliably improve on the
    simple constant in testing here). That residual variance means any
    sustained yaw rate produces some baseline elevated flow reading even
    with nothing nearby - see AVOID_TURN_RATE and BOUNDARY_TURN_RATE in
    Simulator/reflex_controller.py, which are kept moderate partly for this reason,
    and safety_layer.py's already_avoiding flag, which stops
    that residual from triggering a second, conflicting turn decision on
    top of a turn already in progress.

    pixels_per_radian is exposed because the default was calibrated for the
    simulator's DroneCamera and is wrong for the Tello, and because how much
    that matters depends entirely on what the caller does with the result.
    Everything that existed before this argument feeds the output into
    grid_flow_strengths, which takes a magnitude - there a scale error is
    absorbed by the hand-tuned thresholds downstream. signed_hemifield_flow
    below uses the signed RESIDUAL instead, where a too-small correction
    leaves part of the drone's own commanded yaw in the signal with the same
    sign as a genuine drift, which is positive feedback in any loop built on
    it. So: calibrate it against real telemetry before closing a loop on this
    (Drone/Tests/tello_dng02_test.py --mode calibrate-derotation does that), and
    leave it alone otherwise.
    """
    corrected = flow.copy()
    corrected[..., 0] -= pixels_per_radian * yaw_rate * dt
    return corrected


def signed_hemifield_flow(flow, ground_fraction=0.8, side_margin=0.15):
    """Signed mean horizontal flow per hemifield, decomposed into the rotation
    and expansion components - the part grid_flow_strengths throws away when it
    takes a magnitude.

    Pass flow that has already been through derotate_flow(), so the drone's own
    COMMANDED yaw has been subtracted and only rotation it did not ask for
    survives. That residual is what an optomotor reflex should null; the
    commanded part is not an error.

    Why signed, and why this decomposition: magnitude cannot tell "the scene is
    sliding right" from "the scene is sliding left", so it cannot drive a
    steering response at all. And a per-hemifield value on its own cannot tell
    rotation from approach. The two separate cleanly, because they have
    different symmetry:

        rotating right  -> scene slides left  in BOTH hemifields (same sign)
        moving forward  -> scene slides out to EACH side         (opposite signs)

    so the common mode isolates rotation and the difference isolates translation,
    for the cost of two means over an array the caller already has. Without
    that, a drone flying at a wall reads the outward flow as a turn and yaws at
    everything it approaches.

    side_margin drops that fraction of each outer edge, where Farneback's
    border extrapolation is worst and where rotational flow is least uniform -
    the same non-uniformity derotate_flow's docstring describes as its main
    limitation.

    Returns:
        left, right   signed mean horizontal flow per hemifield, px/frame,
                      positive = image content moving RIGHT
        rotation      (left + right) / 2, the common mode: positive means the
                      scene is sliding right, i.e. the drone is rotating LEFT
        translation   (right - left) / 2, the differential: positive means the
                      scene is spreading outwards, i.e. moving forward.
                      Named translation, not expansion, to keep it clearly
                      apart from LoomingDetector's expansion - that one is a
                      flow DIVERGENCE in 1/s (~2/time-to-contact), this is a
                      horizontal difference in px/frame. Different quantity,
                      different units, different circuit.
    """
    h, w = flow.shape[:2]
    band = flow[: int(h * ground_fraction), :]
    margin = int(w * side_margin)
    half = w // 2

    u = band[..., 0]
    left = float(np.mean(u[:, margin:half]))
    right = float(np.mean(u[:, half:w - margin]))
    return {
        "left": left,
        "right": right,
        "rotation": (left + right) / 2.0,
        "translation": (right - left) / 2.0,
    }


def grid_flow_strengths(flow, ground_fraction=0.8):
    """Average flow magnitude over a 3x3 grid (top/center/bottom rows x
    left/center/right columns), restricted to the upper `ground_fraction`
    of the frame.

    Why crop at all (rather than a full 3x3 of the whole image): with a
    level-ish camera, the ground fills the very bottom of the frame and is
    always the closest textured surface in view - it produces huge flow
    purely from proximity ("ventral flow") regardless of whether there's
    a real obstacle below. Cropping only the bottom-most sliver (not
    almost half the frame, like the old 2D version did) keeps a
    meaningful "bottom" row for genuine low-obstacle detection while still
    excluding the worst of the ground-proximity noise.

    Returns a dict with the 9 raw grid cells (e.g. "top_left") plus 5
    aggregated directions used for navigation decisions:
        left   = mean of the left column   (top/center/bottom-left)
        right  = mean of the right column
        top    = mean of the top row
        bottom = mean of the bottom row
        center = the single center cell
    Aggregating a whole column/row (not just the center-row/center-column
    cell) makes the L/R/T/B signal more robust to an obstacle that's
    off-center diagonally, not just dead level with the camera.
    """
    h, w = flow.shape[:2]
    band = flow[: int(h * ground_fraction), :]
    bh, bw = band.shape[:2]

    row_bounds = [0, bh // 3, 2 * bh // 3, bh]
    col_bounds = [0, bw // 3, 2 * bw // 3, bw]
    row_names = ["top", "center", "bottom"]
    col_names = ["left", "center", "right"]

    grid = {}
    for ri, rname in enumerate(row_names):
        for ci, cname in enumerate(col_names):
            region = band[row_bounds[ri]:row_bounds[ri + 1], col_bounds[ci]:col_bounds[ci + 1]]
            grid[f"{rname}_{cname}"] = float(np.mean(np.linalg.norm(region, axis=2)))

    return {
        **grid,
        "left": (grid["top_left"] + grid["center_left"] + grid["bottom_left"]) / 3,
        "right": (grid["top_right"] + grid["center_right"] + grid["bottom_right"]) / 3,
        "top": (grid["top_left"] + grid["top_center"] + grid["top_right"]) / 3,
        "bottom": (grid["bottom_left"] + grid["bottom_center"] + grid["bottom_right"]) / 3,
        "center": grid["center_center"],
    }


# Camera frame axes (x right, y down, z forward) in terms of the drone body
# frame (x forward, y left, z up) - DroneCamera looks along body +X, up = +Z.
_BODY_TO_CAMERA = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], dtype=float)

# Smoother than compute_flow()'s settings: expansion is a spatial
# *derivative* of flow, so it needs a cleaner field than magnitude does.
_LOOMING_FARNEBACK = dict(pyr_scale=0.5, levels=3, winsize=21, iterations=3,
                          poly_n=7, poly_sigma=1.5, flags=0)


def _quat_to_matrix(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


class LoomingDetector:
    """Per-column (left/center/right) image expansion rate, in 1/s - the
    looming signal LC4/LPLC2 respond to, as opposed to grid_flow_strengths'
    plain flow magnitude.

    Why expansion instead of magnitude: magnitude can't tell "something is
    approaching" from "I'm moving", and it's near zero at the dead center of
    a head-on approach (the focus of expansion barely moves). Expansion
    (flow divergence) of an approaching surface is ~2 / time-to-contact
    whatever is doing the moving, and it's strongest exactly there.

    How:
    - Full rotation removal: the older frame is warped by the camera's
      known rotation between the two frames (homography K R K^-1, from the
      drone's orientation) before computing flow, so pitch/roll/yaw add
      nothing. derotate_flow() above only removes the mean yaw shift, and
      rotation on a wide-FOV camera also produces divergence toward the
      image edges (~3 x rotation rate) that a mean shift can't remove.
    - Two-frame baseline: at 320x240/30Hz an object 2m out expands by under
      a pixel per frame, below what Farneback resolves reliably.
    - Expansion per cell is a least-squares affine fit (du/dx + dv/dy) over
      a 40px cell rather than a per-pixel derivative, which spikes at
      occlusion edges; the column value is its most-expanding cell, since
      an object 2m away only fills about one cell.
    - A 3-sample median then EMA over time: single-frame Farneback glitches
      otherwise read as brief huge expansions.

    Calibrated in the sim against ground-truth time-to-contact: tracks
    ~2/TTC on a head-on approach from ~1.7s out; background while
    maneuvering (yaw, strafe, hard braking) stays mostly below ~2-3.5/s.
    """

    def __init__(self, width=320, height=240, fov=75, baseline=2, cell=40,
                 margin=16, smoothing=0.5, median_window=3):
        f = (height / 2) / math.tan(math.radians(fov) / 2)
        self._K = np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]])
        self._K_inv = np.linalg.inv(self._K)
        self._size = (width, height)
        self._cell = cell
        self._margin = margin
        self._smoothing = smoothing
        c = np.arange(cell) - (cell - 1) / 2
        self._offsets = c
        self._offset_var = (c ** 2).sum() * cell
        self._frames = deque(maxlen=baseline + 1)
        self._recent = deque(maxlen=median_window)
        self._smoothed = np.zeros(3)

    def reset(self):
        self._frames.clear()
        self._recent.clear()
        self._smoothed = np.zeros(3)

    def update(self, gray, orientation, dt):
        """gray: current grayscale frame. orientation: the drone's body
        quaternion (x, y, z, w) when it was captured. dt: seconds between
        update() calls. Returns {"left", "center", "right"} expansion rates."""
        self._frames.append((gray, orientation))
        if len(self._frames) < 2:
            return self._as_dict()

        old_gray, old_q = self._frames[0]
        frames_apart = len(self._frames) - 1
        R_old, R_new = _quat_to_matrix(old_q), _quat_to_matrix(orientation)
        C = _BODY_TO_CAMERA
        H = self._K @ C @ R_new.T @ R_old @ C.T @ self._K_inv
        old_aligned = cv2.warpPerspective(old_gray, H, self._size, flags=cv2.INTER_LINEAR,
                                          borderMode=cv2.BORDER_REPLICATE)
        flow = cv2.calcOpticalFlowFarneback(old_aligned, gray, None, **_LOOMING_FARNEBACK)

        expansion = self._cell_expansion(flow) / (frames_apart * dt)
        columns = np.array_split(np.arange(expansion.shape[1]), 3)
        self._recent.append([expansion[:, cols].max() for cols in columns])
        a = self._smoothing
        self._smoothed = a * np.median(np.array(self._recent), axis=0) + (1 - a) * self._smoothed
        return self._as_dict()

    def _cell_expansion(self, flow):
        m, cell = self._margin, self._cell
        region = flow[m:flow.shape[0] - m, m:flow.shape[1] - m]
        rows, cols = region.shape[0] // cell, region.shape[1] // cell
        cells = region[:rows * cell, :cols * cell].reshape(rows, cell, cols, cell, 2)
        c = self._offsets
        du_dx = (cells[..., 0] * c[None, None, None, :]).sum(axis=(1, 3)) / self._offset_var
        dv_dy = (cells[..., 1] * c[None, :, None, None]).sum(axis=(1, 3)) / self._offset_var
        return du_dx + dv_dy

    def _as_dict(self):
        left, center, right = self._smoothed
        return {"left": float(left), "center": float(center), "right": float(right)}


class FlowVisualizer:
    """Color-coded flow image for debugging: hue = direction, brightness =
    speed. Same HSV trick used by most optical-flow demos."""

    def __init__(self, width, height):
        self.hsv = np.zeros((height, width, 3), dtype=np.uint8)
        self.hsv[..., 1] = 255

    def render(self, flow):
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        self.hsv[..., 0] = angle * 180 / np.pi / 2
        self.hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        return cv2.cvtColor(self.hsv, cv2.COLOR_HSV2BGR)
