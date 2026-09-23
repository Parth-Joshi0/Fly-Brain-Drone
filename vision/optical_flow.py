"""
Dense (Farneback) optical flow between consecutive camera frames, split
into a 3x3 grid (TOP/CENTER/BOTTOM x LEFT/CENTER/RIGHT). These are the
"eyes" of the autonomous navigation controller and safety layer - and
later, the FlyBrain network.
"""

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


def derotate_flow(flow, yaw_rate, dt):
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
    reflex_controller.py, which are kept moderate partly for this reason,
    and controllers/safety_layer.py's already_avoiding flag, which stops
    that residual from triggering a second, conflicting turn decision on
    top of a turn already in progress.
    """
    corrected = flow.copy()
    corrected[..., 0] -= _PIXELS_PER_RADIAN * yaw_rate * dt
    return corrected


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
