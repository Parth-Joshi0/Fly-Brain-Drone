"""
Dense (Farneback) optical flow between consecutive camera frames, split
into LEFT / CENTER / RIGHT thirds. These three numbers are the "eyes" of
the reflex controller - and later, the FlyBrain network.
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
# vertical FOV) by measuring actual Farneback output while yawing in an
# empty scene - see the derivation note in derotate_flow().
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
    """
    corrected = flow.copy()
    corrected[..., 0] -= _PIXELS_PER_RADIAN * yaw_rate * dt
    return corrected


def region_flow_strengths(flow, ground_fraction=0.55):
    """Average flow magnitude in the LEFT / CENTER / RIGHT thirds of the
    frame, restricted to the upper `ground_fraction` of the image.

    Why crop: with a level-ish camera, the ground fills the bottom of the
    frame and is always the closest textured surface in view - it produces
    huge flow purely from proximity ("ventral flow"), which drowns out the
    actual obstacle signal if you don't exclude it.
    """
    h, w = flow.shape[:2]
    band = flow[: int(h * ground_fraction), :]

    third = band.shape[1] // 3
    left = band[:, :third]
    center = band[:, third: 2 * third]
    right = band[:, 2 * third:]

    left_flow = float(np.mean(np.linalg.norm(left, axis=2)))
    center_flow = float(np.mean(np.linalg.norm(center, axis=2)))
    right_flow = float(np.mean(np.linalg.norm(right, axis=2)))

    return left_flow, center_flow, right_flow


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
