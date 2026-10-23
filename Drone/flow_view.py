"""
Live optical-flow window for the real Tello. Display only: it draws the
picture and never feeds the brain or the flight controls.

Uses the same Farneback flow and yaw cancelling (derotation) as the
simulator, from NeuralPathways/EscapeNeuron/optical_flow.py. The yaw
rate comes from the Tello's compass, flipped to the project's
convention the same way Drone/tello_drone.py does it.
"""

import math
import time

import cv2

from NeuralPathways.EscapeNeuron.optical_flow import compute_flow, derotate_flow, FlowVisualizer


# The width the flow was calibrated for (see optical_flow.py)
FLOW_WIDTH = 320

# Window name; fly_tello.py places this window on screen
FLOW_WINDOW = "Optical Flow"


class FlowView:

    def __init__(self):

        self.prev_gray = None
        self.prev_yaw = None
        self.prev_time = None
        self.viz = None

    def update(self, frame_bgr, yaw_deg):

        now = time.time()

        height, width = frame_bgr.shape[:2]

        small = cv2.resize(frame_bgr, (FLOW_WIDTH, int(height * FLOW_WIDTH / width)))

        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        if self.viz is None:
            self.viz = FlowVisualizer(gray.shape[1], gray.shape[0])

        if self.prev_gray is not None and yaw_deg is not None and self.prev_yaw is not None:

            dt = max(now - self.prev_time, 1e-3)

            # Tello reports clockwise yaw as positive; flip it to the
            # project's counter-clockwise-positive convention. Wrap the
            # difference so a jump across 180 degrees doesn't spike.
            d_yaw = -((yaw_deg - self.prev_yaw + 180) % 360 - 180)

            yaw_rate = math.radians(d_yaw) / dt

            flow = compute_flow(self.prev_gray, gray)

            flow = derotate_flow(flow, yaw_rate, dt)

            cv2.imshow(FLOW_WINDOW, self.viz.render(flow))

        self.prev_gray = gray
        self.prev_yaw = yaw_deg
        self.prev_time = now
