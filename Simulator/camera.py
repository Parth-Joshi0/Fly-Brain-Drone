"""
Forward-facing virtual camera mounted on the drone. Returns plain BGR
NumPy frames, same as you'd get from cv2.VideoCapture / a real FPV feed.
"""

import pybullet as p
import numpy as np
import cv2


class DroneCamera:
    def __init__(self, width=320, height=240, fov=75, near=0.05, far=50):
        """
        width/height: ~320x240 matches a small FPV drone's typical feed.
        fov: 75 degrees is in the 60-90 range typical of FPV cameras.
        """
        self.width = width
        self.height = height
        self.fov = fov
        self.near = near
        self.far = far
        self.aspect = width / height

        self.projection_matrix = p.computeProjectionMatrixFOV(
            fov=fov, aspect=self.aspect, nearVal=near, farVal=far
        )

    def capture(self, position, orientation_quat):
        """position/orientation_quat: the drone's current pose. Returns a
        (height, width, 3) BGR uint8 frame, forward-facing along the
        drone's own +X axis."""
        rotation_matrix = p.getMatrixFromQuaternion(orientation_quat)

        forward = [rotation_matrix[0], rotation_matrix[3], rotation_matrix[6]]
        up = [rotation_matrix[2], rotation_matrix[5], rotation_matrix[8]]

        eye = position
        target = [position[i] + forward[i] * 5 for i in range(3)]

        view_matrix = p.computeViewMatrix(
            cameraEyePosition=eye,
            cameraTargetPosition=target,
            cameraUpVector=up,
        )

        _, _, rgb, _, _ = p.getCameraImage(
            width=self.width,
            height=self.height,
            viewMatrix=view_matrix,
            projectionMatrix=self.projection_matrix,
            lightDirection=[0, 0, 3],
            lightColor=[1, 1, 1],
            lightDistance=3,
            shadow=0,
            # In GUI mode pybullet defaults to ER_BULLET_HARDWARE_OPENGL,
            # which renders this frame on the GUI's own OpenGL thread - the
            # one place the 30Hz control loop reaches across threads every
            # cycle, and the unreliable path on macOS. Ask for the software
            # renderer explicitly: it runs in this thread, costs ~5ms at
            # 320x240 (well inside the 33ms decision budget), and makes the
            # GUI run and evaluation/run_trials.py's DIRECT run see exactly
            # the same pixels instead of two different renderers.
            renderer=p.ER_TINY_RENDERER,
        )

        frame = np.array(rgb, dtype=np.uint8).reshape(self.height, self.width, 4)
        return cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
