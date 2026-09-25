"""
Concrete SimulatorInterface backed by PyBullet: owns the GUI window, the
test course, keyboard input, and the debug 3D view. Everything in this
file is what would need to change (or disappear) to fly the real drone -
main.py itself stays the same either way.
"""

import math
import time

import pybullet as p
import pybullet_data

from simulation.environment import build_environment
from interfaces.drone_interface import DroneInterface  # noqa: F401 (documents create_drone's return type)
from interfaces.pybullet_drone import PyBulletDrone, PHYSICS_DT, GRAVITY
from interfaces.simulator_interface import SimulatorInterface

# --- "Easy to watch" debug camera - follows the drone at a fixed angle/
# distance rather than a tight close-up, so nearby obstacles and the
# direction of travel stay visible. ---
DEBUG_CAMERA_DISTANCE = 7.0
DEBUG_CAMERA_YAW = 50
DEBUG_CAMERA_PITCH = -35
HEADING_LINE_LENGTH = 1.5  # m - length of the drawn forward-direction line


class PyBulletSimulator(SimulatorInterface):

    def __init__(self):
        self._env = None
        self._heading_line_id = None

    def connect(self):
        p.connect(p.GUI)
        time.sleep(0.5)  # let the renderer finish initializing before loading meshes
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetSimulation()
        p.setGravity(0, 0, -GRAVITY)

        self._env = build_environment()
        return {"bounds": self._env["bounds"], "goal_x": self._env["goal_x"]}

    def create_drone(self, start_pos):
        return PyBulletDrone(
            start_pos=start_pos,
            ground_id=self._env["plane"],
            obstacle_ids=set(self._env["obstacles"]),
        )

    def poll_input(self):
        keys = p.getKeyboardEvents()

        def down(code):
            return code in keys and keys[code] & p.KEY_IS_DOWN

        def pressed(code):
            return code in keys and keys[code] & p.KEY_WAS_TRIGGERED

        return {
            "forward": down(p.B3G_UP_ARROW),
            "backward": down(p.B3G_DOWN_ARROW),
            "strafe_left": down(p.B3G_LEFT_ARROW),
            "strafe_right": down(p.B3G_RIGHT_ARROW),
            "up": down(ord('w')),
            "down": down(ord('s')),
            "yaw_left": down(ord('q')),
            "yaw_right": down(ord('e')),
            "hover": down(ord(' ')),
            "land_pressed": pressed(ord('l')),
            "reset_pressed": pressed(ord('r')),
            "mode_toggle_pressed": pressed(ord('m')),
        }

    def update_debug_view(self, position, yaw_degrees):
        p.resetDebugVisualizerCamera(
            cameraDistance=DEBUG_CAMERA_DISTANCE,
            cameraYaw=DEBUG_CAMERA_YAW,
            cameraPitch=DEBUG_CAMERA_PITCH,
            cameraTargetPosition=position,
        )

        yaw = math.radians(yaw_degrees)
        end = [
            position[0] + HEADING_LINE_LENGTH * math.cos(yaw),
            position[1] + HEADING_LINE_LENGTH * math.sin(yaw),
            position[2],
        ]
        if self._heading_line_id is None:
            self._heading_line_id = p.addUserDebugLine(position, end, lineColorRGB=[1, 1, 0], lineWidth=3)
        else:
            self._heading_line_id = p.addUserDebugLine(
                position, end, lineColorRGB=[1, 1, 0], lineWidth=3,
                replaceItemUniqueId=self._heading_line_id,
            )

    @property
    def physics_dt(self):
        return PHYSICS_DT

    def tick(self):
        time.sleep(PHYSICS_DT)
