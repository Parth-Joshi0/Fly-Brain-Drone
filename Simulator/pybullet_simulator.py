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

from Simulator.banana import add_banana
from Simulator.environment import build_environment, OBSTACLE_TEXTURE
from Drone.drone_interface import DroneInterface  # noqa: F401 (documents create_drone's return type)
from Simulator.pybullet_drone import PyBulletDrone, PHYSICS_DT, GRAVITY
from Simulator.simulator_interface import SimulatorInterface

# pybullet raises this for any command the physics server refuses or never
# answers - including every command after the GUI window is closed, which
# surfaces as an unhelpful mid-loop traceback like "GetBasePositionAndOrientation
# failed". Re-exported so main.py can shut down cleanly on it while still
# not importing pybullet itself.
SimulatorError = p.error

# --- "Easy to watch" debug camera - follows the drone at a fixed angle/
# distance rather than a tight close-up, so nearby obstacles and the
# direction of travel stay visible. ---
DEBUG_CAMERA_DISTANCE = 7.0
DEBUG_CAMERA_YAW = 50
DEBUG_CAMERA_PITCH = -35
HEADING_LINE_LENGTH = 1.5  # m - length of the drawn forward-direction line

# --- Debug tool: left-click anywhere in the sim window to spawn a block
# that flies in a straight line at the drone - an on-demand looming
# stimulus for testing the escape reflex without waiting to stumble into
# a real obstacle. Spawned ahead of wherever the drone is currently
# facing (so it's immediately in the camera's view), not at the actual
# clicked screen position - see the module note in the commit this
# shipped with for why (unprojecting a screen click into a 3D world
# point needs the debug camera's view/projection matrices, which is far
# more fragile to get right than just placing it along a heading we
# already know). ---
TEST_OBSTACLE_DISTANCE = 4.0    # m - spawned this far ahead of the drone
TEST_OBSTACLE_SPEED = 1.5       # m/s - straight-line closing speed
TEST_OBSTACLE_HALF_EXTENTS = [0.3, 0.3, 0.5]


class PyBulletSimulator(SimulatorInterface):

    def __init__(self, banana_position=None):
        """banana_position: (x, y) to put a banana on a stand at (see
        Simulator/banana.py), or None for none."""
        self._banana_position = banana_position
        self._env = None
        self._drone = None
        self._heading_line_id = None
        self._test_obstacle_texture = None
        self._test_obstacles = []  # in-flight test obstacles: see _spawn_test_obstacle

    def connect(self):
        p.connect(p.GUI)
        time.sleep(0.5)  # let the renderer finish initializing before loading meshes
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
        # Left-click is the spawn-a-test-obstacle button (see
        # update_test_obstacles). PyBullet's default mouse picking would also
        # grab whatever body is under the cursor from the GUI thread - a
        # click in the window crashed main.py with "GetBasePositionAndOrientation
        # failed" on the drone, which 120 programmatic spawns never did.
        p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetSimulation()
        p.setGravity(0, 0, -GRAVITY)

        self._env = build_environment()
        if self._banana_position is not None:
            _, stand = add_banana(self._banana_position)
            self._env["obstacles"].append(stand)
        self._test_obstacle_texture = p.loadTexture(OBSTACLE_TEXTURE)
        return {"bounds": self._env["bounds"], "goal_x": self._env["goal_x"]}

    def create_drone(self, start_pos):
        self._drone = PyBulletDrone(
            start_pos=start_pos,
            ground_id=self._env["plane"],
            obstacle_ids=set(self._env["obstacles"]),
        )
        return self._drone

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

    def is_connected(self):
        return p.isConnected()

    @property
    def physics_dt(self):
        return PHYSICS_DT

    def tick(self):
        time.sleep(PHYSICS_DT)

    def update_test_obstacles(self, drone_position, drone_yaw_degrees, dt):
        if self._left_click_triggered():
            self._spawn_test_obstacle(drone_position, drone_yaw_degrees)

        still_flying = []
        for obstacle in self._test_obstacles:
            obstacle["remaining"] -= TEST_OBSTACLE_SPEED * dt
            if obstacle["remaining"] <= 0:
                p.removeBody(obstacle["id"])
                self._drone.obstacle_ids.discard(obstacle["id"])
                continue
            obstacle["position"] = [
                obstacle["position"][i] + obstacle["direction"][i] * TEST_OBSTACLE_SPEED * dt
                for i in range(3)
            ]
            p.resetBasePositionAndOrientation(obstacle["id"], obstacle["position"], [0, 0, 0, 1])
            still_flying.append(obstacle)
        self._test_obstacles = still_flying

    _MOUSE_BUTTON_EVENT = 2  # pybullet's quickstart guide documents this value, but
                              # (unlike KEY_WAS_TRIGGERED etc.) doesn't expose it as
                              # a module attribute - no p.MOUSE_BUTTON_EVENT to use.

    def _left_click_triggered(self):
        for event_type, _x, _y, button_index, button_state in p.getMouseEvents():
            if (event_type == self._MOUSE_BUTTON_EVENT and button_index == 0
                    and button_state & p.KEY_WAS_TRIGGERED):
                return True
        return False

    def _spawn_test_obstacle(self, drone_position, drone_yaw_degrees):
        yaw = math.radians(drone_yaw_degrees)
        forward = (math.cos(yaw), math.sin(yaw), 0.0)
        spawn_position = [drone_position[i] + forward[i] * TEST_OBSTACLE_DISTANCE for i in range(3)]

        collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=TEST_OBSTACLE_HALF_EXTENTS)
        visual = p.createVisualShape(p.GEOM_BOX, halfExtents=TEST_OBSTACLE_HALF_EXTENTS)
        body_id = p.createMultiBody(
            baseMass=0, baseCollisionShapeIndex=collision, baseVisualShapeIndex=visual,
            basePosition=spawn_position,
        )
        p.changeVisualShape(body_id, -1, textureUniqueId=self._test_obstacle_texture, rgbaColor=[1, 1, 1, 1])

        self._test_obstacles.append({
            "id": body_id,
            "position": list(spawn_position),
            "direction": (-forward[0], -forward[1], -forward[2]),  # straight back at the drone
            "remaining": TEST_OBSTACLE_DISTANCE,
        })
        self._drone.obstacle_ids.add(body_id)
        print(
            f"[test obstacle] spawned {TEST_OBSTACLE_DISTANCE:.1f}m ahead, "
            f"closing at {TEST_OBSTACLE_SPEED:.1f}m/s",
            flush=True,
        )
