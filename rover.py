import pybullet as p
import pybullet_data
import time
import cv2
import numpy as np

p.connect(p.GUI)
time.sleep(0.5)  # let the GL/Metal renderer finish initializing before loading meshes

# Trim GUI chrome/shadows - cuts render overhead that was causing input lag
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)

p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.resetSimulation()

# Gravity keeps rover on ground
p.setGravity(0, 0, -10)

# Ground
p.loadURDF("plane.urdf")

# Rover
rover = p.loadURDF(
    "racecar/racecar.urdf",
    [0, 0, 0.2]
)

# Create a box obstacle
box_collision = p.createCollisionShape(
    p.GEOM_BOX,
    halfExtents=[0.5, 0.5, 0.5]
)

box_visual = p.createVisualShape(
    p.GEOM_BOX,
    halfExtents=[0.5, 0.5, 0.5]
)

p.createMultiBody(
    baseMass=0,                  # 0 = fixed in place
    baseCollisionShapeIndex=box_collision,
    baseVisualShapeIndex=box_visual,
    basePosition=[3, 0, 0.5]
)


def make_wall(position):
    collision = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=[5, 0.1, 0.5]
    )

    visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[5, 0.1, 0.5]
    )

    p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=collision,
        baseVisualShapeIndex=visual,
        basePosition=position
    )


make_wall([4, 2, 0.5])
make_wall([4, -2, 0.5])

# All four wheels are driven (this model is 4WD)
drive_wheels = [2, 3, 5, 7]
steering_joints = [4, 6]

# Autonomous forward speed - the reflex controller only decides steering
FORWARD_SPEED = 25

prev_gray = None

while True:

    # --- Camera → grayscale frame ---
    position, orientation = p.getBasePositionAndOrientation(rover)
    rotation_matrix = p.getMatrixFromQuaternion(orientation)

    # Rover's forward direction
    forward = [
        rotation_matrix[0],
        rotation_matrix[3],
        rotation_matrix[6]
    ]

    # Camera position slightly above rover
    camera_pos = [
        position[0],
        position[1],
        position[2] + 0.4
    ]

    # Point camera forward
    camera_target = [
        camera_pos[0] + forward[0] * 5,
        camera_pos[1] + forward[1] * 5,
        camera_pos[2] + forward[2] * 5
    ]

    view_matrix = p.computeViewMatrix(
        cameraEyePosition=camera_pos,
        cameraTargetPosition=camera_target,
        cameraUpVector=[0, 0, 1]
    )

    projection_matrix = p.computeProjectionMatrixFOV(
        fov=90,
        aspect=320 / 240,
        nearVal=0.1,
        farVal=100
    )

    width, height, rgb, depth, segmentation = p.getCameraImage(
        width=320,
        height=240,
        viewMatrix=view_matrix,
        projectionMatrix=projection_matrix
    )

    frame = np.array(rgb, dtype=np.uint8)
    frame = frame.reshape(height, width, 4)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # --- Optical flow → reflex controller ---
    steering = 0
    left_strength = 0
    right_strength = 0

    if prev_gray is not None:
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None,
            pyr_scale=0.5, levels=2, winsize=13,
            iterations=1, poly_n=5, poly_sigma=1.2, flags=0
        )

        left_flow = flow[:, :flow.shape[1] // 2]
        right_flow = flow[:, flow.shape[1] // 2:]

        left_strength = np.mean(np.linalg.norm(left_flow, axis=2))
        right_strength = np.mean(np.linalg.norm(right_flow, axis=2))

        # Turn away from the side with stronger flow (closer obstacle)
        if left_strength > right_strength:
            steering = -0.4  # obstacle closer on left -> turn right
        elif right_strength > left_strength:
            steering = 0.4   # obstacle closer on right -> turn left
        else:
            steering = 0

    prev_gray = gray

    # --- Reflex controller → wheels ---
    for wheel in drive_wheels:
        p.setJointMotorControl2(
            rover,
            wheel,
            p.VELOCITY_CONTROL,
            targetVelocity=FORWARD_SPEED,
            force=60
        )

    for joint in steering_joints:
        p.setJointMotorControl2(
            rover,
            joint,
            p.POSITION_CONTROL,
            targetPosition=steering
        )

    cv2.imshow("Rover Camera", frame)
    cv2.waitKey(1)

    p.stepSimulation()
    time.sleep(1 / 240)
