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

# High-contrast, textured surfaces - optical flow needs real gradients to
# track. A flat gray box on a flat gray floor gives Farneback almost
# nothing to latch onto, which is why flow strength was weak/noisy before.
floor_texture = p.loadTexture("floor_checker.png")
obstacle_texture = p.loadTexture("obstacle_stripes.png")

# Ground
plane = p.loadURDF("plane.urdf")
p.changeVisualShape(plane, -1, textureUniqueId=floor_texture, rgbaColor=[1, 1, 1, 1])

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

box = p.createMultiBody(
    baseMass=0,                  # 0 = fixed in place
    baseCollisionShapeIndex=box_collision,
    baseVisualShapeIndex=box_visual,
    basePosition=[3, 0, 0.5]
)
p.changeVisualShape(box, -1, textureUniqueId=obstacle_texture, rgbaColor=[1, 1, 1, 1])


def make_wall(position):
    collision = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=[5, 0.1, 0.5]
    )

    visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[5, 0.1, 0.5]
    )

    wall = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=collision,
        baseVisualShapeIndex=visual,
        basePosition=position
    )
    p.changeVisualShape(wall, -1, textureUniqueId=obstacle_texture, rgbaColor=[1, 1, 1, 1])


make_wall([4, 2, 0.5])
make_wall([4, -2, 0.5])

# All four wheels are driven (this model is 4WD)
drive_wheels = [2, 3, 5, 7]
steering_joints = [4, 6]

# Autonomous forward speed - the reflex controller only decides steering
FORWARD_SPEED = 15

prev_gray = None
steering = 0
smoothed_diff = 0
smoothed_total = 0

# HSV canvas for the optical flow visualization window
flow_hsv = np.zeros((240, 320, 3), dtype=np.uint8)
flow_hsv[..., 1] = 255

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
        projectionMatrix=projection_matrix,
        lightDirection=[0, 0, 3],
        lightColor=[1, 1, 1],
        lightDistance=3,
        shadow=0
    )

    frame = np.array(rgb, dtype=np.uint8)
    frame = frame.reshape(height, width, 4)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # --- Optical flow → reflex controller ---
    left_strength = 0
    right_strength = 0

    if prev_gray is not None:
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None,
            pyr_scale=0.5, levels=2, winsize=13,
            iterations=1, poly_n=5, poly_sigma=1.2, flags=0
        )

        # The camera looks level, so the ground fills roughly the bottom
        # half of the frame and is always the closest textured surface in
        # view - it produces huge flow purely from proximity ("ventral
        # flow"), which was getting misread as an obstacle. Only look at
        # the upper band of the frame, where actual obstacles/walls at
        # roughly the rover's own height show up, not the floor beneath it.
        obstacle_band = flow[:int(height * 0.55), :]

        left_flow = obstacle_band[:, :obstacle_band.shape[1] // 2]
        right_flow = obstacle_band[:, obstacle_band.shape[1] // 2:]

        left_strength = np.mean(np.linalg.norm(left_flow, axis=2))
        right_strength = np.mean(np.linalg.norm(right_flow, axis=2))

        # diff: which side has more flow (asymmetry -> turn direction)
        # total: how much flow overall (proximity/looming -> is anything
        #        actually close enough right now to react to at all)
        diff = left_strength - right_strength
        total = left_strength + right_strength

        smoothed_diff = 0.5 * smoothed_diff + 0.5 * diff
        smoothed_total = 0.5 * smoothed_total + 0.5 * total

        NEAR_THRESHOLD = 0.03  # below this, nothing meaningful is close
        DEADBAND = 0.015

        if smoothed_total < NEAR_THRESHOLD:
            # Open space - go straight. This is what keeps the rover from
            # drifting into a wall it was never actually reacting to.
            steering = 0
        elif abs(smoothed_diff) > DEADBAND:
            # Clear left/right imbalance - turn away from the stronger
            # side, harder the more lopsided the flow is (continuous,
            # not a fixed ±0.5 snap), so it responds fresh to whatever
            # is closest right now, including a second obstacle right
            # after clearing the first one.
            steering = float(np.clip(-8.0 * smoothed_diff, -0.5, 0.5))
        else:
            # Something's close and dead-center (no left/right signal to
            # go on) - force a turn, harder the closer it looms, so it
            # commits fast enough to clear it instead of holding straight.
            steering = float(np.clip(5.0 * smoothed_total, 0.2, 0.5))

        # Color-coded flow visualization (hue = direction, brightness = speed)
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        flow_hsv[..., 0] = angle * 180 / np.pi / 2
        flow_hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        flow_bgr = cv2.cvtColor(flow_hsv, cv2.COLOR_HSV2BGR)

        cv2.imshow("Optical Flow", flow_bgr)

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
            targetPosition=steering,
            force=100,
            positionGain=0.5,
            velocityGain=1.0,
            maxVelocity=15
        )

    cv2.imshow("Rover Camera", frame)
    cv2.waitKey(1)

    p.stepSimulation()
    time.sleep(1 / 240)
