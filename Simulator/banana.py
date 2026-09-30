"""
A banana on a stand, for the banana-seek behaviour (NeuralPathways/
FoodNeuron/food_orbit.py) to find in the simulator.

The sim runs the same BananaModel/ detector the real Tello uses, straight
off the sim camera frame - so the banana has to be something stock COCO
YOLOv8n will box. A modelled 3D banana was not: PyBullet's tiny renderer
shades flat, and a curved, tapered, photo-textured tube still read as a pale
boat to YOLO (det_conf 0.00 at every distance). What does work is the real
photo in BananaModel/testImage.jpg on a card facing the drone, its countertop
background painted out white.

Measured with the default 320x240 DroneCamera: detected (det_conf > 0.2)
from 2.5 m in, ~0.9 from 1.5 m, nothing from 3 m. So place it within ~2.5 m
of where the drone will be looking. Two choices behind that range:

  * The card is big - BANANA_WIDTH is about five real bananas. The sim
    camera is 320x240 against the Tello's 960x720; at real size the banana
    would be a handful of pixels wide until the drone was nearly on it.
  * The photo is shrunk to TEXTURE_ROWS before use. The tiny renderer
    samples textures nearest-neighbour with no mipmaps, so at a distance the
    full-resolution photo's brown speckles alias into noise, and YOLO lost
    the banana from 2.5 m. Pre-shrinking it is the mipmap the renderer
    doesn't do.

Rendering at a higher resolution did not extend the range either.
"""

import os
import tempfile

import cv2
import numpy as np
import pybullet as p

BANANA_PHOTO = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir,
                            "BananaModel", "testImage.jpg")

BANANA_WIDTH = 1.2         # m, the card's long side
BANANA_HEIGHT = 1.45       # m, card centre - about NORMAL_ALTITUDE, so it
                           # sits mid-frame without the drone climbing
STAND_HALF_WIDTH = 0.06    # m
TEXTURE_ROWS = 48          # see module docstring

_asset_dir = None


def _write_texture(path):
    """Returns the card's height/width ratio."""
    photo = cv2.imread(BANANA_PHOTO)
    photo = cv2.rotate(photo, cv2.ROTATE_90_CLOCKWISE)    # long axis horizontal
    # The peel is saturated and bright; the countertop behind it is neither.
    hsv = cv2.cvtColor(photo, cv2.COLOR_BGR2HSV)
    peel = ((hsv[..., 1] > 80) & (hsv[..., 2] > 70)).astype(np.uint8)
    peel = cv2.morphologyEx(peel, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    peel = cv2.dilate(peel, np.ones((5, 5), np.uint8))
    photo[peel == 0] = (235, 235, 235)
    rows, cols = photo.shape[:2]
    small = cv2.resize(photo, (round(TEXTURE_ROWS * cols / rows), TEXTURE_ROWS),
                       interpolation=cv2.INTER_AREA)
    cv2.imwrite(path, small)
    return rows / cols


def _write_card(path, aspect):
    """A flat card in the local y-z plane, facing -x (and +x - both windings,
    so it isn't culled from behind). y is the texture's u, z its v."""
    half_w, half_h = BANANA_WIDTH / 2, BANANA_WIDTH * aspect / 2
    with open(path, "w") as f:
        for y, z in ((-half_w, -half_h), (half_w, -half_h), (half_w, half_h), (-half_w, half_h)):
            f.write(f"v 0 {y:.4f} {z:.4f}\n")
        # A camera looking along +x has +y on its left, so u runs +y -> -y
        # to keep the photo unmirrored from the front.
        for u, v in ((1, 0), (0, 0), (0, 1), (1, 1)):
            f.write(f"vt {u} {v}\n")
        f.write("f 1/1 2/2 3/3\nf 1/1 3/3 4/4\nf 1/1 3/3 2/2\nf 1/1 4/4 3/3\n")
    return half_h


def _assets():
    global _asset_dir
    if _asset_dir is None:
        _asset_dir = tempfile.mkdtemp(prefix="sim_banana_")
    card, texture = os.path.join(_asset_dir, "banana.obj"), os.path.join(_asset_dir, "banana.png")
    half_h = _write_card(card, _write_texture(texture))
    return card, texture, half_h


def add_banana(position, yaw_degrees=180.0):
    """Places a stand at (x, y) = position[:2] with the banana card on top.
    yaw_degrees is the direction the card faces, world frame - the default
    faces -x, i.e. back toward a drone that starts at the origin looking
    along +x. Returns (banana_id, stand_id); the stand has collision (so
    Simulator/pybullet_drone.py's distance safety net sees it), the card
    doesn't - it only needs to be seen."""
    card, texture, half_h = _assets()
    x, y = position[0], position[1]

    stand_top = BANANA_HEIGHT - half_h
    half = [STAND_HALF_WIDTH, STAND_HALF_WIDTH, stand_top / 2]
    stand = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=half),
        baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=half,
                                                 rgbaColor=[0.45, 0.3, 0.2, 1]),
        basePosition=[x, y, stand_top / 2],
    )

    # The card's front is its local -x; turn that to face yaw_degrees.
    facing = np.radians(yaw_degrees) + np.pi
    banana = p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=p.createVisualShape(p.GEOM_MESH, fileName=card),
        basePosition=[x, y, BANANA_HEIGHT],
        baseOrientation=p.getQuaternionFromEuler([0, 0, facing]),
    )
    p.changeVisualShape(banana, -1, textureUniqueId=p.loadTexture(texture), rgbaColor=[1, 1, 1, 1])
    return banana, stand
