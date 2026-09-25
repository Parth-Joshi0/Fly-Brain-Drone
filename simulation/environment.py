"""
Builds the test course: a textured floor plus a sequence of obstacles
(box, narrow passage, pillar) the drone has to fly through to reach a goal
line. All obstacle surfaces use a high-contrast striped texture on purpose -
optical flow needs real pixel-level gradients to track, and a flat color
gives it almost nothing to work with.
"""

import os
import pybullet as p

_ASSET_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FLOOR_TEXTURE = os.path.join(_ASSET_DIR, "floor_checker.png")
OBSTACLE_TEXTURE = os.path.join(_ASSET_DIR, "obstacle_stripes.png")

GOAL_X = 16.0  # how far down the course counts as "completed"
COURSE_MARGIN = 3.0  # m added around the obstacle course for the soft flight
                      # boundary - parameter #4 the caller asked about: this is
                      # what controls how far past the obstacles the drone is
                      # allowed to wander before BOUNDARY_RETURN kicks in


def _add_box(half_extents, position, texture_id):
    collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
    visual = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents)
    body = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=collision,
        baseVisualShapeIndex=visual,
        basePosition=position,
    )
    p.changeVisualShape(body, -1, textureUniqueId=texture_id, rgbaColor=[1, 1, 1, 1])
    return body


def build_environment():
    """Returns {"plane": id, "obstacles": [ids], "goal_x": float}."""

    floor_texture = p.loadTexture(FLOOR_TEXTURE)
    obstacle_texture = p.loadTexture(OBSTACLE_TEXTURE)

    plane = p.loadURDF("plane.urdf")
    p.changeVisualShape(plane, -1, textureUniqueId=floor_texture, rgbaColor=[1, 1, 1, 1])

    obstacles = []

    # Perimeter side walls (contain the whole course, y = +3 / -3)
    obstacles.append(_add_box([GOAL_X / 2 + 2, 0.1, 1.0], [GOAL_X / 2, 3, 1.0], obstacle_texture))
    obstacles.append(_add_box([GOAL_X / 2 + 2, 0.1, 1.0], [GOAL_X / 2, -3, 1.0], obstacle_texture))

    # Open space: nothing until x ~ 4

    # 1. A single box obstacle to dodge (tall enough to poke above the
    # drone's ~1.2m hover altitude - otherwise it would just fly over it)
    obstacles.append(_add_box([0.4, 0.4, 0.8], [4, 0, 0.8], obstacle_texture))

    # 2. A narrow passage (a "doorway" made of two inward walls with a gap)
    gap_half_width = 0.8  # 1.6 m wide opening - tight but flyable
    wall_half_y = (3 - gap_half_width) / 2
    wall_center_y = gap_half_width + wall_half_y
    obstacles.append(_add_box([0.1, wall_half_y, 1.0], [8, wall_center_y, 1.0], obstacle_texture))
    obstacles.append(_add_box([0.1, wall_half_y, 1.0], [8, -wall_center_y, 1.0], obstacle_texture))

    # 3. A pillar
    obstacles.append(_add_box([0.25, 0.25, 1.0], [11.5, 0.8, 1.0], obstacle_texture))

    # 4. Open space again, then the goal line (visual only, no collision)
    goal_marker = p.createVisualShape(
        p.GEOM_BOX, halfExtents=[0.02, 3, 1.0], rgbaColor=[0.1, 1.0, 0.1, 0.5]
    )
    p.createMultiBody(baseMass=0, baseVisualShapeIndex=goal_marker,
                       basePosition=[GOAL_X, 0, 1.0])

    # Soft flight-area boundary for controllers/reflex_controller.py's
    # BOUNDARY_RETURN state. X range covers the whole obstacle course plus
    # COURSE_MARGIN on each end; Y is pulled in from the y=+-3 perimeter
    # walls (physical obstacles the flow system also avoids on its own) by
    # enough real distance (0.9m, not the earlier 0.3m) to give
    # BOUNDARY_RETURN room to actually turn the drone around before it
    # reaches the wall - found by testing: a fast excursion (e.g. during
    # EMERGENCY_ESCAPE's reverse+turn, which has real momentum and doesn't
    # stop the drone instantly) could cross a thinner margin in a single
    # decision cycle and hit the wall before BOUNDARY_RETURN got a chance
    # to respond.
    bounds = {
        "min_x": -COURSE_MARGIN,
        "max_x": GOAL_X + COURSE_MARGIN,
        "min_y": -2.0,
        "max_y": 2.0,
        "center_x": GOAL_X / 2,
        "center_y": 0.0,
    }

    return {"plane": plane, "obstacles": obstacles, "goal_x": GOAL_X, "bounds": bounds}
