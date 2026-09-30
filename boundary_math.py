"""
Helper functions for reflex_conroller.py
"""

import math


def outside_bounds(position, bounds):
    x, y = position[0], position[1]
    b = bounds
    return x < b["min_x"] or x > b["max_x"] or y < b["min_y"] or y > b["max_y"]


def well_inside_bounds(position, bounds, margin):
    x, y = position[0], position[1]
    b = bounds
    return (b["min_x"] + margin < x < b["max_x"] - margin
            and b["min_y"] + margin < y < b["max_y"] - margin)


def near_or_outside_bounds(position, bounds, margin):
    x, y = position[0], position[1]
    b = bounds
    return (x < b["min_x"] + margin or x > b["max_x"] - margin
            or y < b["min_y"] + margin or y > b["max_y"] - margin)


def heading_error_toward(position, yaw_degrees, target):
    """Signed angle (radians) from the current heading to target, wrapped
    to [-pi, pi] - positive means target is to the left (matches this
    project's yaw convention: positive yaw = turn left)."""
    yaw = math.radians(yaw_degrees)
    dx = target[0] - position[0]
    dy = target[1] - position[1]
    target_heading = math.atan2(dy, dx)
    return (target_heading - yaw + math.pi) % (2 * math.pi) - math.pi


def heading_rate_toward(position, yaw_degrees, target, max_rate, gain):
    heading_error = heading_error_toward(position, yaw_degrees, target)
    return max(-max_rate, min(max_rate, heading_error * gain))
