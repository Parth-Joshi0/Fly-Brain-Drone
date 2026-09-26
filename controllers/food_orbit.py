"""
Food orbit behaviour for the DJI Tello.

For testing, ANY detected banana is considered a valid target.

Behaviour:
SEARCH
    -> find banana
APPROACH
    -> move toward it while keeping it centered
HOVER
    -> hold near it briefly
CIRCLE
    -> strafe sideways while turning toward it
"""

import time
from dataclasses import dataclass


# ============================================================
# SETTINGS
# ============================================================

# For now, ALL banana ripeness classes are accepted.
TARGET_LABELS = {
    "freshripe",
    "freshunripe",
    "overripe",
    "ripe",
    "rotten",
    "unripe",
}

# Ignore extremely uncertain YOLO detections
MIN_DETECTION_CONFIDENCE = 0.20

# Keep this low for testing so ripeness mistakes don't matter much
MIN_CLASSIFIER_CONFIDENCE = 0.10

# How large the banana should appear before the drone considers
# itself close enough.
#
# Increase this if the drone gets TOO close.
# Decrease this if it stops TOO far away.
TARGET_BOX_RATIO = 0.12

# How long to hover before starting the circle
HOVER_TIME = 2.0

# If YOLO loses the banana briefly, don't immediately start searching
TARGET_LOST_GRACE_TIME = 0.7


# ============================================================
# MOVEMENT SPEEDS
# ============================================================

# Slow rotation while looking for a banana
SEARCH_YAW_SPEED = 10

# Maximum forward speed while approaching
MAX_APPROACH_SPEED = 15

# Sideways speed while circling
#
# Positive = circle one direction
# Negative = circle the opposite direction
CIRCLE_SPEED = 10

# Maximum forward/back correction while orbiting
MAX_DISTANCE_CORRECTION = 10


# ============================================================
# CONTROL SETTINGS
# ============================================================

# How strongly the drone turns toward the banana
YAW_GAIN = 0.08

# Controls forward/back correction based on banana size
DISTANCE_GAIN = 300.0

# Maximum yaw speed
MAX_YAW = 20

# Ignore tiny left/right errors so it doesn't constantly shake
CENTER_DEADZONE = 35


# ============================================================
# RC COMMAND
# ============================================================

@dataclass
class RCCommand:
    lr: int = 0
    fb: int = 0
    ud: int = 0
    yaw: int = 0


def clamp(value, minimum, maximum):
    return max(
        minimum,
        min(maximum, value)
    )


# ============================================================
# FOOD ORBIT BEHAVIOUR
# ============================================================

class FoodOrbitBehaviour:

    def __init__(self):

        self.state = "SEARCH"

        self.hover_start_time = None

        self.last_target_time = 0

        self.last_target_label = None

        self.last_box_ratio = 0.0


    # ========================================================
    # CHOOSE BANANA
    # ========================================================

    def _choose_target(self, detections):

        valid_targets = []

        for detection in detections:

            label = detection.label.lower()

            # Accept ANY known banana ripeness class
            if label not in TARGET_LABELS:
                continue

            # Make sure YOLO is reasonably confident
            if detection.det_conf < MIN_DETECTION_CONFIDENCE:
                continue

            # Keep classifier threshold low for testing
            if detection.cls_conf < MIN_CLASSIFIER_CONFIDENCE:
                continue

            valid_targets.append(
                detection
            )

        if not valid_targets:
            return None

        # Choose the strongest/largest banana
        def score(d):

            x1, y1, x2, y2 = d.box

            area = max(
                1,
                (x2 - x1) * (y2 - y1)
            )

            return (
                d.det_conf
                * max(d.cls_conf, 0.1)
                * area
            )

        return max(
            valid_targets,
            key=score
        )


    # ========================================================
    # TURN TOWARD BANANA
    # ========================================================

    def _target_yaw(
        self,
        target,
        frame_width
    ):

        x1, _, x2, _ = target.box

        banana_x = (
            x1 + x2
        ) / 2.0

        camera_center_x = (
            frame_width / 2.0
        )

        error = (
            banana_x
            - camera_center_x
        )

        # Don't react to tiny errors
        if abs(error) < CENTER_DEADZONE:
            return 0

        yaw = int(
            error * YAW_GAIN
        )

        return int(
            clamp(
                yaw,
                -MAX_YAW,
                MAX_YAW
            )
        )


    # ========================================================
    # BANANA SIZE
    # ========================================================

    def _box_ratio(
        self,
        target,
        frame_width,
        frame_height
    ):

        x1, y1, x2, y2 = target.box

        banana_area = max(
            1,
            (x2 - x1)
            * (y2 - y1)
        )

        frame_area = (
            frame_width
            * frame_height
        )

        return (
            banana_area
            / frame_area
        )


    # ========================================================
    # DISTANCE CONTROL
    # ========================================================

    def _distance_control(
        self,
        box_ratio
    ):

        # Banana too small:
        # move forward
        #
        # Banana too large:
        # move backward

        error = (
            TARGET_BOX_RATIO
            - box_ratio
        )

        fb = int(
            error
            * DISTANCE_GAIN
        )

        return int(
            clamp(
                fb,
                -MAX_DISTANCE_CORRECTION,
                MAX_DISTANCE_CORRECTION
            )
        )


    # ========================================================
    # UPDATE BEHAVIOUR
    # ========================================================

    def update(
        self,
        detections,
        frame_width,
        frame_height
    ):

        now = time.time()

        target = self._choose_target(
            detections
        )

        # ====================================================
        # NO BANANA
        # ====================================================

        if target is None:

            time_missing = (
                now
                - self.last_target_time
            )

            # If banana disappeared for just a moment,
            # don't immediately rotate away.
            if (
                self.last_target_time > 0
                and
                time_missing
                < TARGET_LOST_GRACE_TIME
            ):

                return RCCommand(
                    lr=0,
                    fb=0,
                    ud=0,
                    yaw=0
                )

            self.state = "SEARCH"

            self.hover_start_time = None

            return RCCommand(
                lr=0,
                fb=0,
                ud=0,
                yaw=SEARCH_YAW_SPEED
            )


        # ====================================================
        # BANANA FOUND
        # ====================================================

        self.last_target_time = now

        self.last_target_label = (
            target.label
        )

        box_ratio = self._box_ratio(
            target,
            frame_width,
            frame_height
        )

        self.last_box_ratio = (
            box_ratio
        )

        yaw = self._target_yaw(
            target,
            frame_width
        )


        # ====================================================
        # SEARCH -> APPROACH
        # ====================================================

        if self.state == "SEARCH":

            self.state = "APPROACH"


        # ====================================================
        # APPROACH BANANA
        # ====================================================

        if self.state == "APPROACH":

            # Banana is still too small/far away
            if (
                box_ratio
                < TARGET_BOX_RATIO
            ):

                size_error = (
                    TARGET_BOX_RATIO
                    - box_ratio
                )

                forward_speed = int(
                    size_error
                    * DISTANCE_GAIN
                )

                forward_speed = int(
                    clamp(
                        forward_speed,
                        5,
                        MAX_APPROACH_SPEED
                    )
                )

                return RCCommand(
                    lr=0,
                    fb=forward_speed,
                    ud=0,
                    yaw=yaw
                )

            # Close enough
            self.state = "HOVER"

            self.hover_start_time = now


        # ====================================================
        # HOVER
        # ====================================================

        if self.state == "HOVER":

            distance_correction = (
                self._distance_control(
                    box_ratio
                )
            )

            if (
                self.hover_start_time
                is None
            ):

                self.hover_start_time = now

            # After hovering for a little while,
            # begin circling
            if (
                now
                - self.hover_start_time
                >= HOVER_TIME
            ):

                self.state = "CIRCLE"

            return RCCommand(
                lr=0,
                fb=distance_correction,
                ud=0,
                yaw=yaw
            )


        # ====================================================
        # CIRCLE / ORBIT
        # ====================================================

        if self.state == "CIRCLE":

            distance_correction = (
                self._distance_control(
                    box_ratio
                )
            )

            # Move sideways while turning toward banana.
            # This creates the orbit.
            return RCCommand(
                lr=CIRCLE_SPEED,
                fb=distance_correction,
                ud=0,
                yaw=yaw
            )


        # ====================================================
        # FALLBACK
        # ====================================================

        self.state = "SEARCH"

        return RCCommand(
            lr=0,
            fb=0,
            ud=0,
            yaw=0
        )