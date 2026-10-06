"""
fly_tello.py's flight log (one CSV row per picture) and the loop-rate
summary it prints at the end of a run.
"""

import csv
import os
import statistics
import time
from datetime import datetime


# Every run writes one row per frame to flight_logs/flight_<time>.csv
# so we can see exactly what the drone saw and decided.
LOG_COLUMNS = [
    "time_s", "mode", "state", "bananas_seen",
    "label", "det_conf", "cls_conf", "size_pct",
    "hunger", "lr", "fb", "ud", "yaw", "battery",
    "brain", "escape_dir", "loom_l", "loom_c", "loom_r",
    "escape_level", "wobble_floor", "loom_in_l", "loom_in_r", "hand",
    "self_moving",
    "fps", "brain_ms",
    "dng02_rotation", "dng02_l", "dng02_r", "dng02_steer", "dng02_thrust",
    "intended_yaw", "dng02_yaw",
]

# Pictures/s the fly brain needs to catch a quick hand swipe: at ~16/s
# it caught 4/4, at ~11/s 0/4 (see fear_brain.py MAX_BRAIN_STEPS).
# Measured on one MacBook Air - so check it on whatever this runs on.
MIN_LOOP_FPS = 15


def open_flight_log(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, datetime.now().strftime("flight_%Y%m%d_%H%M%S.csv"))
    log_file = open(path, "w", newline="")
    writer = csv.writer(log_file)
    writer.writerow(LOG_COLUMNS)
    print("Flight log:", path)
    return log_file, writer


def dng02_log_fields(fear):
    """The flight-log columns for the DNg02 stabilizer."""
    d = fear.brain.dng02
    return [
        f"{fear.dng02_rotation:.3f}",
        d.get("n_left", 0),
        d.get("n_right", 0),
        f"{d.get('steer', 0.0):.3f}",
        f"{d.get('thrust', 0.0):.3f}",
        fear.intended_yaw_rc,
        fear.dng02_yaw_rc,
    ]


def flight_log_row(elapsed, flying, behaviour, detections, sent, battery, fear, hand_mark, fps):
    """One row, in LOG_COLUMNS order. `fear` is None without --scared/--stabilize."""
    target = behaviour.current_target
    return [
        f"{elapsed:.2f}",
        "FLYING" if flying else "DRY RUN",
        behaviour.state,
        len(detections),
        target.label if target else "",
        f"{target.det_conf:.2f}" if target else "",
        f"{target.cls_conf:.2f}" if target else "",
        f"{behaviour.last_box_ratio * 100:.1f}" if target else "",
        f"{behaviour.hunger:.0f}",
        *sent,
        battery,
        (fear.brain.state if fear.armed else "ARMING") if fear else "",
        fear.escape_direction if fear and fear.escaping else "",
        f"{fear.expansion['left']:.2f}" if fear else "",
        f"{fear.expansion['center']:.2f}" if fear else "",
        f"{fear.expansion['right']:.2f}" if fear else "",
        f"{fear.escape_level:.2f}" if fear else "",
        f"{fear.wobble_floor:.2f}" if fear else "",
        f"{fear.loom_in[0]:.2f}" if fear else "",
        f"{fear.loom_in[1]:.2f}" if fear else "",
        "HAND" if hand_mark else "",
        ("yes" if fear.self_moving else "") if fear else "",
        f"{fps:.1f}",
        f"{fear.brain_ms:.0f}" if fear else "",
        *(dng02_log_fields(fear) if fear and fear.stabilize else [""] * 7),
    ]


class LoopStats:
    """Pictures per second through the brain, and the brain's ms per
    picture - for the summary at the end of the run."""

    def __init__(self):
        self.fps = 0.0
        self.fps_seen = []
        self.brain_ms_seen = []
        self._last_step_time = None

    def after_step(self, fear):
        step_time = time.time()
        if self._last_step_time is not None:
            self.fps = 1.0 / max(1e-3, step_time - self._last_step_time)
            self.fps_seen.append(self.fps)
        self._last_step_time = step_time
        if fear is not None and fear.armed:
            self.brain_ms_seen.append(fear.brain_ms)

    def print_summary(self):
        """Loop rate and brain cost for this run on this machine."""
        if len(self.fps_seen) < 10:
            return
        fps_median = statistics.median(self.fps_seen)
        print(f"Loop: median {fps_median:.1f} pictures/s over {len(self.fps_seen)} pictures")
        if self.brain_ms_seen:
            brain_ms_seen = sorted(self.brain_ms_seen)
            p95 = brain_ms_seen[int(0.95 * (len(brain_ms_seen) - 1))]
            print(f"Brain: median {statistics.median(brain_ms_seen):.0f} ms, "
                  f"p95 {p95:.0f} ms per picture")
            if fps_median < MIN_LOOP_FPS:
                print(f"WARNING: under {MIN_LOOP_FPS} pictures/s - quick hand "
                      f"swipes may be missed on this machine")
