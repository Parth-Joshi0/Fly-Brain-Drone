"""
DJI Tello live banana detection + fly-inspired food behaviour.

Run from the repo root:

Normal:
    python Drone/fly_tello.py

This is DRY RUN mode.
The drone DOES NOT take off.

Real flight:
    python Drone/fly_tello.py --fly

Scared while eating (fly brain backs away from anything swooping at it,
then comes back to finish the banana - needs .venv-brain, see
NeuralPathways/EscapeNeuron/fear_brain.py):
    python Drone/fly_tello.py --fly --scared

DNg02 stabilizer (the fly brain's flight-motor neurons hold the heading
steady in every state, on top of the eating behaviour's own turns -
works with or without --scared, see fear_brain.py):
    python Drone/fly_tello.py --fly --scared --stabilize

Keys:
    q = land and quit
    h = mark "I'm waving my hand NOW" in the flight log (for tuning --scared)
    e = stop moving right away and land gently
    x = EMERGENCY motor stop (drone falls - only if about to hit something)
"""

import os
import sys
import csv
import time
import argparse
import platform
import statistics
from datetime import datetime
from pathlib import Path

# Repo root on the path, so BananaModel/, NeuralPathways/ etc. import
# no matter where this is run from (same as Drone/tests/*.py).
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2

from djitellopy import Tello

from BananaModel.banana_detector import BananaDetector
from NeuralPathways.ScaredEating.scared_eating_brain import ScaredEatingBrain


# ============================================================
# SAFETY
# ============================================================

MIN_BATTERY_FOR_FLIGHT = 30

# Don't take off until the camera has sent a real picture.
VIDEO_START_TIMEOUT = 10.0

# While flying: no new picture for this long -> stop and hover...
VIDEO_STALL_HOVER = 0.5

# ...and for this long -> land (we're flying blind).
VIDEO_LOST_LAND = 3.0


def video_is_running(frame_read, placeholder):
    """
    True once the video thread is alive and has delivered a real
    picture. Until then djitellopy hands out a black placeholder
    frame (not None), so checking for None never caught a dead stream.
    """

    return (
        frame_read.worker.is_alive()
        and frame_read.frame is not placeholder
    )


def wait_for_video(frame_read, placeholder, timeout):

    deadline = time.time() + timeout

    while time.time() < deadline:

        if video_is_running(frame_read, placeholder):
            return True

        if not frame_read.worker.is_alive():
            return False

        time.sleep(0.1)

    return False


# ============================================================
# FLIGHT LOG
# ============================================================

# Every run writes one row per frame to flight_logs/flight_<time>.csv
# so we can see exactly what the drone saw and decided.
LOG_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "flight_logs"
)

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


def open_flight_log():

    os.makedirs(LOG_DIR, exist_ok=True)

    path = os.path.join(
        LOG_DIR,
        datetime.now().strftime("flight_%Y%m%d_%H%M%S.csv")
    )

    log_file = open(path, "w", newline="")

    writer = csv.writer(log_file)

    writer.writerow(LOG_COLUMNS)

    print("Flight log:", path)

    return log_file, writer


def dng02_log_fields(fear, behaviour):
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


# Pictures/s the fly brain needs to catch a quick hand swipe: at ~16/s
# it caught 4/4, at ~11/s 0/4 (see fear_brain.py MAX_BRAIN_STEPS).
# Measured on one MacBook Air - so check it on whatever this runs on.
MIN_LOOP_FPS = 15


def print_loop_summary(fps_seen, brain_ms_seen):
    """Loop rate and brain cost for this run on this machine."""

    if len(fps_seen) < 10:
        return

    fps_median = statistics.median(fps_seen)

    print(f"Loop: median {fps_median:.1f} pictures/s over {len(fps_seen)} pictures")

    if brain_ms_seen:

        brain_ms_seen = sorted(brain_ms_seen)

        p95 = brain_ms_seen[int(0.95 * (len(brain_ms_seen) - 1))]

        print(f"Brain: median {statistics.median(brain_ms_seen):.0f} ms, "
              f"p95 {p95:.0f} ms per picture")

        if fps_median < MIN_LOOP_FPS:

            print(f"WARNING: under {MIN_LOOP_FPS} pictures/s - quick hand "
                  f"swipes may be missed on this machine")


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fly",
        action="store_true",
        help="actually take off and fly"
    )

    parser.add_argument(
        "--scared",
        action="store_true",
        help="run the fly brain's looming escape circuit: dodge "
             "anything swooping at the drone, then come back to eat"
    )

    parser.add_argument(
        "--stabilize",
        action="store_true",
        help="run the fly brain's DNg02 flight-motor neurons as a yaw "
             "stabilizer in every state (with or without --scared)"
    )

    args = parser.parse_args()

    # The loop-rate tuning was done on one laptop - say which one this is
    print("Machine:", platform.node(), platform.machine(), platform.mac_ver()[0])


    # ========================================================
    # LOAD AI / BEHAVIOUR
    # ========================================================

    print("Loading banana detector...")

    # Full Tello resolution (960px) - sees the banana from further away
    detector = BananaDetector(detector_img_size=960)

    # "Sharp eyes" for scanning / looking while holding still: YOLOv8
    # small instead of nano. Far-away bananas (filling 1% of the picture)
    # found 22/24 vs 7/24; with the zoomed-in look and a 0.30 sureness
    # bar, 20-24/24 at 0.5-1% with no wrong boxes. 2x slower, so only
    # used while still - nano keeps the loop fast for catching hands.
    # Downloads itself the first time if missing (needs internet).
    print("Loading sharp-eyes banana detector...")

    sharp_detector = BananaDetector(
        detector_img_size=960,
        detector_conf_threshold=0.30,
        yolo_weights=str(ROOT / "BananaModel" / "yolov8s.pt")
    )


    # ========================================================
    # CONNECT TO TELLO
    # ========================================================

    tello = Tello()

    print("Connecting to Tello...")

    tello.connect()

    battery = tello.get_battery()

    print(
        "Connected!"
    )

    print(
        "Battery:",
        battery,
        "%"
    )


    # ========================================================
    # VIDEO
    # ========================================================

    try:

        tello.streamoff()

    except Exception:

        pass

    time.sleep(1)

    tello.streamon()

    print("Camera stream ON")

    time.sleep(3)

    frame_read = (
        tello.get_frame_read()
    )

    # The black picture djitellopy shows before real video arrives
    placeholder_frame = frame_read.frame


    # ========================================================
    # FLYING STATE
    # ========================================================

    flying = False

    brain = None

    fear = None

    log_file, log_writer = open_flight_log()

    start_time = time.time()


    try:

        # ====================================================
        # CHECK THE CAMERA WORKS BEFORE ANYTHING ELSE
        # ====================================================

        print("Waiting for camera picture...")

        if not wait_for_video(
            frame_read,
            placeholder_frame,
            VIDEO_START_TIMEOUT
        ):

            print(
                "NO CAMERA PICTURE - not taking off."
            )

            print(
                "Turn the drone off and on, reconnect to its "
                "Wi-Fi, and try again."
            )

            return


        print("Camera OK")


        # ====================================================
        # BRAIN: eating behaviour (+ fly brain with --scared)
        # ====================================================

        if args.scared or args.stabilize:

            print("Starting the fly brain (takes a few seconds)...")

        brain = ScaredEatingBrain(
            detector,
            tello,
            frame_read,
            scared=args.scared,
            stabilize=args.stabilize,
            sharp_detector=sharp_detector
        )

        # Shorthands for the screen / flight log below
        behaviour = brain.behaviour

        fear = brain.fear

        print("Brain ready")


        # ====================================================
        # TAKE OFF ONLY WITH --fly
        # ====================================================

        if args.fly:

            if battery < MIN_BATTERY_FOR_FLIGHT:

                print(
                    "Battery too low to fly."
                )

                print(
                    f"Need at least "
                    f"{MIN_BATTERY_FOR_FLIGHT}%."
                )

                return


            print(
                "FLIGHT MODE ENABLED"
            )

            tello.takeoff()

            flying = True

            time.sleep(2)

        else:

            print(
                "DRY RUN MODE"
            )

            print(
                "Drone will NOT take off."
            )


        # The fly brain ignores the first ~2 s (takeoff looks like a loom)
        brain.start()


        print(
            "Press q to quit."
        )

        print(
            "Press e to stop and land gently."
        )

        print(
            "Press x for EMERGENCY motor stop (drone falls)."
        )


        # ====================================================
        # MAIN CAMERA LOOP
        # ====================================================

        last_frame = None

        last_new_frame_time = time.time()

        # Set by the 'h' key, written into the next flight-log row
        hand_mark = False

        # Pictures per second through the brain, and the brain's ms per
        # picture - for the summary at the end (see print_loop_summary)
        last_step_time = None

        fps = 0.0

        fps_seen = []

        brain_ms_seen = []


        while True:

            frame = (
                frame_read.frame
            )


            # =================================================
            # VIDEO SAFETY: no new picture -> hover, then land
            # =================================================

            if frame is last_frame or frame is None:

                stalled_for = time.time() - last_new_frame_time

                if flying and stalled_for > VIDEO_LOST_LAND:

                    print(
                        "CAMERA LOST - landing."
                    )

                    break

                if flying and stalled_for > VIDEO_STALL_HOVER:

                    tello.send_rc_control(0, 0, 0, 0)

                time.sleep(0.01)

                continue


            last_frame = frame

            last_new_frame_time = time.time()


            # Tello gives RGB
            # OpenCV uses BGR.
            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_RGB2BGR
            )


            # =================================================
            # BRAIN: picture in, movement out (on the clean
            # picture - see scared_eating_brain.py)
            # =================================================

            # Compass heading - lets the room scan count its turns
            try:

                yaw_deg = tello.get_yaw()

            except Exception:

                yaw_deg = None

            cmd = brain.step(frame, yaw_deg, flying)

            detections = brain.detections

            step_time = time.time()

            if last_step_time is not None:

                fps = 1.0 / max(1e-3, step_time - last_step_time)

                fps_seen.append(fps)

            last_step_time = step_time

            if fear is not None and fear.armed:

                brain_ms_seen.append(fear.brain_ms)

            # Boxes from the latest detection (may be 1-2 pictures old)
            detector.annotate(frame, detections)


            h, w = (
                frame.shape[:2]
            )


            # =================================================
            # SEND MOVEMENT
            # =================================================

            sent = (cmd.lr, cmd.fb, cmd.ud, cmd.yaw)

            scared = behaviour.state == "SCARED"


            if flying:

                tello.send_rc_control(*sent)


            # =================================================
            # FLIGHT LOG
            # =================================================

            target = behaviour.current_target

            log_writer.writerow([
                f"{time.time() - start_time:.2f}",
                "FLYING" if flying else "DRY RUN",
                behaviour.state,
                len(detections),
                target.label if target else "",
                f"{target.det_conf:.2f}" if target else "",
                f"{target.cls_conf:.2f}" if target else "",
                f"{behaviour.last_box_ratio * 100:.1f}" if target else "",
                f"{behaviour.hunger:.0f}",
                *sent,
                tello.get_battery(),
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
                *(dng02_log_fields(fear, behaviour) if fear and fear.stabilize
                  else [""] * 7),
            ])

            log_file.flush()

            hand_mark = False


            # Done eating, or no banana anywhere -> land (in finally).
            if behaviour.should_land:

                print(
                    f"Landing: {behaviour.land_reason}."
                )

                break


            # =================================================
            # DISPLAY INFORMATION
            # =================================================

            if flying:

                mode_text = "FLYING"

            else:

                mode_text = "DRY RUN"


            # -------------------------------------------------
            # MODE
            # -------------------------------------------------

            cv2.putText(
                frame,
                f"MODE: {mode_text}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2
            )


            # -------------------------------------------------
            # STATE
            # -------------------------------------------------

            cv2.putText(
                frame,
                f"STATE: {behaviour.state}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2
            )


            # -------------------------------------------------
            # HUNGER
            # -------------------------------------------------

            hunger = int(
                behaviour.hunger
            )

            cv2.putText(
                frame,
                f"HUNGER: {hunger}%",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2
            )


            # -------------------------------------------------
            # TARGET
            # -------------------------------------------------

            target_label = (
                behaviour.last_target_label
            )

            if target_label is None:

                target_label = "NONE"


            cv2.putText(
                frame,
                f"TARGET: {target_label}",
                (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2
            )


            # -------------------------------------------------
            # BANANA SIZE (how much of the screen it fills)
            # -------------------------------------------------

            size_text = (
                f"SIZE: {behaviour.last_box_ratio:.1%}"
            )

            cv2.putText(
                frame,
                size_text,
                (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2
            )


            # -------------------------------------------------
            # FLY BRAIN
            # -------------------------------------------------

            if fear is not None:

                brain_text = (
                    f"BRAIN: {fear.brain.state if fear.armed else 'ARMING'}  "
                    f"LOOM L/C/R: {fear.expansion['left']:.1f}/"
                    f"{fear.expansion['center']:.1f}/"
                    f"{fear.expansion['right']:.1f}  "
                    f"ESCAPE: {fear.escape_level:.2f}/0.60  "
                    f"SCARES: {fear.scares}"
                    + ("  (moving - ignoring)" if fear.self_moving else "")
                )

                cv2.putText(
                    frame,
                    brain_text,
                    (10, 180),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 0, 255) if scared else (0, 255, 255),
                    2
                )

                if fear.stabilize:

                    d = fear.brain.dng02

                    cv2.putText(
                        frame,
                        f"DNg02 L={d.get('n_left', 0):2d} R={d.get('n_right', 0):2d}  "
                        f"steer={d.get('steer', 0.0):+.2f}  "
                        f"rot={fear.dng02_rotation:+.2f}px  "
                        f"yaw {int(fear.intended_yaw_rc):+d} + {fear.dng02_yaw_rc:+d}",
                        (10, 210),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 255, 255),
                        2
                    )

                cv2.putText(
                    frame,
                    f"LOOP: {fps:.0f}/s  BRAIN: {fear.brain_ms:.0f} ms",
                    (10, 240),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255),
                    2
                )

                if scared:

                    cv2.putText(
                        frame,
                        "SCARED! DNp01 fired - BACKING AWAY",
                        (10, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (0, 0, 255),
                        3
                    )

                elif behaviour.state == "WAIT":

                    cv2.putText(
                        frame,
                        "WAITING - is it safe? (need to see the banana)",
                        (10, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 165, 255),
                        2
                    )


            # -------------------------------------------------
            # MOVEMENT COMMANDS (what was actually sent)
            # -------------------------------------------------

            command_text = (
                f"LR:{sent[0]}  "
                f"FB:{sent[1]}  "
                f"UD:{sent[2]}  "
                f"YAW:{sent[3]}"
            )

            cv2.putText(
                frame,
                command_text,
                (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )


            # =================================================
            # SHOW VIDEO
            # =================================================

            cv2.imshow(
                "Tello Fly Brain",
                frame
            )


            # =================================================
            # KEYBOARD
            # =================================================

            key = (
                cv2.waitKey(1)
                & 0xFF
            )


            # q = quit
            if key == ord("q"):

                break


            # h = mark the moment a hand is waved, for tuning
            if key == ord("h"):

                hand_mark = True

                print("HAND marked")


            # e = stop moving now, then land gently (in finally)
            if key == ord("e"):

                print(
                    "STOP - landing gently"
                )

                if flying:

                    tello.send_rc_control(
                        0,
                        0,
                        0,
                        0
                    )

                break


            # x = emergency: motors off, drone falls
            if (
                key == ord("x")
                and flying
            ):

                print(
                    "EMERGENCY MOTOR STOP"
                )

                tello.emergency()

                flying = False

                break


    # ========================================================
    # CLEANUP
    # ========================================================

    finally:

        if flying:

            try:

                print(
                    "Landing..."
                )

                tello.send_rc_control(
                    0,
                    0,
                    0,
                    0
                )

                # Let it settle into a still hover first,
                # so it comes straight down softly.
                time.sleep(1.0)

                tello.land()

            except Exception as err:

                print(
                    "Landing error:",
                    err
                )


        try:

            tello.streamoff()

        except Exception:

            pass


        try:

            tello.end()

        except Exception:

            pass


        cv2.destroyAllWindows()

        log_file.close()

        print_loop_summary(fps_seen, brain_ms_seen)

        if brain is not None:

            try:

                brain.close()

            except Exception:

                pass


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    main()