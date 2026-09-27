"""
DJI Tello live banana detection + fly-inspired food behaviour.

Normal:
    python tello_camera.py

This is DRY RUN mode.
The drone DOES NOT take off.

Real flight:
    python tello_camera.py --fly

Keys:
    q = land and quit
    e = stop moving right away and land gently
    x = EMERGENCY motor stop (drone falls - only if about to hit something)
"""

import os
import csv
import time
import argparse
from datetime import datetime

import cv2

from djitellopy import Tello

from food_detection.liveDetect import BananaDetector
from controllers.food_orbit import FoodOrbitBehaviour


# ============================================================
# SAFETY
# ============================================================

MIN_BATTERY_FOR_FLIGHT = 30


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

    args = parser.parse_args()


    # ========================================================
    # LOAD AI / BEHAVIOUR
    # ========================================================

    print("Loading banana detector...")

    detector = BananaDetector()

    print("Loading food behaviour...")

    behaviour = FoodOrbitBehaviour()


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


    # ========================================================
    # FLYING STATE
    # ========================================================

    flying = False

    log_file, log_writer = open_flight_log()

    start_time = time.time()


    try:

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

        while True:

            frame = (
                frame_read.frame
            )

            if frame is None:

                time.sleep(0.01)

                continue


            # Tello gives RGB
            # OpenCV uses BGR.
            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_RGB2BGR
            )


            # =================================================
            # BANANA DETECTION
            # =================================================

            frame, detections = (
                detector.detect_and_annotate(
                    frame
                )
            )


            h, w = (
                frame.shape[:2]
            )


            # =================================================
            # FOOD BRAIN
            # =================================================

            cmd = behaviour.update(
                detections,
                w,
                h
            )


            # =================================================
            # SEND MOVEMENT
            # =================================================

            if flying:

                tello.send_rc_control(
                    cmd.lr,
                    cmd.fb,
                    cmd.ud,
                    cmd.yaw
                )


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
                cmd.lr,
                cmd.fb,
                cmd.ud,
                cmd.yaw,
                tello.get_battery(),
            ])

            log_file.flush()


            # Done eating + waited 5 s -> land (in finally).
            if behaviour.should_land:

                print(
                    "Finished eating - landing."
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
            # MOVEMENT COMMANDS
            # -------------------------------------------------

            command_text = (
                f"LR:{cmd.lr}  "
                f"FB:{cmd.fb}  "
                f"UD:{cmd.ud}  "
                f"YAW:{cmd.yaw}"
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


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    main()