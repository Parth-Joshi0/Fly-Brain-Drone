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
import time
import argparse
import platform
from pathlib import Path

# Repo root on the path, so BananaModel/, NeuralPathways/ etc. import
# no matter where this is run from (same as Drone/FlightTests/*.py).
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2

from djitellopy import Tello

from BananaModel.banana_detector import BananaDetector
from Drone.flight_hud import draw_hud
from Drone.flight_log import LoopStats, flight_log_row, open_flight_log
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


# Every run writes one CSV here - see flight_log.py
LOG_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "flight_logs"
)


# ============================================================
# SETUP
# ============================================================

def parse_args():

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

    return parser.parse_args()


def load_detectors():
    """The banana detector used every picture, and a slower "sharp eyes"
    one used only while holding still."""

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

    return detector, sharp_detector


def connect_tello():

    tello = Tello()

    print("Connecting to Tello...")

    tello.connect()

    battery = tello.get_battery()

    print("Connected!")

    print("Battery:", battery, "%")

    return tello, battery


def start_video(tello):
    """Restarts the stream. Returns the frame reader and the black
    placeholder picture djitellopy shows before real video arrives."""

    try:
        tello.streamoff()
    except Exception:
        pass

    time.sleep(1)

    tello.streamon()

    print("Camera stream ON")

    time.sleep(3)

    frame_read = tello.get_frame_read()

    return frame_read, frame_read.frame


def take_off(args, tello, battery):
    """Takes off with --fly (if the battery allows). Returns whether the
    drone is flying, or None if it should have but can't."""

    if not args.fly:

        print("DRY RUN MODE")

        print("Drone will NOT take off.")

        return False

    if battery < MIN_BATTERY_FOR_FLIGHT:

        print("Battery too low to fly.")

        print(f"Need at least {MIN_BATTERY_FOR_FLIGHT}%.")

        return None

    print("FLIGHT MODE ENABLED")

    tello.takeoff()

    return True


# ============================================================
# KEYS
# ============================================================

def handle_key(key, tello, flying):
    """Returns (stop, flying, hand_mark).

        q = quit
        h = mark the moment a hand is waved, for tuning
        e = stop moving now, then land gently (in finally)
        x = emergency: motors off, drone falls
    """

    if key == ord("q"):
        return True, flying, False

    if key == ord("h"):
        print("HAND marked")
        return False, flying, True

    if key == ord("e"):

        print("STOP - landing gently")

        if flying:
            tello.send_rc_control(0, 0, 0, 0)

        return True, flying, False

    if key == ord("x") and flying:

        print("EMERGENCY MOTOR STOP")

        tello.emergency()

        return True, False, False

    return False, flying, False


# ============================================================
# CLEANUP
# ============================================================

def land_and_disconnect(tello, flying):

    if flying:

        try:

            print("Landing...")

            tello.send_rc_control(0, 0, 0, 0)

            # Let it settle into a still hover first,
            # so it comes straight down softly.
            time.sleep(1.0)

            tello.land()

        except Exception as err:

            print("Landing error:", err)

    try:
        tello.streamoff()
    except Exception:
        pass

    try:
        tello.end()
    except Exception:
        pass


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    # The loop-rate tuning was done on one laptop - say which one this is
    print("Machine:", platform.node(), platform.machine(), platform.mac_ver()[0])

    detector, sharp_detector = load_detectors()

    tello, battery = connect_tello()

    frame_read, placeholder_frame = start_video(tello)

    flying = False

    brain = None

    stats = LoopStats()

    log_file, log_writer = open_flight_log(LOG_DIR)

    start_time = time.time()

    try:

        # Check the camera works before anything else
        print("Waiting for camera picture...")

        if not wait_for_video(frame_read, placeholder_frame, VIDEO_START_TIMEOUT):

            print("NO CAMERA PICTURE - not taking off.")

            print("Turn the drone off and on, reconnect to its "
                  "Wi-Fi, and try again.")

            return

        print("Camera OK")

        # Brain: eating behaviour (+ fly brain with --scared/--stabilize)
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

        flying = take_off(args, tello, battery)

        if flying is None:
            flying = False
            return

        if flying:
            # After `flying` is set, so Ctrl-C here still lands (finally)
            time.sleep(2)

        # The fly brain ignores the first ~2 s (takeoff looks like a loom)
        brain.start()

        print("Press q to quit.")

        print("Press e to stop and land gently.")

        print("Press x for EMERGENCY motor stop (drone falls).")

        last_frame = None

        last_new_frame_time = time.time()

        # Set by the 'h' key, written into the next flight-log row
        hand_mark = False

        while True:

            frame = frame_read.frame

            # Video safety: no new picture -> hover, then land
            if frame is last_frame or frame is None:

                stalled_for = time.time() - last_new_frame_time

                if flying and stalled_for > VIDEO_LOST_LAND:
                    print("CAMERA LOST - landing.")
                    break

                if flying and stalled_for > VIDEO_STALL_HOVER:
                    tello.send_rc_control(0, 0, 0, 0)

                time.sleep(0.01)

                continue

            last_frame = frame

            last_new_frame_time = time.time()

            # Tello gives RGB, OpenCV uses BGR.
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Brain: picture in, movement out (on the clean picture -
            # see scared_eating_brain.py). The compass heading lets the
            # room scan count its turns.
            try:
                yaw_deg = tello.get_yaw()
            except Exception:
                yaw_deg = None

            cmd = brain.step(frame, yaw_deg, flying)

            detections = brain.detections

            stats.after_step(fear)

            # Boxes from the latest detection (may be 1-2 pictures old)
            detector.annotate(frame, detections)

            sent = (cmd.lr, cmd.fb, cmd.ud, cmd.yaw)

            if flying:
                tello.send_rc_control(*sent)

            log_writer.writerow(flight_log_row(
                time.time() - start_time, flying, behaviour, detections, sent,
                tello.get_battery(), fear, hand_mark, stats.fps))

            log_file.flush()

            hand_mark = False

            # Done eating, or no banana anywhere -> land (in finally).
            if behaviour.should_land:
                print(f"Landing: {behaviour.land_reason}.")
                break

            draw_hud(frame, flying, behaviour, fear, stats.fps, sent)

            cv2.imshow("Tello Fly Brain", frame)

            stop, flying, hand_mark = handle_key(cv2.waitKey(1) & 0xFF, tello, flying)

            if stop:
                break

    finally:

        land_and_disconnect(tello, flying)

        cv2.destroyAllWindows()

        log_file.close()

        stats.print_summary()

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
