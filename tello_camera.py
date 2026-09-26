"""
Live banana detection + ripeness classification over a DJI Tello video feed,
with a food-orbit behaviour: when it sees food it likes, it circles it
(see controllers/food_orbit.py).

Usage:
    python tello_camera.py         # dry run - shows what it WOULD do, never takes off
    python tello_camera.py --fly   # takes off and actually flies the behaviour

Keys: 'q' = land and quit, 'e' = EMERGENCY motor stop (drone drops!)
"""

import sys
import os
import time
import argparse
import cv2
from djitellopy import Tello

# liveDetect.py lives in the food_detection subdirectory
from food_detection.liveDetect import BananaDetector
from controllers.food_orbit import FoodOrbitBehaviour



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fly', action='store_true',
                        help='take off and fly the behaviour (default: dry run, stays on the ground)')
    args = parser.parse_args()

    detector = BananaDetector()
    behaviour = FoodOrbitBehaviour()
    flying = False

    tello = Tello()
    tello.connect()
    print("Connected! Battery:", tello.get_battery())

    tello.streamoff()
    time.sleep(1)
    tello.streamon()
    print("Stream ON")
    time.sleep(3)

    frame_read = tello.get_frame_read()

    print("Press 'q' to land and quit, 'e' for emergency motor stop.")

    try:
        if args.fly:
            tello.takeoff()
            flying = True
            print("Taken off - running food-orbit behaviour")
        else:
            print("DRY RUN - not taking off (use --fly to fly)")

        while True:
            frame = frame_read.frame

            if frame is None:
                time.sleep(0.01)
                continue

            # Tello frames come in as RGB — convert to BGR for OpenCV/YOLO
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            frame, detections = detector.detect_and_annotate(frame)

            h, w = frame.shape[:2]
            cmd = behaviour.update(detections, w, h)
            if flying:
                tello.send_rc_control(cmd.lr, cmd.fb, cmd.ud, cmd.yaw)

            mode = "FLYING" if flying else "DRY RUN"
            cv2.putText(frame, f"{mode} | {behaviour.state} | lr={cmd.lr} fb={cmd.fb} yaw={cmd.yaw}",
                        (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.imshow('Tello Banana Ripeness Detection', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('e') and flying:
                tello.emergency()
                flying = False
                break
    finally:
        if flying:
            try:
                tello.send_rc_control(0, 0, 0, 0)
                tello.land()
            except Exception as err:
                print("Landing failed:", err)
        tello.streamoff()
        tello.end()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()