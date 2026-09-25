"""
Live banana detection + ripeness classification over a DJI Tello video feed.

Usage:
    python tello_camera.py
"""

import sys
import os
import time
import cv2
from djitellopy import Tello

# liveDetect.py lives in the food_detection subdirectory
from food_detection.liveDetect import BananaDetector



def main():
    detector = BananaDetector()

    tello = Tello()
    tello.connect(wait_for_state=False)
    print("Connected! Battery:", tello.get_battery())

    tello.streamoff()
    time.sleep(1)
    tello.streamon()
    print("Stream ON")
    time.sleep(3)

    frame_read = tello.get_frame_read()

    print("Press 'q' to quit.")

    try:
        while True:
            frame = frame_read.frame

            if frame is None:
                time.sleep(0.01)
                continue

            # Tello frames come in as RGB — convert to BGR for OpenCV/YOLO
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            frame, detections = detector.detect_and_annotate(frame)

            cv2.imshow('Tello Banana Ripeness Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        tello.streamoff()
        tello.end()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()