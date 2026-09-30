"""
Show the Tello's battery level.

Usage:
    python Drone/Tests/tello_battery_test.py           # print once
    python Drone/Tests/tello_battery_test.py --watch   # keep printing every 5 seconds (Ctrl+C to stop)

Connect your Mac to the TELLO-xxxxxx Wi-Fi first.
"""

import time
import argparse

from djitellopy import Tello


# Same limit tello_camera.py uses before it will take off
MIN_BATTERY_FOR_FLIGHT = 30


def show_battery(tello):

    battery = tello.get_battery()

    if battery >= MIN_BATTERY_FOR_FLIGHT:
        status = "OK to fly"
    else:
        status = f"TOO LOW - charge to at least {MIN_BATTERY_FOR_FLIGHT}%"

    print(f"Battery: {battery}%  ({status})")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--watch",
        action="store_true",
        help="keep printing the battery every 5 seconds"
    )

    args = parser.parse_args()

    tello = Tello()

    print("Connecting to Tello...")

    tello.connect()

    try:

        show_battery(tello)

        while args.watch:

            time.sleep(5)

            show_battery(tello)

    except KeyboardInterrupt:

        pass

    finally:

        tello.end()


if __name__ == "__main__":

    main()
