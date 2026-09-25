from djitellopy import Tello
import time

tello = Tello()

tello.connect(wait_for_state=False)
print("Connected!")

battery = tello.query_battery()
print("Battery:", battery, "%")

tello.takeoff()
print("Flying!")

time.sleep(2)

tello.land()
print("Landed!")