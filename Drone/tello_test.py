from djitellopy import Tello

print("STEP 1: Python file started")

tello = Tello()

print("STEP 2: Library loaded")

print("STEP 3: Trying to connect...")

tello.connect(wait_for_state=False)

print("STEP 4: Connected!")

