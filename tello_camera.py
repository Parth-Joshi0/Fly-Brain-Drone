from djitellopy import Tello
import cv2

print("STEP 1: File started")

tello = Tello()

print("STEP 2: Tello created")

tello.connect(wait_for_state=False)
print("STEP 3: Connected")

tello.streamon()
print("STEP 4: Camera stream ON")

frame_reader = tello.get_frame_read()
print("STEP 5: Frame reader started")

while True:
    frame = frame_reader.frame

    cv2.imshow("Tello Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

tello.streamoff()
cv2.destroyAllWindows()