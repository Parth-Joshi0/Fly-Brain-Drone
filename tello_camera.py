from djitellopy import Tello
import cv2
import time

tello = Tello()

tello.connect(wait_for_state=False)
print("Connected!")

tello.streamoff()
time.sleep(1)

tello.streamon()
print("Stream ON")

time.sleep(3)

frame_read = tello.get_frame_read()

while True:
    frame = frame_read.frame

    if frame is None:
        print("Waiting for video...")
        continue

    print("Frame received:", frame.shape)

    cv2.imshow("Tello Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

tello.streamoff()
tello.end()
cv2.destroyAllWindows()