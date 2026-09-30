import cv2
import numpy as np

cap = cv2.VideoCapture(0)   # keep whatever camera source you already use

ret, frame1 = cap.read()
prev_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,
        gray,
        None,
        0.5,
        3,
        15,
        3,
        5,
        1.2,
        0
    )

    # Flow magnitude
    x_flow = flow[..., 0]
    y_flow = flow[..., 1]

    magnitude = np.sqrt(x_flow**2 + y_flow**2)

    height, width = magnitude.shape

    # Split camera into thirds
    third = width // 3

    left = magnitude[:, :third]
    center = magnitude[:, third:2 * third]
    right = magnitude[:, 2 * third:]

    # Average optical flow in each section
    left_flow = np.mean(left)
    center_flow = np.mean(center)
    right_flow = np.mean(right)

    print(
        f"LEFT: {left_flow:.2f} | "
        f"CENTER: {center_flow:.2f} | "
        f"RIGHT: {right_flow:.2f}"
    )

    # Draw sections so you can see them
    cv2.line(frame, (third, 0), (third, height), (0, 255, 0), 2)
    cv2.line(frame, (2 * third, 0), (2 * third, height), (0, 255, 0), 2)

    cv2.putText(
        frame,
        f"L: {left_flow:.2f}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"C: {center_flow:.2f}",
        (third + 20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"R: {right_flow:.2f}",
        (2 * third + 20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.imshow("Camera + Optical Flow", frame)

    prev_gray = gray

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
 
cap.release()
cv2.destroyAllWindows()