"""
Reusable optical flow function — pass in a video path, get back
dense optical flow visualizations (and optionally a saved output video).

Usage:
    from optical_flow import compute_optical_flow

    frames = compute_optical_flow("my_video.mp4")
    # or save the result to a new video file:
    compute_optical_flow("my_video.mp4", output_path="flow_output.mp4")
"""

import numpy as np
import cv2 as cv


def compute_optical_flow(video_path, output_path=None, show=False):
    """
    Run dense (Farneback) optical flow over a video.

    Args:
        video_path (str): path to the input video file.
        output_path (str, optional): if given, writes a color-coded
            flow visualization video to this path.
        show (bool): if True, displays the flow live in a window
            (press 'q' or Esc to stop early).

    Returns:
        list[np.ndarray]: BGR frames visualizing the flow for each
            frame transition (hue = direction, brightness = speed).
    """
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    ret, first_frame = cap.read()
    if not ret:
        raise IOError("Video has no readable frames.")

    prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
    hsv = np.zeros_like(first_frame)
    hsv[..., 1] = 255

    writer = None
    if output_path:
        fps = cap.get(cv.CAP_PROP_FPS) or 30
        h, w = first_frame.shape[:2]
        fourcc = cv.VideoWriter_fourcc(*"mp4v")
        writer = cv.VideoWriter(output_path, fourcc, fps, (w, h))

    flow_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        flow = cv.calcOpticalFlowFarneback(
            prev_gray, gray, None,
            pyr_scale=0.5, levels=2, winsize=13,
            iterations=1, poly_n=5, poly_sigma=1.2, flags=0,
        )

        magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = angle * 180 / np.pi / 2
        hsv[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)
        flow_bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

        flow_frames.append(flow_bgr)

        if writer:
            writer.write(flow_bgr)

        if show:
            small = cv.resize(flow_bgr, (640, 480))
            cv.imshow("Optical Flow", small)
            key = cv.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break

        prev_gray = gray

    cap.release()
    if writer:
        writer.release()
    if show:
        cv.destroyAllWindows()

    return flow_frames


if __name__ == "__main__":
    import sys
    compute_optical_flow("TestVideo.mp4", show=True)