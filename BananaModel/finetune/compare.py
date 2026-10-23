"""
Step 4: did fine-tuning help? Scores the COCO and fine-tuned detectors
on held-out flights, at the size and sureness bar fly_tello.py uses.

    python BananaModel/finetune/compare.py --flights flight_20261022_103000

Per model: bananas found (all / far-away ones), wrong boxes, and ms per
picture. Only fly with --finetuned if it finds more without adding
wrong boxes - a wrong box on the wall sends the drone to eat the wall.
The detector alone: the zoomed-in look of "sharp eyes" isn't included.
"""

import argparse
import time

import cv2
from ultralytics import YOLO

from common import (COCO_WEIGHTS, FINETUNED_DIR, FLIGHT_CONF, FLIGHT_IMG_SIZE, flight_dirs,
                    image_paths, iou, pick_device, read_boxes)
from BananaModel.banana_detector import banana_class_id

# A banana under this fraction of the picture counts as far away
# (fly_tello.py's notes: COCO nano finds 7/24 bananas at ~1%)
FAR_AREA = 0.01

MATCH_IOU = 0.5


def score(weights, conf, pictures, device):
    model = YOLO(str(weights))
    class_id = banana_class_id(model)
    # First call sets up the model - not part of the per-picture time
    model(pictures[0][0], imgsz=FLIGHT_IMG_SIZE, device=device, verbose=False)
    found = total = found_far = total_far = wrong = wrong_pictures = 0
    seconds = 0.0

    for frame, truth in pictures:
        h, w = frame.shape[:2]
        start = time.time()
        result = model(frame, conf=conf, imgsz=FLIGHT_IMG_SIZE, classes=[class_id],
                       device=device, verbose=False)[0]
        seconds += time.time() - start
        boxes = result.boxes.xyxy.tolist()

        # Greedy: each true banana takes the free box overlapping it most
        unmatched = list(boxes)
        for t in truth:
            far = (t[2] - t[0]) * (t[3] - t[1]) / (w * h) < FAR_AREA
            total += 1
            total_far += far
            best = max(unmatched, key=lambda b: iou(t, b), default=None)
            if best is not None and iou(t, best) >= MATCH_IOU:
                unmatched.remove(best)
                found += 1
                found_far += far
        wrong += len(unmatched)
        wrong_pictures += bool(unmatched)

    return {
        "found": f"{found}/{total} ({found / max(1, total):.0%})",
        "far": f"{found_far}/{total_far} ({found_far / max(1, total_far):.0%})",
        "wrong": f"{wrong} in {wrong_pictures} pics",
        "ms": f"{1000 * seconds / max(1, len(pictures)):.0f}",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--flights", nargs="+", required=True,
                        help="held-out flight folder name(s) - not ones trained on")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    device = args.device or pick_device()

    pictures = []
    for flight in flight_dirs(args.flights):
        for path in image_paths(flight):
            frame = cv2.imread(str(path))
            pictures.append((frame, read_boxes(path, frame.shape[1], frame.shape[0])))
    print(f"{len(pictures)} pictures, "
          f"{sum(1 for _, t in pictures if not t)} with no banana\n")

    rows = []
    for size in ("n", "s"):
        candidates = [("COCO", COCO_WEIGHTS[size]), ("fine-tuned", FINETUNED_DIR / f"banana_{size}.pt")]
        for name, weights in candidates:
            if not weights.exists():
                continue
            label = f"YOLOv8{size} {name} @{FLIGHT_CONF[size]:.2f}"
            rows.append((label, score(weights, FLIGHT_CONF[size], pictures, device)))

    print(f"{'model':<28}{'found':>16}{'far (<1%)':>16}{'wrong boxes':>20}{'ms/pic':>8}")
    for label, r in rows:
        print(f"{label:<28}{r['found']:>16}{r['far']:>16}{r['wrong']:>20}{r['ms']:>8}")


if __name__ == "__main__":
    main()
