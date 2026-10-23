"""
Step 1: draft banana boxes for saved flight pictures with a big YOLO
("teacher") model, ready to be checked in review.py.

    python BananaModel/finetune/autolabel.py Drone/flight_frames/flight_20261022_101500 [more flights...]

Each flight becomes finetune_data/<flight name>/ (see common.py).
Pictures barely different from the last one kept (drone hovering) are
skipped - near-copies add training time, not information.

The teacher misses the same far-away bananas our models do, so it also
looks at 4 zoomed-in quarters of each picture, and the sureness bar is
low: a wrong box costs one right-click in review.py, a missed banana
costs a drag.
"""

import argparse
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.ops import nms
from ultralytics import YOLO

from common import DATA_DIR, ROOT, pick_device, write_boxes
from BananaModel.banana_detector import banana_class_id

# A quarter overlaps its neighbours by this much of the picture, so a
# banana on the middle line is whole in at least one of them
TILE_FRACTION = 0.6

# Boxes this close (px) to a quarter's inner edge are probably a cut-off
# banana - the full picture or the next quarter has the whole one
EDGE_MARGIN = 4


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("flights", nargs="+", help="Drone/flight_frames/<flight> folders")
    parser.add_argument("--teacher", default=str(ROOT / "BananaModel" / "yolo11x.pt"),
                        help="big YOLO model (downloads itself the first time)")
    parser.add_argument("--conf", type=float, default=0.15, help="teacher sureness bar")
    parser.add_argument("--imgsz", type=int, default=1280, help="teacher input size")
    parser.add_argument("--min-change", type=float, default=3.0,
                        help="skip a picture whose mean pixel change from the last "
                             "kept one is below this (0-255); 0 keeps everything")
    parser.add_argument("--no-tiles", action="store_true", help="full picture only (faster)")
    return parser.parse_args()


def teacher_boxes(teacher, class_id, frame, args, device):
    """Banana boxes [x1, y1, x2, y2] + sureness, from the full picture
    and (unless --no-tiles) its 4 overlapping quarters."""
    h, w = frame.shape[:2]
    views = [(0, 0, w, h)]
    if not args.no_tiles:
        tw, th = int(w * TILE_FRACTION), int(h * TILE_FRACTION)
        views += [(x, y, x + tw, y + th) for x in (0, w - tw) for y in (0, h - th)]

    boxes, scores = [], []
    for vx1, vy1, vx2, vy2 in views:
        result = teacher(frame[vy1:vy2, vx1:vx2], conf=args.conf, imgsz=args.imgsz,
                         classes=[class_id], device=device, verbose=False)[0]
        for box, score in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist()):
            x1, y1, x2, y2 = box
            # Inner edges only - a quarter's outer edges are the picture's
            is_tile = (vx2 - vx1) < w
            if is_tile and ((vx1 > 0 and x1 < EDGE_MARGIN) or
                            (vx2 < w and x2 > vx2 - vx1 - EDGE_MARGIN) or
                            (vy1 > 0 and y1 < EDGE_MARGIN) or
                            (vy2 < h and y2 > vy2 - vy1 - EDGE_MARGIN)):
                continue
            boxes.append([x1 + vx1, y1 + vy1, x2 + vx1, y2 + vy1])
            scores.append(score)

    if not boxes:
        return []
    keep = nms(torch.tensor(boxes), torch.tensor(scores), iou_threshold=0.5)
    return [boxes[i] for i in keep.tolist()]


def thumbnail(frame):
    return cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 48)).astype(np.float32)


def label_flight(flight, teacher, class_id, args, device):
    out = DATA_DIR / flight.name
    if out.exists():
        print(f"{out} already exists - skipping (delete it to redo)")
        return
    (out / "images").mkdir(parents=True)

    frames = sorted(flight.glob("*.jpg"))
    kept = with_banana = 0
    last_thumb = None
    start = time.time()

    for i, path in enumerate(frames):
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        thumb = thumbnail(frame)
        if last_thumb is not None and np.abs(thumb - last_thumb).mean() < args.min_change:
            continue
        last_thumb = thumb

        image_path = out / "images" / path.name
        shutil.copy2(path, image_path)
        boxes = teacher_boxes(teacher, class_id, frame, args, device)
        write_boxes(image_path, boxes, frame.shape[1], frame.shape[0])
        kept += 1
        with_banana += bool(boxes)
        print(f"\r{flight.name}: {i + 1}/{len(frames)} pictures", end="", flush=True)

    print(f"\r{flight.name}: kept {kept}/{len(frames)} pictures, {with_banana} with a "
          f"banana, in {time.time() - start:.0f} s -> {out}")


def main():
    args = parse_args()
    device = pick_device()
    print("Loading teacher", args.teacher, "on", device)
    teacher = YOLO(args.teacher)
    class_id = banana_class_id(teacher)
    for flight in args.flights:
        label_flight(Path(flight).resolve(), teacher, class_id, args, device)
    print("Next: python BananaModel/finetune/review.py <flight name>")


if __name__ == "__main__":
    main()
