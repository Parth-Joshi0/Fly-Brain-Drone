"""
Paths and the label format shared by the fine-tuning scripts.

One folder per flight in finetune_data/:

    finetune_data/flight_20261022_101500/
        images/frame_00012.jpg
        labels/frame_00012.txt   one "0 cx cy w h" line per banana (0-1),
                                 empty = no banana in the picture
        rejected/                pictures dropped in review.py
        reviewed.txt             names of pictures checked in review.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "BananaModel" / "finetune_data"
FINETUNED_DIR = ROOT / "BananaModel" / "finetuned"
COCO_WEIGHTS = {
    "n": ROOT / "BananaModel" / "yolov8n.pt",
    "s": ROOT / "BananaModel" / "yolov8s.pt",
}

# Same as Drone/fly_tello.py's load_detectors()
FLIGHT_IMG_SIZE = 960
FLIGHT_CONF = {"n": 0.20, "s": 0.30}


def flight_dirs(names=None):
    """Flight folders in finetune_data/ (all of them, or just `names`)."""
    if names:
        dirs = [DATA_DIR / name for name in names]
        missing = [d.name for d in dirs if not (d / "images").is_dir()]
        if missing:
            sys.exit(f"Not in {DATA_DIR}: {', '.join(missing)}")
        return dirs
    return sorted(d for d in DATA_DIR.iterdir() if (d / "images").is_dir())


def image_paths(flight_dir):
    return sorted((flight_dir / "images").glob("*.jpg"))


def label_path(image_path):
    return image_path.parent.parent / "labels" / (image_path.stem + ".txt")


def read_boxes(image_path, width, height):
    """Pixel (x1, y1, x2, y2) boxes from an image's label file."""
    path = label_path(image_path)
    if not path.exists():
        return []
    boxes = []
    for line in path.read_text().split("\n"):
        if not line.strip():
            continue
        _, cx, cy, w, h = map(float, line.split())
        boxes.append((
            (cx - w / 2) * width, (cy - h / 2) * height,
            (cx + w / 2) * width, (cy + h / 2) * height,
        ))
    return boxes


def write_boxes(image_path, boxes, width, height):
    path = label_path(image_path)
    path.parent.mkdir(exist_ok=True)
    lines = []
    for x1, y1, x2, y2 in boxes:
        x1, x2 = sorted((max(0, x1), min(width, x2)))
        y1, y2 = sorted((max(0, y1), min(height, y2)))
        lines.append(f"0 {(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f} "
                     f"{(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def reviewed_names(flight_dir):
    path = flight_dir / "reviewed.txt"
    return set(path.read_text().split()) if path.exists() else set()


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def pick_device():
    import torch
    if torch.cuda.is_available():
        return 0
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
