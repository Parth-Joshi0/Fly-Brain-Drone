"""
Step 3: fine-tune the flight's banana detectors (YOLOv8 nano and small,
starting from their COCO weights) on the checked flight pictures.

    python BananaModel/finetune/train_detector.py --val flight_20261022_103000

--val names whole flights to hold out for scoring; every other flight
in finetune_data/ trains. Not a random split: pictures half a second
apart are near-copies, so a random split scores the model on pictures
it has practically trained on.

Trains as COCO's 80 classes with the bananas as class 46, rather than a
new 1-class model: that keeps COCO's banana knowledge in the model's
last layer, where a 1-class model starts it from scratch - and a few
hundred pictures are too few to relearn it. (The other 79 classes
fade, which nothing here uses.)

Writes BananaModel/finetuned/banana_n.pt and banana_s.pt (the previous
ones are kept as *.prev.pt), which `fly_tello.py --finetuned` flies with.
Then check it actually got better: compare.py.
"""

import argparse
import shutil
import time

import yaml
from ultralytics import YOLO

from common import (COCO_WEIGHTS, DATA_DIR, FINETUNED_DIR, FLIGHT_IMG_SIZE, flight_dirs,
                    image_paths, label_path, pick_device, reviewed_names)
from BananaModel.banana_detector import banana_class_id

RUNS_DIR = DATA_DIR / "_runs"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val", nargs="+", required=True,
                        help="flight folder name(s) to hold out for scoring")
    parser.add_argument("--train", nargs="+",
                        help="flight folder name(s) to train on (default: all others)")
    parser.add_argument("--models", nargs="+", default=["n", "s"], choices=["n", "s"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=8)
    # The first 10 layers are YOLOv8's backbone (general picture
    # features from COCO) - a few hundred pictures should only retune
    # the parts that decide "banana", or it forgets more than it learns.
    parser.add_argument("--freeze", type=int, default=10,
                        help="layers kept as they are (0 = train everything)")
    parser.add_argument("--device", default=None, help="mps / cpu / 0 (default: best available)")
    return parser.parse_args()


def check_reviewed(flights):
    """Warns about pictures review.py hasn't been through - their boxes
    are the teacher's guesses."""
    for flight in flights:
        images = image_paths(flight)
        unchecked = len({p.name for p in images} - reviewed_names(flight))
        if unchecked:
            print(f"WARNING: {flight.name}: {unchecked}/{len(images)} pictures not "
                  f"checked in review.py")


def copy_as_coco(flights, split, class_id):
    """Copies the flights' pictures to _runs/coco/<split>/, with the
    labels' banana class 0 renumbered to COCO's."""
    out = RUNS_DIR / "coco" / split
    shutil.rmtree(out, ignore_errors=True)
    (out / "images").mkdir(parents=True)
    (out / "labels").mkdir()
    for flight in flights:
        for image in image_paths(flight):
            name = f"{flight.name}_{image.name}"
            shutil.copy2(image, out / "images" / name)
            lines = label_path(image).read_text().split("\n") if label_path(image).exists() else []
            (out / "labels" / name).with_suffix(".txt").write_text("".join(
                f"{class_id} {line.split(' ', 1)[1]}\n" for line in lines if line.strip()))
    return out / "images"


def write_data_yaml(train, val):
    coco = YOLO(str(COCO_WEIGHTS["n"]))
    class_id = banana_class_id(coco)
    data = {
        "train": str(copy_as_coco(train, "train", class_id)),
        "val": str(copy_as_coco(val, "val", class_id)),
        "names": coco.names,
    }
    path = RUNS_DIR / "data.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def train_one(size, data_yaml, args, device):
    start = time.time()
    model = YOLO(str(COCO_WEIGHTS[size]))
    model.train(
        data=str(data_yaml),
        imgsz=FLIGHT_IMG_SIZE,
        epochs=args.epochs,
        patience=15,
        batch=args.batch,
        freeze=args.freeze,
        device=device,
        project=str(RUNS_DIR),
        name=f"banana_{size}",
        exist_ok=True,
        cache="ram",
        # The Tello's pictures swing dark/bright and tilt as it flies
        hsv_v=0.5,
        degrees=5,
        close_mosaic=10,
        plots=False,
    )
    best = RUNS_DIR / f"banana_{size}" / "weights" / "best.pt"

    FINETUNED_DIR.mkdir(exist_ok=True)
    out = FINETUNED_DIR / f"banana_{size}.pt"
    if out.exists():
        shutil.copy2(out, FINETUNED_DIR / f"banana_{size}.prev.pt")
    shutil.copy2(best, out)
    print(f"\nYOLOv8{size}: {(time.time() - start) / 60:.1f} min -> {out}\n")


def main():
    args = parse_args()
    val = flight_dirs(args.val)
    train = flight_dirs(args.train) if args.train else [
        f for f in flight_dirs() if f.name not in args.val]
    if not train:
        raise SystemExit("No flights left to train on - --val holds out all of them")
    print("Train:", ", ".join(f.name for f in train))
    print("Val:  ", ", ".join(f.name for f in val))
    check_reviewed(train + val)

    device = args.device or pick_device()
    data_yaml = write_data_yaml(train, val)
    for size in args.models:
        train_one(size, data_yaml, args, device)
    print("Next: python BananaModel/finetune/compare.py --flights", " ".join(args.val))


if __name__ == "__main__":
    main()
