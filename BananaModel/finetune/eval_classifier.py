"""
Score the ripeness classifier on held-out flights, before vs after
fine-tuning, without training anything - reprints what
train_classifier.py printed, plus per-class recall / precision / F1.

    python BananaModel/finetune/eval_classifier.py --val flight_20261022_103000

"Before" is checkpoints/best_banana_model.pth, "after" is
finetuned/banana_classifier.pth. Classifier only: the green-peel
check banana_detector.py applies on the drone isn't included.
"""

import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models

from common import CLASSIFIER_WEIGHTS, FINETUNED_DIR, RIPENESS_CLASSES, flight_dirs, pick_device
from train_classifier import CropDataset, class_counts, load_boxes, predict, print_score


def load_classifier(path, device):
    model = models.mobilenet_v2(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(model.last_channel, len(RIPENESS_CLASSES))
    )
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model.to(device)


def print_per_class(preds, truth):
    print(f"{'class':<15}{'n':>5}{'recall':>9}{'precision':>11}{'F1':>8}")
    f1s = []
    for i, name in enumerate(RIPENESS_CLASSES):
        n = sum(t == i for t in truth)
        picked = sum(p == i for p in preds)
        if n == 0 and picked == 0:
            continue
        hit = sum(p == t == i for p, t in zip(preds, truth))
        recall = hit / n if n else 0.0
        precision = hit / picked if picked else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if n:
            f1s.append(f1)
        print(f"{name:<15}{n:>5}{recall:>9.1%}{precision:>11.1%}{f1:>8.1%}")
    print(f"{'macro F1':<15}{'':>25}{sum(f1s) / max(1, len(f1s)):>8.1%}  (classes in this flight)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val", nargs="+", required=True,
                        help="held-out flight folder name(s) - the same --val as training")
    parser.add_argument("--device", default=None, help="mps / cpu / cuda (default: best available)")
    args = parser.parse_args()

    after = FINETUNED_DIR / "banana_classifier.pth"
    if not after.exists():
        raise SystemExit(f"No fine-tuned classifier at {after}")

    items = load_boxes(flight_dirs(args.val))
    if not items:
        raise SystemExit("No ripeness labels in these flights - check the --val names")
    print("Held out:", ", ".join(args.val), class_counts(items))

    device = args.device or pick_device()
    device = torch.device("cuda" if device == 0 else device)
    loader = DataLoader(CropDataset(items, train=False), batch_size=64)

    for title, path in [("BEFORE fine-tuning", CLASSIFIER_WEIGHTS), ("AFTER fine-tuning", after)]:
        preds, truth = predict(load_classifier(path, device), loader, device)
        print_score(f"Held-out flight, {title}", preds, truth)
        print()
        print_per_class(preds, truth)


if __name__ == "__main__":
    main()
