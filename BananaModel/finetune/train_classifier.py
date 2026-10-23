"""
Step 4b: fine-tune the ripeness classifier (MobileNetV2) on the flight
pictures' banana boxes, labelled in label_ripeness.py.

    python BananaModel/finetune/train_classifier.py --val flight_20261022_103000

Starts from checkpoints/best_banana_model.pth (trained on close-up,
well-lit photos) and teaches it what bananas look like through the
Tello's camera. --val holds out whole flights for scoring, like
train_detector.py; it prints the held-out score before and after.

Writes BananaModel/finetuned/banana_classifier.pth (the previous one is
kept as .prev.pth), which `fly_tello.py --finetuned-classifier` uses.

Only the last few MobileNetV2 layers and the final layer train: a few
hundred crops of the same couple of bananas would wear the rest down
to "this exact banana". Classes with no flight examples aren't
retrained - after fine-tuning, the model will rarely pick them.
"""

import argparse
import copy
import random
import shutil
import time

import cv2
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from common import (CLASSIFIER_WEIGHTS, FINETUNED_DIR, RIPENESS_CLASSES, classifier_crop,
                    flight_dirs, image_paths, line_to_box, pick_device, read_label_lines,
                    read_ripeness)

IMG_SIZE = 224
NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# Same as BananaDetector.classify_transform - what the drone does
VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    NORMALIZE,
])

# No hue change: colour is what ripeness is
TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.35, contrast=0.3, saturation=0.2),
    transforms.RandomApply([transforms.GaussianBlur(5, sigma=(0.1, 2.0))], p=0.3),
    transforms.ToTensor(),
    NORMALIZE,
])

# In flight the crop comes from YOLO's box, which doesn't sit exactly
# where the hand-checked box does - each side moves up to this much of
# the box's size in training
BOX_JITTER = 0.08


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val", nargs="+", required=True,
                        help="flight folder name(s) to hold out for scoring")
    parser.add_argument("--train", nargs="+",
                        help="flight folder name(s) to train on (default: all others)")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--unfreeze", type=int, default=4,
                        help="last MobileNetV2 feature layers that train (0 = final layer only)")
    parser.add_argument("--device", default=None, help="mps / cpu / cuda (default: best available)")
    return parser.parse_args()


def load_boxes(flights):
    """(picture, pixel box, class index) for every box with a ripeness label."""
    items = []
    for flight in flights:
        ripeness = read_ripeness(flight)
        for image in image_paths(flight):
            labels = ripeness.get(image.name, {})
            lines = [line for line in read_label_lines(image) if labels.get(line) in RIPENESS_CLASSES]
            if not lines:
                continue
            frame = cv2.imread(str(image))
            h, w = frame.shape[:2]
            for line in lines:
                items.append((frame, line_to_box(line, w, h),
                              RIPENESS_CLASSES.index(labels[line])))
    return items


class CropDataset(Dataset):

    def __init__(self, items, train):
        self.items = items
        self.train = train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        frame, (x1, y1, x2, y2), label = self.items[index]
        if self.train:
            bw, bh = x2 - x1, y2 - y1
            x1 += random.uniform(-BOX_JITTER, BOX_JITTER) * bw
            x2 += random.uniform(-BOX_JITTER, BOX_JITTER) * bw
            y1 += random.uniform(-BOX_JITTER, BOX_JITTER) * bh
            y2 += random.uniform(-BOX_JITTER, BOX_JITTER) * bh
        crop = classifier_crop(frame, (x1, y1, x2, y2))
        image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        return (TRAIN_TRANSFORM if self.train else VAL_TRANSFORM)(image), label


def build_classifier(unfreeze):
    """BananaDetector's MobileNetV2, with the current weights, and all
    but the last `unfreeze` feature layers frozen."""
    model = models.mobilenet_v2(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(model.last_channel, len(RIPENESS_CLASSES))
    )
    model.load_state_dict(torch.load(CLASSIFIER_WEIGHTS, map_location="cpu"))
    frozen = len(model.features) - unfreeze
    for param in model.features[:frozen].parameters():
        param.requires_grad = False
    return model, frozen


def predict(model, loader, device):
    model.eval()
    preds, truth = [], []
    with torch.no_grad():
        for images, labels in loader:
            preds += model(images.to(device)).argmax(1).cpu().tolist()
            truth += labels.tolist()
    return preds, truth


def print_score(title, preds, truth):
    correct = sum(p == t for p, t in zip(preds, truth))
    print(f"\n{title}: {correct}/{len(truth)} right ({correct / max(1, len(truth)):.0%})")
    shown = [i for i in range(len(RIPENESS_CLASSES)) if i in truth or i in preds]
    print("true \\ picked".ljust(15) + "".join(f"{RIPENESS_CLASSES[i]:>12}" for i in shown))
    for t in shown:
        if t in truth:
            row = [sum(1 for p, tt in zip(preds, truth) if tt == t and p == i) for i in shown]
            print(f"{RIPENESS_CLASSES[t]:<15}" + "".join(f"{n:>12}" for n in row))


def class_counts(items):
    counts = {}
    for _, _, label in items:
        counts[RIPENESS_CLASSES[label]] = counts.get(RIPENESS_CLASSES[label], 0) + 1
    return counts


def main():
    args = parse_args()
    val_flights = flight_dirs(args.val)
    train_flights = flight_dirs(args.train) if args.train else [
        f for f in flight_dirs() if f.name not in args.val]
    if not train_flights:
        raise SystemExit("No flights left to train on - --val holds out all of them")

    train_items, val_items = load_boxes(train_flights), load_boxes(val_flights)
    print("Train:", ", ".join(f.name for f in train_flights), class_counts(train_items))
    print("Val:  ", ", ".join(f.name for f in val_flights), class_counts(val_items))
    if not train_items or not val_items:
        raise SystemExit("No ripeness labels - run label_ripeness.py on these flights first")
    missing = [c for c in RIPENESS_CLASSES if c not in class_counts(train_items)]
    if missing:
        print(f"No training examples of {', '.join(missing)} - it will rarely pick those")

    device = args.device or pick_device()
    device = torch.device("cuda" if device == 0 else device)
    model, frozen = build_classifier(args.unfreeze)
    model.to(device)
    train_loader = DataLoader(CropDataset(train_items, train=True), batch_size=32, shuffle=True)
    val_loader = DataLoader(CropDataset(val_items, train=False), batch_size=64)

    print_score("Held-out flight, BEFORE fine-tuning", *predict(model, val_loader, device))

    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                                 lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    best_acc, best_state = -1.0, None
    start = time.time()

    for epoch in range(args.epochs):
        model.train()
        # Frozen layers keep the batch statistics they were trained with
        model.features[:frozen].eval()
        total_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += len(labels)

        preds, truth = predict(model, val_loader, device)
        val_acc = sum(p == t for p, t in zip(preds, truth)) / len(truth)
        print(f"Epoch {epoch + 1}/{args.epochs} | train loss {total_loss / total:.3f} "
              f"acc {correct / total:.0%} | held-out acc {val_acc:.0%}")
        if val_acc > best_acc:
            best_acc, best_state = val_acc, copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    print_score("Held-out flight, AFTER fine-tuning (best epoch)", *predict(model, val_loader, device))

    FINETUNED_DIR.mkdir(exist_ok=True)
    out = FINETUNED_DIR / "banana_classifier.pth"
    if out.exists():
        shutil.copy2(out, FINETUNED_DIR / "banana_classifier.prev.pth")
    torch.save({k: v.cpu() for k, v in best_state.items()}, out)
    print(f"\n{(time.time() - start) / 60:.1f} min -> {out}")
    print("Fly with it: python Drone/fly_tello.py --fly --finetuned --finetuned-classifier")


if __name__ == "__main__":
    main()
