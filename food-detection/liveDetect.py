"""
Live banana detection + ripeness classification.

Uses a pretrained YOLOv8 (COCO weights) to LOCATE bananas in the frame,
then runs each detected crop through YOUR trained ripeness classifier.

Usage:
    python live_detect_classify.py
    python live_detect_classify.py --source 0          # webcam (default)
    python live_detect_classify.py --source video.mp4  # video file
"""

import argparse
import json
import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from ultralytics import YOLO

# ---- Config (match your training setup) ----
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = 224
CHECKPOINT_PATH = 'checkpoints/best_banana_model.pth'
CLASS_NAMES_PATH = 'checkpoints/class_names.json'
DEFAULT_CLASS_NAMES = ['freshripe', 'freshunripe', 'overripe', 'ripe', 'rotten', 'unripe']

COCO_BANANA_CLASS_ID = 46          # "banana" class index in COCO
DETECTOR_CONF_THRESHOLD = 0.4      # min confidence for the detector to count a box as "banana"
CLASSIFIER_CONF_THRESHOLD = 0.0    # set >0 (e.g. 0.5) to hide low-confidence ripeness labels
PADDING_RATIO = 0.08               # expand crop slightly beyond the box so the whole banana is included


def load_class_names():
    try:
        with open(CLASS_NAMES_PATH) as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: {CLASS_NAMES_PATH} not found, using default class list.")
        return DEFAULT_CLASS_NAMES


def build_classifier(num_classes):
    model = models.mobilenet_v2(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(model.last_channel, num_classes)
    )
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='0', help="Webcam index (e.g. 0) or path to a video file")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source

    # ---- Load your ripeness classifier ----
    class_names = load_class_names()
    classifier = build_classifier(len(class_names)).to(DEVICE)
    classifier.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    classifier.eval()

    classify_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # ---- Load pretrained banana detector (COCO weights, not your data) ----
    detector = YOLO('yolov8n.pt')  # auto-downloads on first run

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Could not open video source: {source}")
        return

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        cv2.flip(frame, 1)
        if not ret:
            break

        h, w = frame.shape[:2]

        # 1) Detect bananas in the frame
        results = detector(frame, conf=DETECTOR_CONF_THRESHOLD, classes=[COCO_BANANA_CLASS_ID], verbose=False)

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            det_conf = float(box.conf[0])

            # Expand box slightly so the crop isn't too tight
            pad_x = int((x2 - x1) * PADDING_RATIO)
            pad_y = int((y2 - y1) * PADDING_RATIO)
            cx1, cy1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
            cx2, cy2 = min(w, x2 + pad_x), min(h, y2 + pad_y)

            crop = frame[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                continue

            # 2) Classify ripeness on the cropped banana
            img_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            input_tensor = classify_transform(pil_img).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                output = classifier(input_tensor)
                probs = torch.softmax(output, dim=1)[0]
                cls_conf, pred_idx = torch.max(probs, 0)

            cls_conf = cls_conf.item()
            if cls_conf < CLASSIFIER_CONF_THRESHOLD:
                continue

            label = f"{class_names[pred_idx.item()]} {cls_conf:.0%} (det {det_conf:.0%})"

            # 3) Draw box + label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow('Banana Ripeness Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()