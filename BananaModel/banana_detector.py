"""
Reusable banana detection + ripeness classification model.

Wraps a pretrained YOLOv8 (COCO weights) for LOCATING bananas, and your
trained classifier for RIPENESS classification on each detected crop.

Designed to be reused across different input sources (webcam, Tello
video feed, PyBullet simulation renders, etc.) — just call `detect()`
on whatever BGR frame you have.
"""

import json
import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from ultralytics import YOLO
import os
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), 'checkpoints')


def banana_class_id(yolo):
    """The "banana" class index of a loaded YOLO model, read from its own
    class names rather than assumed - 46 in COCO (and in our fine-tuned
    weights, see finetune/train_detector.py)."""
    for class_id, name in yolo.names.items():
        if name == 'banana':
            return class_id
    raise ValueError(f"YOLO model has no 'banana' class: {yolo.names}")


# The ripeness classifier barely looks at colour: a yellow banana photo
# recoloured green still scores 100% freshripe, and on the Tello's dark,
# blurry frames a green banana gets voted rotten. So peel colour gets
# the final say on green bananas. OpenCV hue is 0-180: a yellow peel
# measures 16-25, green starts around 35.
GREEN_HUE_RANGE = (33, 85)
PEEL_MIN_SATURATION = 60  # below this it's background/grey, not peel
PEEL_MIN_VALUE = 40       # below this it's shadow (or rotten black)


class Detection:
    """A single detected + classified banana."""

    def __init__(self, box, det_conf, label, cls_conf):
        self.box = box              # (x1, y1, x2, y2)
        self.det_conf = det_conf    # detector confidence
        self.label = label          # predicted ripeness class name
        self.cls_conf = cls_conf    # classifier confidence

    def __repr__(self):
        return f"Detection(box={self.box}, label={self.label}, cls_conf={self.cls_conf:.2f}, det_conf={self.det_conf:.2f})"


class BananaDetector:
    def __init__(
        self,
        checkpoint_path=os.path.join(CHECKPOINT_DIR, 'best_banana_model.pth'),
        class_names_path=os.path.join(CHECKPOINT_DIR, 'class_names.json'),
        default_class_names=None,
        img_size=224,
        detector_conf_threshold=0.15,
        detector_img_size=640,
        classifier_conf_threshold=0.0,
        padding_ratio=0.08,
        unripe_green_fraction=0.5,
        yolo_weights=os.path.join(os.path.dirname(__file__), 'yolov8n.pt'),
        device=None,
    ):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.img_size = img_size
        self.detector_conf_threshold = detector_conf_threshold
        # YOLO's input size. 640 (YOLO's default) shrinks a 960px Tello
        # frame; 960 keeps full detail, so small far-away bananas get
        # found (banana filling 1.5% of the frame: 1/12 -> 9/12 found),
        # at ~2x the time.
        self.detector_img_size = detector_img_size
        self.classifier_conf_threshold = classifier_conf_threshold
        self.padding_ratio = padding_ratio
        # A banana whose peel is at least this fraction green is
        # labelled 'unripe', whatever the classifier says.
        self.unripe_green_fraction = unripe_green_fraction
        self.default_class_names = default_class_names or [
            'freshripe', 'freshunripe', 'overripe', 'ripe', 'rotten', 'unripe'
        ]

        self.class_names = self._load_class_names(class_names_path)
        self.classifier = self._build_classifier(len(self.class_names)).to(self.device)
        self.classifier.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.classifier.eval()

        self.classify_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.detector = YOLO(yolo_weights)  # auto-downloads on first run
        self.banana_class_id = banana_class_id(self.detector)

    def _load_class_names(self, class_names_path):
        try:
            with open(class_names_path) as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: {class_names_path} not found, using default class list.")
            return self.default_class_names

    def _build_classifier(self, num_classes):
        model = models.mobilenet_v2(weights=None)
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(model.last_channel, num_classes)
        )
        return model

    def detect(self, frame):
        """
        Run detection + classification on a single BGR frame.

        Returns a list of Detection objects. Does not modify the frame.
        """
        h, w = frame.shape[:2]
        detections = []

        results = self.detector(
            frame,
            conf=self.detector_conf_threshold,
            imgsz=self.detector_img_size,
            classes=[self.banana_class_id],
            verbose=False
        )

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            det_conf = float(box.conf[0])

            pad_x = int((x2 - x1) * self.padding_ratio)
            pad_y = int((y2 - y1) * self.padding_ratio)
            cx1, cy1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
            cx2, cy2 = min(w, x2 + pad_x), min(h, y2 + pad_y)

            crop = frame[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                continue

            img_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            input_tensor = self.classify_transform(pil_img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.classifier(input_tensor)
                probs = torch.softmax(output, dim=1)[0]
                cls_conf, pred_idx = torch.max(probs, 0)

            cls_conf = cls_conf.item()
            if cls_conf < self.classifier_conf_threshold:
                continue

            label = self.class_names[pred_idx.item()]
            green_fraction = self._peel_green_fraction(frame[y1:y2, x1:x2])
            if green_fraction >= self.unripe_green_fraction:
                label, cls_conf = 'unripe', green_fraction
            detections.append(Detection((x1, y1, x2, y2), det_conf, label, cls_conf))

        return detections

    def _peel_green_fraction(self, box_img):
        """Fraction of the coloured (peel) pixels in the middle of the
        box that are green. Only the middle half is used - a curved
        banana leaves the box corners as background."""
        h, w = box_img.shape[:2]
        middle = box_img[h // 4: h - h // 4, w // 4: w - w // 4]
        if middle.size == 0:
            return 0.0
        hsv = cv2.cvtColor(middle, cv2.COLOR_BGR2HSV)
        peel = (hsv[..., 1] > PEEL_MIN_SATURATION) & (hsv[..., 2] > PEEL_MIN_VALUE)
        if not peel.any():
            return 0.0
        hue = hsv[..., 0][peel]
        green = (hue >= GREEN_HUE_RANGE[0]) & (hue <= GREEN_HUE_RANGE[1])
        return float(green.mean())

    def annotate(self, frame, detections):
        """
        Draw boxes + labels for the given detections onto frame (in place)
        and return it, for convenience.
        """
        for d in detections:
            x1, y1, x2, y2 = d.box
            text = f"{d.label} {d.cls_conf:.0%} (det {d.det_conf:.0%})"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, text, (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return frame

    def detect_and_annotate(self, frame):
        """Convenience wrapper: detect() + annotate() in one call."""
        detections = self.detect(frame)
        self.annotate(frame, detections)
        return frame, detections