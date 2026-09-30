# Banana Detection Model

Finds bananas in a camera frame and classifies how ripe each one is.
Used by the drone (`Drone/tello_camera.py`) to find food for the
fly-inspired "hover → eat" behaviour in `NeuralPathways/FoodNeuron/food_orbit.py`.

## How it works

Two models run one after the other on every frame:

```
camera frame (BGR)
   │
   ▼
1. YOLOv8n (COCO)  ──►  box around each banana
   │
   ▼  crop each box (+8% padding), resize to 224×224
2. MobileNetV2     ──►  ripeness label + confidence
   │
   ▼
list of Detection(box, det_conf, label, cls_conf)
```

| Stage | Model | Job | Weights |
|---|---|---|---|
| 1. Locate | YOLOv8n (Ultralytics) | Find bananas (COCO class 46) | `yolov8n.pt`, pretrained on COCO, not retrained |
| 2. Classify | MobileNetV2 (torchvision) | Pick 1 of 6 ripeness classes | `checkpoints/best_banana_model.pth`, trained by us |

YOLO is only used to **find** the banana; our trained classifier
decides **how ripe** it is.

## Ripeness classes

In this order (alphabetical, from the dataset folder names; the
model's outputs follow this order):

| # | Class |
|---|---|
| 0 | `freshripe` |
| 1 | `freshunripe` |
| 2 | `overripe` |
| 3 | `ripe` |
| 4 | `rotten` |
| 5 | `unripe` |

## Dataset

Image folders in `dataset/{train,valid,test}/<class>/`. The source
of the images isn't recorded in the repo (`.gitignore` lists a
`README.roboflow.txt`, so it was probably exported from Roboflow).

| Class | Train | Valid | Test |
|---|---:|---:|---:|
| freshripe | 704 | 207 | 102 |
| freshunripe | 501 | 136 | 83 |
| overripe | 783 | 229 | 113 |
| ripe | 470 | 132 | 52 |
| rotten | 1,340 | 388 | 185 |
| unripe | 133 | 31 | 27 |
| **Total** | **3,931** | **1,123** | **562** |

The classes are unbalanced: `rotten` has 10× more training images
than `unripe`.

## Training

Transfer learning: start from MobileNetV2 pretrained on ImageNet,
freeze the feature layers, and train only a new final layer.

| Setting | Value |
|---|---|
| Base model | MobileNetV2, ImageNet weights (`IMAGENET1K_V1`) |
| Frozen | All feature layers (only the classifier head trains) |
| Head | Dropout(0.2) → Linear(1280 → 6) |
| Input | 224×224, ImageNet mean/std normalisation |
| Augmentation (train only) | Horizontal flip, rotation ±15°, colour jitter (brightness/contrast/saturation 0.2) |
| Loss | Cross-entropy |
| Optimiser | Adam, learning rate 1e-3 |
| Batch size | 32 |
| Epochs | Up to 50, early stopping after 10 epochs without validation improvement |
| Saved checkpoint | Best validation accuracy |
| Parameters | 2.23 M |

Train it (from inside `BananaModel/`):

```bash
python train.py
```

Settings live in `config.py`.

## Results

Measured on the held-out **test** set (562 images, never used in
training):

**Overall accuracy: 93.6% (526 / 562)**

| Class | Correct | Accuracy |
|---|---:|---:|
| freshripe | 101 / 102 | 99% |
| freshunripe | 83 / 83 | 100% |
| overripe | 107 / 113 | 95% |
| ripe | 41 / 52 | 79% |
| rotten | 169 / 185 | 91% |
| unripe | 25 / 27 | 93% |

Confusion matrix (rows = true class, columns = predicted):

| | freshripe | freshunripe | overripe | ripe | rotten | unripe |
|---|---:|---:|---:|---:|---:|---:|
| **freshripe** | 101 | 0 | 0 | 1 | 0 | 0 |
| **freshunripe** | 0 | 83 | 0 | 0 | 0 | 0 |
| **overripe** | 0 | 0 | 107 | 6 | 0 | 0 |
| **ripe** | 9 | 0 | 0 | 41 | 2 | 0 |
| **rotten** | 4 | 0 | 6 | 2 | 169 | 4 |
| **unripe** | 0 | 0 | 0 | 0 | 2 | 25 |

Most mistakes are between neighbouring ripeness levels, e.g.
`ripe` ↔ `freshripe` and `overripe` ↔ `ripe`.

**Speed:** about 19 ms per image for the classifier on a Mac CPU.
The full pipeline (YOLO + classifier) runs at about 22 frames per
second on the live Tello feed.

## Using it in code

```python
from BananaModel.liveDetect import BananaDetector

detector = BananaDetector()

detections = detector.detect(frame)           # frame = BGR image (OpenCV)
frame, detections = detector.detect_and_annotate(frame)   # also draws boxes

for d in detections:
    print(d.box, d.label, d.cls_conf, d.det_conf)
```

Each `Detection` has:

| Field | Meaning |
|---|---|
| `box` | `(x1, y1, x2, y2)` in pixels |
| `det_conf` | YOLO's confidence that it's a banana |
| `label` | Ripeness class name |
| `cls_conf` | Classifier's confidence in that label |

`BananaDetector` options: `detector_conf_threshold` (default 0.4),
`classifier_conf_threshold` (default 0.0), `padding_ratio` (default
0.08), `device` (defaults to CUDA if available, otherwise CPU).

## Files

| File | What it does |
|---|---|
| `liveDetect.py` | `BananaDetector`: the full two-stage pipeline used by the drone |
| `model.py` | Builds the MobileNetV2 classifier |
| `dataset.py` | Loads images and applies augmentation |
| `train.py` | Training loop with early stopping |
| `config.py` | Training settings and paths |
| `test.py` | Classify a single image |
| `interface.py` | Live webcam demo of the classifier |
| `checkpoints/best_banana_model.pth` | Trained classifier weights |
| `yolov8n.pt` | YOLOv8 nano weights (COCO) |

## Known limitations

- **Labels flicker on a real drone.** In flight tests the same banana
  switched between `rotten` and `freshripe` from frame to frame, often
  with high confidence. The training images are close-up, well-lit
  photos, unlike the drone's view (farther away, moving, compressed
  video). Because of this, the drone behaviour now treats **any**
  banana as food instead of relying on the ripeness label.
- **`checkpoints/class_names.json` is missing.** `liveDetect.py` falls
  back to the built-in class list, which does match the training order
  above. Saving `class_names.json` from `train.py` would make this
  explicit.
- **`ripe` is the weakest class** (79%), and `unripe` has very few
  training images (133).
- **YOLO is not fine-tuned.** It uses the stock COCO "banana" class, so
  very close-up, partly visible or unusual-looking bananas can be
  missed.
