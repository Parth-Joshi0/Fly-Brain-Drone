"""
Test the trained banana ripeness model on a single image.

Usage:
    python test_image.py path/to/image.jpg
"""

import sys
import json
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# ---- Config (match your training setup) ----
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = 224
CHECKPOINT_PATH = 'checkpoints/best_banana_model.pth'
CLASS_NAMES_PATH = 'checkpoints/class_names.json'

# Fallback class list if class_names.json doesn't exist
DEFAULT_CLASS_NAMES = ['freshripe', 'freshunripe', 'overripe', 'ripe', 'rotten', 'unripe']


def load_class_names():
    try:
        with open(CLASS_NAMES_PATH) as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: {CLASS_NAMES_PATH} not found, using default class list.")
        return DEFAULT_CLASS_NAMES


def build_model(num_classes):
    model = models.mobilenet_v2(weights=None)  # weights loaded from checkpoint, not ImageNet
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(model.last_channel, num_classes)
    )
    return model


def main():
    if len(sys.argv) != 2:
        print("Usage: python test_image.py path/to/image.jpg")
        sys.exit(1)

    image_path = sys.argv[1]

    class_names = load_class_names()

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    model = build_model(len(class_names)).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()

    img = Image.open(image_path).convert('RGB')
    input_tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)[0]
        conf, pred = torch.max(probs, 0)

    print(f"\nImage: {image_path}")
    print(f"Predicted class: {class_names[pred.item()]}")
    print(f"Confidence: {conf.item():.2%}\n")

    print("All class probabilities:")
    for name, p in sorted(zip(class_names, probs.tolist()), key=lambda x: -x[1]):
        print(f"  {name:15s} {p:.2%}")


if __name__ == '__main__':
    main()