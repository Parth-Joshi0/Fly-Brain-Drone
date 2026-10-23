"""
Step 3b: give every checked banana box its ripeness, for fine-tuning
the ripeness classifier (train_classifier.py). Run after review.py.

    python BananaModel/finetune/label_ripeness.py flight_20261022_101500

Shows one box at a time (yellow), with a zoomed-in view on the right.
Keys:
    1-6     = ripeness (listed on screen), next box
    space   = take the suggestion (highlighted), next box
    s       = skip - can't tell, not used for training
    b       = back one box
    q       = quit (saved as you go, starts where you left off)

The suggestion is what you called the same banana (nearest box) in the
picture before, so a flight with one or two bananas is mostly space.
Label what the banana really is, not what the drone's picture makes it
look like - the point is teaching it to see through the dark, blurry
camera.
"""

import argparse

import cv2
import numpy as np

from common import (RIPENESS_CLASSES, SKIP, flight_dirs, image_paths, line_to_box,
                    read_label_lines, read_ripeness, reviewed_names, write_ripeness)

WINDOW = "ripeness - 1-6 label, space take suggestion, s skip, b back, q quit"
PANEL_WIDTH = 360

# A box further than this (fraction of picture width) from every box in
# the picture before is a different banana - no suggestion from them
SAME_BANANA_DISTANCE = 0.25


def centre(box):
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


class RipenessLabeller:

    def __init__(self, flight_dir):
        self.flight_dir = flight_dir
        self.ripeness = read_ripeness(flight_dir)
        # One entry per box: (picture, label line)
        self.boxes = [(image, line) for image in image_paths(flight_dir)
                      for line in read_label_lines(image)]
        self.index = next((i for i, (image, line) in enumerate(self.boxes)
                           if self.label_of(image, line) is None), len(self.boxes) - 1)
        self.frames = {}

    def label_of(self, image, line):
        return self.ripeness.get(image.name, {}).get(line)

    def set_label(self, label):
        image, line = self.boxes[self.index]
        self.ripeness.setdefault(image.name, {})[line] = label
        write_ripeness(self.flight_dir, self.ripeness)

    def frame(self, image):
        if image not in self.frames:
            self.frames = {image: cv2.imread(str(image))}
        return self.frames[image]

    def suggestion(self):
        """The label of the nearest box in the closest earlier picture
        with labels, if near enough to be the same banana."""
        image, line = self.boxes[self.index]
        frame = self.frame(image)
        h, w = frame.shape[:2]
        cx, cy = centre(line_to_box(line, w, h))
        earlier = [(i, l) for i, l in self.boxes[:self.index]
                   if i != image and self.label_of(i, l) not in (None, SKIP)]
        if not earlier:
            return None
        last_image = earlier[-1][0]
        candidates = [(np.hypot(*np.subtract(centre(line_to_box(l, w, h)), (cx, cy))) / w,
                       self.label_of(i, l)) for i, l in earlier if i == last_image]
        distance, label = min(candidates)
        return label if distance < SAME_BANANA_DISTANCE else None

    def draw(self, suggestion):
        image, line = self.boxes[self.index]
        view = self.frame(image).copy()
        h, w = view.shape[:2]
        for other in read_label_lines(image):
            if other != line:
                x1, y1, x2, y2 = map(int, line_to_box(other, w, h))
                cv2.rectangle(view, (x1, y1), (x2, y2), (150, 150, 150), 1)
        box = line_to_box(line, w, h)
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(view, (x1, y1), (x2, y2), (0, 255, 255), 3)

        panel = np.zeros((h, PANEL_WIDTH, 3), np.uint8)
        crop = self.frame(image)[max(0, y1):y2, max(0, x1):x2]
        if crop.size:
            scale = min((PANEL_WIDTH - 20) / crop.shape[1], 300 / crop.shape[0])
            crop = cv2.resize(crop, None, fx=scale, fy=scale)
            panel[10:10 + crop.shape[0], 10:10 + crop.shape[1]] = crop

        current = self.label_of(image, line)
        y = 340
        for key, name in enumerate(RIPENESS_CLASSES, start=1):
            colour = (255, 255, 255)
            if name == suggestion:
                colour = (0, 255, 255)
            if name == current:
                colour = (0, 255, 0)
            mark = "  <- labelled" if name == current else ("  <- space" if name == suggestion else "")
            cv2.putText(panel, f"{key}  {name}{mark}", (15, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, colour, 2)
            y += 36
        cv2.putText(panel, "s  skip (can't tell)" + ("  <- labelled" if current == SKIP else ""),
                    (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (0, 255, 0) if current == SKIP else (180, 180, 180), 2)

        done = sum(1 for i, l in self.boxes if self.label_of(i, l) is not None)
        text = f"box {self.index + 1}/{len(self.boxes)}  {image.name}  ({done} labelled)"
        cv2.putText(view, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(view, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imshow(WINDOW, np.hstack([view, panel]))

    def run(self):
        cv2.namedWindow(WINDOW)
        while True:
            suggestion = self.suggestion()
            self.draw(suggestion)
            key = cv2.waitKey(0) & 0xFF
            char = chr(key)
            label = None
            if char in "123456":
                label = RIPENESS_CLASSES[int(char) - 1]
            elif char == " " and suggestion:
                label = suggestion
            elif char == "s":
                label = SKIP
            elif char == "b":
                self.index = max(0, self.index - 1)
            elif char == "q":
                break
            if label:
                self.set_label(label)
                if self.index == len(self.boxes) - 1:
                    print("Last box labelled.")
                    break
                self.index += 1
        cv2.destroyAllWindows()
        self.print_summary()

    def print_summary(self):
        labels = [self.label_of(i, l) for i, l in self.boxes]
        counts = {name: labels.count(name) for name in RIPENESS_CLASSES + [SKIP] if labels.count(name)}
        print(f"{self.flight_dir.name}: {len(labels) - labels.count(None)}/{len(labels)} "
              f"boxes labelled {counts}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("flight", help="folder name in BananaModel/finetune_data/")
    args = parser.parse_args()
    flight_dir = flight_dirs([args.flight])[0]

    images = image_paths(flight_dir)
    unchecked = len({p.name for p in images} - reviewed_names(flight_dir))
    if unchecked:
        print(f"WARNING: {unchecked}/{len(images)} pictures not checked in review.py - "
              f"boxes changed there later lose their ripeness label")

    labeller = RipenessLabeller(flight_dir)
    if not labeller.boxes:
        raise SystemExit(f"No banana boxes in {flight_dir}")
    labeller.run()


if __name__ == "__main__":
    main()
