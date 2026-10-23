"""
Step 2: check and fix autolabel.py's banana boxes, one picture at a time.

    python BananaModel/finetune/review.py flight_20261022_101500

Mouse:
    left-drag   = draw a box around a banana the teacher missed
    right-click = delete the box under the cursor (not a banana)
Keys:
    space / n   = picture is right, next
    b           = back one picture
    z           = undo the last box drawn
    c           = clear all boxes (no banana here)
    x           = drop this picture from training (too blurry to tell)
    q           = quit (everything is saved as you go)

Box every banana you can recognise, however small - those far-away
ones are the whole point. Starts at the first picture not yet checked,
so it can be stopped and picked up again.
"""

import argparse
import shutil

import cv2

from common import flight_dirs, image_paths, label_path, read_boxes, reviewed_names, write_boxes

WINDOW = "review - space next, b back, right-click delete, drag add, x drop, q quit"

# cv2.waitKeyEx() codes for the arrow keys on macOS / Linux / Windows
LEFT_KEYS = {63234, 65361, 2424832}
RIGHT_KEYS = {63235, 65363, 2555904}


class Review:

    def __init__(self, flight_dir):
        self.flight_dir = flight_dir
        self.images = image_paths(flight_dir)
        self.reviewed = reviewed_names(flight_dir)
        self.index = next((i for i, p in enumerate(self.images)
                           if p.name not in self.reviewed), 0)
        self.drag_start = None
        self.drag_end = None
        self.load()

    def load(self):
        self.frame = cv2.imread(str(self.images[self.index]))
        self.height, self.width = self.frame.shape[:2]
        self.boxes = read_boxes(self.images[self.index], self.width, self.height)

    def save(self):
        write_boxes(self.images[self.index], self.boxes, self.width, self.height)

    def mark_reviewed(self):
        self.reviewed.add(self.images[self.index].name)
        (self.flight_dir / "reviewed.txt").write_text("\n".join(sorted(self.reviewed)) + "\n")

    def go(self, step):
        """Returns False past the last picture."""
        self.index += step
        if self.index >= len(self.images):
            return False
        self.index = max(0, self.index)
        self.load()
        return True

    def drop(self):
        """Moves the picture and its labels to rejected/."""
        image = self.images.pop(self.index)
        rejected = self.flight_dir / "rejected"
        rejected.mkdir(exist_ok=True)
        shutil.move(str(image), rejected / image.name)
        if label_path(image).exists():
            label_path(image).unlink()
        if not self.images:
            return False
        self.index = min(self.index, len(self.images) - 1)
        self.load()
        return True

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_start = self.drag_end = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.drag_start:
            self.drag_end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.drag_start:
            (x1, y1), (x2, y2) = self.drag_start, (x, y)
            self.drag_start = None
            if abs(x2 - x1) > 3 and abs(y2 - y1) > 3:
                self.boxes.append((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)))
                self.save()
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Smallest box under the cursor, so a small box inside a big one can be picked
            under = [b for b in self.boxes if b[0] <= x <= b[2] and b[1] <= y <= b[3]]
            if under:
                self.boxes.remove(min(under, key=lambda b: (b[2] - b[0]) * (b[3] - b[1])))
                self.save()

    def draw(self):
        view = self.frame.copy()
        for x1, y1, x2, y2 in self.boxes:
            cv2.rectangle(view, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        if self.drag_start:
            cv2.rectangle(view, self.drag_start, self.drag_end, (0, 200, 255), 2)
        name = self.images[self.index].name
        done = "checked" if name in self.reviewed else "NOT CHECKED"
        text = (f"{self.index + 1}/{len(self.images)}  {name}  {len(self.boxes)} banana(s)  "
                f"{done}  ({len(self.reviewed & {p.name for p in self.images})} checked)")
        cv2.putText(view, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(view, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imshow(WINDOW, view)

    def run(self):
        cv2.namedWindow(WINDOW)
        cv2.setMouseCallback(WINDOW, self.on_mouse)
        while True:
            self.draw()
            key = cv2.waitKeyEx(20)
            if key == -1:
                continue
            char = chr(key & 0xFF) if key < 256 else ""
            if char in (" ", "n") or key in RIGHT_KEYS:
                self.mark_reviewed()
                if not self.go(1):
                    print("Last picture checked.")
                    break
            elif char == "b" or key in LEFT_KEYS:
                self.go(-1)
            elif char == "z" and self.boxes:
                self.boxes.pop()
                self.save()
            elif char == "c":
                self.boxes = []
                self.save()
            elif char == "x":
                if not self.drop():
                    break
            elif char == "q":
                break
        cv2.destroyAllWindows()
        left = len([p for p in self.images if p.name not in self.reviewed])
        print(f"{self.flight_dir.name}: {len(self.images) - left}/{len(self.images)} checked")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("flight", help="folder name in BananaModel/finetune_data/")
    args = parser.parse_args()
    flight_dir = flight_dirs([args.flight])[0]
    if not image_paths(flight_dir):
        raise SystemExit(f"No pictures in {flight_dir}")
    Review(flight_dir).run()


if __name__ == "__main__":
    main()
