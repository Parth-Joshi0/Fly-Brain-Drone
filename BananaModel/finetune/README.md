# Fine-tuning the banana detector on our own flight pictures

COCO's YOLOv8 has seen bananas in kitchen photos, not on a dark, blurry,
tilting Tello camera from across a room. This retrains it on pictures
from our own flights, in the place we film.

```
fly_tello.py --save-frames   ->  autolabel.py  ->  review.py  ->  train_detector.py  ->  compare.py  ->  fly_tello.py --finetuned
(2 pictures/s, clean)            (big YOLO          (fix the       (nano + small,         (better than
                                  drafts boxes)      boxes)         ~50 epochs)            COCO?)
```

Run everything from the repo root, in `venv`.

## 1. Fly and save pictures

```
python Drone/fly_tello.py --fly --save-frames
```

Saves the clean camera picture (no boxes / HUD) twice a second to
`Drone/flight_frames/flight_<time>/`. Fly it the way the real take
flies: far from the banana, scanning, approaching, eating, hands
around it. **Do at least 2 flights** - one is held out for scoring
(step 4), and it can't be one the model trained on.

Pictures with no banana in them matter too: they teach it what the
wall, floor and your hands are *not*.

## 2. Draft boxes

```
python BananaModel/finetune/autolabel.py Drone/flight_frames/flight_<time> [more flights...]
```

YOLO11x (110 MB, downloads itself the first time - do that at home)
looks at each picture and 4 zoomed-in quarters of it. Hover pictures
that barely changed are skipped. ~2 s per picture on the M2 Air.

## 3. Check every picture

```
python BananaModel/finetune/review.py flight_<time>
```

| | |
|---|---|
| left-drag | add a box (a banana it missed) |
| right-click | delete a box (not a banana) |
| space / n | right, next picture |
| b | back |
| z | undo last box |
| c | clear boxes (no banana here) |
| x | drop the picture (too blurry to tell) |
| q | quit - saved as you go, starts where you left off |

Box every banana you can recognise, however small - the far-away ones
are what COCO misses and what this is for. A banana the teacher
missed and you didn't box is taught as "not a banana".

## 4. Train

```
python BananaModel/finetune/train_detector.py --val flight_<held-out time>
```

Trains nano and small at 960px from the COCO weights (`--models n` for
just one). Writes `BananaModel/finetuned/banana_n.pt` and `banana_s.pt`
(the previous ones become `*.prev.pt`). Warns about pictures not yet
checked in review.py.

Rough times for ~300 pictures, 50 epochs (it stops early once it stops
improving): M4 Pro ~15 min nano + ~25 min small; M2 Air about double.

If MPS training misbehaves (crashes, or losses go `nan`), add
`--device cpu` (slow) or train on Colab (copy the repo plus
`BananaModel/finetune_data/` over, then run the same command; ~5-10
min per model on a T4, then copy `BananaModel/finetuned/` back).

## 5. Did it get better?

```
python BananaModel/finetune/compare.py --flights flight_<held-out time>
```

```
model                                  found       far (<1%)         wrong boxes  ms/pic
YOLOv8n COCO @0.20                ...
YOLOv8n fine-tuned @0.20          ...
```

Same picture size and sureness bars as `fly_tello.py`. Fly with it only
if it finds more bananas **without** more wrong boxes - a wrong box on
the wall sends the drone to eat the wall. Also check `ms/pic` hasn't
grown: the loop needs ~15 pictures/s to catch hand swats.

## 6. Fly with it

```
python Drone/fly_tello.py --fly --finetuned [--scared ...]
```

Without `--finetuned` it's the COCO detectors as before, so going back
is just dropping the flag.

## Notes

- **Why COCO's 80 classes, not a new 1-class model:** a new model's
  last layer starts from scratch, and a few hundred pictures are too
  few to relearn "banana" (in a test it found 0/8 after 30 epochs; kept
  as COCO class 46 it started at 0.94 mAP50). The other 79 classes fade,
  which nothing here uses.
- **Why hold out a whole flight:** pictures half a second apart are
  near-copies, so a random split scores it on pictures it trained on.
- Pictures, labels and weights stay on your computer (gitignored):
  `Drone/flight_frames/`, `BananaModel/finetune_data/`,
  `BananaModel/finetuned/`.
