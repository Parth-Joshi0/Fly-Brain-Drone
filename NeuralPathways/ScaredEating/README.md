# ScaredEating

The "scared while eating" brain for the real Tello: the drone finds a banana, flies to it and eats — and if something is waved at it while it's holding still, the real fly connectome's escape circuit makes it back straight off, wait until the coast is clear, then come back and finish eating. Joins the two pathways next door:

- **`FoodNeuron/food_orbit.py`** — the eating behaviour (state machine, no Brian2)
- **`EscapeNeuron/fear_brain.py`** — looming → LC4/LPLC2 → DNp01 Giant Fiber → "scared!" (Brian2, via `flybrain_controller.py`)

## Files

| File | What it does |
|---|---|
| `scared_eating_brain.py` | `ScaredEatingBrain`: one `step(picture) -> rc command` call per camera picture. Runs the fly brain on the clean picture first, the banana AI every 3rd picture (so the looming detector sees ~21 pictures/s, fast enough to catch a quick hand), then the eating behaviour, and feeds the command back to the fly brain as its efference copy. Without `scared=True` it's just the eating behaviour. |

Used by `Drone/tello_camera.py`, which does the camera, safety checks, screen, keys and flight log:

```bash
python Drone/tello_camera.py --fly --scared
```

## Behaviour

`SEARCH` (turn and look) → `APPROACH` (turn to face the banana and fly to it, slowing down near it) → `FEED` (hold still, eat) → `DONE` → `LAND`.

Scared (Giant Fiber fired while holding still): `SCARED` (back straight off, 1 s) → `WAIT` (hover until the banana is seen again, at least 1 s) → `APPROACH` → `FEED` again, hunger where it left off. Repeats for every new wave.

## Things the fly brain ignores, on purpose

- **Its own movement.** Flying forward, turning or going up/down makes the whole scene loom, as strongly as a real hand. Like a real fly (efference copy), looming is ignored while the drone moves on purpose and for 0.6 s after — so waves only count while it's eating or waiting. See `SELF_MOTION_RC` in `EscapeNeuron/fear_brain.py`.
- **The bottom third of the picture** — feet and things on the floor. See `IGNORE_BOTTOM_FRACTION`.

## Needs

The fly brain needs Brian2 in its own env. `fear_brain.py` uses `.venv-brain/` at the repo root if it exists (`python3 -m venv .venv-brain && .venv-brain/bin/pip install "brian2==2.5.1" "numpy<2" pandas pyarrow`), otherwise `flybrain_controller.py`'s usual search / `FLYBRAIN_PYTHON`.
