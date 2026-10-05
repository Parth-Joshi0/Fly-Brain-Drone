# ScaredEating

The "scared while eating" brain for the real Tello: the drone finds a banana, flies to it and eats — and if something is waved at it while it's holding still, the real fly connectome's escape circuit makes it back straight off, wait until the coast is clear, then come back and finish eating. Joins the two pathways next door:

- **`FoodNeuron/feeding_behaviour.py`** — the eating behaviour (state machine, no Brian2)
- **`EscapeNeuron/fear_brain.py`** — looming → LC4/LPLC2 → DNp01 Giant Fiber → "scared!" (Brian2, via `flybrain_controller.py`)

## Files

| File | What it does |
|---|---|
| `scared_eating_brain.py` | `ScaredEatingBrain`: one `step(picture) -> rc command` call per camera picture. Runs the fly brain on the clean picture first, then the eating behaviour with the latest banana AI result. The banana AI runs on its own background thread (YOLOv8 nano normally; the slower sharp-eyes zoomed look with YOLOv8 small while holding still to search), so the camera + fly brain loop never waits for it and stays fast enough (~15+ pictures/s) to catch a quick hand. Then it adds the DNg02 stabilizer's yaw (`stabilize=True`), and feeds the command back to the fly brain as its efference copy. Without `scared=True` or `stabilize=True` it's just the eating behaviour. |
| `Tests/test_stabilizer_wiring.py` | Synthetic sliding pictures + fake Tello + the real brain: checks the DNg02 correction's sign, that a *commanded* turn isn't fought (efference copy), and that the sent yaw is behaviour + DNg02. `python NeuralPathways/ScaredEating/Tests/test_stabilizer_wiring.py` |

Used by `Drone/fly_tello.py`, which does the camera, safety checks, screen, keys and flight log:

```bash
python Drone/fly_tello.py --fly --scared
python Drone/fly_tello.py --fly --scared --stabilize   # + DNg02 stabilizer
```

## Behaviour

`SEARCH` (turn and look) → `APPROACH` (turn to face the banana and fly to it, slowing down near it) → `FEED` (hold still, eat) → `DONE` → `LAND`.

Scared (Giant Fiber fired while holding still): `SCARED` (back straight off, a quick 0.7 s jump at 60%) → `WAIT` (hover until the banana is seen again, at least 1 s) → `APPROACH` → `FEED` again, hunger where it left off. Repeats for every new wave.

## DNg02 stabilizer (`--stabilize`)

The same Brian2 network also runs the 24-cell DNg02 flight-motor population (`StabilizerNeuron/`), and its steering output is added to the eating behaviour's yaw **in every state** - scanning, approaching, eating, backing off - so the heading holds against drift and bumps. Thrust is logged, not flown (on the real Tello it sits near saturation at a plain hover).

What keeps it from fighting the 360 scan and the turns toward the banana is an **efference copy**, as in a real fly: before DNg02 sees the optic flow, the image motion the behaviour's own yaw command should cause is subtracted (`EFFERENCE_PIXELS_PER_RADIAN` in `EscapeNeuron/fear_brain.py`). Only rotation nobody asked for is corrected. If that constant is off, deliberate turns come out a bit faster or slower (0.75-1.3x over the measured range), never reversed.

Not flown yet in this form: the gain (`DNG02_TELLO_YAW_GAIN = 0.3`) is set to match the loop gain `Drone/Tests/tello_optomotor_flight_test.py` flew at `--yaw-gain 0.6`. And DNg02 makes each brain step slower (~22 ms vs ~13 ms measured on a MacBook Air), which costs pictures/s - `fly_tello.py` prints the loop rate and brain time at the end of every run and warns under 15/s.

## Things the fly brain ignores, on purpose

- **Its own movement.** Flying forward, turning or going up/down makes the whole scene loom, as strongly as a real hand. Like a real fly (efference copy), looming is ignored while the drone moves on purpose and for 0.6 s after — so waves only count while it's eating or waiting. See `SELF_MOTION_RC` in `EscapeNeuron/fear_brain.py`.
- **The bottom third of the picture** — feet and things on the floor. See `IGNORE_BOTTOM_FRACTION`.

## Needs

The fly brain needs Brian2 in its own env. `fear_brain.py` uses `.venv-brain/` at the repo root if it exists (`python3 -m venv .venv-brain && .venv-brain/bin/pip install "brian2==2.5.1" "numpy<2" pandas pyarrow`), otherwise `flybrain_controller.py`'s usual search / `FLYBRAIN_PYTHON`.
