# FoodNeuron

The fly-inspired "hover → eat" feeding behaviour, separate from the escape/DNg02 circuit in the rest of `NeuralPathways/` — this pathway doesn't touch Brian2 or the connectome at all; it's a plain state machine driven by `BananaModel`'s detections.

## Files

| File | What it does |
|---|---|
| `food_orbit.py` | `FoodOrbitBehaviour`: a small state machine — `SEARCH` (turn a little, hover, repeat, looking for a banana) → `FEED` (banana seen: hold position, keep it in frame, eat) → `DONE` (full: slide right, hover) → `LAND` (5s after eating). Any banana counts as food (see `BANANA_LABELS`); hunger only drops while a banana is actually in view. The drone never flies closer to it — it eats from wherever it first saw the banana, only making small corrective moves to keep it in frame. Consumed by `Drone/tello_camera.py`, which feeds it detections from `BananaModel`, and by the simulator's `Simulator/banana_seek_controller.py` (`main.py`'s `USE_BANANA`), which passes physics time in as its `clock` so its timers follow the sim rather than the wall. |

## Tests

No unit tests — this pathway is exercised end to end by `Simulator/Tests/test_banana_sim.py` (real detector on a sim banana, with and without the fly brain scaring it mid-meal; see `Simulator/README.md`), by flying `Drone/tello_camera.py` against a real banana, and by `BananaModel`'s own checks on the detector it depends on. A scripted state-machine test (feed it synthetic detection sequences and assert on the SEARCH → FEED → DONE → LAND transitions, the way `NeuralPathways/StabilizerNeuron/Tests/test_optomotor_sign.py` does for the optomotor sign) would be a natural first one to add here.
