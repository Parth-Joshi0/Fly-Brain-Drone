# Testing

Scripts for checking that the FlyBrain looming circuit fires, that the drone actually escapes, and that the vision signal feeding the brain is calibrated. All of them run headless, with no PyBullet GUI window and no keyboard.

Run everything from the repo root.

## Environments

Two Python environments are involved:

| Environment | Needs | Used by |
|---|---|---|
| **brian2** (conda env) | `brian2`, `pandas`, `pyarrow` | `test_brain_circuit.py`, and the brain subprocess that the other scripts start |
| **sim** (e.g. conda `base`) | `pybullet`, `opencv-python`, `numpy` | `test_escape_sim.py`, `calibrate_looming.py`, `drone_step_response.py` |

You don't have to activate the brian2 env for the sim scripts. `controllers/flybrain_controller.py` finds it on its own (common conda locations or `conda run -n brian2`). If it can't, point `FLYBRAIN_PYTHON` at that env's python:

```bash
export FLYBRAIN_PYTHON=/path/to/envs/brian2/bin/python
```

## Scripts

### `test_brain_circuit.py`: are the neurons firing?

Drives the LC4/LPLC2 → DNp01/DNp03/DNp06 circuit (`fly_brain_controller.py`) directly with fixed looming inputs, with no simulator involved. It first prints the synapse wiring into each output neuron. Then, for each stimulus (none, weak, strong, left-only, right-only), it prints:

- spike counts per output neuron over 1 s
- peak escape signal, and whether it crosses the ESCAPE threshold (0.6)
- mean turn signal (positive = turn left)
- compute time per 20 ms brain step, which must stay under ~33 ms for the 30 Hz control loop

```bash
conda run -n brian2 python Testing/test_brain_circuit.py
```

**Expected:** silent with no stimulus. From loom ≈0.25 the Giant Fiber (DNp01) fires and ESCAPE = YES. A left-only loom gives a negative turn (turn right, away from it), and a right-only loom gives a positive one.

### `test_escape_sim.py`: does the drone actually escape?

Replicates `main.py`'s decision loop (camera → optic flow + `LoomingDetector` → `FlyBrainController` → `SafetyLayer` → drone) in a headless PyBullet session. It spawns the same test obstacle a left-click does in the GUI.

| Scenario | What happens | Pass looks like |
|---|---|---|
| `hover` | Behaves like `NEURON_TEST_MODE = True`: hovers unless the brain triggers ESCAPE; obstacle fired straight at the drone | ESCAPE triggers, `collided=False`, closest approach well above 0 m |
| `hover2` | Like `hover`, then a second obstacle about 1 s after the first dodge ends | Both dodged; still `flying` at ~1.5 m; brain state back to `CRUISE` |
| `fly` | Brain flies the drone forward; same obstacle fired head-on | `collided=False` |
| `cruise` | Brain flies the course for 20 s, no test obstacle | No escapes unless something is actually close; `collided=False` |

```bash
python Testing/test_escape_sim.py hover
python Testing/test_escape_sim.py hover2
python Testing/test_escape_sim.py fly
python Testing/test_escape_sim.py cruise
```

Options:

- `-v`: print every decision cycle (the default prints every 5th cycle plus state changes)
- `--gui`: open the PyBullet GUI window instead of running headless, to watch it. Headless and GUI runs have produced the same results so far.
- `--magnitude`: feed the brain plain optic-flow strength instead of `LoomingDetector`'s expansion, as a before/after baseline
- `CONST=value`: override any constant in `controllers/flybrain_controller.py` for that run, e.g. `LOOM_EXPANSION_FLOOR=1.2 ESCAPE_BACK_SPEED=1.0`

The harness doesn't read the keyboard, so it can't catch bugs in `main.py`'s key handling. Still try changes in `python main.py` itself.

The summary at the end lists state counts and each escape trigger. Each trigger shows: when it fired, the distance to the test obstacle and to the nearest course obstacle, the expansion reading, and the dodge direction. The brain's inputs are random (Poisson spike trains), so results vary run to run. Run each scenario a few times before drawing conclusions.

### `test_main_gui.py`: does the real `main.py` work?

Runs the actual `main.py` (GUI, key handling, `NEURON_TEST_MODE`, the whole loop) and fakes the left-click that spawns a test box at chosen decision cycles. Then it prints, for each box, whether the brain escaped and which way, and whether the drone was hit. Use it after changing `main.py`: `test_escape_sim.py` re-implements the loop, so it has missed bugs that only exist in `main.py`.

```bash
python Testing/test_main_gui.py                       # boxes at cycles 200, 330, 460
python Testing/test_main_gui.py 180,260,400,520,640   # custom click cycles
```

Takeoff plus the pre-explore hover take about 140 decision cycles. A box sent before then is never seen by the brain. Opens the PyBullet and camera windows. A line starting `[sim] transient simulator error` means PyBullet's GUI dropped one command and `main.py` retried it. That's expected occasionally.

### `calibrate_looming.py`: is the vision signal right?

Flies scripted maneuvers (no brain, no SafetyLayer) and runs the real `LoomingDetector` over the recorded frames:

- **Obstacle approaches** (`hover_obstacle`, `fly_obstacle`, `fly_box`): prints measured expansion next to the ideal 2 / time-to-contact. It should track that from about 1.4 s out.
- **Maneuvers with nothing approaching** (`yaw_hover`, `strafe_dodge`, `start_stop`, `fly_open`): prints the background noise distribution (median, p90, p98, max).

```bash
python Testing/calibrate_looming.py                  # all scenarios
python Testing/calibrate_looming.py hover_obstacle   # just one
```

Re-run it and re-tune the `LOOM_EXPANSION_*` constants in `controllers/flybrain_controller.py` whenever you change the camera, `LoomingDetector`, or the drone's flight dynamics. The loom floor needs to sit above the background noise while the approach readings still clear it early enough to dodge.

### `drone_step_response.py`: how fast can the drone move?

Hovers, commands 2 m/s for 1 s and then 0 for 1 s. Every 0.1 s it prints velocity, displacement and tilt. This is the physical limit on how late an escape can trigger and still get out of the way: with the current gains, about 0.6 m sideways takes ~1 s.

```bash
python Testing/drone_step_response.py strafe
python Testing/drone_step_response.py forward
python Testing/drone_step_response.py strafe TILT_KP=0.2 TILT_KD=0.04   # try other flight gains
```

`CONST=value` overrides constants in `interfaces/pybullet_drone.py` for that run only.

## Notes

- The scripts that start the brain overwrite `flybrain_spikes.log` in the repo root, just as `main.py` does. It's gitignored.
- `calibrate_looming.py` writes its recordings to a new temp directory each run, not into the repo.
