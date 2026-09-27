# Testing

Scripts for checking that the FlyBrain looming circuit fires, that the drone actually escapes, and that the vision signal feeding the brain is calibrated. They run headless, with no PyBullet GUI window and no keyboard — the exception is `tello_neuron_test.py` and `tello_escape_flight_test.py`, which run against the real Tello and open a window so you can mark swats / abort a flight.

Run everything from the repo root.

## Environments

Two Python environments are involved:

| Environment | Needs | Used by |
|---|---|---|
| **brian2** (conda env) | `brian2`, `pandas`, `pyarrow` | `test_brain_circuit.py`, and the brain subprocess that the other scripts start |
| **sim** (e.g. conda `base`) | `pybullet`, `opencv-python`, `numpy` | `test_escape_sim.py`, `calibrate_looming.py`, `drone_step_response.py` |
| **tello** | `djitellopy`, `opencv-python`, `numpy` (no pybullet) | `tello_neuron_test.py`, `tello_escape_flight_test.py` |

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

### `tello_neuron_test.py`: do the neurons fire on the REAL drone?

The only script here that talks to real hardware, and the first real-drone test to run. It puts the Tello on a desk, props off, and runs the perception half of the escape path off the live video feed:

```
Tello video -> LoomingDetector -> FlyBrainController -> log
```

**It never flies the drone.** No `takeoff()`, no `land()`, no `send_rc_control()` — it only reads the video stream and the attitude telemetry, so there is nothing to crash. Leave the propellers off.

A stationary drone is the right first test because `FlyBrainController._loom_floor` subtracts the expansion the drone's own motion would cause, and on a desk that term is zero. The floor collapses to `LOOM_EXPANSION_FLOOR`, so any expansion the circuit sees is genuinely your hand. That isolates the one unknown: whether Farneback can recover clean expansion from the Tello's H.264 stream at all. Every constant in the pipeline was calibrated against PyBullet's clean renders, and expansion is a spatial *derivative* of flow — far more sensitive to compression artifacts, rolling shutter and auto-exposure hunting than plain flow magnitude.

```bash
python -m pip install djitellopy    # note: python -m pip, not bare pip
python Testing/tello_neuron_test.py
```

Use `python -m pip`, not bare `pip` — on this machine they are different interpreters, and installing into the wrong one leaves the script reporting `djitellopy is not installed` while `pip` insists it already is.

Options: `--seconds N` (live phase, default 60), `--baseline N` (quiet phase first, default 10), `--fov DEG`, `--no-video`, `--log PATH`.

**Watch which interpreter you launch from.** `_find_python_with_brian2()` returns the *current* interpreter if it can import brian2, before it ever looks for the conda env. So launching from a python that happens to have its own brian2 silently hosts the brain there instead — and the two installs are not equally fast. Measured here: 47 ms per brain step under a stray brian2 vs 20 ms under the conda env, which is the difference between missing and meeting the 33 ms budget for a 30 Hz loop. Pin it if in doubt:

```bash
FLYBRAIN_PYTHON=/path/to/envs/brian2/bin/python python Testing/tello_neuron_test.py
```

The log records `brain_interpreter` and `FLYBRAIN_PYTHON` in its header and prints a warning in the summary if the mean brain step exceeds 33 ms, so a run made under the wrong one is self-evident afterwards rather than a mystery.

It runs in three phases: a discarded warm-up while the exposure settles, a **baseline** phase of nothing happening (this measures the noise floor), then the **live** phase. During live, press SPACE the instant you swat and Q to stop. The SPACE markers are the most valuable thing in the log — they give ground truth to line the neuron response up against, so a count of DNp01 firings can be scored as hits vs false positives instead of guessed at.

Everything lands in one self-contained plain-text log (`Testing/tello_neuron_test.log`): a metadata header with every constant in effect, tab-separated per-cycle rows, the swat markers, and a summary. It can be handed over offline, which matters because reaching the Tello means joining its wifi and losing internet.

**Two things to do beforehand:**

- **Do one dry run with internet** (no Tello needed — it will just fail to connect). brian2 compiles its generated C++ on first use, and you don't want to discover that while on the Tello's wifi.
- Note `--fov` defaults to **55.6**, the *vertical* FOV. The Tello's published 82.6° is a *diagonal* spec, and `LoomingDetector`'s `fov` argument is vertical (`f = (height/2)/tan(fov/2)`, same convention as the sim's `DroneCamera(fov=75)`). Passing 82.6 would set the focal length ~35% short and mis-scale the rotation-removal homography.

**Expected:** near-zero expansion during the baseline phase, and DNp01 spiking with `escape` crossing 0.6 on a swat. If baseline expansion is already up near `LOOM_EXPANSION_FLOOR` (0.8), the Tello's stream is noisier than the sim's renders and the floor needs raising before any flight test.

The summary also reports a latency breakdown (`flow_ms`, `brain_ms`, `cycle_ms`, `effective_fps`). Watch these: the escape is only useful if the loop is fast enough to react before a hand arrives, and Tello video latency stacks on top of the compute time measured here.

### `tello_escape_flight_test.py`: does it actually dodge, for real?

The flight step up from `tello_neuron_test.py` — same perception pipeline, but now via `interfaces/tello_drone.py` (a real `DroneInterface` implementation) it actually takes off, hovers, and lets the brain's ESCAPE dodge command through to the motors. Default behavior otherwise is a plain hover (same as `main.py`'s `NEURON_TEST_MODE = True` and `test_escape_sim.py`'s `hover` scenario) — the brain runs every cycle, but only its ESCAPE output is ever applied.

```bash
python Testing/tello_escape_flight_test.py            # 60s hard auto-land
python Testing/tello_escape_flight_test.py --seconds 30
```

Two things this is the first real test of, and that are still unverified assumptions rather than measured constants (see the script's docstring and `interfaces/tello_drone.py`): the RC speed/yaw-rate scale (`RC_SPEED_SCALE`, `_rate_to_rc`), and the escape dodge commanding full-speed RC on two axes at once. Fly it once with no one near it and confirm the `l`/`q` abort keys land it promptly before trusting a dodge near a hand.

Keys (video window focused): `l` lands immediately; `q` or SPACE forces a hover, then lands ~1s later.

## Notes

- The scripts that start the brain overwrite `flybrain_spikes.log` in the repo root, just as `main.py` does. It's gitignored.
- `calibrate_looming.py` writes its recordings to a new temp directory each run, not into the repo.
