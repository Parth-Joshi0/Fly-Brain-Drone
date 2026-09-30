# Testing

Tests now live next to what they exercise instead of in one flat folder: the circuit-only tests under `NeuralPathways/Tests/` and `NeuralPathways/EscapeNeuron/Tests/`, the PyBullet/simulator tests under `Simulator/Tests/`, and the real-Tello hardware tests (plus their log files) under `Drone/Tests/`.

Scripts for checking that the FlyBrain looming circuit fires, that the drone actually escapes, and that the vision signal feeding the brain is calibrated. They run headless, with no PyBullet GUI window and no keyboard — the exception is the `tello_*` scripts, which run against the real Tello and open a window so you can mark ground truth / abort a flight.

Two circuits are modelled, and the scripts split along that line. `test_brain_circuit.py`, `test_escape_sim.py` and `tello_escape_flight_test.py` cover the **looming/escape** circuit (LC4/LPLC2 → DNp01/DNp03/DNp06). `test_dng02_circuit.py` and `tello_dng02_test.py` cover the **DNg02 flight-motor** circuit — a population of 24 descending neurons whose *recruitment count* sets thrust and steering. The DNg02 half is off unless a controller is constructed with `optomotor=True`, and when it's off the brain subprocess doesn't even build it, so the escape scripts run the same 274-neuron network they always have.

Run everything from the repo root.

## Environments

Two Python environments are involved:

| Environment | Needs | Used by |
|---|---|---|
| **brian2** (conda env) | `brian2`, `pandas`, `pyarrow` | `test_brain_circuit.py`, `test_dng02_circuit.py`, `NeuralPathways/build_dng02_circuit.py`, and the brain subprocess that the other scripts start |
| **sim** (e.g. conda `base`) | `pybullet`, `opencv-python`, `numpy` | `test_escape_sim.py`, `calibrate_looming.py`, `drone_step_response.py` |
| **tello** | `djitellopy`, `opencv-python`, `numpy` (no pybullet) | `tello_neuron_test.py`, `tello_escape_flight_test.py`, `tello_dng02_test.py` |

You don't have to activate the brian2 env for the sim scripts. `NeuralPathways/flybrain_controller.py` finds it on its own (common conda locations or `conda run -n brian2`). If it can't, point `FLYBRAIN_PYTHON` at that env's python:

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
conda run -n brian2 python NeuralPathways/Tests/test_brain_circuit.py
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
python Simulator/Tests/test_escape_sim.py hover
python Simulator/Tests/test_escape_sim.py hover2
python Simulator/Tests/test_escape_sim.py fly
python Simulator/Tests/test_escape_sim.py cruise
```

Options:

- `-v`: print every decision cycle (the default prints every 5th cycle plus state changes)
- `--gui`: open the PyBullet GUI window instead of running headless, to watch it. Headless and GUI runs have produced the same results so far.
- `--magnitude`: feed the brain plain optic-flow strength instead of `LoomingDetector`'s expansion, as a before/after baseline
- `CONST=value`: override any constant in `NeuralPathways/flybrain_controller.py` for that run, e.g. `LOOM_EXPANSION_FLOOR=1.2 ESCAPE_BACK_SPEED=1.0`

The harness doesn't read the keyboard, so it can't catch bugs in `main.py`'s key handling. Still try changes in `python main.py` itself.

The summary at the end lists state counts and each escape trigger. Each trigger shows: when it fired, the distance to the test obstacle and to the nearest course obstacle, the expansion reading, and the dodge direction. The brain's inputs are random (Poisson spike trains), so results vary run to run. Run each scenario a few times before drawing conclusions.

### `test_main_gui.py`: does the real `main.py` work?

Runs the actual `main.py` (GUI, key handling, `NEURON_TEST_MODE`, the whole loop) and fakes the left-click that spawns a test box at chosen decision cycles. Then it prints, for each box, whether the brain escaped and which way, and whether the drone was hit. Use it after changing `main.py`: `test_escape_sim.py` re-implements the loop, so it has missed bugs that only exist in `main.py`.

```bash
python Simulator/Tests/test_main_gui.py                       # boxes at cycles 200, 330, 460
python Simulator/Tests/test_main_gui.py 180,260,400,520,640   # custom click cycles
```

Takeoff plus the pre-explore hover take about 140 decision cycles. A box sent before then is never seen by the brain. Opens the PyBullet and camera windows. A line starting `[sim] transient simulator error` means PyBullet's GUI dropped one command and `main.py` retried it. That's expected occasionally.

### `calibrate_looming.py`: is the vision signal right?

Flies scripted maneuvers (no brain, no SafetyLayer) and runs the real `LoomingDetector` over the recorded frames:

- **Obstacle approaches** (`hover_obstacle`, `fly_obstacle`, `fly_box`): prints measured expansion next to the ideal 2 / time-to-contact. It should track that from about 1.4 s out.
- **Maneuvers with nothing approaching** (`yaw_hover`, `strafe_dodge`, `start_stop`, `fly_open`): prints the background noise distribution (median, p90, p98, max).

```bash
python NeuralPathways/EscapeNeuron/Tests/calibrate_looming.py                  # all scenarios
python NeuralPathways/EscapeNeuron/Tests/calibrate_looming.py hover_obstacle   # just one
```

Re-run it and re-tune the `LOOM_EXPANSION_*` constants in `NeuralPathways/flybrain_controller.py` whenever you change the camera, `LoomingDetector`, or the drone's flight dynamics. The loom floor needs to sit above the background noise while the approach readings still clear it early enough to dodge.

### `drone_step_response.py`: how fast can the drone move?

Hovers, commands 2 m/s for 1 s and then 0 for 1 s. Every 0.1 s it prints velocity, displacement and tilt. This is the physical limit on how late an escape can trigger and still get out of the way: with the current gains, about 0.6 m sideways takes ~1 s.

```bash
python Simulator/Tests/drone_step_response.py strafe
python Simulator/Tests/drone_step_response.py forward
python Simulator/Tests/drone_step_response.py strafe TILT_KP=0.2 TILT_KD=0.04   # try other flight gains
```

`CONST=value` overrides constants in `Simulator/pybullet_drone.py` for that run only.

### `tello_neuron_test.py`: do the neurons fire on the REAL drone?

The only script here that talks to real hardware, and the first real-drone test to run. It puts the Tello on a desk, props off, and runs the perception half of the escape path off the live video feed:

```
Tello video -> LoomingDetector -> FlyBrainController -> log
```

**It never flies the drone.** No `takeoff()`, no `land()`, no `send_rc_control()` — it only reads the video stream and the attitude telemetry, so there is nothing to crash. Leave the propellers off.

A stationary drone is the right first test because `FlyBrainController._loom_floor` subtracts the expansion the drone's own motion would cause, and on a desk that term is zero. The floor collapses to `LOOM_EXPANSION_FLOOR`, so any expansion the circuit sees is genuinely your hand. That isolates the one unknown: whether Farneback can recover clean expansion from the Tello's H.264 stream at all. Every constant in the pipeline was calibrated against PyBullet's clean renders, and expansion is a spatial *derivative* of flow — far more sensitive to compression artifacts, rolling shutter and auto-exposure hunting than plain flow magnitude.

```bash
python -m pip install djitellopy    # note: python -m pip, not bare pip
python Drone/Tests/tello_neuron_test.py
```

Use `python -m pip`, not bare `pip` — on this machine they are different interpreters, and installing into the wrong one leaves the script reporting `djitellopy is not installed` while `pip` insists it already is.

Options: `--seconds N` (live phase, default 60), `--baseline N` (quiet phase first, default 10), `--fov DEG`, `--no-video`, `--log PATH`.

**Watch which interpreter you launch from.** `_find_python_with_brian2()` returns the *current* interpreter if it can import brian2, before it ever looks for the conda env. So launching from a python that happens to have its own brian2 silently hosts the brain there instead — and the two installs are not equally fast. Measured here: 47 ms per brain step under a stray brian2 vs 20 ms under the conda env, which is the difference between missing and meeting the 33 ms budget for a 30 Hz loop. Pin it if in doubt:

```bash
FLYBRAIN_PYTHON=/path/to/envs/brian2/bin/python python Drone/Tests/tello_neuron_test.py
```

The log records `brain_interpreter` and `FLYBRAIN_PYTHON` in its header and prints a warning in the summary if the mean brain step exceeds 33 ms, so a run made under the wrong one is self-evident afterwards rather than a mystery.

It runs in three phases: a discarded warm-up while the exposure settles, a **baseline** phase of nothing happening (this measures the noise floor), then the **live** phase. During live, press SPACE the instant you swat and Q to stop. The SPACE markers are the most valuable thing in the log — they give ground truth to line the neuron response up against, so a count of DNp01 firings can be scored as hits vs false positives instead of guessed at.

Everything lands in one self-contained plain-text log (`Drone/flight_logs/tello_neuron_test.log`): a metadata header with every constant in effect, tab-separated per-cycle rows, the swat markers, and a summary. It can be handed over offline, which matters because reaching the Tello means joining its wifi and losing internet.

**Two things to do beforehand:**

- **Do one dry run with internet** (no Tello needed — it will just fail to connect). brian2 compiles its generated C++ on first use, and you don't want to discover that while on the Tello's wifi.
- Note `--fov` defaults to **55.6**, the *vertical* FOV. The Tello's published 82.6° is a *diagonal* spec, and `LoomingDetector`'s `fov` argument is vertical (`f = (height/2)/tan(fov/2)`, same convention as the sim's `DroneCamera(fov=75)`). Passing 82.6 would set the focal length ~35% short and mis-scale the rotation-removal homography.

**Expected:** near-zero expansion during the baseline phase, and DNp01 spiking with `escape` crossing 0.6 on a swat. If baseline expansion is already up near `LOOM_EXPANSION_FLOOR` (0.8), the Tello's stream is noisier than the sim's renders and the floor needs raising before any flight test.

The summary also reports a latency breakdown (`flow_ms`, `brain_ms`, `cycle_ms`, `effective_fps`). Watch these: the escape is only useful if the loop is fast enough to react before a hand arrives, and Tello video latency stacks on top of the compute time measured here.

### `tello_escape_flight_test.py`: does it actually dodge, for real?

The flight step up from `tello_neuron_test.py` — same perception pipeline, but now via `Drone/tello_drone.py` (a real `DroneInterface` implementation) it actually takes off, hovers, and lets the brain's ESCAPE dodge command through to the motors. Default behavior otherwise is a plain hover (same as `main.py`'s `NEURON_TEST_MODE = True` and `test_escape_sim.py`'s `hover` scenario) — the brain runs every cycle, but only its ESCAPE output is ever applied.

```bash
python Drone/Tests/tello_escape_flight_test.py            # 15s hard auto-land
python Drone/Tests/tello_escape_flight_test.py --seconds 30
```

Two things this is the first real test of, and that are still unverified assumptions rather than measured constants (see the script's docstring and `Drone/tello_drone.py`): the RC speed/yaw-rate scale (`RC_SPEED_SCALE`, `_rate_to_rc`), and the escape dodge commanding full-speed RC on two axes at once. Fly it once with no one near it and confirm the `l`/`q` abort keys land it promptly before trusting a dodge near a hand.

Keys (video window focused): `l` lands immediately; `q` or SPACE forces a hover, then lands ~1s later.

### `test_dng02_circuit.py`: does the population code work?

The DNg02 counterpart to `test_brain_circuit.py`, and the **gate for everything downstream** — nothing DNg02-related should reach a real drone until this passes. It drives the flight-motor half directly and prints:

- the recruitment ladder: each of the 24 DNg02 cells' excitatory input weight from the drive pool, which is the order they're predicted to recruit in
- every synapse running from the drive pool back into the escape circuit, split by sign
- a symmetric drive sweep (0 → 1) reporting recruited cells per side, `thrust` and `steer`, plus a per-level map of *which* cells fired
- an asymmetric sweep, which is where the steering signal has to show up
- a 2-D `loom × drive` sweep, because the two channels interact

```bash
conda run -n brian2 python NeuralPathways/Tests/test_dng02_circuit.py
```

It ends in hard assertions and a non-zero exit on failure. The load-bearing ones: step time p95 under the 33 ms budget, the population silent when the path is idle, recruitment monotone *and* graded (not all-or-nothing — a population that switches on at once has no code left to read), `steer` correctly signed both ways with no standing bias at symmetric drive, **DNp01 never spiking from optic-flow drive alone**, and the ESCAPE threshold not moving between zero and full drive.

It also prints `DNG02_RECRUIT_SAT` and the two recruitment curves as paste-ready lines. Those are measured calibration constants in `fly_brain_controller.py`, so if you change `MAX_DRIVE_RATE` or the drive pool, re-run this and paste the new values back.

**Expected:** all checks pass, recruitment climbing smoothly from 0 to ~13 of 24 cells, ~18 ms/step. Worth knowing: the bottom 8 cells of the ladder never recruit at all, so the effective population is ~16 — which is why thrust normalizes by a measured saturation rather than by 24.

### `test_tello_harness_smoke.py`: do the drone scripts actually run?

Runs the real `main()` of `tello_dng02_test.py` and `tello_neuron_test.py` against a **fake** drone — fake attitude, fake moving frames, `cv2`'s display calls stubbed so the HUD code still executes without a window. No hardware, ~1 minute.

```bash
venv/bin/python Drone/Tests/test_tello_harness_smoke.py
```

Run it before every hardware session. It exists because the per-cycle body and the cv2 HUD were the only code in this project whose first execution was always on a real drone, mid-session, with a battery burning — and a stale dict key in one HUD f-string (invisible to import, to `--analyze`, and to `--no-video`) once crashed a calibration run one cycle after warmup and cost the whole session.

It covers every mode × the video path, asserts each log comes out with a header, data rows and a summary, and asserts `analyse()` can read back what the loop just wrote — those two drifted apart once already when a column was added. It is a smoke test: it says the loops run and log cleanly, nothing about whether the numbers are right.

### `test_optomotor_sign.py`: is the feedback loop the right way round?

Small, fast, and the one to run after touching anything in the optomotor path. It drives the whole adapter chain with synthetic flow and asserts on the **commanded yaw** — the thing that actually reaches the motors.

```bash
python NeuralPathways/Tests/test_optomotor_sign.py      # normal env, brain runs as a subprocess
```

It exists because the DNg02 steering term uses the *opposite* sign convention to DNp06 two lines away in the same function, for a real biological reason (DNg02 tracks the contralateral wing; DNp06 steers away from looming). If that sign ever flips, the drone doesn't fly slightly worse — it yaws harder into its own drift until it hits something. `test_dng02_circuit.py` can't catch it, because it never sees a yaw command.

Checks: the drive mapping favours the right DNg02 for rightward scene motion and nothing below the flow floor; uncommanded left rotation commands a **right** turn and vice versa (negative feedback); the contribution stays inside `DNG02_YAW_AUTHORITY`; a hard loom plus a hard steering request still yields `ESCAPE` with the dodge's yaw at exactly 0; and with `optomotor=False` the subprocess doesn't build the DNg02 half and residual rotation is ignored entirely.

**Note:** with nothing moving, `steer` sits around +0.06 rather than 0, giving a standing yaw of about −0.009 rad/s (~0.5°/s). That's residual population asymmetry the recruitment-curve inversion doesn't fully cancel. It's well under `OPTOMOTOR_STATE_THRESHOLD`, but it is a slow drift, so watch for it on the first flight rather than assuming dead-centre is dead-centre.

### `tello_dng02_test.py`: does DNg02 respond to the REAL camera?

Props-off table-top test, same deal as `tello_neuron_test.py` — it never calls `takeoff()`, `land()` or `send_rc_control()`. It imports that script's harness helpers rather than copying them, so the placeholder-frame guard, duplicate rejection and attitude handling stay in one place.

Two modes, and the order matters:

```bash
python Drone/Tests/tello_dng02_test.py --mode calibrate     # RUN THIS FIRST
python Drone/Tests/tello_dng02_test.py --mode pan --ppr 260
```

**`--mode calibrate` is a hard prerequisite for any optomotor flight.** `NeuralPathways/EscapeNeuron/optical_flow._PIXELS_PER_RADIAN` is 75.0, calibrated against the *simulator's* camera; the Tello at 320px across ~70.3° horizontal should be nearer 260. That has never mattered, because every existing caller feeds derotated flow into a *magnitude*, where a scale error gets absorbed by hand-tuned thresholds. The optomotor loop uses the signed **residual**, where a too-small correction leaves part of the drone's own commanded yaw in the signal *with the same sign as a real drift* — positive feedback, i.e. the drone chasing its own turn.

**Rotate slowly — under ~2°/frame, so roughly 40°/s at 20 fps.** A deliberate quarter-turn over about three seconds, alternating direction, repeatedly, about the drone's own vertical axis. Too fast and `compute_flow`'s Farneback settings (`levels=2, winsize=13`) stop tracking the displacement and under-report it, which corrupts the fitted magnitude. Measured on the first real run: 2.1°/frame effective already recovered only 3.3 px of an implied 18, and the fitted slope was still growing with window size — the script now detects and says so.

**The fit is windowed, and that is not a refinement.** The Tello streams attitude on its own socket at ~9 Hz while frames arrive at 20–25 Hz, so **56% of frames report `d_yaw` exactly 0** and the rest report a jump that didn't happen over the interval the flow was measured across. The same real data fits R² **0.38** per-frame and R² **0.89** accumulated over 20 frames. So `--mode calibrate` reports the fit at every window size in `CALIBRATE_WINDOWS` and picks the best R² — and the *trend* across windows is itself the diagnosis: a slope still climbing at the largest window means the flow is under-tracked, not that the fit converged.

Below R² 0.5 at every window, don't fly it. A negative slope means the Tello's yaw sign is opposite to what `derotate_flow` assumes — see the note below, because that affects more than this circuit.

**`--mode pan`** is the population-code check. Sweep a textured card (a newspaper, a book cover — a blank wall gives Farneback nothing) across the view; `d` and `a` latch a direction label, `x` clears it. The HUD draws the population as two rows of cells in ladder order, so you can watch them light up in order as you sweep harder.

**Expected:** near-zero `rotation` and no recruitment while still; sweeping right giving `rotation > 0`, more right-side than left-side cells, `steer > 0`, and a *negative* would-be yaw command (turn right). The summary prints an explicit `SIGN CHECK`. If it says `INVERTED`, do not fly — in closed loop that's positive feedback, and the failure mode is a growing oscillation rather than a steady wrong heading.

If quiet-scene `|rotation|` exceeds `OPTOMOTOR_FLOW_FLOOR` (0.15), the summary warns: raise the floor before flying, or the drone will steer at its own Farneback noise.

#### `--analyze`: score the run instead of eyeballing it

```bash
python Drone/Tests/tello_dng02_test.py --analyze                 # scores --log's path
python Drone/Tests/tello_dng02_test.py --analyze path/to.log
```

No drone, no brain — it reads a log back and reports eight PASS/FAIL checks. Run it after every `--mode pan` session.

This isn't optional polish. The escape circuit could be scored by counting discrete events against SPACE markers; DNg02 has no events, only the shape of a distribution, and the HUD updates 25 times a second while your eye averages it. "The cells lit up in roughly the right order" is not something you can honestly judge by watching.

What it reports:

- **per-cell recruitment table** in committed ladder order — how often each cell fired, and the "onset" level (how much of the population has to be active before it joins)
- **is it a population code?** Spearman rho between ladder position and firing rate. Wants **negative** — cells earlier in the ladder should fire more often. Passes below −0.4.
- **is it graded?** distinct levels, deciles, and the fraction of cycles pinned at 0 or the peak. Fails if bimodal.
- **steering channel** binned by `|rotation|`: `mean|steer|` and `mean|nR-nL|` must rise. `_optomotor_drive` caps `drive_common` at `1 − |offset|` so this channel always has headroom — without that cap a common drive near 1.0 pins both sides at the brain's ±1 clip and the opponent channel goes dead, which is what the first real run did on 18% of cycles.
- **thrust channel** binned by `drive_common` — usually *not scored* on a desk, and it says so: `drive_common` comes from the translational set-point error, and a sideways card sweep produces almost no translational flow. That channel only becomes testable in flight.
- **baseline noise** vs the steering floor, with a suggested new floor if it fails
- **sign**, recomputed from the rows rather than trusting the summary line
- **timing** against the 33 ms budget

Two things it prints that look like faults but aren't. Total recruitment **falls** as rotation grows — a large opponent offset saturates one side and suppresses the other, so a hard turn request costs thrust. And roughly the bottom third of the ladder never recruits at all, which is why thrust normalizes by a measured saturation rather than by 24.

It averages *magnitudes*, not signed values: a bin mixes leftward and rightward sweeps, so a signed mean of `steer` averages +0.4 and −0.4 to nearly nothing however strong the response was.

## Notes

- The scripts that start the brain overwrite `flybrain_spikes.log` in the repo root, just as `main.py` does. It's gitignored.
- `calibrate_looming.py` writes its recordings to a new temp directory each run, not into the repo.
- `NeuralPathways/build_dng02_circuit.py` regenerates `dng02_circuit_neurons.json` (the 24 DNg02 cells and their 120-neuron drive pool). It downloads the FlyWire annotation table once and caches it at `NeuralPathways/Data/flywire_neuron_annotations_630.tsv` (gitignored, 31 MB). Run it under the brian2 env; every selection rule is a constant at the top of the file, and it prints the resulting circuit so the output is self-checking.
- **The one modelling assumption in the DNg02 circuit, stated plainly:** DNg02 has essentially no visual input in this connectome — 137 of 12,848 incoming synapses come from visual projection neurons, none at all from the HS/VS wide-field tangential cells that carry optic flow, and none from the LC4/LPLC2 pool. Its input is central (posterior slope, inferior bridge, clamp, LAL, ascending from the VNC). So driving those central cells with optic flow is *stipulated*, not something the connectome supports; the pathway from wide-field motion to DNg02 is a published open question (Schnell, *Trends in Neurosciences* 2026). What *is* connectome-derived is everything downstream: which cell excites or inhibits which DNg02, how strongly, and therefore the recruitment order. That order is the scientifically interesting part. This is not "we found DNg02's visual pathway."
- `main.py` is deliberately **not** wired for optomotor yet. It builds its `flow` dict without the `rotation`/`translation` keys, so setting `optomotor=True` there today would silently get a zero drive request. Wiring it up means calling `signed_hemifield_flow` in its loop as well — a separate change, and not one to make before a flight test has passed.
- **The Tello reports yaw clockwise-positive; this project is counter-clockwise-positive. That is now converted once, at the boundary,** by `TELLO_YAW_SIGN = -1.0` in `Drone/tello_drone.py`. Measured 2026-09-27: `--mode calibrate` fitted **−86.6 at R² 0.888**, negative at every window size from 1 to 30 frames. Before the fix nothing flipped it, so `derotate_flow` *added* the drone's own rotation instead of cancelling it and `LoomingDetector`'s homography warped the wrong way — both inflate apparent looming during any turn. **This affected the escape path too, not just DNg02,** and is worth ruling in or out against a hover flight test that spent 37% of cycles in ESCAPE.
  - The conversion happens in `get_state()`, so every consumer of `DroneInterface` gets it. `tello_neuron_test.py` and `tello_dng02_test.py` read attitude themselves rather than through `get_state`, so they apply the same constant explicitly — change one and you must change the others, or the perception tests and the flight path disagree about which way a turn went.
  - `--mode calibrate` now applies the constant *before* fitting, so a **positive** slope means it is still correct. Re-run after a firmware change.
  - **Yaw only.** Pitch and roll feed the same quaternion and are still unverified — they sit near zero on a desk, so no props-off test can measure them. Re-flying escape is the way to find out.
  - The **magnitude** is still unmeasured: that run rotated too fast (2.14°/frame effective, peaks at 43) for Farneback to track, so `_PIXELS_PER_RADIAN` stays at its sim-derived 75.0 pending a slow re-run.
- DNg02 steering uses the **opposite** sign convention to DNp06, for a real reason: DNg02 activity tracks wingbeat amplitude in the *contralateral* wing, so more right-side DNg02 yaws the fly right, whereas DNp06 steers *away* from a looming object. `NeuralPathways/flybrain_controller.py`'s `decide()` therefore subtracts the DNg02 term where it adds the DNp06 one, two lines apart. Both are commented; don't "fix" either.
