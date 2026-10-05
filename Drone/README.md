# Drone

Everything specific to flying — the abstract drone contract, the real DJI Tello implementation of it, manual keyboard control, and the live banana-eating flight script. `Simulator/pybullet_drone.py` implements the same `DroneInterface` contract for the simulated drone; nothing above this layer (`main.py`, `safety_layer.py`, `Simulator/reflex_controller.py`, `NeuralPathways/`) needs to know which one it's talking to.

## Files

| File | What it does |
|---|---|
| `drone_interface.py` | Abstract `DroneInterface` contract (`takeoff`, `land`, `move_forward`, `get_state`, ...) between the autonomous/manual controllers and whatever is actually flying the drone. |
| `tello_drone.py` | Concrete `DroneInterface` backed by a real DJI Tello over `djitellopy`. No physics/PID cascade — the Tello's own onboard flight controller handles that; this class's job is bookkeeping (flight state, a dead-reckoned position estimate) and translating `DroneInterface` calls into `send_rc_control()`. Also where `TELLO_YAW_SIGN` lives — see Notes. |
| `manual_controller.py` | Manual-flight controller. Reads the normalized input dict any `SimulatorInterface.poll_input()` produces, so it works unchanged whether input came from the PyBullet GUI or a real controller flying the real drone. |
| `fly_tello.py` | The live "find and eat a banana" flight script: DJI Tello + `BananaModel`'s detector + `NeuralPathways/FoodNeuron`'s `FeedingBehaviour`. Defaults to a dry run (no takeoff); `--fly` for a real flight, `--scared` for the escape reflex, `--stabilize` for the DNg02 yaw stabilizer (see `NeuralPathways/ScaredEating/README.md`). Writes one CSV per run to `flight_logs/` and prints the loop rate / brain time at the end. |
| `flight_logs/` | All log output lands here — both real-flight telemetry from `fly_tello.py` (`flight_<timestamp>.csv`, one row per frame) and the diagnostic text logs the `Tests/` scripts write (see below). |

## Tests (`Tests/`)

The exception to "headless" in this repo — these run against the real Tello and open a window so you can mark ground truth / abort a flight. Needs `djitellopy` (`pip install djitellopy` — use `python -m pip`, not bare `pip`, since on some machines they're different interpreters).

### `tello_battery_test.py`: quick connectivity + battery check

Not a test in the assert sense — connects to the Tello and prints its battery level (and whether it's above the 30% flight minimum). `--watch` keeps printing every 5s.

```bash
python Drone/Tests/tello_battery_test.py
python Drone/Tests/tello_battery_test.py --watch
```

### `tello_neuron_test.py`: do the neurons fire on the REAL drone?

The first real-drone test to run. Props off, drone on a desk — runs the perception half of the escape path off the live video feed (`Tello video -> LoomingDetector -> FlyBrainController -> log`). **Never flies the drone** — no `takeoff()`/`land()`/`send_rc_control()`, so there's nothing to crash.

A stationary drone is the right first test because `FlyBrainController`'s loom floor subtracts the expansion the drone's own motion would cause — on a desk that term is zero, so any expansion the circuit sees is genuinely your hand. Isolates the one real unknown: whether Farneback can recover clean expansion from the Tello's H.264 stream at all.

```bash
python Drone/Tests/tello_neuron_test.py
```

Options: `--seconds N` (live phase, default 60), `--baseline N` (quiet phase first, default 10), `--fov DEG` (default 55.6 — the Tello's published 82.6° is a *diagonal* spec; this needs the *vertical* one), `--no-video`, `--log PATH` (default `Drone/flight_logs/tello_neuron_test.log`).

**Watch which interpreter you launch from** — `_find_python_with_brian2()` uses the *current* interpreter if it can import brian2, before looking for the conda env, and the two installs are not equally fast (measured: 47ms/step under a stray brian2 vs 20ms under the conda env — the difference between missing and meeting the 33ms budget). Pin it if in doubt: `FLYBRAIN_PYTHON=/path/to/envs/brian2/bin/python python Drone/Tests/tello_neuron_test.py`.

During the live phase: press SPACE the instant you swat, Q to stop. The SPACE markers give ground truth to score DNp01 firings as hits vs false positives. **Do one dry run with internet first** (no Tello needed) — brian2 compiles generated C++ on first use, and you don't want to discover that on the Tello's wifi.

**Expected:** near-zero expansion during baseline, DNp01 spiking with `escape` crossing 0.6 on a swat.

### `tello_escape_flight_test.py`: does it actually dodge, for real?

The flight step up from `tello_neuron_test.py` — same perception pipeline, but now via `tello_drone.py` it actually takes off, hovers, and lets the brain's ESCAPE dodge through to the motors. Otherwise a plain hover — the brain runs every cycle, but only its ESCAPE output is applied.

```bash
python Drone/Tests/tello_escape_flight_test.py            # 15s hard auto-land
python Drone/Tests/tello_escape_flight_test.py --seconds 30
```

First real test of the RC speed/yaw-rate scale (`RC_SPEED_SCALE`, `_rate_to_rc`) and of the escape dodge commanding full-speed RC on two axes at once — still unverified assumptions, not measured constants. Fly it once with no one near it and confirm `l`/`q` land it promptly before trusting a dodge near a hand. Keys (video window focused): `l` lands immediately; `q`/SPACE forces a hover, lands ~1s later.

### `tello_dng02_test.py`: does DNg02 respond to the REAL camera?

Props-off table-top test, same deal as `tello_neuron_test.py` (never calls `takeoff()`/`land()`/`send_rc_control()`). Two modes, order matters:

```bash
python Drone/Tests/tello_dng02_test.py --mode calibrate     # RUN THIS FIRST
python Drone/Tests/tello_dng02_test.py --mode pan --ppr 260
```

`--mode calibrate` is a hard prerequisite for any optomotor flight — `optical_flow._PIXELS_PER_RADIAN` (75.0) is calibrated against the simulator's camera; the Tello should be nearer 260. The optomotor loop uses the signed *residual*, where a scale error is positive feedback rather than an absorbed threshold error. Rotate slowly (under ~2°/frame) for a clean fit; below R² 0.5 at every window, don't fly it.

`--mode pan` is the population-code check — sweep a textured card across the view (`d`/`a` latch a direction, `x` clears it) and watch the HUD's two rows of cells light up in ladder order.

```bash
python Drone/Tests/tello_dng02_test.py --analyze                 # scores --log's path
python Drone/Tests/tello_dng02_test.py --analyze path/to.log
```

`--analyze` reads a log back (no drone, no brain) and reports eight PASS/FAIL checks — per-cell recruitment, whether it's graded, the steering channel, baseline noise, sign, and timing. Run it after every `--mode pan` session.

### `tello_optomotor_flight_test.py`: first FLYING optomotor test (yaw only)

The drone takes off and hovers; the *only* thing the brain is allowed to command is yaw from the DNg02 opponent channel (forward/strafe stay zero, DNp06 avoidance is dropped, ESCAPE becomes a plain hover) — so exactly one loop closes through the airframe. Hold a textured sheet in front of the drone and slide it sideways; the drone should turn with it.

```bash
python Drone/Tests/tello_optomotor_flight_test.py [--seconds 20] [--dry-run] [--yaw-gain G] [--log PATH]
```

`--dry-run` skips takeoff and never sends RC — doubles as a props-off steering check (hold and turn the drone by hand, watch the HUD's yaw command push back). Keys: `l` lands immediately; `q`/SPACE hovers, lands shortly after.

### `test_tello_harness_smoke.py`: do the drone scripts actually run?

Runs the real `main()` of `tello_dng02_test.py`, `tello_neuron_test.py`, `tello_optomotor_flight_test.py` (dry run) and `fly_tello.py --scared --stabilize` (dry run, no banana model) against a **fake** drone (fake attitude, fake moving frames, `cv2` display calls stubbed) — no hardware, ~1 minute.

```bash
venv/bin/python Drone/Tests/test_tello_harness_smoke.py
```

**Run it before every hardware session.** It exists because the per-cycle body and the cv2 HUD were the only code whose first execution was always on a real drone, mid-session, with a battery burning — a stale dict key in one HUD f-string once crashed a calibration run a cycle after warmup. Covers every mode × the video path, asserts each log comes out with a header/data rows/summary, and that `analyse()` can read back what the loop just wrote. It's a smoke test: says the loops run and log cleanly, nothing about whether the numbers are right.

## Running on a different laptop

Every timing number in this repo (swat catching 4/4 at ~16 pictures/s, `MAX_BRAIN_STEPS`, `BANANA_EVERY_N_SCARED`) was measured on one MacBook Air. Before flying from another machine:

1. Create `.venv-brain` at the repo root (it's gitignored, so it doesn't come with the clone): `python3 -m venv .venv-brain && .venv-brain/bin/pip install "brian2==2.5.1" "numpy<2" pandas pyarrow`. Brian2 compiles C++ on first use - needs Xcode command-line tools, and do it once with internet, not on the Tello's wifi.
2. `venv/bin/python Drone/Tests/test_tello_harness_smoke.py` - also prints `Machine:` and the loop/brain timing for that laptop.
3. `python Drone/Tests/tello_neuron_test.py` again (props off) - the 5/5 swat result is from the Air.
4. A dry run of `python Drone/fly_tello.py --scared --stabilize`, waving a hand: check the end-of-run `Loop:` line stays at 15+/s with the banana model running.

## Notes

- **The Tello reports yaw clockwise-positive; this project is counter-clockwise-positive.** Converted once, at the boundary, by `TELLO_YAW_SIGN = -1.0` in `tello_drone.py`. The conversion happens in `get_state()`, so every `DroneInterface` consumer gets it automatically — but `tello_neuron_test.py` and `tello_dng02_test.py` read attitude themselves rather than through `get_state()`, so they apply the same constant explicitly. Change one and you must change the others.
- A negative `--mode calibrate` slope means the Tello's yaw sign is opposite to what `derotate_flow` assumes.
