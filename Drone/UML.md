# Drone — UML

Class, state and activity diagrams for the real-drone layer. See [README.md](README.md) for what each file is, how to run the flight tests, and the hardware caveats.

## Architecture (UML)

Six structural views: the drone contract and its Tello implementation, the module-level helpers that have no class, the live flight script, the shared flight harness, the `FlightTests/` scripts, and the analysis/smoke-test layer. The activity diagrams below cover the control flow inside them. Methods marked `*` are abstract; `-` marks the `_private` helpers (their real names keep the leading underscore).

### 1. The drone contract and the Tello implementation

`DroneInterface` is the only thing `main.py` and the controllers know about. `TelloDrone` and `Simulator/pybullet_drone.py`'s `PyBulletDrone` are the two implementations — swapping one for the other changes nothing above this layer.

Unlike the sim there is no PID cascade here: the Tello's own onboard flight controller already turns an RC velocity request into motor output, so this class is bookkeeping plus unit conversion.

```mermaid
classDiagram
    direction TB

    class DroneInterface {
        <<abstract>>
        +takeoff()*
        +land()*
        +hover()*
        +move_forward(speed)*
        +move_backward(speed)*
        +move_left(speed)*
        +move_right(speed)*
        +turn_left(rate)*
        +turn_right(rate)*
        +get_camera_frame()*
        +get_state()*
        +step()*
        +reset()*
    }

    class TelloDrone {
        -_tello
        -_frame_read
        +proc_width : int
        +proc_height : int
        +state : str
        +collided = False
        +emergency = False
        +min_obstacle_distance = None
        +safety_override = False
        +target_vx : float
        +target_vy : float
        +target_yaw_rate : float
        +target_altitude_delta : float
        -_pos : list
        -_last_step_time
        -_prev_yaw_rad
        -_yaw_rate : float
        -_last_frame
        +__init__(tello, frame_read, proc_width=320, proc_height=240)
        +takeoff() / land() / hover()
        +move_forward(speed) / move_backward(speed)
        +move_left(speed) / move_right(speed)
        +move_up() / move_down() / relax_altitude()
        +turn_left(rate) / turn_right(rate)
        +get_camera_frame() ndarray
        +get_state() dict
        +step()
        +reset()
    }

    class PyBulletDrone {
        <<Simulator/pybullet_drone.py>>
        the simulated sibling
    }

    DroneInterface <|-- TelloDrone : implements
    DroneInterface <|-- PyBulletDrone : implements

    note for TelloDrone "No collision sensor and no forward range sensor,\nso collided / min_obstacle_distance are always\nFalse / None - the fields exist to satisfy the\ncontract the sim's consumers already read."
    note for PyBulletDrone "Same contract, physics instead of hardware.\nSee Simulator/UML.md."
```

`TelloDrone` has no real position: there is no motion capture or GPS, and the Tello's own `vgx`/`vgy` telemetry is frequently unreliable or zero on non-EDU units. `get_state()["position"]` is dead-reckoned by integrating the last *commanded* body velocity, which is only meant to feed the ESCAPE displacement tracking in `NeuralPathways/flybrain_controller.py` — never absolute navigation. Passing `bounds=None` to the controller skips the boundary logic that would need a real position.

### 2. `tello_drone.py`'s module-level helpers and conventions

The three constants at the top are the whole unit-conversion story, and two of the three are **assumed rather than measured** — see the module docstring.

```mermaid
classDiagram
    direction LR

    class tello_drone {
        <<module>>
        +RC_SPEED_SCALE = 60.0
        +RC_YAW_RATE_AT_100 = 1.5
        +MAX_RC_PERCENT = 100
        +TELLO_YAW_SIGN = -1.0
        -_mps_to_rc(speed_mps) int
        +rate_to_rc(rate_rad_s, rate_at_100) int
        -_euler_deg_to_quat(roll_deg, pitch_deg, yaw_deg) tuple
    }

    class drone_interface {
        <<module>>
        +DroneInterface
    }

    tello_drone ..> drone_interface : implements the contract
    tello_drone *-- TelloDrone : defines

    note for tello_drone "RC_SPEED_SCALE and RC_YAW_RATE_AT_100 are ASSUMED,\nnot measured - verify at low speed before trusting\nthem for a real dodge.\n\nTELLO_YAW_SIGN is MEASURED: --mode calibrate fit\n-86.6 with R-squared 0.888 on 2026-09-27."
```

`TELLO_YAW_SIGN` exists because the Tello reports yaw increasing clockwise while this project's convention is counter-clockwise-positive. It is applied once, at the boundary in `get_state()`, so everything downstream is in one convention. It matters because `derotate_flow` subtracts the drone's own rotation and `LoomingDetector` warps frames by orientation — with the sign wrong, both *add* self-motion instead of removing it, so every turn inflates apparent looming.

`tello_neuron_test.py` and `tello_dng02_test.py` read attitude themselves rather than through `get_state()`, so they apply the same constant explicitly through `YawTracker`. Change one and all of them must change.

### 3. `fly_tello.py` and its HUD / log modules

The live "find and eat a banana" script. It is the only thing here that drives the drone through `ScaredEatingBrain` rather than `DroneInterface` — it talks to `djitellopy` directly and sends its own `send_rc_control`.

```mermaid
classDiagram
    direction TB

    class fly_tello {
        <<module>>
        +MIN_BATTERY_FOR_FLIGHT = 30
        +VIDEO_START_TIMEOUT = 10.0
        +VIDEO_STALL_HOVER = 0.5
        +VIDEO_LOST_LAND = 3.0
        +LOG_DIR : str
        +video_is_running(frame_read, placeholder) bool
        +wait_for_video(frame_read, placeholder, timeout) bool
        +parse_args() Namespace
        +load_detectors() tuple
        +connect_tello() tuple
        +start_video(tello) tuple
        +take_off(args, tello, battery) bool
        +handle_key(key, tello, flying) tuple
        +land_and_disconnect(tello, flying)
        +main()
    }

    class flight_hud {
        <<module>>
        +YELLOW / RED / ORANGE / WHITE
        -_line(frame, text, y, color, scale, thickness)
        +draw_hud(frame, flying, behaviour, fear, fps, sent)
    }

    class flight_log {
        <<module>>
        +LOG_COLUMNS : list
        +MIN_LOOP_FPS = 15
        +open_flight_log(log_dir) tuple
        +dng02_log_fields(fear) list
        +flight_log_row(elapsed, flying, behaviour, detections, sent, battery, fear, hand_mark, fps) list
    }

    class LoopStats {
        +fps : float
        +fps_seen : list
        +brain_ms_seen : list
        -_last_step_time
        +after_step(fear)
        +print_summary()
    }

    class ScaredEatingBrain {
        <<NeuralPathways/ScaredEating>>
        +behaviour
        +fear
        +detections
        +start()
        +step(frame, yaw_deg, flying) cmd
        +close()
    }

    class BananaDetector {
        <<BananaModel>>
        +detect(frame)
        +annotate(frame, detections)
    }

    flight_log *-- LoopStats : defines
    fly_tello ..> flight_hud : draw_hud()
    fly_tello ..> flight_log : row writer and LoopStats
    fly_tello ..> ScaredEatingBrain : picture in, movement out
    fly_tello ..> BananaDetector : two of them, nano and sharp-eyes

    note for fly_tello "Defaults to a DRY RUN: it never takes off\nwithout --fly. --scared adds the escape reflex,\n--stabilize the DNg02 yaw stabilizer,\n--show-brain the live diagram."
    note for LoopStats "Loop rate matters: at ~16 pictures/s the brain\ncaught 4/4 hand swipes, at ~11/s it caught 0/4.\nUnder MIN_LOOP_FPS the summary warns."
```

Two detectors are loaded on purpose: a nano model every picture to keep the loop fast enough to catch a hand, and a slower "sharp eyes" YOLOv8s used only while holding still, which found far-away bananas 22/24 against the nano's 7/24.

### 4. `flight_harness.py` — what the `FlightTests/` share

Camera geometry, stream start-up, the plain-text log writer and the attitude readout. Two flavours of the connect/stream helpers live here, as they did in the scripts they came from: the `open_tello`/`open_stream` pair records telemetry availability into a `TelloLogger` for the desk tests, and the `_for_flight` pair takes a print-like callable and warns on low battery for the flight tests. Scripts import whichever pair they need under the names `open_tello`/`open_stream`, which is also what the smoke test patches.

```mermaid
classDiagram
    direction TB

    class flight_harness {
        <<module>>
        +ROOT : Path
        +TELLO_DIAGONAL_FOV = 82.6
        +DEFAULT_VERTICAL_FOV = 55.6
        +PROC_WIDTH = 320
        +PROC_HEIGHT = 240
        +STREAM_SETTLE_SECONDS = 3.0
        +WARMUP_FRAMES = 15
        +FIRST_FRAME_TIMEOUT = 20.0
        +PLACEHOLDER_SHAPE = (300, 400)
        +ARM_GRACE_SECONDS = 2.0
        +EMERGENCY_HOVER_SECONDS = 1.0
        +frame_not_ready(frame) bool
        +euler_deg_to_quat(roll_deg, pitch_deg, yaw_deg) tuple
        +percentile(values, q) float
        +git_commit() str
        +open_tello(log) Tello
        +open_stream(tello, log) frame_read
        +open_tello_for_flight(log) Tello
        +open_stream_for_flight(tello, log) frame_read
        +read_attitude(tello) dict
        +wait_for_state(tello, log, timeout) bool
        +capture_brain_requests(controller) dict
    }

    class TelloLogger {
        +COLUMNS : list
        +path : Path
        -_f
        +__init__(path)
        +meta(key, value)
        +raw(line)
        +header()
        +row(values)
        +close()
    }

    class YawTracker {
        -_prev_yaw
        +update(att, dt) tuple
    }

    flight_harness *-- TelloLogger : defines
    flight_harness *-- YawTracker : defines
    YawTracker ..> flight_harness : euler_deg_to_quat(), TELLO_YAW_SIGN

    note for flight_harness "The Tello's published 82.6 deg is a DIAGONAL spec;\nLoomingDetector wants the VERTICAL one, ~55.6 at 4:3.\nPassing 82.6 would make the focal length ~35% short\nand mis-scale the rotation-removal homography."
    note for TelloLogger "One self-contained text file: '#' metadata,\nmarkers and summary plus tab-separated rows.\nFlushed every line - a run may end by yanking\nwifi, and a truncated log still reads."
```

`PLACEHOLDER_SHAPE` is load-bearing. djitellopy 2.5.0's `BackgroundFrameRead` seeds `.frame` with a 400x300 black placeholder and **never** returns `None`, so "no real frame yet" has to be detected by shape and content. Everything here is resized to 320x240 before use, which would hide the shape difference and quietly push black frames through the whole pipeline — zero flow, zero expansion, and a log that looks exactly like "the circuit never fired".

### 5. The `FlightTests/` scripts

Each script is a module with a `main()`; the two logging ones subclass `TelloLogger` with their own `COLUMNS` and keep a `Run` accumulator for the end-of-run summary. The two props-off desk tests never call `takeoff()`, `land()` or `send_rc_control()` at all, so there is nothing to crash.

```mermaid
classDiagram
    direction TB

    class tello_battery_test {
        <<script>>
        +MIN_BATTERY_FOR_FLIGHT = 30
        +show_battery(tello)
        +main()
    }

    class tello_neuron_test {
        <<script, props off>>
        +Logger
        +Run
        +write_header(log, args)
        +draw_hud(small, phase, t_abs, hz, run, expansion, floor, loom_l, loom_r, escape, state, spikes)
        +write_summary(log, run)
        +main()
    }

    class tello_dng02_test {
        <<script, props off>>
        +ADAPTER_CONSTANTS : tuple
        +Logger
        +Run
        +write_header(log, args, controller)
        +draw_population(view, labels, counts, left_mask, x0, y0)
        +draw_hud(...)
        +write_summary(log, args, run, labels)
        +write_sweep_summary(log, run, labels)
        +main()
    }

    class tello_escape_flight_test {
        <<script, flies>>
        +main()
    }

    class tello_optomotor_flight_test {
        <<script, flies>>
        +WARMUP_FRAMES = 15
        +YAW_RATE_WINDOW_S = 0.25
        +RUNAWAY_YAW_RATE = 1.2
        +RUNAWAY_SECONDS = 0.5
        +WindowedYawRate
        +main()
    }

    class WindowedYawRate {
        +window_s : float
        -_hist : deque
        +update(t, yaw_deg) float
    }

    class Run {
        +accepted : int
        +duplicates : int
        +att_missing : int
        +marks : list
        +stats : dict
        +state_counts : dict
        +escape_events : list
        +baseline_exp : list
        +note_state(state, t_abs)
    }

    TelloLogger <|-- Logger : per-script COLUMNS
    tello_neuron_test *-- Run : accumulator
    tello_dng02_test *-- Run : its own variant
    tello_optomotor_flight_test *-- WindowedYawRate : defines

    tello_neuron_test ..> flight_harness : open_tello, YawTracker, read_attitude
    tello_dng02_test ..> flight_harness : same, plus percentile
    tello_escape_flight_test ..> flight_harness : the _for_flight pair
    tello_optomotor_flight_test ..> flight_harness : the _for_flight pair
    tello_escape_flight_test ..> TelloDrone : flies through the contract
    tello_optomotor_flight_test ..> TelloDrone : flies through the contract

    note for tello_neuron_test "Tello video to LoomingDetector to FlyBrainController\nto log. A stationary drone is the right first test:\nthe loom floor's self-motion term is zero, so any\nexpansion seen is genuinely your hand."
    note for tello_optomotor_flight_test "Yaw is the ONLY thing the brain may command -\nforward and strafe stay zero, DNp06 is dropped,\nESCAPE becomes a plain hover. Exactly one loop\ncloses through the airframe."
```

`tello_dng02_test.py` deliberately does not import the network's own constants: `connectome_worker` needs brian2, which this interpreter does not have. They arrive from the subprocess's ready handshake instead, which is strictly better — the log then records what the network was actually built with rather than what this side assumed.

### 6. `dng02_analysis.py` and the smoke test's fakes

`dng02_analysis.py` reads a log back with no drone and no brain, and reports eight PASS/FAIL checks. `test_tello_harness_smoke.py` runs the real `main()` of four scripts against a fake drone, so the per-cycle body and the cv2 HUD are not executing for the first time mid-flight on a live battery.

```mermaid
classDiagram
    direction TB

    class dng02_analysis {
        <<module>>
        +DEFAULT_LOG : Path
        +CALIBRATE_WINDOWS = (1, 5, 10, 20, 30)
        +CALIBRATE_MIN_WINDOW_RAD = 0.035
        +CALIBRATE_MAX_DEG_PER_FRAME = 2.0
        +linear_fit(xs, ys) tuple
        +windowed_fit(dyaw, raw, window, min_rad) tuple
        +report_calibration(log, cal)
        +parse_log(path) tuple
        +spearman(a, b) float
        -_bin_by(rows, key, value_keys, n_bins, min_rows) list
        -_recruitment(ladder, live) tuple
        -_score_ladder(ladder, fired_at, n_live, verdict)
        -_score_graded(totals, n_live, verdict)
        -_score_steering(live, verdict)
        -_score_thrust(live, verdict)
        -_score_baseline(base, meta, verdict)
        -_score_sign(live, mode, verdict)
        -_score_timing(live, verdict)
        +analyse(path) int
    }

    class _Verdicts {
        +results : list
        +__call__(ok, message)
    }

    class test_tello_harness_smoke {
        <<script>>
        +check(ok, message)
        +stub_cv2_display()
        +run_fly_tello(log_dir, seconds)
        +run_script(module, argv, label)
        +main()
    }

    class FakeTello {
        +get_current_state() dict
        +get_battery() int
        +streamoff() / streamon() / end()
        +takeoff() / land()
        +send_rc_control(*a)
    }

    class FakeCameraTello {
        +connect()
        +get_yaw() float
        +get_frame_read() FakeCameraFrameRead
    }

    class FakeFrameRead {
        +width : int
        +height : int
        +frame : ndarray
    }

    class NoBananaDetector {
        +detect(frame) list
        +annotate(frame, detections)
    }

    dng02_analysis *-- _Verdicts : defines
    dng02_analysis ..> _Verdicts : one per analyse() run
    FakeTello <|-- FakeCameraTello : adds the camera
    FakeFrameRead <|-- FakeCameraFrameRead : moving frames
    test_tello_harness_smoke *-- FakeTello : defines
    test_tello_harness_smoke *-- FakeFrameRead : defines
    test_tello_harness_smoke *-- NoBananaDetector : defines
    test_tello_harness_smoke ..> dng02_analysis : analyse() reads back what the loop wrote

    note for dng02_analysis "--mode calibrate is a hard prerequisite for any\noptomotor flight: _PIXELS_PER_RADIAN (75.0) is\ncalibrated against the sim camera, the Tello should\nbe nearer 260. Below R-squared 0.5, do not fly it."
    note for test_tello_harness_smoke "Run before every hardware session. A stale dict key\nin one HUD f-string once crashed a calibration run\na cycle after warmup - that is the class of bug\nthis exists to catch."
```

## Activity diagrams

### 7. `TelloDrone` flight state machine

Much simpler than the sim's, because the Tello's SDK calls block: `takeoff()` sets `taking_off`, calls through, and is already `flying` when it returns. There is no `emergency` state — no collision sensor exists to enter it.

```mermaid
stateDiagram-v2
    [*] --> idle : __init__

    idle --> taking_off : takeoff()
    landed --> taking_off : takeoff()
    taking_off --> flying : tello.takeoff() returned

    flying --> landing : land()
    taking_off --> landing : land()
    landing --> landed : tello.land() returned

    landed --> taking_off : reset()
    flying --> landing : reset() lands first

    note right of flying
        step() only sends RC while flying;
        in every other state it returns
        immediately and the Tello holds
        whatever it was doing
    end note
    note right of idle
        no emergency state: the Tello has
        no collision sensor, so collided
        and emergency are always False
    end note
```

### 8. `TelloDrone.get_state()` and `step()` — one decision cycle

The two halves of a cycle: read the telemetry and dead-reckon a position, then convert the commanded body velocity into an RC packet. Note the two sign flips, which are the easiest thing in this file to get wrong.

```mermaid
flowchart TD
    A([get_state called]) --> B["tello.get_current_state()<br/>empty dict on any exception"]
    B --> C["read pitch, roll, yaw<br/>missing keys fall back to 0.0"]
    C --> D["yaw_deg = TELLO_YAW_SIGN * reported yaw<br/>clockwise-positive becomes counter-clockwise"]
    D --> E["dt = now - last step time"]
    E --> F{"a previous yaw to compare against?"}
    F -->|yes| F1["yaw_rate = wrapped yaw difference / dt"]
    F -->|no| F2["yaw_rate stays 0.0"]
    F1 --> G
    F2 --> G

    G["dead-reckon: integrate the COMMANDED<br/>target_vx / target_vy, rotated by yaw"] --> H["altitude from h, or tof, or 0.0"]
    H --> I["orientation = _euler_deg_to_quat(roll, pitch, yaw)"]
    I --> J(["one dict shaped exactly like<br/>PyBulletDrone.get_state()"])

    K([step called]) --> L{"state is flying?"}
    L -->|no| L1([return, send nothing])
    L -->|yes| M["lr = _mps_to_rc(-target_vy)<br/>send_rc_control is +right, move_right is -target_vy"]
    M --> N["fb = _mps_to_rc(target_vx)"]
    N --> O["ud = 0, the Tello holds its own altitude"]
    O --> P["yaw = rate_to_rc(-target_yaw_rate)<br/>this project is ccw-positive, Tello is cw-positive"]
    P --> Q["tello.send_rc_control(lr, fb, ud, yaw)<br/>each an int in -100..100"]
    Q --> R([return])
```

Integrating the command rather than a measured velocity is reasonable here precisely because the Tello's flight controller tracks a commanded RC velocity fairly directly — unlike the sim's mass and PID model, it does not coast much once told to stop. Over a roughly one-second dodge that is good enough for the displacement tracking it feeds.

### 9. `fly_tello.py`'s main loop

The live banana flight. The video-stall ladder is the safety spine: a missing picture first forces a hover, then lands, because the drone is flying blind.

```mermaid
flowchart TD
    A([main]) --> B["parse_args, print the machine<br/>load both detectors"]
    B --> C["connect_tello, start_video<br/>open the flight log CSV"]
    C --> D{"a real picture within<br/>VIDEO_START_TIMEOUT?"}
    D -->|no| D1([print NO CAMERA PICTURE and return])
    D -->|yes| E["build ScaredEatingBrain<br/>attach the brain diagram with --show-brain"]
    E --> F{"--fly, and battery at or above<br/>MIN_BATTERY_FOR_FLIGHT?"}
    F -->|"no --fly"| F1["DRY RUN: flying = False"]
    F -->|"battery too low"| F2([refuse to take off and return])
    F -->|yes| F3["tello.takeoff(), settle 2s"]
    F1 --> G
    F3 --> G

    G["brain.start(), which ignores the first ~2s<br/>because takeoff itself looks like a loom"] --> H{"new picture this iteration?"}
    H -->|"no, stalled"| I{"how long stalled?"}
    I -->|"over VIDEO_LOST_LAND"| I1["CAMERA LOST - break and land"]
    I -->|"over VIDEO_STALL_HOVER"| I2["send a zero RC packet, hover"]
    I2 --> H
    I -->|"briefly"| H

    H -->|yes| J["convert RGB to BGR<br/>read tello.get_yaw() for the room scan"]
    J --> K["cmd = brain.step(frame, yaw_deg, flying)<br/>picture in, movement out"]
    K --> L["stats.after_step(fear)<br/>annotate the boxes onto the frame"]
    L --> M{"flying?"}
    M -->|yes| M1["tello.send_rc_control(lr, fb, ud, yaw)"]
    M -->|"dry run"| N
    M1 --> N

    N["write one flight-log row and flush"] --> O{"behaviour.should_land?"}
    O -->|yes| O1["print the land reason, break"]
    O -->|no| P["draw_hud, imshow"]
    P --> Q["handle_key on waitKey:<br/>q quit, h mark a hand, e stop and land, x motor stop"]
    Q --> R{"stop requested?"}
    R -->|yes| S
    R -->|no| H

    I1 --> S
    O1 --> S
    S["finally: land_and_disconnect<br/>close the log, print the loop and brain summary<br/>brain.close()"] --> T([end])
```

### 10. `tello_neuron_test.py`'s three phases

The props-off desk test, and the one place duplicate-frame rejection is explicit. Processing the same background buffer twice yields zero flow and therefore zero expansion — which looks exactly like "the circuit never fired" while actually being a sampling artifact, so duplicates are skipped and counted separately.

```mermaid
flowchart TD
    A([main]) --> B["open the Logger, write the header<br/>including every flybrain constant in effect"]
    B --> C["open_tello, open_stream<br/>LoomingDetector at the vertical fov"]
    C --> D["FlyBrainController(bounds=None)<br/>capture_brain_requests wraps the subprocess"]
    D --> E["phase = warmup"]

    E --> F{"frame ready?"}
    F -->|no| F1["sleep 5ms"]
    F1 --> F
    F -->|yes| G["to BGR, resize to 320x240, to gray"]
    G --> H{"identical to the last accepted frame?"}
    H -->|"yes, duplicate"| H1["run.duplicates += 1<br/>poll for q, continue"]
    H1 --> F
    H -->|no| I["accept it: dt from the last ACCEPTED frame,<br/>not the nominal loop period"]

    I --> J["read_attitude, YawTracker gives<br/>yaw_deg, quaternion, yaw_rate"]
    J --> K["looming.update(gray, quat, dt)<br/>plus compute_flow and derotate_flow for magnitude"]
    K --> L{"which phase?"}

    L -->|warmup| M{"WARMUP_FRAMES accepted?"}
    M -->|yes| M1["looming.reset(), drop prev_gray<br/>phase = baseline, start the clock"]
    M -->|no| F
    M1 --> F

    L -->|baseline| N{"baseline seconds elapsed?"}
    N -->|yes| N1["phase = live, print the swat prompt"]
    N -->|no| P
    N1 --> P

    L -->|live| O{"live seconds elapsed?"}
    O -->|yes| Z1["break"]
    O -->|no| P

    P["state: position and actual_vx forced to zero<br/>so nothing self-motion-related is subtracted"] --> Q["controller.decide(flow, state)"]
    Q --> R["read the captured payload and result:<br/>escape, loom_left, loom_right, spike counts"]
    R --> S["reconstruct the floor rather than calling<br/>_loom_floor again, which would advance its state twice"]
    S --> T["log one row, accumulate run.stats<br/>note_state for the escape timeline"]
    T --> U{"video enabled?"}
    U -->|yes| V["draw_hud, then waitKey:<br/>q breaks, SPACE marks a swat"]
    U -->|no| F
    V --> F

    Z1 --> Z["finally: write_summary, close the log,<br/>controller.close(), streamoff, end()"]
    Z --> ZZ([print accepted, duplicates, marks, escape triggers])
```

### 11. The two flying tests' abort paths

`tello_escape_flight_test.py` and `tello_optomotor_flight_test.py` share the arming ladder and the two-mode abort. The optomotor test adds a runaway guard, because it is the one test that closes a loop through the airframe and so is the one that can oscillate.

```mermaid
flowchart TD
    A([takeoff]) --> B["phase = warmup"]
    B --> C["poll waitKey EVERY raw iteration,<br/>not just accepted frames, so an abort key<br/>never hides behind a run of duplicates"]
    C --> D{"abort key?"}
    D -->|"l"| D1["abort_mode = land_now<br/>deadline = now"]
    D -->|"q or SPACE"| D2["abort_mode = hover_then_land<br/>deadline = now + EMERGENCY_HOVER_SECONDS<br/>brain.reset()"]
    D -->|none| E

    D1 --> K
    D2 --> K

    E{"frame ready and not a duplicate?"}
    E -->|no| C
    E -->|yes| F["expansion, flow, derotation<br/>from drone.get_state()"]
    F --> G{"phase?"}
    G -->|"warmup, WARMUP_FRAMES reached"| G1["looming.reset()<br/>phase = arming"]
    G -->|"arming, ARM_GRACE_SECONDS passed"| G2["looming.reset()<br/>phase = armed, brain output now trusted"]
    G -->|otherwise| H
    G1 --> H
    G2 --> H

    H{"optomotor only:<br/>yaw rate over RUNAWAY_YAW_RATE<br/>for RUNAWAY_SECONDS?"}
    H -->|yes| H1["ABORT: yaw runaway<br/>abort_mode = hover_then_land"]
    H -->|no| K
    H1 --> K

    K{"abort_mode set?"}
    K -->|yes| K1{"deadline reached?"}
    K1 -->|yes| Z1["break, land in finally"]
    K1 -->|no| K2["command a plain hover"]
    K2 --> L
    K -->|"no, and armed"| M["brain.decide(flow, state)"]

    M --> N{"which test?"}
    N -->|escape| N1["keep the command only in ESCAPE,<br/>otherwise hover; then safety.apply()"]
    N -->|optomotor| N2["ESCAPE becomes a hover;<br/>otherwise yaw only, from the DNg02 steer channel,<br/>clamped to DNG02_YAW_AUTHORITY"]
    N1 --> L
    N2 --> L

    L["apply_command(drone, final)<br/>drone.step() sends the RC packet"] --> O["draw the HUD, imshow"]
    O --> P{"hard time limit reached?"}
    P -->|yes| Z1
    P -->|no| C

    Z1 --> Z["finally: drone.land(), brain.close()<br/>streamoff, end(), write the summary"]
    Z --> ZZ([end])
```

### 12. `dng02_analysis.analyse()` — the eight checks

Run after every `--mode pan` session. A failure here is a measurement, not a crash.

```mermaid
flowchart TD
    A([analyse path]) --> B{"the log file exists?"}
    B -->|no| B1([print where it would be, return 1])
    B -->|yes| C["parse_log: metadata, the DNg02 ladder,<br/>and the rows"]
    C --> D["split rows into baseline and live by phase"]
    D --> E{"a ladder in the header?"}
    E -->|no| E1([written by an older version, return 1])
    E -->|yes| F{"any live rows?"}
    F -->|no| F1([never got past baseline, return 1])
    F -->|yes| G["_recruitment: per-cell activity and the<br/>level each cell needs to come in at"]
    G --> H{"any usable recruitment rows?"}
    H -->|no| H1([return 1])
    H -->|yes| I["_score_ladder: do cells recruit in ladder order?"]
    I --> J["_score_graded: is the response graded, not binary?"]
    J --> K["_score_steering: does the steer channel track flow?"]
    K --> L["_score_thrust: does the thrust channel behave?"]
    L --> M{"baseline rows present?"}
    M -->|yes| M1["_score_baseline: is the quiet phase quiet?"]
    M -->|no| N
    M1 --> N
    N["_score_sign: does it push the right way?"] --> O["_score_timing: does it respond fast enough?"]
    O --> P["_Verdicts printed each PASS or FAIL as it was decided"]
    P --> Q{"all checks passed?"}
    Q -->|yes| Q1([print ALL CHECKS PASSED, return 0])
    Q -->|no| Q2([print the failed count and the warning, return 1])
```

`--mode calibrate` has its own path through `report_calibration()`, which runs `windowed_fit()` at each of `CALIBRATE_WINDOWS` and reports the slope and R-squared per window. That fit is what established `TELLO_YAW_SIGN`: a negative slope means the Tello's yaw sign is opposite to what `derotate_flow` assumes, and with the constant applied a correct value now yields a positive slope.
