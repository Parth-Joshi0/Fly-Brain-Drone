# NeuralPathways UML Diagrams

UML diagrams for the `NeuralPathways/` folder. See [README.md](README.md)
and each subfolder's README for the biology behind the circuits.

The folder has three parts:

- **Runtime brain**: the code that runs once per camera frame on the
  drone or in the simulator. `ScaredEatingBrain` joins the banana-eating
  behaviour (`FoodNeuron/`) to the fear check (`EscapeNeuron/`).
  `FlyBrainController` is the adapter to the real connectome. The
  connectome itself (`connectome_worker.py`) runs in a **separate Python
  process**, because it needs Brian2. The two processes talk by sending
  one JSON object per line over stdin/stdout. Nothing outside that
  process imports `connectome_worker.py`.
- **Offline build scripts**: they create the data files the runtime
  reads (`dng02_circuit_neurons.json`, `brain_atlas.npz`), plus a
  calibration script for the looming detector.
- **BrainView**: an optional live window that shows which neurons are
  spiking. It hooks into an existing `FlyBrainController` without
  changing any flight script.

Test files under the `Tests/` folders are not shown.

## 1. Class diagrams

Python files that only contain functions and constants are shown as
`<<module>>` classes. Classes from outside this folder have a
stereotype naming where they come from.

### 1a. Runtime brain

Most constants are left out. Only the attributes and methods you need to
follow the data flow are shown.

```mermaid
classDiagram
    direction TB

    class ScaredEatingBrain {
        +BananaDetector detector
        +BananaDetector sharp_detector
        +FeedingBehaviour behaviour
        +FearBrain fear
        +_BananaWorker banana
        +bool scared
        +bool stabilize
        +list~Detection~ detections
        +ScaredEatingBrain(detector, tello, frame_read, scared, stabilize, sharp_detector)
        +start()
        +step(frame_bgr, yaw_deg, flying) RCCommand
        +close()
    }

    class _BananaWorker {
        +BananaDetector detector
        +BananaDetector sharp_detector
        +list~Detection~ result
        +int result_id
        +float result_time
        -Thread _thread
        -Lock _lock
        -Event _wake
        -bool _busy
        +submit(frame_bgr, sharp) bool
        -_run()
        +close()
    }

    class scared_eating_brain {
        <<module>>
        +float MAX_DETECTION_AGE = 0.8
        +float ZOOM_PIECE = 0.6
        +detect_zoomed(detector, frame_bgr) list~Detection~
        -_overlap(a, b) float
    }

    class FeedingBehaviour {
        +str state
        +float hunger
        +Detection current_target
        +float last_box_ratio
        +float heading
        +str scan_phase
        +int scans_done
        +deque sightings
        +float escape_level
        +float backed_off
        +bool should_land
        +str land_reason
        +bool wants_zoom
        +FeedingBehaviour(clock)
        +update(detections, frame_width, frame_height, yaw_deg, fresh) RCCommand
        +scare() bool
        -_sense(...) Detection
        -_choose_target(detections) Detection
        -_keep_in_frame(target, w, h, forward, center_deadzone) RCCommand
        -_update_scan(...) RCCommand
        -_update_approach(...) RCCommand
        -_update_look(now) RCCommand
        -_update_feed(...) RCCommand
        -_update_done(now) RCCommand
        -_update_scared(now, dt) RCCommand
        -_update_wait(now) RCCommand
        -_smooth_command(lr, fb, ud, yaw) RCCommand
        -_direct(lr, fb, ud, yaw) RCCommand
    }

    class RCCommand {
        <<dataclass>>
        +int lr
        +int fb
        +int ud
        +int yaw
    }

    class FearBrain {
        +LoomingDetector looming
        +FlyBrainController brain
        +TelloDrone drone
        +bool stabilize
        +bool armed
        +bool self_moving
        +bool escaping
        +str escape_direction
        +float escape_level
        +int scares
        +int dng02_yaw_rc
        +int intended_yaw_rc
        +float brain_ms
        -bool _ready
        +FearBrain(tello, frame_read, stabilize)
        +start()
        +update(frame_bgr) bool
        +record_command(lr, fb, ud, yaw, intended_yaw)
        +close()
    }

    class LoomingDetector {
        -ndarray _K
        -deque _frames
        -deque _recent
        -ndarray _smoothed
        +LoomingDetector(width, height, fov, baseline, cell, margin, smoothing, median_window)
        +reset()
        +update(gray, orientation, dt) dict
        -_cell_expansion(flow) ndarray
    }

    class FlowVisualizer {
        +ndarray hsv
        +FlowVisualizer(width, height)
        +render(flow) ndarray
    }

    class optical_flow {
        <<module>>
        +compute_flow(prev_gray, gray) ndarray
        +derotate_flow(flow, yaw_rate, dt, pixels_per_radian) ndarray
        +signed_hemifield_flow(flow, ground_fraction, side_margin) dict
        +grid_flow_strengths(flow, ground_fraction) dict
    }

    class FlyBrainController {
        +dict bounds
        +bool optomotor
        +str state
        +str escape_direction
        +float escape_level
        +float loom_floor_offset
        +dict dng02
        -_FlyBrainProcess _brain
        -int _escape_timer
        -int _refractory_timer
        -float _rotation_floor
        +FlyBrainController(bounds, optomotor)
        +decide(flow, state) dict
        +dng02_yaw_rate(escape) float
        +reset()
        +close()
        -_optomotor_drive(flow)$ dict
        -_loom_floor(state) float
        -_loom(expansion, floor)$ float
        -_pick_escape_dir(yaw, flow, state) int
        -_log_spikes(...)
        -_command(forward_speed, yaw_rate, strafe_speed) dict
    }

    class _FlyBrainProcess {
        +dict info
        +dict constants
        +list dng02_labels
        -Popen _proc
        +_FlyBrainProcess(with_dng02)
        +request(payload) dict
        +close()
    }

    class ConnectomeNetwork {
        <<runs in brian2 subprocess>>
        +bool with_dng02
        +list root_ids
        +Network net
        -PoissonGroup _poisson
        -SpikeMonitor _spk_mon
        -ndarray _smoothed_rate_hz
        +ConnectomeNetwork(step_dt, with_dng02)
        +step(loom_left, loom_right, drive_common, drive_left, drive_right) dict
        +reset()
        -_read_dng02(delta) dict
    }

    class connectome_worker {
        <<module>>
        -_load_circuit(with_dng02) dict
        -_load_synapses(root_to_local) tuple
        -_serve_stdio()
    }

    class Detection {
        <<BananaModel>>
    }
    class BananaDetector {
        <<BananaModel>>
        +detect(frame) list~Detection~
    }
    class TelloDrone {
        <<Drone>>
        +get_state() dict
    }

    ScaredEatingBrain *-- "1" FeedingBehaviour : behaviour
    ScaredEatingBrain *-- "1" _BananaWorker : banana
    ScaredEatingBrain *-- "0..1" FearBrain : fear (if scared or stabilize)
    ScaredEatingBrain o-- "1..2" BananaDetector : detector, sharp_detector
    ScaredEatingBrain ..> RCCommand : returns
    _BananaWorker o-- "1..2" BananaDetector
    _BananaWorker ..> scared_eating_brain : detect_zoomed()
    _BananaWorker ..> Detection : result
    FeedingBehaviour ..> Detection : reads
    FeedingBehaviour ..> RCCommand : creates

    FearBrain *-- "1" LoomingDetector : looming
    FearBrain *-- "1" FlyBrainController : brain
    FearBrain *-- "1" TelloDrone : drone (telemetry only)
    FearBrain ..> optical_flow : uses
    optical_flow *-- LoomingDetector : defines
    optical_flow *-- FlowVisualizer : defines

    FlyBrainController *-- "1" _FlyBrainProcess : _brain
    _FlyBrainProcess ..> connectome_worker : starts as subprocess
    _FlyBrainProcess ..> ConnectomeNetwork : JSON lines over stdin/stdout
    connectome_worker *-- "1" ConnectomeNetwork : _serve_stdio() owns
```

`wants_zoom` is a read-only property.

Who uses these classes from outside the folder:

| Class | Used by |
|---|---|
| `ScaredEatingBrain` | `Drone/fly_tello.py` |
| `FeedingBehaviour` | `ScaredEatingBrain`, `Controllers/banana_seek_controller.py` (simulator) |
| `FlyBrainController` | `FearBrain`, `main.py`, the `Drone/FlightTests/` scripts, the simulator tests |
| `LoomingDetector`, `FlowVisualizer` | `FearBrain`, `main.py`, the flight tests, `EscapeNeuron/Tools/calibrate_looming.py` |

### 1b. BrainView and the offline build scripts

```mermaid
classDiagram
    direction LR

    class BrainView {
        +str window
        +bool display
        +float min_interval
        +list points
        +ndarray heat
        +ndarray counts
        +ndarray peak
        +ndarray rows
        +str state
        +int spikes
        +ndarray last_frame
        +BrainView(window, max_fps, display)
        +bind(controller)
        +record(payload, result)
        +show(force)
        -_decay()
        -_draw_static(bg, names, colors) ndarray
    }

    class brain_diagram {
        <<module>>
        +Path ATLAS_PATH
        +str WINDOW_NAME
        +float DEFAULT_MAX_FPS = 10
        +attach(controller, window, max_fps, display) BrainView
    }

    class show_brain {
        <<module>>
        +install(max_fps, window)
        +main()
    }

    class WatchedController {
        <<FlyBrainController subclass made in install()>>
        +__init__(*args, **kwargs)
    }

    class FlyBrainController {
        <<flybrain_controller.py>>
        +decide(flow, state) dict
    }

    class build_brain_atlas {
        <<module>>
        +circuit_cells() dict
        +build()
    }

    class build_dng02_circuit {
        <<module>>
        +load_annotations(path, url) dict
        +load_completeness(path) set
        +group_label(ipsi_frac) str
        +build(verbose) dict
        +report(...)
        +main()
    }

    class calibrate_looming {
        <<module>>
        +time_to_contact(name, t, vx, x, gap, collided) float
        +record(name, out_path)
        +analyze(name, path)
    }

    class LoomingDetector {
        <<optical_flow.py>>
    }

    class brain_atlas_npz {
        <<file: brain_atlas.npz>>
    }
    class dng02_json {
        <<file: dng02_circuit_neurons.json>>
    }
    class looming_json {
        <<file: looming_circuit_neurons.json>>
    }
    class flywire_data {
        <<files: Data/ connectome + annotations>>
    }

    brain_diagram *-- BrainView : defines
    show_brain ..> brain_diagram : attach()
    show_brain *-- WatchedController : defines
    WatchedController --|> FlyBrainController
    brain_diagram ..> FlyBrainController : wraps request() and decide()
    BrainView ..> brain_atlas_npz : reads

    build_dng02_circuit ..> flywire_data : reads
    build_dng02_circuit ..> dng02_json : writes
    build_brain_atlas ..> build_dng02_circuit : load_annotations()
    build_brain_atlas ..> looming_json : reads
    build_brain_atlas ..> dng02_json : reads
    build_brain_atlas ..> brain_atlas_npz : writes
    calibrate_looming ..> LoomingDetector : runs on recorded frames
```

`looming_circuit_neurons.json` was made by hand. No script in this
folder writes it. `connectome_worker.py` reads both neuron JSON files
and the connectivity parquet in `Data/`.

## 2. Activity diagrams

### 2a. One camera frame: `ScaredEatingBrain.step(frame_bgr)`

`Drone/fly_tello.py` calls this once for every picture and sends the
returned command to the Tello. The banana detector runs on its own
thread (right-hand lane), so the frame loop never waits for it.

```mermaid
flowchart TD
    start([frame arrives]) --> hasFear{"fear brain on?<br/>(scared or stabilize)"}
    hasFear -- yes --> fearUpdate["fear.update(frame)<br/>see 2b"]
    fearUpdate --> fired{"Giant Fiber fired<br/>and scared=True?"}
    fired -- yes --> scare["behaviour.scare()"]
    fired -- no --> submit
    scare --> submit
    hasFear -- no --> submit

    submit["banana.submit(copy of frame,<br/>sharp = behaviour.wants_zoom)<br/>ignored if the worker is busy"]
    submit --> fresh["fresh = a new result came in<br/>since the last frame"]
    fresh --> age{"latest result<br/>at most 0.8 s old?"}
    age -- yes --> useDet["detections = latest result"]
    age -- no --> noDet["detections = []"]
    useDet --> passEsc
    noDet --> passEsc
    passEsc["behaviour.escape_level =<br/>fear.escape_level (if fear brain on)"]
    passEsc --> behave["cmd = behaviour.update(detections, w, h, yaw_deg, fresh)<br/>see 2e<br/>intended_yaw = cmd.yaw"]
    behave --> stab{"stabilize?"}
    stab -- yes --> addYaw["cmd.yaw = clamp(cmd.yaw + fear.dng02_yaw_rc, -100, 100)"]
    stab -- no --> efference
    addYaw --> efference
    efference{"fear brain on?"}
    efference -- no --> done([return cmd])
    efference -- yes --> flying{"flying?"}
    flying -- yes --> rec["fear.record_command(cmd, intended_yaw)<br/>efference copy + dead reckoning"]
    flying -- no --> recZero["fear.record_command(0, 0, 0, 0)<br/>dry run: nothing moves"]
    rec --> done
    recZero --> done

    subgraph worker["Background thread: _BananaWorker._run()"]
        wake["wait for a submitted frame"] --> sharp{"sharp?"}
        sharp -- yes --> zoom["detect_zoomed(sharp_detector, frame)<br/>whole frame + 4 zoomed pieces,<br/>duplicate boxes removed"]
        sharp -- no --> plain["detector.detect(frame)"]
        zoom --> store["store result, result_id += 1,<br/>result_time = now, not busy"]
        plain --> store
        store --> wake
    end

    submit -. wakes .-> wake
    store -. read on a later frame .-> fresh
```

### 2b. Fear check: `FearBrain.update(frame_bgr)`

Returns `True` only on the frame where a new escape starts.

```mermaid
flowchart TD
    start([frame in]) --> prep["resize to 320x240, grayscale,<br/>blank the bottom third (floor)"]
    prep --> tele["state = drone.get_state()"]
    tele --> loom["expansion = looming.update(gray, orientation, dt)<br/>left / center / right"]
    loom --> havePrev{"previous frame?"}
    havePrev -- yes --> flow["compute_flow, derotate by the measured yaw rate,<br/>grid_flow_strengths"]
    flow --> stab1{"stabilize?"}
    stab1 -- yes --> eff["derotate by the COMMANDED yaw (efference copy),<br/>signed_hemifield_flow gives rotation, translation"]
    stab1 -- no --> armed
    eff --> armed
    havePrev -- no --> armed

    armed{"armed?"}
    armed -- no --> warm{"at least 15 frames and<br/>2 s since start()?"}
    warm -- yes --> arm["looming.reset(), armed = True"] --> retF1([return False])
    warm -- no --> retF1

    armed -- yes --> decide["brain.decide(flow, state)<br/>see 2c"]
    decide --> extra["replay the same input for extra brain steps<br/>(up to 2 per frame, so brain time keeps up with real time)<br/>level = highest escape level"]
    extra --> stab2{"stabilize?"}
    stab2 -- yes --> dng["dng02_yaw_rc from DNg02 steer,<br/>scaled by (1 - level), capped"]
    stab2 -- no --> self
    dng --> self

    self{"moved on purpose in the last 0.6 s?<br/>(any rc above 8)"}
    self -- yes --> disarm["not ready, not escaping"] --> retF2([return False])
    self -- no --> esc{"level at least 0.6<br/>and ready?"}
    esc -- yes --> fire["not ready, escaping = True,<br/>direction away from the bigger loom,<br/>scares += 1"] --> retT([return True])
    esc -- no --> rearm{"level below 0.3?"}
    rearm -- yes --> ready["ready = True, escaping = False"] --> retF3([return False])
    rearm -- no --> retF3
```

### 2c. Connectome adapter: `FlyBrainController.decide(flow, state)`

Used by `FearBrain` on the real drone and by `main.py` in the simulator.
`FearBrain` passes `bounds=None`, so the boundary steps never run on the
Tello.

```mermaid
flowchart TD
    start([flow, state in]) --> bounds{"bounds set?"}
    bounds -- yes --> outside["update returning_to_bounds:<br/>on when outside, off once well inside"]
    bounds -- no --> floor
    outside --> floor
    floor["floor = 0.8 + 0.5 x forward speed<br/>+ rotation floor + loom_floor_offset"]
    floor --> loomIn["loom_left = scale(max(expansion_left, expansion_center) - floor)<br/>loom_right = scale(max(expansion_right, expansion_center) - floor)"]
    loomIn --> opto{"optomotor?"}
    opto -- yes --> drive["add the DNg02 drive request<br/>from rotation / translation"]
    opto -- no --> req
    drive --> req
    req["_brain.request(payload)<br/>(Brian2 subprocess, see 2d)<br/>gives yaw, forward, escape, dng02"]

    req --> timer{"escape timer running?"}
    timer -- yes --> tick["timer -= 1<br/>reached 0: start 30-cycle refractory"]
    timer -- no --> refr{"refractory running?"}
    refr -- yes --> tickR["refractory -= 1"]
    refr -- no --> trig{"escape at least 0.6?"}
    trig -- yes --> startEsc["escape timer = 30 cycles,<br/>pick dodge side (DNp06 yaw, then flow, then alternate),<br/>skip backing off if that would leave the bounds"]
    trig -- no --> mode
    tick --> mode
    tickR --> mode
    startEsc --> mode

    mode{"which mode?"}
    mode -- "escape timer > 0" --> escape["state = ESCAPE<br/>strafe until 0.4 m to the side,<br/>back off until 0.4 m back, no yaw"]
    mode -- "returning to bounds" --> ret["state = BOUNDARY_RETURN<br/>fly toward the area centre"]
    mode -- otherwise --> cruise["forward = 1.0 x forward x (1 - escape)<br/>yaw_rate = 0.9 x DNp06 yaw"]
    cruise --> opto2{"optomotor?"}
    opto2 -- yes --> addDng["yaw_rate += dng02_yaw_rate()<br/>forward += 0.4 x thrust x (1 - escape)"]
    opto2 -- no --> label
    addDng --> label
    label["state = AVOID_LEFT / AVOID_RIGHT if |yaw| > 0.05,<br/>else OPTOMOTOR if |steer| > 0.2,<br/>else CRUISE"]

    escape --> log
    ret --> log
    label --> log
    log{"any output neuron spiked?"}
    log -- yes --> write["print + write flybrain_spikes.log"] --> out
    log -- no --> out
    out([return command dict:<br/>forward_speed, strafe_speed, yaw_rate])
```

### 2d. One brain step: `ConnectomeNetwork.step(...)` in the Brian2 process

`_serve_stdio()` reads one JSON request per line. A `{"reset": true}`
request calls `reset()` instead. Any other request runs this step and
the result is written back as one JSON line.

```mermaid
flowchart TD
    start([request line in]) --> loomRates["Poisson rates of the LC4/LPLC2 inputs:<br/>left eye = loom_left x 30 Hz<br/>right eye = loom_right x 30 Hz"]
    loomRates --> hasPool{"DNg02 built (--dng02)?"}
    hasPool -- no --> run
    hasPool -- yes --> anyDrive{"all drive requests 0?"}
    anyDrive -- yes --> silent["drive pool silent (0 Hz)"] --> run
    anyDrive -- no --> poolRates["each drive-pool neuron's rate from the<br/>left/right requests, its connectome side split<br/>and sign (inhibitory neurons are flipped)"] --> run
    run["net.run(20 ms)"]
    run --> counts["spike counts since the last step"]
    counts --> dn["smooth DNp01/03/06 firing rates"]
    dn --> outs["escape = DNp01 left + right, saturates at 1<br/>yaw = DNp06 right - left<br/>forward = 1 - (DNp03 + DNp06) drive"]
    outs --> dng{"DNg02 built?"}
    dng -- yes --> recruit["count DNg02 cells that fired per side, smooth,<br/>thrust = total / 13.1<br/>steer = right - left through each side's<br/>recruitment curve"]
    dng -- no --> zero["dng02 = zeros"]
    recruit --> reply
    zero --> reply
    reply([reply line: yaw, forward, escape,<br/>spike_counts, dng02, spiked])
```

### 2e. Banana-eating behaviour: `FeedingBehaviour.update(...)` over a flight

Each call to `update()` runs one step of the current state and may move
to the next one. `scare()` is called from outside by `ScaredEatingBrain`
when the Giant Fiber fires. It only works in `SCAN`, `APPROACH`, `LOOK`
and `FEED`.

```mermaid
flowchart TD
    start([takeoff]) --> SCAN

    SCAN["SCAN<br/>pause and look (zoomed), turn 45 deg, repeat"]
    SCAN --> seen{"banana confirmed?<br/>(2 sightings at least 0.15 s apart)"}
    seen -- yes --> APPROACH
    seen -- no --> full{"2 full 360s done?"}
    full -- no --> SCAN
    full -- yes --> LAND

    APPROACH["APPROACH<br/>turn to face the banana, fly to it,<br/>slower as it gets bigger (dash 35 after a scare)"]
    APPROACH --> lost{"banana unseen for 2.5 s?"}
    lost -- "yes, after a scare<br/>and not back yet" --> creep["creep forward blind at 12<br/>until seen, or distance made up"]
    creep -- seen --> APPROACH
    creep -- "made up, still unseen" --> SCAN
    lost -- yes, otherwise --> LOOK
    lost -- no --> blocked{"dashing back and banana<br/>blocked for 0.5 s?"}
    blocked -- yes --> WAIT
    blocked -- no --> close{"big enough to eat<br/>or 15 s passed?"}
    close -- no --> APPROACH
    close -- yes --> FEED

    LOOK["LOOK<br/>hover, zoomed look"]
    LOOK --> lookSeen{"banana confirmed<br/>within 1 s?"}
    lookSeen -- yes --> APPROACH
    lookSeen -- no --> SCAN

    FEED["FEED<br/>hold still, keep banana in frame,<br/>hunger drops 6 per second while in view"]
    FEED --> fedLost{"banana unseen for 3 s?"}
    fedLost -- yes --> SCAN
    fedLost -- no --> full2{"hunger at most 10?"}
    full2 -- no --> FEED
    full2 -- yes --> DONE

    DONE["DONE<br/>slide right 1.5 s, hover"]
    DONE -- "5 s after eating" --> LAND
    LAND(["LAND<br/>should_land = True,<br/>fly_tello.py lands"])

    scareIn(["scare() from<br/>SCAN / APPROACH / LOOK / FEED"]) --> SCARED
    SCARED["SCARED<br/>back straight off at 60 for 0.7 s"]
    SCARED --> WAIT
    WAIT["WAIT<br/>hover, more scares ignored"]
    WAIT --> clear{"banana in view, settled 0.3 s,<br/>nothing looming for 0.3 s?"}
    clear -- yes --> dash["dash back most of the distance"] --> APPROACH
    clear -- no --> giveUp{"3 s with no banana<br/>and nothing looming?"}
    giveUp -- yes --> creepWait["come back blind"] --> APPROACH
    giveUp -- no --> WAIT
```
