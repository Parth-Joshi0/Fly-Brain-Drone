# Controllers — UML

Class, state and activity diagrams for the decision layer. See [README.md](README.md) for what each file is and when each controller is used.

## Architecture (UML)

Five structural views: the shared `decide()` contract and the controllers that implement it, the two stateless modules, `SafetyLayer`, `ReflexController`, and `BananaSeekController`. The activity diagrams below cover the control flow inside them. `-` marks the `_private` helpers (their real names keep the leading underscore).

The whole directory is one pipeline, and the order is the point:

```
    ManualController      \
    ReflexController       >--  SafetyLayer.apply()  -->  apply_command()  -->  DroneInterface
    FlyBrainController    /          (the veto)            (the dispatch)
    BananaSeekController /
```

A controller's output is only ever a *request*. `SafetyLayer` has the final say, which is what makes "hold forward into a wall" impossible. `BananaSeekController` is the exception: `main.py` flies it without a `SafetyLayer`, exactly as the Tello does.

### 1. The `decide()` contract and the controllers

There is no abstract base class here — the contract is duck-typed: a `decide()` method returning a command dict with `EMPTY_CMD`'s keys, plus `.state` and `.reset()`. That is all `main.py` needs, so any of these is a drop-in swap for the others.

```mermaid
classDiagram
    direction TB

    class ReflexController {
        +bounds : dict
        +state : str
        +turn_command : str
        +current_cell : tuple
        +least_visited_cell : tuple
        +wall_time_seconds : float
        +decide(flow, state) dict
        +reset()
    }

    class BananaSeekController {
        +detector
        +brain
        +food : FeedingBehaviour
        +detections : list
        +state : str
        +scares : int
        +ignored_scares : int
        +escape_direction : str
        +see(frame)
        +decide(flow, state) dict
        +reset()
        +close()
    }

    class ManualController {
        +decide(input_state) dict
    }

    class FlyBrainController {
        <<NeuralPathways>>
        +state : str
        +decide(flow, state) dict
        +reset()
        +close()
    }

    class SafetyLayer {
        +apply(cmd, flow, position, already_avoiding) tuple
        +reset()
    }

    class commands {
        <<module>>
        +EMPTY_CMD : dict
        +AVOIDING_STATES : tuple
        +apply_command(drone, cmd) bool
    }

    ReflexController ..> commands : command shape
    BananaSeekController ..> commands : command shape
    ManualController ..> commands : command shape
    FlyBrainController ..> commands : command shape
    commands ..> SafetyLayer : AVOIDING_STATES feeds already_avoiding
    SafetyLayer ..> commands : returns a final command for apply_command()

    note for ManualController "The one different signature: decide(input_state)\ntakes the normalized dict any\nSimulatorInterface.poll_input() produces,\nnever a platform keyboard API, so it works\nunchanged on sim or real hardware."
    note for BananaSeekController "The only controller that looks at the PICTURE\nrather than at optic flow, so it needs one extra\ncall: see(frame) each cycle before decide().\nFlown without SafetyLayer, as on the Tello."
    note for FlyBrainController "Lives in NeuralPathways/, not here, but\nimplements the same contract - ReflexController\nis the hand-written alternative to it."
```

`main.py` picks between them with `USE_FLYBRAIN` and `USE_BANANA`. `ReflexController` is also what `Simulator/Evaluation/run_trials.py` uses exclusively; no real-Tello script imports it.

### 2. `commands.py` and `boundary_math.py`

Both are deliberately dependency-free. `commands.py` has no pybullet or cv2 imports, so the real-Tello scripts can use it from an environment that only has djitellopy, opencv and numpy. `boundary_math.py` is pure geometry with no state at all.

```mermaid
classDiagram
    direction LR

    class commands {
        <<module>>
        +AVOIDING_STATES : tuple
        +EMPTY_CMD : dict
        +apply_command(drone, cmd) bool
    }

    class boundary_math {
        <<module>>
        +outside_bounds(position, bounds) bool
        +well_inside_bounds(position, bounds, margin) bool
        +near_or_outside_bounds(position, bounds, margin) bool
        +heading_error_toward(position, yaw_degrees, target) float
        +heading_rate_toward(position, yaw_degrees, target, max_rate, gain) float
    }

    class DroneInterface {
        <<Drone/drone_interface.py>>
        +hover() / land() / reset() / takeoff()
        +move_forward(speed) / move_left(speed) / move_right(speed)
        +turn_left(rate) / turn_right(rate)
        +move_up() / move_down() / relax_altitude()
    }

    commands ..> DroneInterface : apply_command dispatches to it
    boundary_math ..> ReflexController : BOUNDARY_RETURN and WALL_ESCAPE
    boundary_math ..> FlyBrainController : the same boundary handling

    note for commands "EMPTY_CMD keys: forward_speed, strafe_speed,\nyaw_rate, altitude_delta, hover, land, reset,\npressed_direction.\n\nAVOIDING_STATES: AVOID_LEFT, AVOID_RIGHT,\nBOUNDARY_RETURN, WALL_ESCAPE, EMERGENCY_ESCAPE,\nESCAPE."
    note for boundary_math "heading_error_toward is positive when the target\nis to the LEFT, matching this project's convention\nthat positive yaw means turning left."
```

`boundary_math`'s helpers only really run against the simulator's bounded course: every real-Tello call site constructs `FlyBrainController(bounds=None, ...)`, and all boundary handling is gated behind a `bounds is not None` check.

### 3. `SafetyLayer` — the override

Sits between whatever produced a command and the motors. It uses the optic-flow 3x3 grid, deliberately *not* the physics-distance net in `Simulator/pybullet_drone.py` — those are two independent backstops, one vision and one ground truth, and either can catch what the other misses.

```mermaid
classDiagram
    direction TB

    class SafetyLayer {
        +warning_threshold : float
        +danger_threshold : float
        +critical_threshold : float
        +clear_threshold : float
        +fast_speed : float
        +normal_speed : float
        +avoidance_speed : float
        +turn_rate : float
        +emergency_turn_rate : float
        +smoothing = 0.3
        +stuck_radius : float
        +stuck_escape_cycles : int
        -_smoothed : dict
        -_position_history : deque
        -_escape_sequence : list
        -_escape_index : int
        -_escape_action : str
        -_escape_timer : int
        +__init__(thresholds, speeds, rates, smoothing, stuck params)
        +apply(cmd, flow, position, already_avoiding) tuple
        +reset()
        -_steer_toward(final, direction, rate)
        -_is_stuck() bool
        -_run_escape(final, info) tuple
    }

    class safety_layer {
        <<module>>
        +FAST_SPEED = 1.6
        +NORMAL_SPEED = 1.0
        +AVOIDANCE_SPEED = 0.2
        +CLEAR_THRESHOLD = 0.35
        +WARNING_THRESHOLD = 0.6
        +DANGER_THRESHOLD = 1.4
        +CRITICAL_THRESHOLD = 2.2
        +TURN_RATE = 0.8
        +EMERGENCY_TURN_RATE = 1.0
        +STUCK_WINDOW = 300
        +STUCK_RADIUS = 0.35
        +STUCK_ESCAPE_CYCLES = 60
    }

    safety_layer *-- SafetyLayer : defines

    note for SafetyLayer "Flow is smoothed first, then the preferred\ndirection is recomputed from the smoothed values\nevery call - never locked to an early guess.\nThat fixed a real bug where one noisy first frame\nmade it steer TOWARD a distant obstacle."
    note for safety_layer "These thresholds must stay ABOVE\nReflexController's AVOID_TRIGGER_THRESHOLD.\nThis layer escalates only when that controller's\ngentler response was not enough - it must not\nrace it on the same reading."
```

`STUCK_WINDOW` is 300 cycles (~10 s), up from an earlier 90 (~3 s). The shorter window fired mid-turn, overrode the turn with an unrelated escape, and the two fought: the turn never completed because the detector kept interrupting it.

### 4. `ReflexController` — the hand-written state machine

The non-neural alternative to `FlyBrainController`. This is deliberately where "which side is safer" gets **decided and committed to** — re-picking fresh every frame, which an earlier version did, flip-flops on noisy readings.

```mermaid
classDiagram
    direction TB

    class ReflexController {
        +bounds : dict
        +state : str
        +turn_command : str
        +current_cell : tuple
        +least_visited_cell : tuple
        +wall_time_seconds : float
        -_avoid_direction : str
        -_avoid_forward_fraction : float
        -_avoid_timer : int
        -_avoid_streak : int
        -_wall_escape_timer : int
        -_emergency_timer : int
        -_emergency_direction : str
        -_emergency_toward_interior : bool
        -_visit_counts : list
        +__init__(bounds=None)
        +decide(flow, state=None) dict
        +reset()
        -_enter_avoid(left, center, right)
        -_enter_boundary_return()
        -_enter_wall_escape()
        -_enter_emergency_escape(position, yaw_degrees, left, right)
        -_exit_to_cruise()
        -_cruise_command(position, yaw_degrees) dict
        -_avoid_command() dict
        -_wall_escape_command(position, yaw_degrees) dict
        -_emergency_escape_command() dict
        -_boundary_return_command(position, yaw_degrees) dict
        -_command(forward_speed, yaw_rate) dict
        -_update_exploration_grid(position)
        -_cell_index(position) tuple
        -_cell_center(col, row) tuple
        -_least_visited_cell_center() tuple
    }

    class reflex_controller {
        <<module>>
        +AVOID_TRIGGER_THRESHOLD = 1.2
        +AVOID_CLEAR_THRESHOLD = 0.9
        +AVOID_MIN_CYCLES = 30
        +CRUISE_SPEED = 1.0
        +AVOID_TURN_RATE = 0.15
        +SIDE_FORWARD_FRACTION = 0.28
        +CENTER_FORWARD_FRACTION = 0.2
        +WALL_FOLLOW_TIMEOUT_CYCLES = 150
        +WALL_ESCAPE_CYCLES = 45
        +WALL_ESCAPE_TURN_RATE = 0.2
        +EMERGENCY_TRIGGER_THRESHOLD = 1.8
        +EMERGENCY_ESCAPE_CYCLES = 20
        +EMERGENCY_REVERSE_SPEED = 0.3
        +EMERGENCY_INTERIOR_SPEED = 0.15
        +EMERGENCY_TURN_RATE = 0.5
        +NEAR_BOUNDARY_MARGIN = 1.0
        +BOUNDARY_FORWARD_SPEED = 0.4
        +BOUNDARY_TURN_RATE = 0.2
        +BOUNDARY_TURN_GAIN = 1.6
        +BOUNDARY_RELEASE_MARGIN = 1.0
        +EXPLORATION_GRID_SIZE = 4
        +EXPLORATION_BIAS_GAIN = 0.5
        +EXPLORATION_MAX_YAW_BIAS = 0.1
        +DECISION_HZ = 30.0
    }

    reflex_controller *-- ReflexController : defines
    ReflexController ..> boundary_math : all five helpers

    note for ReflexController "Only EMERGENCY_ESCAPE may drop forward_speed to\nzero or below. Every other state, avoidance\nincluded, keeps moving forward while it steers -\nfrom an explicit bug report that the drone\nstopped beside a wall instead of passing it."
    note for reflex_controller "Every turn rate is capped under the residual\nself-motion flow measured at 0.20 rad/s, where\nflow spikes past 3 from an aliasing effect against\nthe striped texture and falsely trips\nEMERGENCY_ESCAPE from the turn itself."
```

`bounds=None` disables boundary containment **and** exploration, since both need a finite area to work within. `AVOID_CLEAR_THRESHOLD` (0.9) deliberately sits above the ~0.6 residual flow a turn generates on its own — otherwise the AVOID state could never see "clear" and would never exit, because its own turning would permanently look like an obstacle to itself.

### 5. `BananaSeekController` — the Tello's banana flight, in the sim

Three pieces wired as on the drone: `BananaModel`'s detector finds the banana, `FeedingBehaviour` decides what to do about it, and the fly brain runs alongside as a background fear reflex.

```mermaid
classDiagram
    direction TB

    class BananaSeekController {
        +detector
        +brain : FlyBrainController
        +food : FeedingBehaviour
        +detections : list
        +state : str
        +scares : int
        +ignored_scares : int
        +escape_direction : str
        -_frames : int
        -_escaping : bool
        -_dodging : bool
        -_frame_size : tuple
        +__init__(detector, brain=None, clock=time.time)
        +see(frame)
        +decide(flow, state=None) dict
        +reset()
        +close()
        -_speed_scale() float
    }

    class _ScaledDetection {
        +box : tuple
        +label : str
        +det_conf : float
        +cls_conf : float
        +__init__(d, sx, sy)
    }

    class banana_seek_controller {
        <<module>>
        +TELLO_FRAME = (960, 720)
        +DETECT_EVERY_N = 3
        +FORWARD_SPEED_SCALE = 2.5
        +FAST_UNTIL_SIZE_FRACTION = 0.5
        +APPROACH_LOOM_FLOOR_OFFSET = 0.8
    }

    class FeedingBehaviour {
        <<NeuralPathways/FoodNeuron>>
        +state : str
        +hunger : float
        +last_box_ratio : float
        +size_before_scare : float
        +should_land : bool
        +update(detections, w, h) rc
        +scare() bool
    }

    class BananaDetector {
        <<BananaModel>>
        +detect(frame) list
    }

    banana_seek_controller *-- BananaSeekController : defines
    banana_seek_controller *-- _ScaledDetection : defines
    BananaSeekController *-- FeedingBehaviour : food
    BananaSeekController ..> BananaDetector : every DETECT_EVERY_N cycles
    BananaSeekController ..> FlyBrainController : the fear reflex alongside
    BananaSeekController ..> _ScaledDetection : one per detection, per cycle

    note for _ScaledDetection "FeedingBehaviour's pixel deadzones were tuned on\nthe Tello's 960x720 frame; the sim camera is\n320x240. Detections are scaled up to TELLO_FRAME\nrather than retuning it - box-size ratios do not\nchange and the deadzones stay the same fraction\nof the picture. The original is left alone, since\nit is still drawn on the sim frame."
    note for banana_seek_controller "FeedingBehaviour speaks Tello RC percentages.\nThey are converted to this project's m/s and rad/s\nusing the same assumed scales Drone/tello_drone.py\nuses in the other direction."
```

Two things differ from the Tello **on purpose**. First, the dodge: on the Tello the brain only decides *when* to be scared and `FeedingBehaviour`'s slow back-off is the whole reaction, but the sim's click-to-spawn obstacle closes at 1.5 m/s and that back-off alone gets hit, so in the sim the brain's own tuned dodge flies the drone. Second, the frame scaling above. `APPROACH_LOOM_FLOOR_OFFSET` is added to the loom floor while approaching and eating, because the card growing in view and the pitch as the drone brakes read up to ~1.0 over the floor and fired the Giant Fiber on most arrivals; the thrown test box goes 2–19 over, so it still gets through.

## Activity diagrams

### 6. `ReflexController`'s state machine

Six states, checked in strict priority order every cycle. The `AVOID_LEFT`/`AVOID_RIGHT` names describe **which side the obstacle is on**, not which way the drone turns — an obstacle on the left means turning right, and that is `AVOID_LEFT`.

```mermaid
stateDiagram-v2
    [*] --> CRUISE

    CRUISE --> EMERGENCY_ESCAPE : flow over 1.8
    AVOID_LEFT --> EMERGENCY_ESCAPE : flow over 1.8
    AVOID_RIGHT --> EMERGENCY_ESCAPE : flow over 1.8
    WALL_ESCAPE --> EMERGENCY_ESCAPE : flow over 1.8
    BOUNDARY_RETURN --> EMERGENCY_ESCAPE : flow over 1.8
    EMERGENCY_ESCAPE --> CRUISE : after EMERGENCY_ESCAPE_CYCLES

    CRUISE --> BOUNDARY_RETURN : outside_bounds
    BOUNDARY_RETURN --> CRUISE : well_inside_bounds<br/>by BOUNDARY_RELEASE_MARGIN

    CRUISE --> AVOID_LEFT : flow over 1.2, obstacle on the left
    CRUISE --> AVOID_RIGHT : flow over 1.2, obstacle on the right
    AVOID_LEFT --> CRUISE : timer expired AND all three clear
    AVOID_RIGHT --> CRUISE : timer expired AND all three clear

    AVOID_LEFT --> WALL_ESCAPE : avoid streak over<br/>WALL_FOLLOW_TIMEOUT_CYCLES
    AVOID_RIGHT --> WALL_ESCAPE : avoid streak over<br/>WALL_FOLLOW_TIMEOUT_CYCLES
    WALL_ESCAPE --> CRUISE : after WALL_ESCAPE_CYCLES

    note right of EMERGENCY_ESCAPE
        the only state allowed to stop or
        reverse; time-boxed rather than
        flow-cleared, so the self-motion
        flow residual cannot strand it
    end note
    note right of BOUNDARY_RETURN
        skipped entirely when bounds is None,
        which is every real-Tello call site
    end note
```

Exit from `AVOID_*` needs the minimum commitment time elapsed **and** the path actually clear — not either one. With only the timer, a lingering elevated reading right as it expired would still cut the turn short before it got around the obstacle.

### 7. `ReflexController.decide()` — the priority ladder

```mermaid
flowchart TD
    A([decide flow, state]) --> B["read position and yaw<br/>worst = max of left, center, right"]
    B --> C["_update_exploration_grid<br/>wall_time_seconds = streak / DECISION_HZ"]

    C --> D{"already in EMERGENCY_ESCAPE<br/>with time left?"}
    D -->|yes| D1["_emergency_escape_command<br/>tick the timer down"]
    D -->|no| E{"worst over EMERGENCY_TRIGGER_THRESHOLD?"}
    E -->|yes| E1["_enter_emergency_escape<br/>near the boundary? steer toward the interior,<br/>otherwise pick the lower-flow side"]
    E1 --> D1
    E -->|no| F{"bounds configured?"}

    F -->|no| H
    F -->|yes| G{"in BOUNDARY_RETURN?"}
    G -->|yes| G1{"well inside by the release margin?"}
    G1 -->|yes| G2["_exit_to_cruise"]
    G1 -->|no| G3["_boundary_return_command<br/>steer toward the arena center"]
    G -->|no| G4{"outside_bounds?"}
    G4 -->|yes| G5["_enter_boundary_return"]
    G5 --> G3
    G4 -->|no| H
    G2 --> H

    H{"in WALL_ESCAPE with time left?"}
    H -->|yes| H1["_wall_escape_command<br/>harder turn toward the interior"]
    H -->|no| I{"avoid streak at<br/>WALL_FOLLOW_TIMEOUT_CYCLES?"}
    I -->|yes| I1["_enter_wall_escape"]
    I1 --> H1
    I -->|no| J{"in AVOID_LEFT or AVOID_RIGHT?"}

    J -->|yes| J1["tick _avoid_timer down<br/>cleared = all three under AVOID_CLEAR_THRESHOLD"]
    J1 --> J2{"timer at zero AND cleared?"}
    J2 -->|"no, hold the turn"| J3["streak += 1<br/>_avoid_command"]
    J2 -->|yes| J4["_exit_to_cruise"]
    J -->|no| K
    J4 --> K

    K{"worst over AVOID_TRIGGER_THRESHOLD?"}
    K -->|yes| K1["_enter_avoid: commit to the side with<br/>lower flow, pick the forward fraction<br/>by whether it is centered or one-sided"]
    K1 --> K2["streak += 1<br/>_avoid_command"]
    K -->|no| L["streak = 0<br/>_cruise_command with a gentle bias<br/>toward the least-visited cell"]

    D1 --> Z([command dict])
    G3 --> Z
    H1 --> Z
    J3 --> Z
    K2 --> Z
    L --> Z
```

The exploration bias is only ever layered on top of `CRUISE`, capped at `EXPLORATION_MAX_YAW_BIAS` (0.1 rad/s) — deliberately below `AVOID_TURN_RATE`, so exploration alone cannot generate enough residual self-motion flow to look like an obstacle and falsely trigger avoidance.

### 8. `SafetyLayer.apply()` — the four tiers

Each tier can override everything below it. Note that `worst` is `max(left, right, center)` — **top and bottom deliberately do not contribute** to the trigger decision, only to picking an escape direction once something else triggered.

```mermaid
flowchart TD
    A([apply cmd, flow, position, already_avoiding]) --> B["exponentially smooth every flow cell<br/>at smoothing = 0.3"]
    B --> C["worst = max of left, right, center<br/>top and bottom excluded on purpose"]
    C --> D["final = copy of cmd<br/>info = CLEAR, inactive, FORWARD, not stuck"]

    D --> E{"did the caller ask to hover?"}
    E -->|"yes, deliberate hold"| G
    E -->|no| F["append the position to the history"]
    F --> F1{"escape already running,<br/>or _is_stuck?"}
    F1 -->|yes| F2["TIER 1 _run_escape:<br/>step the left, right, up, down sequence<br/>commit for STUCK_ESCAPE_CYCLES"]
    F2 --> Z1([final, info with stuck true])
    F1 -->|no| G

    G{"worst under WARNING_THRESHOLD?"}
    G -->|yes| G1["TIER 4 speed staging only:<br/>under CLEAR_THRESHOLD gives FAST_SPEED,<br/>otherwise NORMAL_SPEED"]
    G1 --> Z2([final, info still CLEAR])

    G -->|no| H["safest = LEFT if left flow is lower, else RIGHT<br/>recomputed fresh from the smoothed values"]
    H --> I{"worst at or over CRITICAL_THRESHOLD?"}
    I -->|yes| I1["TIER 2 danger reflex:<br/>reverse at AVOIDANCE_SPEED, direction BACK"]
    I1 --> I2{"already_avoiding?"}
    I2 -->|yes| I3["leave the existing turn alone -<br/>only the stop and retreat are forced"]
    I2 -->|no| I4["_steer_toward safest at<br/>EMERGENCY_TURN_RATE"]
    I3 --> Z3([final, info DANGER])
    I4 --> Z3

    I -->|no| J{"not already_avoiding, and center or worst<br/>at or over DANGER_THRESHOLD?"}
    J -->|yes| J1["TIER 3 obstacle avoidance:<br/>cap forward at zero,<br/>_steer_toward safest at TURN_RATE"]
    J1 --> Z4([final, info DANGER])

    J -->|no| K["between WARNING and DANGER:<br/>let the controller's own turn through unchanged,<br/>just hold speed at NORMAL rather than FAST"]
    K --> Z5([final, info CLEAR])
```

`already_avoiding` exists because two independent turn-deciders reacting to the same flow reading disagreed often enough in testing to fight each other and spin — an observed bug where oscillating yaw inflated the flow, which fed back into more oscillation. When it is set, this layer still caps speed and still escalates to CRITICAL, but does not layer its own direction choice on top.

`UP` and `DOWN` are excluded from the direction pick for a measured reason: climbing generates genuine optical flow with no de-rotation equivalent for vertical translation, and open sky reads ~0 flow, so `UP` almost always looked safest the instant any tier fired. That created a runaway — climb, the climb generates flow, CRITICAL fires again, climb more — which pinned the drone at the altitude ceiling. They survive only inside the time-boxed stuck-escape sequence.

### 9. `apply_command()` — one command dict to `DroneInterface` calls

The single place a command becomes motion. Called with the **final** command, after the safety layer.

```mermaid
flowchart TD
    A([apply_command drone, cmd]) --> B{"cmd reset?"}
    B -->|yes| B1["drone.reset() then drone.takeoff()"]
    B1 --> B2([return True so the caller clears<br/>stale vision and controller state])

    B -->|no| C{"cmd land?"}
    C -->|yes| C1["drone.land()"]
    C1 --> Z

    C -->|no| D{"drone.state is flying?"}
    D -->|no| D1([return False - the low-level state machine<br/>and safety net own the drone right now])

    D -->|yes| E{"cmd hover?"}
    E -->|yes| E1["drone.hover()"]
    E -->|no| F["drone.move_forward(forward_speed)"]
    F --> G{"strafe_speed sign?"}
    G -->|positive| G1["drone.move_left(strafe_speed)"]
    G -->|negative| G2["drone.move_right(-strafe_speed)"]
    G -->|zero| G3["drone.move_left(0)"]
    G1 --> H
    G2 --> H
    G3 --> H

    H{"yaw_rate sign?"}
    H -->|positive| H1["drone.turn_left(yaw_rate)"]
    H -->|negative| H2["drone.turn_right(-yaw_rate)"]
    H -->|zero| H3["drone.turn_left(0)"]
    H1 --> I
    H2 --> I
    H3 --> I
    E1 --> I

    I{"altitude_delta sign?"}
    I -->|positive| I1["drone.move_up()"]
    I -->|negative| I2["drone.move_down()"]
    I -->|zero| I3["drone.relax_altitude()<br/>drift back toward NORMAL_ALTITUDE"]
    I1 --> Z
    I2 --> Z
    I3 --> Z
    Z([return False])
```

Note the ordering: `reset` and `land` are honoured even when the drone is not `flying`, but every movement command is dropped in any other flight state.

### 10. `BananaSeekController.decide()` — the scare hand-off

`see(frame)` runs first each cycle, caching detections between detector runs. Then `decide()` runs the brain, hands any new Giant Fiber firing to `FeedingBehaviour`, and lets the behaviour fly unless the dodge was taken.

```mermaid
flowchart TD
    A([see frame]) --> A1["record the frame size"]
    A1 --> A2{"frame count divisible by DETECT_EVERY_N?"}
    A2 -->|yes| A3["detections = detector.detect(frame)"]
    A2 -->|"no, the last detections stand"| A4["frame count += 1"]
    A3 --> A4

    A4 --> B([decide flow, state])
    B --> C{"a brain attached?"}
    C -->|yes| C1["loom_floor_offset = APPROACH_LOOM_FLOOR_OFFSET<br/>while in APPROACH or FEED, else 0.0"]
    C1 --> C2["brain_cmd = brain.decide(flow, state)"]
    C2 --> D
    C -->|no| D

    D{"brain just entered ESCAPE<br/>and was not in it last cycle?"}
    D -->|yes| D1["food.scare() decides whether to take it"]
    D1 --> D2{"taken?"}
    D2 -->|yes| D3["_dodging = True, scares += 1"]
    D2 -->|"no, already backed off or done eating"| D4["ignored_scares += 1"]
    D3 --> E
    D4 --> E
    D -->|no| E

    E["scale every detection up to TELLO_FRAME<br/>rc = food.update(scaled, 960, 720)"] --> F{"escaping AND the scare was taken?"}
    F -->|yes| F1["state = ESCAPE<br/>return the brain's own dodge command"]
    F1 --> Z1([the brain flies the dodge])

    F -->|no| G["state = food.state<br/>convert rc percentages to m/s and rad/s"]
    G --> G1["forward scaled by _speed_scale,<br/>fading from FORWARD_SPEED_SCALE to 1.0<br/>as the banana grows to eating size"]
    G1 --> G2["strafe and yaw sign-flipped into<br/>this project's conventions<br/>land = food.should_land"]
    G2 --> H{"a brain attached?"}
    H -->|yes| H1["add brain.dng02_yaw_rate()<br/>for course stabilization, 0.0 unless<br/>built with optomotor=True"]
    H -->|no| Z2
    H1 --> Z2([the feeding behaviour flies])
```

The dodge is only flown when `FeedingBehaviour` *takes* the scare. It turns one down while already backed off (`SCARED`/`WAIT`) or done eating (`DONE`/`LAND`), and then the drone does not react at all, as on the Tello. That matters: `DONE`'s slide to the right fired the Giant Fiber 1.5 s in on 4 of 4 runs — sideways motion past the banana card's edge reads as looming, since the loom floor only discounts forward speed — and flying that dodge threw the drone 2 m/s backwards right after it had finished eating.
