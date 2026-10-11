# Simulator — UML

Class, state and activity diagrams for the PyBullet simulator. See [README.md](README.md) for what each file is and how to run the tests, tools and evaluation.

## Architecture (UML)

Three structural views of the same code: the runtime classes, the module-level builders that have no class, and the evaluation/test layer. The activity diagrams below cover the control flow inside them. Methods marked `*` are abstract; `-` marks the `_private` helpers (their real names keep the leading underscore).

### 1. Runtime classes

The two abstract contracts — `SimulatorInterface` here and `DroneInterface` from `Drone/drone_interface.py` — are the only thing `main.py` knows about. Everything below them is PyBullet-specific and swappable.

```mermaid
classDiagram
    direction TB

    class SimulatorInterface {
        <<abstract>>
        +physics_dt: float
        +connect()*
        +create_drone(start_pos)*
        +poll_input()*
        +update_debug_view(position, yaw_degrees)*
        +tick()*
        +is_connected()
        +update_test_obstacles(drone_position, drone_yaw_degrees, dt)
    }

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

    class PyBulletSimulator {
        -_banana_position
        -_env : dict
        -_drone : PyBulletDrone
        -_heading_line_id
        -_test_obstacle_texture
        -_test_obstacles : list
        -_MOUSE_BUTTON_EVENT = 2
        +__init__(banana_position=None)
        +connect() dict
        +create_drone(start_pos) PyBulletDrone
        +poll_input() dict
        +update_debug_view(position, yaw_degrees)
        +is_connected() bool
        +physics_dt : float
        +tick()
        +update_test_obstacles(drone_position, drone_yaw_degrees, dt)
        -_left_click_triggered() bool
        -_spawn_test_obstacle(drone_position, drone_yaw_degrees)
    }

    class PyBulletDrone {
        +start_pos
        +body : QuadcopterBody
        +camera : DroneCamera
        +ground_id
        +obstacle_ids : set
        +state : str
        +target_altitude : float
        +target_vx, target_vy, target_yaw_rate : float
        +collided, emergency : bool
        +emergency_stop_count : int
        +min_obstacle_distance : float
        +safety_override : bool
        +__init__(start_pos, ground_id, obstacle_ids)
        +takeoff() land() hover()
        +move_forward(speed) move_backward(speed)
        +move_left(speed) move_right(speed)
        +move_up() move_down() relax_altitude()
        +turn_left(rate) turn_right(rate)
        +get_camera_frame() ndarray
        +get_state() dict
        +step()
        +reset()
        -_reset_flight_vars()
        -_current_altitude() float
        -_effective_target_altitude() float
        -_closest_obstacle_distance(position, yaw) float
        -_run_state_machine()
        -_check_safety_and_collisions()
    }

    class _PID {
        +kp, ki, kd : float
        +integral : float
        +prev_error : float
        +__init__(kp, ki=0.0, kd=0.0)
        +update(error, dt, derivative=None) float
        +reset()
    }

    class QuadcopterBody {
        +mass : float
        +id : int
        +start_pos
        +__init__(start_pos, mass=1.0, half_extents=(0.15, 0.15, 0.04))
        +apply_control(thrust, torque_roll, torque_pitch, torque_yaw)
        +get_state() dict
        +get_contacts() list
        +closest_distance(other_body_id, max_distance=5.0) float
        +reset(start_pos=None)
    }

    class DroneCamera {
        +width, height : int
        +fov, near, far : float
        +aspect : float
        +projection_matrix
        +__init__(width=320, height=240, fov=75, near=0.05, far=50)
        +capture(position, orientation_quat) ndarray
    }

    SimulatorInterface <|-- PyBulletSimulator : implements
    DroneInterface <|-- PyBulletDrone : implements

    PyBulletSimulator "1" --> "1" PyBulletDrone : creates, owns as _drone
    PyBulletDrone "1" *-- "1" QuadcopterBody : body
    PyBulletDrone "1" *-- "1" DroneCamera : camera
    PyBulletDrone "1" *-- "4" _PID : alt, roll, pitch, yaw
    PyBulletDrone ..> DroneCamera : capture(pose from body.get_state())

    note for PyBulletDrone "Flight controller layer: cascaded PIDs.\nvelocity error to target tilt to torque,\nseparate altitude loop, rate-controlled yaw.\nAlso owns the proximity safety net."
    note for QuadcopterBody "Knows nothing about forward or hover -\njust a rigid box pushed by force and torque."
```

### 2. Module-level builders and helpers

`environment.py` and `banana.py` hold no classes at all — only functions plus the tunable constants. `PyBulletSimulator.connect()` is the single caller of both.

```mermaid
classDiagram
    direction LR

    class environment {
        <<module>>
        -_ASSET_DIR : str
        +FLOOR_TEXTURE : str
        +OBSTACLE_TEXTURE : str
        +GOAL_X = 16.0
        +COURSE_MARGIN = 3.0
        -_add_box(half_extents, position, texture_id) int
        +build_environment() dict
    }

    class banana {
        <<module>>
        +BANANA_PHOTO : str
        +BANANA_WIDTH = 1.2
        +BANANA_HEIGHT = 1.45
        +STAND_HALF_WIDTH = 0.06
        +TEXTURE_ROWS = 48
        -_asset_dir : str
        -_write_texture(path) float
        -_write_card(path, aspect) float
        -_assets() tuple
        +add_banana(position, yaw_degrees=180.0) tuple
    }

    class simulator_interface {
        <<module>>
        +EMPTY_INPUT : dict
        +SimulatorInterface
    }

    class pybullet_simulator {
        <<module>>
        +SimulatorError = p.error
        +DEBUG_CAMERA_DISTANCE, _YAW, _PITCH
        +HEADING_LINE_LENGTH = 1.5
        +TEST_OBSTACLE_DISTANCE = 4.0
        +TEST_OBSTACLE_SPEED = 1.5
        +TEST_OBSTACLE_HALF_EXTENTS
        +PyBulletSimulator
    }

    class pybullet_drone {
        <<module>>
        +GRAVITY = 9.81
        +PHYSICS_DT = 1/240
        +MASS, MIN_ALTITUDE, NORMAL_ALTITUDE, MAX_ALTITUDE
        +MAX_SPEED, MAX_YAW_RATE, MAX_TILT
        +SAFETY_DISTANCE, CRITICAL_DISTANCE, RETREAT_SPEED
        +FORWARD_CONE_DEGREES = 70
        +ALTITUDE_STEP, RETURN_TO_NORMAL_STEP
        +ALT_KP, ALT_KI, ALT_KD
        +VEL_TO_TILT_KP, TILT_KP, TILT_KD, YAW_KP
        +EMERGENCY_HOVER_STEPS = 180
        +_PID
        +PyBulletDrone
    }

    class camera {
        <<module>>
        +DroneCamera
    }

    class quadcopter_body {
        <<module>>
        +QuadcopterBody
    }

    environment ..> environment : build_environment() calls _add_box() 7x
    banana ..> banana : _assets() calls _write_card() calls _write_texture()

    pybullet_simulator ..> environment : build_environment(), OBSTACLE_TEXTURE
    pybullet_simulator ..> banana : add_banana(banana_position)
    pybullet_simulator ..> simulator_interface : SimulatorInterface, EMPTY_INPUT shape
    pybullet_simulator ..> pybullet_drone : PyBulletDrone, PHYSICS_DT, GRAVITY
    pybullet_drone ..> camera : DroneCamera
    pybullet_drone ..> quadcopter_body : QuadcopterBody

    note for environment "Obstacle surfaces are striped on purpose:\noptical flow needs pixel-level gradients.\nReturns plane, obstacles, goal_x, bounds."
    note for banana "A photo on a flat card, not a 3D model -\nstock COCO YOLOv8n never boxed the model.\nThe stand has collision, the card does not."
```

### 3. Evaluation, Tests and Tools

`TrialMetrics` is the only class in this layer; everything else is a script that drives the simulator and prints a summary.

```mermaid
classDiagram
    direction TB

    class TrialMetrics {
        +goal_x : float
        +start_pos
        +decision_steps : int
        +distance_traveled : float
        +min_obstacle_distance : float
        +emergency_stops, collisions : int
        +completed, crashed, successful : bool
        +avoidance_attempts, avoidance_successes : int
        +flight_time, average_speed : float
        -_last_pos
        -_speed_sum, _speed_samples
        -_was_avoiding : bool
        -_avoidance_pending_deadline
        +__init__(goal_x)
        +start(position)
        +update(position, speed, obstacle_distance, emergency_stop_count, collided, is_avoidance_command)
        +finalize(decision_dt) TrialMetrics
        +summary_line() str
    }

    class metrics {
        <<module>>
        +CONFIRM_WINDOW_STEPS = 45
        +TrialMetrics
        +print_summary(trials)
    }

    class run_trials {
        <<module>>
        +DECISION_INTERVAL_STEPS = 8
        +MAX_TRIAL_SECONDS = 25
        +run_trial() TrialMetrics
        +main()
    }

    class test_escape_sim {
        <<script>>
        +MODE : hover, hover2, fly, cruise
        +TEST_MODE, SPAWN, RUN_CYCLES
        -_record_floor(state)
        +prints escape events, closest approach
    }

    class test_banana_sim {
        <<script>>
        +MODE : eat, scare
        +SCARE_AFTER_FEED_S = 3.0
        +MAX_SIM_S = 90.0
        +prints state timeline, Giant Fiber firings
    }

    class test_main_gui {
        <<script>>
        +CLICKS : list
        +RUN_AFTER_LAST_CLICK = 130
        +update_test_obstacles(self, position, yaw_degrees, dt)
        +fake_click(self)
        +decide(self, flow, state=None)
    }

    class drone_step_response {
        <<script>>
        +axis : strafe or forward
        +CONST=value overrides of pybullet_drone
        +prints velocity, displacement, tilt every 0.1s
    }

    metrics *-- TrialMetrics
    run_trials ..> metrics : TrialMetrics, print_summary
    run_trials ..> environment : build_environment()
    run_trials ..> PyBulletDrone : drives directly, no PyBulletSimulator

    test_escape_sim ..> PyBulletSimulator
    test_banana_sim ..> PyBulletSimulator
    drone_step_response ..> PyBulletSimulator
    test_main_gui ..> PyBulletSimulator : monkeypatches update_test_obstacles and _left_click_triggered

    note for test_main_gui "The only test that runs the real main.py.\nThe others re-implement its decision loop,\nso they miss bugs that live in main.py."
    note for run_trials "Bypasses SimulatorInterface entirely:\nconnects p.DIRECT itself and resets the sim\nper trial, with a randomized start y."
```

## Activity diagrams

Where the class diagrams show what exists, these show what *happens* — the branching the code actually takes. Four activities plus the flight state machine they drive.

### 4. `PyBulletDrone.step()` — one physics tick (240 Hz)

The whole control cycle. Note the ordering that matters: the state machine and collision check run *before* the PIDs, and the proximity safety net overwrites `target_vx`/`target_vy` *after* the controller set them but *before* they become a tilt — so it can veto any command, manual or autonomous.

```mermaid
flowchart TD
    A([step called]) --> B["_run_state_machine()"]

    B --> B1{"current state?"}
    B1 -->|taking_off| B2{"within 0.05 m of NORMAL_ALTITUDE<br/>and vertical speed settled?"}
    B2 -->|yes| B3["state = flying"]
    B2 -->|no| C
    B1 -->|landing| B4{"altitude under 0.08 m<br/>and vertical speed settled?"}
    B4 -->|yes| B5["state = landed<br/>zero target_vx, target_vy, target_yaw_rate"]
    B4 -->|no| C
    B1 -->|emergency| B6["_emergency_timer += 1"]
    B6 --> B7{"timer over EMERGENCY_HOVER_STEPS?"}
    B7 -->|yes| B8["land()"]
    B7 -->|no| C
    B1 -->|idle, landed, flying| C
    B3 --> C
    B5 --> C
    B8 --> C

    C["_check_safety_and_collisions()<br/>body.get_contacts()"] --> C1{"hit an obstacle, or hit the<br/>ground while taking off or flying?"}
    C1 -->|"yes, and not already in emergency"| C2["collided = true, emergency = true<br/>emergency_stop_count += 1<br/>_emergency_timer = 0<br/>state = emergency, hover()"]
    C1 -->|"no, and state is not emergency"| C3["emergency = false"]
    C1 -->|"otherwise, leave the flags alone"| D
    C2 --> D
    C3 --> D

    D["read position, orientation, velocity<br/>rotate angular velocity into body frame<br/>so roll and pitch damping stay correct while yawing"] --> E{"state is idle or landed?"}
    E -->|yes| E1["p.stepSimulation()"]
    E1 --> EZ([return, motors off])
    E -->|no| F{"state is landing?"}

    F -->|yes| F1["_landing_altitude -= 0.4 * dt<br/>ramp toward the ground"]
    F -->|no| F2["clamp target_altitude<br/>to MIN_ALTITUDE..MAX_ALTITUDE"]
    F1 --> G
    F2 --> G

    G["altitude loop:<br/>thrust = MASS * GRAVITY + alt_pid.update(error, dt, -vz)<br/>clamped at 0"] --> H["min_obstacle_distance =<br/>_closest_obstacle_distance(position, yaw)"]

    H --> H1{"distance under CRITICAL_DISTANCE?"}
    H1 -->|yes| H2["safety_override = true<br/>override target to reverse at RETREAT_SPEED"]
    H1 -->|no| H3{"distance under SAFETY_DISTANCE?"}
    H3 -->|yes| H4["safety_override = true<br/>cap forward target at 0<br/>braking, backing away and strafing still allowed"]
    H3 -->|"no, or no obstacles registered"| I
    H2 --> I
    H4 --> I

    I["outer loop: rotate measured velocity into body frame<br/>velocity error to desired pitch and roll<br/>clamped to MAX_TILT"] --> J["inner loop: pitch and roll PIDs to torques<br/>yaw rate PID to torque_yaw"]
    J --> K["body.apply_control(thrust, roll, pitch, yaw)"]
    K --> L["p.stepSimulation()"]
    L --> Z([return])
```

### 5. Flight state machine

`PyBulletDrone.state` — what the decisions in activity 4 are moving between. Only `takeoff()`, `land()` and a collision change it from outside.

```mermaid
stateDiagram-v2
    [*] --> idle : __init__ / reset()

    idle --> taking_off : takeoff()
    landed --> taking_off : takeoff()
    taking_off --> flying : reached NORMAL_ALTITUDE<br/>and vertical speed settled

    taking_off --> emergency : collision
    flying --> emergency : collision
    emergency --> landing : _emergency_timer over<br/>EMERGENCY_HOVER_STEPS, calls land()

    taking_off --> landing : land()
    flying --> landing : land()
    emergency --> landing : land()
    landing --> landed : altitude under 0.08 m<br/>and vertical speed settled

    landed --> idle : reset()
    flying --> idle : reset()

    note right of idle
        idle and landed are the only states
        where step() skips the PIDs entirely
        and just calls stepSimulation
    end note
    note right of emergency
        entered by _check_safety_and_collisions
        on any obstacle contact, or on ground
        contact while taking off or flying;
        hover() is issued immediately
    end note
```

### 6. The decision cycle — a full trial

`Evaluation/run_trials.py`'s loop, which is also the shape `main.py` and the three test harnesses use: physics at 240 Hz, decisions every 8th tick at 30 Hz. The pre-exploration hover gate is the subtle part — the safety layer is skipped entirely until the drone has been `flying` long enough, or its stuck-detector primes on the intentional hover.

```mermaid
flowchart TD
    A([run_trial]) --> B["p.resetSimulation(), setGravity<br/>build_environment()"]
    B --> C["PyBulletDrone with randomized start y<br/>ReflexController, SafetyLayer, TrialMetrics"]
    C --> D["drone.takeoff()"]
    D --> E{"physics steps remaining?"}
    E -->|no| ZF["metrics.finalize(decision_dt)"]

    E -->|yes| F["drone.step()<br/>see activity 4"]
    F --> G{"tick index mod<br/>DECISION_INTERVAL_STEPS == 0?"}
    G -->|"no, physics only"| E

    G -->|yes| H["frame = drone.get_camera_frame()<br/>convert to grayscale"]
    H --> I{"flight_state == flying?"}
    I -->|yes| I1["flying_cycle_count += 1"]
    I -->|no| I2["flying_cycle_count = 0"]
    I1 --> J
    I2 --> J

    J{"have a previous frame?"}
    J -->|"no, first cycle"| P
    J -->|yes| K["compute_flow(prev, gray)<br/>derotate_flow by measured yaw rate<br/>grid_flow_strengths"]

    K --> L{"flying_cycle_count over<br/>HOVER_BEFORE_EXPLORE_CYCLES?"}
    L -->|"no, still in the pre-explore hover"| M1["cmd = EMPTY_CMD<br/>safety skipped, level = CLEAR"]
    L -->|yes| M2["raw_cmd = controller.decide(flow, state)"]
    M2 --> M3["safety.apply(raw_cmd, flow, position,<br/>already_avoiding)"]
    M1 --> N
    M3 --> N

    N["apply_command(drone, cmd)<br/>sets target velocities, yaw rate, altitude"] --> O["is_avoidance = safety_info active flag"]
    O --> P["prev_gray = gray"]

    P --> Q["closest_distance to every course obstacle<br/>keep the minimum"]
    Q --> R["metrics.update(position, speed, distance,<br/>emergency_stop_count, collided, is_avoidance)"]
    R --> S{"reached goal_x, or state == landed?"}
    S -->|yes| ZF
    S -->|no| E

    ZF --> ZG(["TrialMetrics, then print_summary across trials"])
```

### 7. `TrialMetrics.update()` — scoring one decision cycle

How an avoidance manoeuvre gets counted. Rising-edge detection means holding a turn for 40 cycles is still one attempt, and an attempt only becomes a success once it has stayed crash-free for `CONFIRM_WINDOW_STEPS` (~1.5 s).

```mermaid
flowchart TD
    A([update called once per decision cycle]) --> B["decision_steps += 1<br/>accumulate distance from last position<br/>accumulate speed sample"]
    B --> C{"obstacle_distance reported?"}
    C -->|yes| C1["min_obstacle_distance = min(so far, this)"]
    C -->|no| D
    C1 --> D

    D{"emergency_stop_count<br/>higher than recorded?"}
    D -->|yes| D1["emergency_stops = collisions = new count"]
    D -->|no| E
    D1 --> E

    E{"collided this cycle?"}
    E -->|yes| E1["crashed = true<br/>drop any pending avoidance deadline,<br/>that attempt just failed"]
    E -->|no| F
    E1 --> F

    F{"position x at or past goal_x?"}
    F -->|yes| F1["completed = true"]
    F -->|no| G
    F1 --> G

    G{"avoidance command active<br/>and it was not last cycle?"}
    G -->|"yes, rising edge"| G1["avoidance_attempts += 1<br/>set deadline = decision_steps + CONFIRM_WINDOW_STEPS"]
    G -->|"no, held or idle"| H
    G1 --> H

    H["_was_avoiding = is_avoidance_command"] --> I{"a deadline is pending,<br/>now reached, and still not crashed?"}
    I -->|yes| I1["avoidance_successes += 1<br/>clear the deadline"]
    I -->|no| Z
    I1 --> Z
    Z([return])
```

### 8. `update_test_obstacles()` — the on-demand looming stimulus

`PyBulletSimulator`'s debug hook, called once per decision cycle. A left-click conjures a block ahead of the drone's current heading, which then flies straight back at it; the harnesses in `Tests/` call `_spawn_test_obstacle()` directly instead of faking a click.

```mermaid
flowchart TD
    A([update_test_obstacles called]) --> B{"_left_click_triggered()?<br/>scan getMouseEvents for a<br/>left button press"}
    B -->|yes| C["_spawn_test_obstacle(position, yaw)"]
    C --> C1["place a textured box<br/>TEST_OBSTACLE_DISTANCE ahead of the heading"]
    C1 --> C2["record direction straight back at the drone<br/>remaining = TEST_OBSTACLE_DISTANCE"]
    C2 --> C3["drone.obstacle_ids.add(body_id)<br/>so the proximity net can see it"]
    C3 --> D
    B -->|no| D

    D{"more live test obstacles<br/>to advance?"}
    D -->|no| Z(["_test_obstacles = the survivors"])
    D -->|yes| E["remaining -= TEST_OBSTACLE_SPEED * dt"]
    E --> F{"remaining at or below 0?"}
    F -->|"yes, it has passed the drone"| G["p.removeBody(id)<br/>drone.obstacle_ids.discard(id)<br/>drop it"]
    F -->|no| H["advance position along direction<br/>resetBasePositionAndOrientation<br/>keep it"]
    G --> D
    H --> D
```
