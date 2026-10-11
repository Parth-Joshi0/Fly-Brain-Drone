# Simulator UML Diagrams

UML diagrams for the `Simulator/` folder. See [README.md](README.md) for what
each file does and how to run the tests, tools and evaluation.

Python modules that only contain functions and constants are shown as
`<<module>>` classes, following `BananaModel/UML.md`.

The folder has three layers:

- **Contracts** (`simulator_interface.py`, plus `Drone/drone_interface.py`):
  the abstract classes `main.py` talks to. Swapping the sim for the real
  Tello means writing one new implementation of each.
- **PyBullet implementation** (`pybullet_simulator.py`, `pybullet_drone.py`,
  `quadcopter_body.py`, `camera.py`, `environment.py`, `banana.py`): the
  virtual drone, its camera, and the course it flies through.
- **Harnesses** (`Tests/`, `Tools/`, `Evaluation/`): scripts that drive the
  above and print a summary.

## 1. Class diagram: contracts and the flight stack

Every member of `SimulatorInterface` and `DroneInterface` is abstract except
`SimulatorInterface.is_connected()` and
`SimulatorInterface.update_test_obstacles()`, which have working defaults so
real hardware does not have to implement them.

```mermaid
classDiagram
    direction TB

    class SimulatorInterface {
        <<abstract>>
        +float physics_dt
        +connect() dict
        +create_drone(start_pos) DroneInterface
        +poll_input() dict
        +update_debug_view(position, yaw_degrees) None
        +tick() None
        +is_connected() bool
        +update_test_obstacles(position, yaw_degrees, dt) None
    }

    class DroneInterface {
        <<abstract>>
        +takeoff() None
        +land() None
        +hover() None
        +move_forward(speed) None
        +move_backward(speed) None
        +move_left(speed) None
        +move_right(speed) None
        +turn_left(rate) None
        +turn_right(rate) None
        +get_camera_frame() ndarray
        +get_state() dict
        +step() None
        +reset() None
    }

    class PyBulletSimulator {
        -tuple _banana_position
        -dict _env
        -PyBulletDrone _drone
        -int _heading_line_id
        -int _test_obstacle_texture
        -list _test_obstacles
        -int _MOUSE_BUTTON_EVENT = 2
        +PyBulletSimulator(banana_position)
        +connect() dict
        +create_drone(start_pos) PyBulletDrone
        +poll_input() dict
        +update_debug_view(position, yaw_degrees) None
        +is_connected() bool
        +tick() None
        +update_test_obstacles(position, yaw_degrees, dt) None
        -_left_click_triggered() bool
        -_spawn_test_obstacle(position, yaw_degrees) None
    }

    class PyBulletDrone {
        +QuadcopterBody body
        +DroneCamera camera
        +tuple start_pos
        +int ground_id
        +set obstacle_ids
        +str state
        +float target_altitude
        +float target_vx
        +float target_vy
        +float target_yaw_rate
        +bool collided
        +bool emergency
        +int emergency_stop_count
        +float min_obstacle_distance
        +bool safety_override
        +PyBulletDrone(start_pos, ground_id, obstacle_ids)
        +takeoff() None
        +land() None
        +hover() None
        +move_forward(speed) None
        +move_backward(speed) None
        +move_left(speed) None
        +move_right(speed) None
        +move_up() None
        +move_down() None
        +relax_altitude() None
        +turn_left(rate) None
        +turn_right(rate) None
        +get_camera_frame() ndarray
        +get_state() dict
        +step() None
        +reset() None
        -_reset_flight_vars() None
        -_current_altitude() float
        -_effective_target_altitude() float
        -_closest_obstacle_distance(position, yaw) float
        -_run_state_machine() None
        -_check_safety_and_collisions() None
    }

    class _PID {
        +float kp
        +float ki
        +float kd
        +float integral
        +float prev_error
        +_PID(kp, ki, kd)
        +update(error, dt, derivative) float
        +reset() None
    }

    class QuadcopterBody {
        +float mass
        +int id
        +tuple start_pos
        +QuadcopterBody(start_pos, mass, half_extents)
        +apply_control(thrust, roll, pitch, yaw) None
        +get_state() dict
        +get_contacts() list
        +closest_distance(other_body_id, max_distance) float
        +reset(start_pos) None
    }

    class DroneCamera {
        +int width
        +int height
        +float fov
        +float near
        +float far
        +float aspect
        +projection_matrix
        +DroneCamera(width, height, fov, near, far)
        +capture(position, orientation_quat) ndarray
    }

    SimulatorInterface <|-- PyBulletSimulator : implements
    DroneInterface <|-- PyBulletDrone : implements

    PyBulletSimulator --> "1" PyBulletDrone : creates and owns
    PyBulletDrone *-- "1" QuadcopterBody : body
    PyBulletDrone *-- "1" DroneCamera : camera
    PyBulletDrone *-- "4" _PID : altitude, roll, pitch, yaw
```

`PyBulletDrone` is the flight-controller layer: cascaded PID loops turn a
desired velocity into a target tilt, that tilt into a torque, and a separate
loop turns a target altitude into thrust, with yaw rate-controlled directly.
It also owns the proximity safety net. `QuadcopterBody` knows nothing about
"forward" or "hover" — it is one rigid box pushed around by force and torque.
`DroneCamera` is handed the pose that `PyBulletDrone` read from
`QuadcopterBody.get_state()`; it never queries the body itself.

`QuadcopterBody` defaults to `mass=1.0` and `half_extents` of 0.15 by 0.15 by
0.04 m; `DroneCamera` defaults to 320 by 240 at a 75 degree field of view.

## 2. Class diagram: modules, builders and constants

`environment.py` and `banana.py` hold no classes at all — only functions plus
the tunable constants. `PyBulletSimulator.connect()` is the single caller of
both.

```mermaid
classDiagram
    direction LR

    class environment {
        <<module>>
        +str FLOOR_TEXTURE
        +str OBSTACLE_TEXTURE
        +float GOAL_X = 16.0
        +float COURSE_MARGIN = 3.0
        -str _ASSET_DIR
        -_add_box(half_extents, position, texture_id) int
        +build_environment() dict
    }

    class banana {
        <<module>>
        +str BANANA_PHOTO
        +float BANANA_WIDTH = 1.2
        +float BANANA_HEIGHT = 1.45
        +float STAND_HALF_WIDTH = 0.06
        +int TEXTURE_ROWS = 48
        -str _asset_dir
        -_write_texture(path) float
        -_write_card(path, aspect) float
        -_assets() tuple
        +add_banana(position, yaw_degrees) tuple
    }

    class simulator_interface {
        <<module>>
        +dict EMPTY_INPUT
    }

    class pybullet_simulator {
        <<module>>
        +SimulatorError
        +float DEBUG_CAMERA_DISTANCE = 7.0
        +int DEBUG_CAMERA_YAW = 50
        +int DEBUG_CAMERA_PITCH = -35
        +float HEADING_LINE_LENGTH = 1.5
        +float TEST_OBSTACLE_DISTANCE = 4.0
        +float TEST_OBSTACLE_SPEED = 1.5
        +list TEST_OBSTACLE_HALF_EXTENTS
    }

    class pybullet_drone {
        <<module>>
        +float GRAVITY = 9.81
        +float PHYSICS_DT
        +float MASS = 1.0
        +float MIN_ALTITUDE = 0.5
        +float NORMAL_ALTITUDE = 1.5
        +float MAX_ALTITUDE = 3.0
        +float MAX_SPEED = 2.0
        +float MAX_YAW_RATE = 1.5
        +float MAX_TILT = 0.45
        +float SAFETY_DISTANCE = 0.6
        +float CRITICAL_DISTANCE = 0.3
        +float RETREAT_SPEED = 0.25
        +int FORWARD_CONE_DEGREES = 70
        +float ALTITUDE_STEP = 0.02
        +float RETURN_TO_NORMAL_STEP = 0.01
        +float ALT_KP = 10.0
        +float ALT_KI = 1.0
        +float ALT_KD = 9.0
        +float VEL_TO_TILT_KP = 0.18
        +float TILT_KP = 0.09
        +float TILT_KD = 0.025
        +float YAW_KP = 0.22
        +int EMERGENCY_HOVER_STEPS = 180
    }

    class camera {
        <<module>>
    }

    class quadcopter_body {
        <<module>>
    }

    simulator_interface *-- SimulatorInterface : defines
    pybullet_simulator *-- PyBulletSimulator : defines
    pybullet_drone *-- PyBulletDrone : defines
    pybullet_drone *-- _PID : defines
    camera *-- DroneCamera : defines
    quadcopter_body *-- QuadcopterBody : defines

    pybullet_simulator ..> environment : build_environment
    pybullet_simulator ..> banana : add_banana
    pybullet_simulator ..> simulator_interface : implements contract
    pybullet_simulator ..> pybullet_drone : creates the drone
    pybullet_drone ..> camera : mounts a camera
    pybullet_drone ..> quadcopter_body : wraps a body
```

`PHYSICS_DT` is 1/240 s, so one `step()` is one 240 Hz tick.
`build_environment()` calls `_add_box()` seven times — two perimeter walls, a
box to dodge, two walls forming a narrow passage, and a pillar — then adds a
collision-free goal marker and returns `plane`, `obstacles`, `goal_x` and
`bounds`. Obstacle surfaces are striped on purpose: optical flow needs real
pixel-level gradients to track.

In `banana.py` the call chain is `_assets()` to `_write_card()` to
`_write_texture()`. The banana is a real photo on a flat card rather than a 3D
model, because stock COCO YOLOv8n never boxed the model in PyBullet's
renderer. The stand has collision so the proximity net sees it; the card does
not, since it only needs to be seen.

## 3. Class diagram: harnesses

`TrialMetrics` is the only class in this layer; everything else is a module or
a script.

```mermaid
classDiagram
    direction TB

    class TrialMetrics {
        +float goal_x
        +tuple start_pos
        +int decision_steps
        +float distance_traveled
        +float min_obstacle_distance
        +int emergency_stops
        +int collisions
        +bool completed
        +bool crashed
        +bool successful
        +int avoidance_attempts
        +int avoidance_successes
        +float flight_time
        +float average_speed
        -tuple _last_pos
        -float _speed_sum
        -int _speed_samples
        -bool _was_avoiding
        -int _avoidance_pending_deadline
        +TrialMetrics(goal_x)
        +start(position) None
        +update(position, speed, obstacle_distance, stops, collided, avoiding) None
        +finalize(decision_dt) TrialMetrics
        +summary_line() str
    }

    class metrics {
        <<module>>
        +int CONFIRM_WINDOW_STEPS = 45
        +print_summary(trials) None
    }

    class run_trials {
        <<module>>
        +int DECISION_INTERVAL_STEPS = 8
        +int MAX_TRIAL_SECONDS = 25
        +run_trial() TrialMetrics
        +main() None
    }

    class test_escape_sim {
        <<script>>
        +str MODE
        +bool TEST_MODE
        +bool SPAWN
        +bool MAGNITUDE
        +int RUN_CYCLES
        -_record_floor(state) float
    }

    class test_banana_sim {
        <<script>>
        +str MODE
        +bool USE_BRAIN
        +bool OPTOMOTOR
        +float SCARE_AFTER_FEED_S = 3.0
        +float MAX_SIM_S = 90.0
    }

    class test_main_gui {
        <<script>>
        +list CLICKS
        +int RUN_AFTER_LAST_CLICK = 130
        +update_test_obstacles(self, position, yaw_degrees, dt) None
        +fake_click(self) bool
        +decide(self, flow, state) dict
    }

    class drone_step_response {
        <<script>>
        +str axis
    }

    metrics *-- TrialMetrics : defines
    run_trials ..> metrics : TrialMetrics and print_summary
    run_trials ..> environment : build_environment
    run_trials ..> PyBulletDrone : drives the drone directly

    test_escape_sim ..> PyBulletSimulator : drives
    test_banana_sim ..> PyBulletSimulator : drives
    drone_step_response ..> PyBulletSimulator : drives
    test_main_gui ..> PyBulletSimulator : monkeypatches
```

Two structural details the table in the README does not show.
`run_trials.py` bypasses `SimulatorInterface` entirely: it connects
`p.DIRECT` itself and resets the simulation once per trial, with a randomized
start `y`. `test_main_gui.py` is the only harness that runs the real
`main.py`; it replaces `PyBulletSimulator.update_test_obstacles` and
`_left_click_triggered` on the class, and `FlyBrainController.decide`, to fake
clicks and record escapes. The others re-implement `main.py`'s decision loop,
so they miss bugs that live in `main.py` itself.

`drone_step_response.py` and `test_escape_sim.py` both accept `CONST=value`
arguments that override module constants for that run, in `pybullet_drone.py`
and `flybrain_controller.py` respectively.

## 4. Activity diagram: `PyBulletDrone.step()`, one physics tick

The whole control cycle, run once per 240 Hz tick. The ordering matters: the
state machine and the collision check run before the PIDs, and the proximity
safety net overwrites `target_vx` and `target_vy` after the controller set
them but before they become a tilt — so it can veto any command, manual or
autonomous.

```mermaid
flowchart TD
    start([step called]) --> sm["_run_state_machine()<br/>read altitude and vertical speed"]
    sm --> which{"current state?"}

    which -- taking_off --> arrived{"within 0.05 m of NORMAL_ALTITUDE<br/>and vertical speed settled?"}
    arrived -- yes --> fly["state = flying"] --> coll
    arrived -- no --> coll

    which -- landing --> down{"altitude under 0.08 m<br/>and vertical speed settled?"}
    down -- yes --> grounded["state = landed<br/>zero all three velocity targets"] --> coll
    down -- no --> coll

    which -- emergency --> timer["_emergency_timer += 1"]
    timer --> expired{"timer past EMERGENCY_HOVER_STEPS?"}
    expired -- yes --> autoland["land()"] --> coll
    expired -- no --> coll

    which -- other --> coll

    coll["_check_safety_and_collisions()<br/>body.get_contacts()"] --> hit{"hit an obstacle, or hit the ground<br/>while taking off or flying?"}
    hit -- "yes, and not already in emergency" --> panic["collided = true, emergency = true<br/>emergency_stop_count += 1<br/>_emergency_timer = 0<br/>state = emergency, hover()"] --> read
    hit -- "no, and state is not emergency" --> clear["emergency = false"] --> read
    hit -- otherwise --> read

    read["read position, orientation, velocity<br/>rotate angular velocity into body frame<br/>so tilt damping survives a yaw"] --> motors{"state is idle or landed?"}
    motors -- yes --> idlestep["p.stepSimulation()"] --> off([return, motors off])

    motors -- no --> isLanding{"state is landing?"}
    isLanding -- yes --> ramp["_landing_altitude drifts down<br/>at 0.4 m per second"] --> thrust
    isLanding -- no --> clamp["clamp target_altitude between<br/>MIN_ALTITUDE and MAX_ALTITUDE"] --> thrust

    thrust["altitude loop<br/>thrust = weight plus alt PID output<br/>clamped at zero"] --> near["min_obstacle_distance =<br/>_closest_obstacle_distance<br/>nearest point inside the forward cone"]

    near --> critical{"closer than CRITICAL_DISTANCE?"}
    critical -- yes --> retreat["safety_override = true<br/>override target to reverse<br/>at RETREAT_SPEED"] --> tilt
    critical -- no --> warn{"closer than SAFETY_DISTANCE?"}
    warn -- yes --> cap["safety_override = true<br/>cap forward target at zero<br/>braking and strafing still allowed"] --> tilt
    warn -- "no, or no obstacles" --> tilt

    tilt["outer loop<br/>rotate measured velocity into body frame<br/>velocity error becomes desired pitch and roll<br/>clamped to MAX_TILT"] --> torque["inner loop<br/>pitch and roll PIDs give torques<br/>yaw rate PID gives torque_yaw"]
    torque --> apply["body.apply_control"]
    apply --> sim["p.stepSimulation()"]
    sim --> done([return])
```

Only obstacles roughly ahead count for the safety net —
`_closest_obstacle_distance()` keeps a point only if it lies within
`FORWARD_CONE_DEGREES` of the heading. Without that cone, a wall merely
alongside the drone, being correctly steered around, zeroed forward speed
anyway.

## 5. State diagram: the flight state machine

`PyBulletDrone.state`, which is what the decisions in activity 4 move
between. Only `takeoff()`, `land()`, `reset()` and a collision change it.

```mermaid
stateDiagram-v2
    [*] --> idle

    idle --> taking_off : takeoff
    landed --> taking_off : takeoff
    taking_off --> flying : reached cruise altitude

    taking_off --> emergency : collision
    flying --> emergency : collision
    emergency --> landing : hover timer expired

    taking_off --> landing : land
    flying --> landing : land
    emergency --> landing : land
    landing --> landed : touched down

    landed --> idle : reset
    flying --> idle : reset
```

`idle` and `landed` are the only states where `step()` skips the PIDs
entirely and just advances the simulation, so the drone sits still with its
motors off. `emergency` is entered by `_check_safety_and_collisions()` on any
obstacle contact, or on ground contact while taking off or flying; `hover()`
is issued immediately, and after `EMERGENCY_HOVER_STEPS` ticks the drone
lands itself.

## 6. Activity diagram: the decision cycle, one trial

`Evaluation/run_trials.py`'s loop, which is the shape `main.py` and the three
test harnesses all use: physics at 240 Hz, a decision every eighth tick at
30 Hz.

```mermaid
flowchart TD
    start([run_trial]) --> build["p.resetSimulation and setGravity<br/>build_environment()"]
    build --> make["PyBulletDrone with a randomized start y<br/>ReflexController, SafetyLayer, TrialMetrics"]
    make --> takeoff["drone.takeoff()"]
    takeoff --> more{"physics steps left?"}
    more -- no --> fin["metrics.finalize(decision_dt)"]

    more -- yes --> tick["drone.step()<br/>see activity 4"]
    tick --> isDecision{"eighth tick?"}
    isDecision -- no --> more

    isDecision -- yes --> frame["frame = drone.get_camera_frame()<br/>convert to grayscale"]
    frame --> airborne{"flight_state is flying?"}
    airborne -- yes --> inc["flying_cycle_count += 1"] --> hasPrev
    airborne -- no --> zero["flying_cycle_count = 0"] --> hasPrev

    hasPrev{"is there a previous frame?"}
    hasPrev -- "no, first cycle" --> keep
    hasPrev -- yes --> flowcalc["compute_flow<br/>derotate_flow by measured yaw rate<br/>grid_flow_strengths"]

    flowcalc --> gate{"flying_cycle_count past<br/>HOVER_BEFORE_EXPLORE_CYCLES?"}
    gate -- "no, pre-explore hover" --> empty["cmd = EMPTY_CMD<br/>safety layer skipped"] --> send
    gate -- yes --> decide["raw_cmd = controller.decide"]
    decide --> safety["safety.apply with the already_avoiding flag"] --> send

    send["apply_command sets target velocities,<br/>yaw rate and altitude on the drone"] --> keep["prev_gray = gray"]
    keep --> measure["closest_distance to every course obstacle,<br/>keep the minimum"]
    measure --> record["metrics.update"]
    record --> stop{"reached goal_x, or landed?"}
    stop -- yes --> fin
    stop -- no --> more

    fin --> out([TrialMetrics, then print_summary across trials])
```

The pre-exploration hover gate is the subtle part: until the drone has been
`flying` for `HOVER_BEFORE_EXPLORE_CYCLES`, the controller and the safety
layer are skipped entirely and an empty command is sent, because the safety
layer's stuck-detector would otherwise prime on the intentional hover.

## 7. Activity diagram: `TrialMetrics.update()`

How one decision cycle is scored, and in particular how an avoidance
manoeuvre gets counted.

```mermaid
flowchart TD
    start([update, once per decision cycle]) --> acc["decision_steps += 1<br/>add distance from the last position<br/>add a speed sample"]
    acc --> hasDist{"an obstacle distance was reported?"}
    hasDist -- yes --> keepMin["min_obstacle_distance keeps the smaller"] --> stops
    hasDist -- no --> stops

    stops{"emergency_stop_count above the record?"}
    stops -- yes --> bump["emergency_stops and collisions<br/>both take the new count"] --> crashed
    stops -- no --> crashed

    crashed{"collided this cycle?"}
    crashed -- yes --> fail["crashed = true<br/>drop any pending avoidance deadline,<br/>that attempt just failed"] --> goal
    crashed -- no --> goal

    goal{"position x at or past goal_x?"}
    goal -- yes --> complete["completed = true"] --> edge
    goal -- no --> edge

    edge{"avoidance command active now<br/>but not last cycle?"}
    edge -- "yes, rising edge" --> attempt["avoidance_attempts += 1<br/>deadline = now plus CONFIRM_WINDOW_STEPS"] --> remember
    edge -- "no, held or idle" --> remember

    remember["_was_avoiding = the current flag"] --> due{"a deadline is pending, now reached,<br/>and still not crashed?"}
    due -- yes --> win["avoidance_successes += 1<br/>clear the deadline"] --> out
    due -- no --> out
    out([return])
```

Rising-edge detection means holding a turn for forty cycles is still one
attempt, and an attempt only becomes a success once it has stayed crash-free
for `CONFIRM_WINDOW_STEPS`, about 1.5 s at 30 Hz.

## 8. Activity diagram: `update_test_obstacles()`

`PyBulletSimulator`'s debug hook, called once per decision cycle. A left
click conjures a block ahead of the drone's current heading, which then flies
straight back at it — a looming stimulus on demand, instead of waiting to
stumble into a real obstacle.

```mermaid
flowchart TD
    start([update_test_obstacles]) --> click{"_left_click_triggered?<br/>scan getMouseEvents for<br/>a left button press"}
    click -- yes --> spawn["_spawn_test_obstacle"]
    spawn --> place["place a textured box<br/>TEST_OBSTACLE_DISTANCE ahead of the heading"]
    place --> aim["record a direction straight back at the drone<br/>remaining = TEST_OBSTACLE_DISTANCE"]
    aim --> register["drone.obstacle_ids.add<br/>so the proximity net can see it"]
    register --> loop
    click -- no --> loop

    loop{"another live test obstacle?"}
    loop -- no --> out([keep only the survivors])
    loop -- yes --> advance["remaining decreases by<br/>TEST_OBSTACLE_SPEED times dt"]
    advance --> past{"remaining at or below zero?"}
    past -- "yes, it has passed the drone" --> drop["p.removeBody<br/>drone.obstacle_ids.discard<br/>forget it"] --> loop
    past -- no --> move["advance the position<br/>resetBasePositionAndOrientation<br/>keep it"] --> loop
```

The harnesses in `Tests/` call `_spawn_test_obstacle()` directly rather than
faking a click, except `test_main_gui.py`, which replaces
`_left_click_triggered()` so that the real `main.py` loop does the spawning.
