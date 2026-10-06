"""
Interactive, visual run of the FlyBrain drone sim:

    Virtual Camera -> Optic Flow -> 3D Navigation Controller -> Safety
    Override -> Drone Interface -> PyBullet Drone

Starts in AUTONOMOUS mode: takes off, hovers briefly, then flies on its
own - no keyboard input needed. With USE_BANANA (below) that means finding
and eating a banana, as the real Tello does; otherwise exploring the course. Every command (autonomous or manual)
passes through the SafetyLayer before reaching the drone, so it can
override even a continuous "go forward" request if something's too
close. Opens a PyBullet GUI window plus two debug windows (camera feed87
with a telemetry overlay, and a color-coded optical-flow visualization).

Keys (work in BOTH modes):
    Space     emergency hover
    L         emergency land
    R         reset (re-takes off after resetting)
    M         switch AUTONOMOUS <-> MANUAL

Manual-only movement (only read while in MANUAL mode):
    Up/Down       move forward / backward (relative to current heading)
    Left/Right    strafe left / right (relative to current heading)
    W/S           altitude up / down
    Q/E           yaw left / right

Debug: left-click anywhere in the sim window to spawn a block that flies
straight at the drone from wherever it's currently facing - an on-demand
looming stimulus for testing the escape reflex (works in either mode).

Run:
    python main.py

Everything PyBullet-specific (GUI window, test course, keyboard input,
debug 3D view) lives behind Simulator/pybullet_simulator.py's
PyBulletSimulator, the SimulatorInterface implementation constructed
below - same pattern as PyBulletDrone/DroneInterface. Swapping to the
real drone means writing one new SimulatorInterface (and DroneInterface)
implementation and changing the two lines below that construct them;
nothing else in this file, or in NeuralPathways/ or Controllers/reflex_controller.py,
needs to change.
"""

import cv2

from Simulator.pybullet_simulator import PyBulletSimulator, SimulatorError
from Controllers.reflex_controller import ReflexController
from Controllers.manual_controller import ManualController
from Controllers.safety_layer import SafetyLayer
from Controllers.commands import AVOIDING_STATES, EMPTY_CMD, apply_command
from NeuralPathways.EscapeNeuron.optical_flow import (compute_flow, derotate_flow, grid_flow_strengths,
                                                      signed_hemifield_flow, FlowVisualizer, LoomingDetector)

# The hand-written CRUISE/AVOID_LEFT/AVOID_RIGHT state machine (default),
# or the real Fly-Brain connectome circuit (NeuralPathways/
# flybrain_controller.py -> connectome_worker.py's LC4/LPLC2 ->
# DNp01/03/06 looming subnetwork) - same decide(flow, state) contract,
# swap one line to try it. Needs a Python env with brian2/pandas/pyarrow
# installed (see connectome_worker.py's docstring); it's spawned as a
# subprocess, so this venv itself doesn't need those.
USE_FLYBRAIN = True

# The DNg02 flight-motor / stabilizer circuit (NeuralPathways/
# StabilizerNeuron/), in the same Brian2 network as the escape circuit.
# Needs USE_FLYBRAIN. Feeds it residual (uncommanded) rotation and
# translational optic flow; it adds a yaw correction and, when cruising, a
# thrust adjustment. See FlyBrainController's optomotor=True.
USE_OPTOMOTOR = True

# Banana seek-and-eat, as flown on the real Tello (Drone/fly_tello.py):
# BananaModel's YOLOv8 + ripeness detector on the camera frame drives
# NeuralPathways/FoodNeuron/feeding_behaviour.py's SEARCH -> APPROACH -> FEED ->
# DONE -> LAND, with the fly brain (if USE_FLYBRAIN) as a background fear
# reflex - click to throw a box at it while it eats. Puts a banana on a stand
# at BANANA_POSITION; replaces course exploration, and NEURON_TEST_MODE
# doesn't apply. Needs torch + ultralytics (requirements.txt).
USE_BANANA = True
BANANA_POSITION = (2.2, -0.5)   # m - within the ~2.5 m the detector sees it
                                 # from in the sim (Simulator/banana.py)

# Debug: when True, the autonomous controller still runs every cycle off
# live optic flow (so FlyBrainController's neurons keep firing off the
# real camera feed, logged to flybrain_spikes.log - see its
# _log_spikes()), but its forward/yaw output is discarded and the drone
# just hovers in place instead of actually flying on it - except for the
# Giant Fiber ESCAPE dodge, which is let through. Isolates "is the
# circuit reacting correctly to something looming" from the flight
# dynamics - useful together with the click-to-spawn test obstacle, since
# the drone no longer drifts/turns out of the obstacle's straight-line path.
NEURON_TEST_MODE = True

DECISION_INTERVAL_STEPS = 8   # 240Hz physics / 8 = 30Hz decision loop, in the
                               # ~20-30 FPS range requested for the camera
MAX_CONSECUTIVE_SIM_FAILURES = 10  # see the SimulatorError retry in main()
HOVER_BEFORE_EXPLORE_CYCLES = 45  # ~1.5s at 30Hz: "take off, hover briefly,
                                   # THEN begin exploring" (item 1)

_ACTION_FOR_STATE = {
    "CRUISE": "FORWARD",
    "AVOID_LEFT": "WALL LEFT -> TURN RIGHT",
    "AVOID_RIGHT": "WALL RIGHT -> TURN LEFT",
    "WALL_ESCAPE": "WALL ESCAPE",
    "EMERGENCY_ESCAPE": "EMERGENCY ESCAPE",
    "BOUNDARY_RETURN": "RETURN TO COURSE",
    # BananaSeekController's (feeding_behaviour.py's) states
    "SEARCH": "LOOKING FOR FOOD",
    "APPROACH": "FLYING TO BANANA",
    "FEED": "EATING",
    "SCARED": "SCARED - BACKING AWAY",
    "WAIT": "WAITING - IS IT SAFE?",
    "DONE": "FULL",
    "LAND": "LANDING",
}


def emergency_keys(input_state):
    """Space/L/R work in both manual and autonomous mode - these are the
    keys item 1 says to keep regardless of mode (emergency hover,
    emergency land, reset)."""
    keys = {
        "hover": input_state["hover"],
        "land": input_state["land_pressed"],
        "reset": input_state["reset_pressed"],
    }
    # Only pressed keys: an unpressed Space must not clear a hover the
    # command already asked for (NEURON_TEST_MODE's hover got overwritten).
    return {name: True for name, pressed in keys.items() if pressed}


def action_label(mode, controller, safety_info):
    """The single, human-readable "what is it doing right now" string -
    an emergency override (SafetyLayer) always takes priority to display
    over the navigation controller's own state, since it means the
    controller's plan just got overruled."""
    if safety_info["stuck"]:
        return f"STUCK -> {safety_info['direction']}"
    if safety_info["level"] == "DANGER":
        return f"EMERGENCY {safety_info['direction']}"
    if mode != "autonomous":
        return "MANUAL"
    if controller.state == "ESCAPE":
        return f"ESCAPE {controller.escape_direction} (Giant Fiber)"
    return _ACTION_FOR_STATE.get(controller.state, controller.state)


def draw_debug_overlay(frame, mode, controller, state, final_cmd, safety_info, flow):
    nav_state = controller.state
    dist = state["min_obstacle_distance"]
    dist_str = f"{dist:.2f}m" if dist is not None else "n/a"
    action = action_label(mode, controller, safety_info)
    lines = [
        f"MODE: {mode.upper()}  (M to switch)   STATE: {nav_state}",
        f"ACTION: {action}",
        f"LEFT FLOW: {flow['left']:.2f}   CENTER FLOW: {flow['center']:.2f}   RIGHT FLOW: {flow['right']:.2f}",
        f"EXPANSION (1/s): L={flow['expansion_left']:.2f} C={flow['expansion_center']:.2f} "
        f"R={flow['expansion_right']:.2f}",
        f"cmd: fwd={final_cmd['forward_speed']:.2f}m/s  yaw={final_cmd['yaw_rate']:.2f}rad/s",
    ]
    if hasattr(controller, "current_cell"):
        # ReflexController-only: FlyBrainController doesn't track an
        # exploration grid, so these fields don't apply to it.
        lines.append(
            f"TURN CMD: {controller.turn_command}   GRID CELL: {controller.current_cell}   "
            f"LEAST-VISITED: {controller.least_visited_cell}   "
            f"TIME NEAR WALL: {controller.wall_time_seconds:.1f}s"
        )
    if hasattr(controller, "food"):
        food = controller.food
        target = food.current_target
        lines.append(
            f"HUNGER: {food.hunger:.0f}%   FOOD: {food.state}   "
            f"TARGET: {target.label if target else 'NONE'}   SIZE: {food.last_box_ratio:.1%}   "
            f"SCARES: {controller.scares}"
        )
    brain = getattr(controller, "brain", controller)
    if getattr(brain, "optomotor", False):
        dng02 = brain.dng02
        lines.append(
            f"DNg02: L={dng02['n_left']} R={dng02['n_right']} steer={dng02['steer']:+.2f} "
            f"thrust={dng02['thrust']:.2f}   flow rot={flow.get('rotation', 0.0):+.2f} "
            f"trans={flow.get('translation', 0.0):+.2f}"
        )
    lines += [
        f"alt: {state['altitude']:.2f}m -> {state['target_altitude']:.2f}m   yaw: {state['yaw_degrees']:.0f}deg",
        f"pos: ({state['position'][0]:.1f}, {state['position'][1]:.1f})   "
        f"forward_vel: {state['actual_vx']:.2f}m/s",
        f"top/bottom flow: T={flow['top']:.2f} B={flow['bottom']:.2f}   physics dist: {dist_str}",
        f"safety override: {'YES' if safety_info['active'] else 'NO'} ({safety_info['level']})"
        f"{'  [STUCK]' if safety_info['stuck'] else ''}   collided: {state['collided']}",
    ]
    out = frame.copy()
    if hasattr(controller, "detector"):
        controller.detector.annotate(out, controller.detections)
    for i, line in enumerate(lines):
        color = (0, 0, 255) if safety_info["active"] else (0, 255, 0)
        cv2.putText(out, line, (6, 16 + i * 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, color, 1, cv2.LINE_AA)
    return out


# safety_info when SafetyLayer isn't consulted this cycle
NO_SAFETY_OVERRIDE = {"level": "CLEAR", "active": False, "direction": "FORWARD", "stuck": False}


class FlowSensor:
    """Camera frame -> the flow dict every controller's decide() reads:
    grid flow strengths, signed rotation/translation for DNg02, and the
    looming expansion per column. Also shows the flow debug window."""

    def __init__(self):
        self.looming = LoomingDetector()
        self.flow_viz = None  # lazily sized from the first camera frame
        self.prev_gray = None
        self.flow = {"left": 0.0, "right": 0.0, "top": 0.0, "bottom": 0.0, "center": 0.0,
                     "expansion_left": 0.0, "expansion_center": 0.0, "expansion_right": 0.0}

    def update(self, frame, capture_state, dt):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.flow_viz is None:
            height, width = gray.shape[:2]
            self.flow_viz = FlowVisualizer(width, height)

        expansion = self.looming.update(gray, capture_state["orientation"], dt)
        if self.prev_gray is not None:
            raw_flow = compute_flow(self.prev_gray, gray)
            derotated = derotate_flow(raw_flow, capture_state["yaw_rate"], dt)
            self.flow = grid_flow_strengths(derotated)
            # Signed rotation/translation for DNg02 - only these
            # two keys: its "left"/"right" are signed means, not
            # the magnitudes grid_flow_strengths put there.
            hemifields = signed_hemifield_flow(derotated)
            self.flow["rotation"] = hemifields["rotation"]
            self.flow["translation"] = hemifields["translation"]
            cv2.imshow("Optical Flow", self.flow_viz.render(derotated))
        self.flow.update({f"expansion_{side}": value for side, value in expansion.items()})
        self.prev_gray = gray
        return self.flow

    def reset(self):
        self.prev_gray = None
        self.looming.reset()


def build_autonomous(env, clock):
    """The autonomous controller the USE_* flags ask for."""
    brain = None
    if USE_FLYBRAIN:
        from NeuralPathways.flybrain_controller import FlyBrainController
        brain = FlyBrainController(bounds=env["bounds"], optomotor=USE_OPTOMOTOR)
    if USE_BANANA:
        # Imported here so the other modes don't need torch/ultralytics
        from BananaModel.banana_detector import BananaDetector
        from Controllers.banana_seek_controller import BananaSeekController
        return BananaSeekController(BananaDetector(), brain=brain, clock=clock)
    if brain is not None:
        return brain
    return ReflexController(bounds=env["bounds"])


def choose_command(mode, exploring, input_state, frame, flow, state_now, autonomous, manual, safety):
    """The command for this cycle, and the SafetyLayer verdict on it.
    Returns (final_cmd, safety_info)."""
    if mode == "manual":
        raw_cmd = manual.decide(input_state)
    elif exploring:
        if USE_BANANA:
            autonomous.see(frame)
        raw_cmd = autonomous.decide(flow, state_now)  # still runs on live
                                                        # flow even in test mode -
                                                        # only its movement gets
                                                        # thrown away below
        if NEURON_TEST_MODE and not USE_BANANA and autonomous.state != "ESCAPE":
            raw_cmd = dict(EMPTY_CMD)
            raw_cmd["hover"] = True
        raw_cmd.update(emergency_keys(input_state))  # Space/L/R still work
    else:
        raw_cmd = dict(EMPTY_CMD)
        raw_cmd.update(emergency_keys(input_state))

    # Banana mode flies without the flow SafetyLayer, as the
    # real Tello does (fly_tello.py sends rc straight to the
    # drone): it's built for exploring - it speed-stages any
    # non-hover command up to cruise speed, reads eating in
    # place as STUCK after STUCK_WINDOW, and steers away from
    # the banana's stand once close. pybullet_drone.py's
    # physics-distance net still applies underneath.
    if mode == "manual" or (exploring and not USE_BANANA):
        # 2. Obstacle safety/reflex layer - final override
        # authority (this is what makes "hold/request forward
        # into a wall" impossible even while flying itself).
        # already_avoiding tells it the FSM is already turning
        # away from something, so it won't add a second,
        # possibly-disagreeing turn decision on top.
        already_avoiding = autonomous.state in AVOIDING_STATES
        return safety.apply(raw_cmd, flow, state_now["position"], already_avoiding)
    return raw_cmd, dict(NO_SAFETY_OVERRIDE)


def print_status(mode, autonomous, safety_info, final_cmd, state, flow):
    action = action_label(mode, autonomous, safety_info)
    # Same split as draw_debug_overlay: the turn/grid fields are
    # ReflexController-only, since FlyBrainController doesn't
    # track an exploration grid.
    grid_fields = ""
    if hasattr(autonomous, "current_cell"):
        grid_fields = (
            f"turn={autonomous.turn_command:>9} "
            f"cell={autonomous.current_cell} least_visited={autonomous.least_visited_cell} "
            f"wall_time={autonomous.wall_time_seconds:.1f}s "
        )
    print(
        f"[{mode:>10}][STATE={autonomous.state:>17}][ACTION={action:>22}] "
        f"fwd={final_cmd['forward_speed']:.2f}m/s {grid_fields}"
        f"pos=({state['position'][0]:.1f},{state['position'][1]:.1f}) "
        f"collided={state['collided']} avoidance={safety_info['active']} stuck={safety_info['stuck']} "
        f"LEFT={flow['left']:.2f} CENTER={flow['center']:.2f} RIGHT={flow['right']:.2f}",
        flush=True,
    )


def main():
    sim = PyBulletSimulator(banana_position=BANANA_POSITION if USE_BANANA else None)
    env = sim.connect()
    drone = sim.create_drone(start_pos=(0, 0, 0.05))
    manual = ManualController()
    step_count = 0
    # Physics time, not wall time: with the detector and the brain in
    # the loop the sim runs slower than real time, and feeding_behaviour's
    # timers (hunger, back-off, waits) are about what the drone did.
    autonomous = build_autonomous(env, clock=lambda: step_count * sim.physics_dt)
    safety = SafetyLayer()
    sensor = FlowSensor()

    mode = "autonomous"  # fully autonomous by default (item 1) - press M for manual
    drone.takeoff()
    final_cmd = dict(EMPTY_CMD)
    safety_info = dict(NO_SAFETY_OVERRIDE)
    flying_cycle_count = 0  # counts decision cycles spent in "flying" state,
                             # for the "hover briefly before exploring" grace period
    was_exploring = False

    consecutive_sim_failures = 0
    try:
        while True:
            try:
                drone.step()

                if step_count % DECISION_INTERVAL_STEPS == 0:
                    decision_dt = DECISION_INTERVAL_STEPS * sim.physics_dt

                    frame = drone.get_camera_frame()
                    flow = sensor.update(frame, drone.get_state(), decision_dt)

                    input_state = sim.poll_input()
                    if input_state["mode_toggle_pressed"]:
                        mode = "manual" if mode == "autonomous" else "autonomous"
                        print(f"--- switched to {mode.upper()} mode ---", flush=True)

                    state_now = drone.get_state()
                    if state_now["flight_state"] == "flying":
                        flying_cycle_count += 1
                    else:
                        flying_cycle_count = 0

                    # Item 1: take off, hover briefly, THEN start exploring. During
                    # that deliberate hover, skip the safety layer entirely rather
                    # than feeding it "not moving" position samples - otherwise
                    # its stuck-detector's window fills with the *intentional*
                    # hover before real navigation ever gets a turn, and it
                    # immediately (and permanently) thinks it's stuck the moment
                    # exploring starts.
                    exploring = mode == "autonomous" and flying_cycle_count > HOVER_BEFORE_EXPLORE_CYCLES
                    if was_exploring and not exploring:
                        # decide() stops being called (collision emergency, landing,
                        # manual mode) - without this the HUD kept showing whatever
                        # state it was last in, e.g. ESCAPE, indefinitely.
                        autonomous.reset()
                    was_exploring = exploring

                    final_cmd, safety_info = choose_command(
                        mode, exploring, input_state, frame, flow, state_now,
                        autonomous, manual, safety)

                    # 3. Send the final (possibly overridden) command to the drone
                    if apply_command(drone, final_cmd):
                        sensor.reset()
                        flying_cycle_count = 0
                        autonomous.reset()
                        safety.reset()

                    state = drone.get_state()
                    sim.update_debug_view(state["position"], state["yaw_degrees"])
                    sim.update_test_obstacles(state["position"], state["yaw_degrees"], decision_dt)

                    cv2.imshow("Drone Camera", draw_debug_overlay(
                        frame, mode, autonomous, state, final_cmd, safety_info, flow))
                    cv2.waitKey(1)

                    if step_count % (DECISION_INTERVAL_STEPS * 15) == 0:
                        print_status(mode, autonomous, safety_info, final_cmd, state, flow)

                    if state["position"][0] >= env["goal_x"]:
                        print("Course completed!", flush=True)
                        drone.land()

                step_count += 1
                sim.tick()
            except SimulatorError as exc:
                # PyBullet's GUI mode intermittently fails a single command
                # (seen as "GetBasePositionAndOrientation failed") while the
                # very next one works - reproduced with the OpenCV windows
                # open. Retry this step; only a lost connection or a run of
                # failures means the simulator is really gone.
                consecutive_sim_failures += 1
                if not sim.is_connected() or consecutive_sim_failures >= MAX_CONSECUTIVE_SIM_FAILURES:
                    raise
                print(f"[sim] transient simulator error ({exc}) - retrying step", flush=True)
                continue
            consecutive_sim_failures = 0

    except KeyboardInterrupt:
        print("\ninterrupted - shutting down", flush=True)
    except SimulatorError as exc:
        # Every pybullet command fails once the GUI window is gone, so
        # closing the window otherwise ends the run with a traceback
        # pointing at whichever call happened to come next (e.g. the
        # drone's own position read) instead of at the real cause.
        print(f"\nsimulator stopped responding ({exc}) - shutting down", flush=True)
    finally:
        # FlyBrainController runs brian2 in a subprocess; without this it
        # outlives main.py on every exit, clean or not. ReflexController
        # has no close() to call.
        if hasattr(autonomous, "close"):
            autonomous.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
