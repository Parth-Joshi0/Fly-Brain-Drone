"""
Interactive, visual run of the FlyBrain drone sim:

    Virtual Camera -> Optic Flow -> 3D Navigation Controller -> Safety
    Override -> Drone Interface -> PyBullet Drone

Starts in AUTONOMOUS mode: takes off, hovers briefly, then explores on
its own - no keyboard input needed. Every command (autonomous or manual)
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
nothing else in this file, or in neural_pathways/ or reflex_controller.py,
needs to change.
"""

import cv2

from Simulator.pybullet_simulator import PyBulletSimulator, SimulatorError
from reflex_controller import ReflexController
from Drone.manual_controller import ManualController
from safety_layer import SafetyLayer
from neural_pathways.escape_neuron.optical_flow import compute_flow, derotate_flow, grid_flow_strengths, FlowVisualizer, LoomingDetector

# The hand-written CRUISE/AVOID_LEFT/AVOID_RIGHT state machine (default),
# or the real Fly-Brain connectome circuit (neural_pathways/
# flybrain_controller.py -> fly_brain_controller.py's LC4/LPLC2 ->
# DNp01/03/06 looming subnetwork) - same decide(flow, state) contract,
# swap one line to try it. Needs a Python env with brian2/pandas/pyarrow
# installed (see fly_brain_controller.py's docstring); it's spawned as a
# subprocess, so this venv itself doesn't need those.
USE_FLYBRAIN = True

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
}

# States where the navigation FSM is already actively steering away from
# something - SafetyLayer won't layer its own (possibly disagreeing) turn
# decision on top of any of these, see already_avoiding in its apply().
_AVOIDING_STATES = ("AVOID_LEFT", "AVOID_RIGHT", "BOUNDARY_RETURN", "WALL_ESCAPE", "EMERGENCY_ESCAPE", "ESCAPE")

EMPTY_CMD = {"forward_speed": 0.0, "strafe_speed": 0.0, "yaw_rate": 0.0,
             "altitude_delta": 0.0, "hover": False, "land": False, "reset": False,
             "pressed_direction": "-"}


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
    lines += [
        f"alt: {state['altitude']:.2f}m -> {state['target_altitude']:.2f}m   yaw: {state['yaw_degrees']:.0f}deg",
        f"pos: ({state['position'][0]:.1f}, {state['position'][1]:.1f})   "
        f"forward_vel: {state['actual_vx']:.2f}m/s",
        f"top/bottom flow: T={flow['top']:.2f} B={flow['bottom']:.2f}   physics dist: {dist_str}",
        f"safety override: {'YES' if safety_info['active'] else 'NO'} ({safety_info['level']})"
        f"{'  [STUCK]' if safety_info['stuck'] else ''}   collided: {state['collided']}",
    ]
    out = frame.copy()
    for i, line in enumerate(lines):
        color = (0, 0, 255) if safety_info["active"] else (0, 255, 0)
        cv2.putText(out, line, (6, 16 + i * 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, color, 1, cv2.LINE_AA)
    return out


def apply_command(drone, cmd):
    """Dispatches one FINAL command dict (already passed through the
    safety layer) to the drone interface. Returns True if a reset
    happened, so the caller can clear stale vision/controller state."""

    if cmd["reset"]:
        drone.reset()
        drone.takeoff()
        return True

    if cmd["land"]:
        drone.land()
        return False

    if drone.state != "flying":
        # Still taking off / landing / recovering from an emergency - the
        # low-level state machine and safety net own the drone right now,
        # not the controller.
        return False

    if cmd["hover"]:
        drone.hover()
    else:
        drone.move_forward(cmd["forward_speed"])
        if cmd["strafe_speed"] > 0:
            drone.move_left(cmd["strafe_speed"])
        elif cmd["strafe_speed"] < 0:
            drone.move_right(-cmd["strafe_speed"])
        else:
            drone.move_left(0)
        if cmd["yaw_rate"] > 0:
            drone.turn_left(cmd["yaw_rate"])
        elif cmd["yaw_rate"] < 0:
            drone.turn_right(-cmd["yaw_rate"])
        else:
            drone.turn_left(0)

    if cmd["altitude_delta"] > 0:
        drone.move_up()
    elif cmd["altitude_delta"] < 0:
        drone.move_down()
    else:
        drone.relax_altitude()  # drift back toward NORMAL_ALTITUDE when idle

    return False


def main():
    sim = PyBulletSimulator()
    env = sim.connect()
    drone = sim.create_drone(start_pos=(0, 0, 0.05))
    manual = ManualController()
    if USE_FLYBRAIN:
        from neural_pathways.flybrain_controller import FlyBrainController
        autonomous = FlyBrainController(bounds=env["bounds"])
    else:
        autonomous = ReflexController(bounds=env["bounds"])
    safety = SafetyLayer()
    flow_viz = None  # lazily sized from the first camera frame (see below)
    looming = LoomingDetector()

    mode = "autonomous"  # fully autonomous by default (item 1) - press M for manual
    drone.takeoff()
    prev_gray = None
    flow = {"left": 0.0, "right": 0.0, "top": 0.0, "bottom": 0.0, "center": 0.0,
            "expansion_left": 0.0, "expansion_center": 0.0, "expansion_right": 0.0}
    final_cmd = dict(EMPTY_CMD)
    safety_info = {"level": "CLEAR", "active": False, "direction": "FORWARD", "stuck": False}
    flying_cycle_count = 0  # counts decision cycles spent in "flying" state,
                             # for the "hover briefly before exploring" grace period
    was_exploring = False

    step_count = 0
    consecutive_sim_failures = 0
    try:
        while True:
            try:
                drone.step()

                if step_count % DECISION_INTERVAL_STEPS == 0:
                    decision_dt = DECISION_INTERVAL_STEPS * sim.physics_dt

                    frame = drone.get_camera_frame()
                    capture_state = drone.get_state()
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    if flow_viz is None:
                        height, width = gray.shape[:2]
                        flow_viz = FlowVisualizer(width, height)

                    expansion = looming.update(gray, capture_state["orientation"], decision_dt)
                    if prev_gray is not None:
                        raw_flow = compute_flow(prev_gray, gray)
                        derotated = derotate_flow(raw_flow, capture_state["yaw_rate"], decision_dt)
                        flow = grid_flow_strengths(derotated)
                        cv2.imshow("Optical Flow", flow_viz.render(derotated))
                    flow.update({f"expansion_{side}": value for side, value in expansion.items()})
                    prev_gray = gray

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

                    if mode == "manual":
                        raw_cmd = manual.decide(input_state)
                    elif exploring:
                        raw_cmd = autonomous.decide(flow, state_now)  # still runs on live
                                                                        # flow even in test mode -
                                                                        # only its movement gets
                                                                        # thrown away below
                        if NEURON_TEST_MODE and autonomous.state != "ESCAPE":
                            raw_cmd = dict(EMPTY_CMD)
                            raw_cmd["hover"] = True
                        raw_cmd.update(emergency_keys(input_state))  # Space/L/R still work
                    else:
                        raw_cmd = dict(EMPTY_CMD)
                        raw_cmd.update(emergency_keys(input_state))

                    if mode == "manual" or exploring:
                        # 2. Obstacle safety/reflex layer - final override
                        # authority (this is what makes "hold/request forward
                        # into a wall" impossible even while flying itself).
                        # already_avoiding tells it the FSM is already turning
                        # away from something, so it won't add a second,
                        # possibly-disagreeing turn decision on top.
                        already_avoiding = autonomous.state in _AVOIDING_STATES
                        final_cmd, safety_info = safety.apply(raw_cmd, flow, state_now["position"], already_avoiding)
                    else:
                        final_cmd = raw_cmd
                        safety_info = {"level": "CLEAR", "active": False, "direction": "FORWARD", "stuck": False}

                    # 3. Send the final (possibly overridden) command to the drone
                    if apply_command(drone, final_cmd):
                        prev_gray = None
                        looming.reset()
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
