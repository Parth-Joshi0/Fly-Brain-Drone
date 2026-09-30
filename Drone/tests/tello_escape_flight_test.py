"""
First real FLYING test of the Giant Fiber escape circuit on the Tello.

    Tello video -> LoomingDetector -> FlyBrainController -> SafetyLayer ->
    TelloDrone (Drone/tello_drone.py) -> real RC control

Default behavior is HOVER - same as main.py's NEURON_TEST_MODE=True and
Simulator/tests/test_escape_sim.py's "hover" scenario: every decision cycle, the
brain still runs on the live camera feed, but its forward/yaw output is
thrown away and replaced with a plain hover UNLESS its state is "ESCAPE",
in which case the dodge command is let through untouched. So the drone
just sits there until something looms at it, dodges, then goes back to
hovering.

This is the flight step up from Drone/tests/tello_neuron_test.py, which
proved the perception half (DNp01 fires on a real swat, 5/5 hits, 0 false
positives on a desk test - see that script's docstring) with the
propellers OFF and nothing ever sent to the motors. This script is the
first time anything here actually commands the Tello to move. Two things
that test could not check are still unverified and load-bearing:

  1. The RC speed/rate scale in Drone/tello_drone.py
     (RC_SPEED_SCALE, _rate_to_rc's rate_at_100) is an ASSUMPTION, not a
     measured constant.
  2. The escape dodge commands full-speed strafe + full-speed backward
     simultaneously (ESCAPE_STRAFE_SPEED = ESCAPE_BACK_SPEED = 2.0 m/s in
     neural_pathways/flybrain_controller.py, tuned in sim) - clamped to 100%
     RC either way, so on real hardware that's "as fast as the Tello's
     RC will go," not a gentle nudge.

STRONGLY recommended before trusting this near a hand: fly it once with
no one near it and use the 'l'/'q' abort keys to confirm they land it
promptly, then trigger a dodge (wave something into frame from a safe
distance) and watch which way/how hard it moves before doing it close up.

Keys (the video window must be focused):
    l         land immediately
    q / SPACE force hover, then land shortly after (use if a dodge looks
              wrong mid-maneuver)

Safety limits:
    --seconds N   hard auto-land this many seconds after takeoff (default
                  15), regardless of anything else going on.

Run (needs djitellopy + opencv + numpy; the brain itself runs as a
subprocess under whichever env has brian2, see
neural_pathways/flybrain_controller.py):

    python Drone/tests/tello_escape_flight_test.py [--seconds 15] [--fov 55.6] [--log PATH]
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from neural_pathways.flybrain_controller import FlyBrainController
from safety_layer import SafetyLayer
from Drone.tello_drone import TelloDrone
from neural_pathways.escape_neuron.optical_flow import LoomingDetector, compute_flow, derotate_flow, grid_flow_strengths

# Deliberately NOT `import main` - main.py pulls in Simulator/
# pybullet_simulator.py at module level, which imports pybullet. Per
# TESTING.md's environment table, the "tello" env this script runs
# under has djitellopy/opencv/numpy only, no pybullet. EMPTY_CMD and
# apply_command are copied from main.py rather than imported.
EMPTY_CMD = {"forward_speed": 0.0, "strafe_speed": 0.0, "yaw_rate": 0.0,
             "altitude_delta": 0.0, "hover": False, "land": False, "reset": False,
             "pressed_direction": "-"}


def apply_command(drone, cmd):
    if cmd["reset"]:
        drone.reset()
        drone.takeoff()
        return True
    if cmd["land"]:
        drone.land()
        return False
    if drone.state != "flying":
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
        drone.relax_altitude()
    return False

# Same convention/derivation as Drone/tests/tello_neuron_test.py: the Tello's
# 82.6deg spec is diagonal; LoomingDetector's fov is vertical.
DEFAULT_VERTICAL_FOV = 55.6
PROC_WIDTH, PROC_HEIGHT = 320, 240

STREAM_SETTLE_SECONDS = 3.0
WARMUP_FRAMES = 15
FIRST_FRAME_TIMEOUT = 20.0
ARM_GRACE_SECONDS = 2.0  # post-takeoff settle before the brain's output is
                          # trusted - the climb itself is a big, non-looming
                          # expansion transient (main.py's equivalent is
                          # HOVER_BEFORE_EXPLORE_CYCLES)
EMERGENCY_HOVER_SECONDS = 1.0  # how long 'q'/SPACE hovers before landing

PLACEHOLDER_SHAPE = (300, 400)


def frame_not_ready(frame):
    return frame is None or frame.shape[:2] == PLACEHOLDER_SHAPE or not frame.any()


def open_tello(log):
    from djitellopy import Tello

    tello = Tello()
    tello.connect(wait_for_state=False)
    log("Connected (SDK mode). Waiting for state telemetry...")
    deadline = time.perf_counter() + 10.0
    while time.perf_counter() < deadline:
        try:
            if tello.get_current_state():
                break
        except Exception:
            pass
        time.sleep(0.1)
    else:
        log("WARNING: no state telemetry yet - continuing anyway.")

    try:
        battery = tello.get_battery()
        log(f"Battery: {battery}%")
        if battery < 20:
            log(f"WARNING: battery is low ({battery}%).")
    except Exception as exc:
        log(f"Battery: unknown ({exc})")

    tello.streamoff()
    time.sleep(1)
    tello.streamon()
    log(f"Stream on - settling for {STREAM_SETTLE_SECONDS}s")
    time.sleep(STREAM_SETTLE_SECONDS)
    return tello


def open_stream(tello, log):
    frame_read = tello.get_frame_read()
    deadline = time.perf_counter() + FIRST_FRAME_TIMEOUT
    while time.perf_counter() < deadline:
        frame = frame_read.frame
        if not frame_not_ready(frame):
            log(f"first real frame: {frame.shape[1]}x{frame.shape[0]}")
            return frame_read
        time.sleep(0.05)
    log(f"ABORTED: no real frame within {FIRST_FRAME_TIMEOUT}s")
    raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seconds", type=float, default=15.0, help="hard auto-land time after takeoff")
    parser.add_argument("--fov", type=float, default=DEFAULT_VERTICAL_FOV)
    parser.add_argument("--log", default=str(Path(__file__).with_suffix(".log")))
    args = parser.parse_args()

    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "w")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_file.write(line + "\n")
        log_file.flush()

    tello = open_tello(log)
    frame_read = open_stream(tello, log)

    looming = LoomingDetector(width=PROC_WIDTH, height=PROC_HEIGHT, fov=args.fov)
    log("starting the brain subprocess (brian2 build takes a few seconds)...")
    brain = FlyBrainController(bounds=None)
    log("brain ready")
    safety = SafetyLayer()
    drone = TelloDrone(tello, frame_read, PROC_WIDTH, PROC_HEIGHT)

    prev_gray = None
    prev_accepted = None
    last_accept_t = None
    accepted = 0
    escape_events = []
    state_counts = {}
    phase = "warmup"
    armed_t0 = None
    abort_mode = None       # None | "land_now" | "hover_then_land"
    abort_deadline = None

    try:
        log("Taking off...")
        drone.takeoff()
        log(f"Airborne (state={drone.state}). Auto-land in {args.seconds:.0f}s.")

        while True:
            # Polled every raw loop iteration (not just accepted frames) so
            # an abort key never gets lost behind a run of duplicate/
            # not-ready frames, which happen often on the Tello's stream.
            key = cv2.waitKey(1) & 0xFF
            if abort_mode is None:
                if key == ord("l"):
                    log("ABORT: 'l' pressed - landing now")
                    abort_mode, abort_deadline = "land_now", time.perf_counter()
                elif key in (ord("q"), ord(" ")):
                    log("ABORT: emergency key pressed - forcing hover, landing shortly")
                    abort_mode = "hover_then_land"
                    abort_deadline = time.perf_counter() + EMERGENCY_HOVER_SECONDS
                    brain.reset()

            frame = frame_read.frame
            if frame_not_ready(frame):
                time.sleep(0.005)
                continue

            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            small = cv2.resize(frame_bgr, (PROC_WIDTH, PROC_HEIGHT), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

            if prev_accepted is not None and np.array_equal(gray, prev_accepted):
                continue
            prev_accepted = gray
            accepted += 1

            now = time.perf_counter()
            dt = max(1e-3, (now - last_accept_t) if last_accept_t else 1.0 / 20.0)
            last_accept_t = now

            state = drone.get_state()
            expansion = looming.update(gray, state["orientation"], dt)
            flow = {"left": 0.0, "right": 0.0, "top": 0.0, "bottom": 0.0, "center": 0.0}
            if prev_gray is not None:
                raw_flow = compute_flow(prev_gray, gray)
                derotated = derotate_flow(raw_flow, state["yaw_rate"], dt)
                flow = grid_flow_strengths(derotated)
            prev_gray = gray
            flow.update({f"expansion_{side}": v for side, v in expansion.items()})

            if phase == "warmup" and accepted >= WARMUP_FRAMES:
                looming.reset()
                prev_gray = None
                phase = "arming"
                armed_t0 = time.perf_counter()
                log(f"warmup done ({accepted} frames) - arming (hover, brain output suppressed for "
                    f"{ARM_GRACE_SECONDS:.1f}s)")

            t_aloft = 0.0 if armed_t0 is None else time.perf_counter() - armed_t0
            exploring = phase in ("arming", "armed") and t_aloft >= ARM_GRACE_SECONDS
            if phase == "arming" and exploring:
                phase = "armed"
                looming.reset()
                prev_gray = None
                log("ARMED - hovering, will dodge on a swat")

            if abort_mode is not None:
                final, sinfo = dict(EMPTY_CMD), {"level": "CLEAR", "active": False, "direction": "FORWARD", "stuck": False}
                final["hover"] = True
                if time.perf_counter() >= abort_deadline:
                    log(f"abort ({abort_mode}): landing now")
                    break
            elif exploring:
                raw_cmd = brain.decide(flow, state)
                if brain.state != "ESCAPE":
                    raw_cmd = dict(EMPTY_CMD)
                    raw_cmd["hover"] = True
                already_avoiding = brain.state == "ESCAPE"
                final, sinfo = safety.apply(raw_cmd, flow, state["position"], already_avoiding)
                state_counts[brain.state] = state_counts.get(brain.state, 0) + 1
                if brain.state == "ESCAPE" and (not escape_events or escape_events[-1][1] != "run"):
                    escape_events.append((round(t_aloft, 2), "run", brain.escape_direction))
                    log(f"ESCAPE triggered t={t_aloft:.2f}s dir={brain.escape_direction} "
                        f"expansion L/C/R={expansion['left']:.2f}/{expansion['center']:.2f}/{expansion['right']:.2f}")
                elif brain.state != "ESCAPE" and escape_events and escape_events[-1][1] == "run":
                    escape_events[-1] = (escape_events[-1][0], "done", escape_events[-1][2])
            else:
                final, sinfo = dict(EMPTY_CMD), {"level": "CLEAR", "active": False, "direction": "FORWARD", "stuck": False}
                final["hover"] = True

            apply_command(drone, final)
            drone.step()

            view = small.copy()
            fired = exploring and brain.state == "ESCAPE"
            color = (0, 0, 255) if fired else (0, 255, 0)
            lines = [
                f"{phase.upper()}  t_aloft={t_aloft:5.1f}s / {args.seconds:.0f}s  frames={accepted}",
                f"EXP L={expansion['left']:.2f} C={expansion['center']:.2f} R={expansion['right']:.2f}",
                f"state={brain.state if exploring else '(suppressed)'}  safety={sinfo['level']}",
                f"cmd fwd={final['forward_speed']:+.2f} strafe={final['strafe_speed']:+.2f} yaw={final['yaw_rate']:+.2f}",
                "l=land  q/SPACE=emergency hover+land",
            ]
            for i, line in enumerate(lines):
                cv2.putText(view, line, (6, 16 + i * 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
            if fired:
                cv2.putText(view, "DNp01 ESCAPE", (60, PROC_HEIGHT - 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow("Tello escape flight test", view)

            if phase == "armed" and t_aloft >= args.seconds:
                log(f"max duration {args.seconds:.0f}s reached - landing")
                break

    except KeyboardInterrupt:
        log("KeyboardInterrupt - landing")
    finally:
        try:
            drone.land()
            log(f"Landed (state={drone.state}).")
        except Exception as exc:
            log(f"land() raised during shutdown: {exc!r}")
        try:
            brain.close()
        except Exception:
            pass
        try:
            tello.streamoff()
            tello.end()
        except Exception:
            pass
        cv2.destroyAllWindows()

        log("=" * 60)
        log(f"accepted_frames={accepted}")
        log(f"state_counts={state_counts}")
        log(f"escape_events={escape_events}")
        log_file.close()
        print(f"\nwrote {log_path}", flush=True)


if __name__ == "__main__":
    main()
