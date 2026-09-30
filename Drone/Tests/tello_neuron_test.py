"""
Table-top test of the Giant Fiber escape circuit on the REAL Tello - the
perception half only, with the drone sitting still on a desk.

    Tello video -> LoomingDetector -> FlyBrainController -> (log only)

THIS SCRIPT NEVER FLIES THE DRONE. It never calls takeoff(), land(), or
send_rc_control(). It only reads the video stream and the attitude
telemetry. Leave the propellers off. The whole point is to answer "does
DNp01 fire when I swat at it" with zero flight risk, before any of the
TelloDrone/DroneInterface control work exists.

Why a stationary drone is the right first test: FlyBrainController's loom
floor (_loom_floor) subtracts the expansion the drone's own motion would
produce, and on a desk that term is zero - LOOM_EXPANSION_FLOOR_PER_MPS *
0 plus a rotation term that stays ~0 while it isn't moving. So the floor
collapses to the LOOM_EXPANSION_FLOOR baseline and any expansion the
circuit sees is genuinely your hand, not self-motion the floor failed to
cancel. That isolates the one thing we don't know yet: whether Farneback
can recover clean expansion from the Tello's H.264 stream at all. Every
constant in the pipeline was calibrated against PyBullet's clean renders;
expansion is a spatial *derivative* of flow, so it is far more sensitive
to compression artifacts, rolling shutter and auto-exposure hunting than
plain flow magnitude is.

Run (needs djitellopy + opencv + numpy; the brain itself runs as a
subprocess under whichever env has brian2, see
NeuralPathways/flybrain_controller.py):

    pip install djitellopy
    python Drone/Tests/tello_neuron_test.py

    --seconds N          live phase length (default 60)
    --baseline N         quiet phase length before it (default 10)
    --fov DEG            camera VERTICAL fov (default 55.6, see below)
    --no-video           headless; disables swat marking (needs the window)
    --log PATH           where to write (default Drone/flight_logs/tello_neuron_test.log)

While the live phase runs: press SPACE the instant you swat, Q to stop
early. The SPACE markers are the most valuable thing in the log - they
give ground truth to line the neuron response up against, so "DNp01 fired
14 times" can be scored as hits vs false positives instead of guessed at.

DO ONE DRY RUN WITH INTERNET FIRST (no Tello needed, it will just fail to
connect): brian2 compiles its generated C++ on first use, and you will be
on the Tello's wifi with no internet during the real test. Getting the
compile cached beforehand avoids discovering that at the worst moment.

Everything lands in one plain-text log file (metadata header, per-cycle
TSV rows, swat markers, summary) so it can be handed over offline without
needing anything else from the session.
"""

import argparse
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

import NeuralPathways.flybrain_controller as fbc
from NeuralPathways.flybrain_controller import FlyBrainController
from NeuralPathways.EscapeNeuron.optical_flow import (LoomingDetector, compute_flow, derotate_flow,
                                 grid_flow_strengths)
from Drone.tello_drone import TELLO_YAW_SIGN

# The Tello's 82.6 deg is a DIAGONAL spec; LoomingDetector's fov argument
# is the VERTICAL one (f = (height/2)/tan(fov/2)), same convention as the
# sim's Simulator/camera.py DroneCamera(fov=75) -> computeProjectionMatrixFOV.
# For a 4:3 frame, 82.6 diagonal works out to ~55.6 vertical / ~70.3
# horizontal. Passing 82.6 here would make the focal length ~35% too short
# and mis-scale the rotation-removal homography, so expansion would come
# out wrong. Worth confirming empirically - the published number varies by
# source and the video stream may be cropped relative to stills - hence
# --fov.
TELLO_DIAGONAL_FOV = 82.6
DEFAULT_VERTICAL_FOV = 55.6

# LoomingDetector was calibrated at 320x240; the Tello streams 960x720
# (same 4:3), so downscale rather than re-tune the cell/margin geometry.
PROC_WIDTH, PROC_HEIGHT = 320, 240

STREAM_SETTLE_SECONDS = 3.0   # tello_camera.py uses the same wait - the
                               # first frames are pre-keyframe garbage
WARMUP_FRAMES = 15             # discarded after that, then the detector is
                               # reset so its EMA/median start clean
FIRST_FRAME_TIMEOUT = 20.0     # give up (loudly) if the decoder never delivers

# djitellopy 2.5.0's BackgroundFrameRead seeds .frame with a 400x300 black
# placeholder - np.zeros([300, 400, 3]) - and NEVER returns None. So "no
# real frame yet" has to be detected by shape/content, not by None. This
# matters because everything here is resized to 320x240 before use, which
# would hide the shape difference and quietly push black frames through the
# whole pipeline: zero flow, zero expansion, and a log that looks exactly
# like "the circuit never fired".
PLACEHOLDER_SHAPE = (300, 400)


def frame_not_ready(frame):
    return (frame is None
            or frame.shape[:2] == PLACEHOLDER_SHAPE
            or not frame.any())


def euler_deg_to_quat(roll_deg, pitch_deg, yaw_deg):
    """Tello attitude (degrees) -> (x, y, z, w), matching the convention
    NeuralPathways/EscapeNeuron/optical_flow._quat_to_matrix and flybrain_controller._pitch_roll
    expect: body frame x forward / y left / z up, rotation composed as
    R = Rz(yaw) @ Ry(pitch) @ Rx(roll) (what PyBullet's
    getQuaternionFromEuler produces, which is what the sim fed them).

    NOTE: the Tello's own sign conventions for pitch/roll are not reliably
    documented and differ between firmware/wrapper versions. On a level
    desk all three are ~0 so it does not matter here, but this must be
    verified against the real thing before any flight test - a sign error
    would make the rotation-removal homography rotate the wrong way and
    *add* apparent expansion during maneuvers instead of removing it.
    """
    r, p_, y = (math.radians(roll_deg), math.radians(pitch_deg), math.radians(yaw_deg))
    cr, sr = math.cos(r / 2), math.sin(r / 2)
    cp, sp = math.cos(p_ / 2), math.sin(p_ / 2)
    cy, sy = math.cos(y / 2), math.sin(y / 2)
    return (
        sr * cp * cy - cr * sp * sy,   # x
        cr * sp * cy + sr * cp * sy,   # y
        cr * cp * sy - sr * sp * cy,   # z
        cr * cp * cy + sr * sp * sy,   # w
    )


def percentile(values, q):
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((q / 100.0) * (len(ordered) - 1)))))
    return ordered[idx]


class Logger:
    """Writes one self-contained plain-text file: '#' metadata/marker/
    summary lines plus tab-separated per-cycle rows under a COLUMNS
    header. Flushed every line - the run may well end by yanking wifi or
    Ctrl-C, and a truncated log is still worth reading."""

    COLUMNS = [
        "t_s", "phase", "dt_ms", "dup",
        "exp_l", "exp_c", "exp_r",
        "floor", "loom_l", "loom_r",
        "escape", "brain_yaw", "brain_fwd", "state", "spikes",
        "flow_l", "flow_c", "flow_r",
        "pitch", "roll", "yaw", "att_ok", "tello_vgx", "tello_h", "mean_px",
        "ms_grab", "ms_flow", "ms_brain", "ms_cycle",
    ]

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "w")

    def meta(self, key, value):
        self._f.write(f"# {key}: {value}\n")
        self._f.flush()

    def raw(self, line):
        self._f.write(f"# {line}\n")
        self._f.flush()

    def header(self):
        self._f.write("# COLUMNS (tab-separated)\n")
        self._f.write("\t".join(self.COLUMNS) + "\n")
        self._f.flush()

    def row(self, values):
        self._f.write("\t".join(str(v) for v in values) + "\n")
        self._f.flush()

    def close(self):
        self._f.close()


def git_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True, timeout=5)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                               capture_output=True, text=True, timeout=5)
        return out.stdout.strip() + (" (dirty)" if dirty.stdout.strip() else "")
    except Exception:
        return "unknown"


def write_header(log, args):
    log.meta("test", "tello_neuron_test - stationary Tello, escape circuit perception only")
    log.meta("started", time.strftime("%Y-%m-%d %H:%M:%S"))
    log.meta("git", git_commit())
    log.meta("python", f"{platform.python_version()} on {platform.platform()}")
    log.meta("opencv", cv2.__version__)
    log.meta("proc_resolution", f"{PROC_WIDTH}x{PROC_HEIGHT}")
    log.meta("vertical_fov_deg", args.fov)
    log.meta("baseline_seconds", args.baseline)
    log.meta("live_seconds", args.seconds)
    log.raw("--- flybrain_controller constants in effect ---")
    for name in ("LOOM_EXPANSION_FLOOR", "LOOM_EXPANSION_FLOOR_PER_MPS",
                 "LOOM_EXPANSION_FLOOR_PER_RAD", "LOOM_ROTATION_FLOOR_DECAY",
                 "LOOM_EXPANSION_SPAN", "ESCAPE_STATE_THRESHOLD",
                 "AVOID_STATE_THRESHOLD", "ESCAPE_CYCLES",
                 "ESCAPE_REFRACTORY_CYCLES", "ESCAPE_SIDE_MIN_YAW",
                 "ESCAPE_SIDE_MIN_FLOW_DIFF"):
        log.meta(name, getattr(fbc, name))
    log.raw("--- note: drone stationary, so actual_vx is forced to 0.0 and the")
    log.raw("--- loom floor reduces to LOOM_EXPANSION_FLOOR + rotation term")
    # Which interpreter hosts the brian2 subprocess is worth recording:
    # _find_python_with_brian2() short-circuits on the CURRENT interpreter if
    # it happens to import brian2, so simply launching from a different
    # python silently changes which brian2 runs - and they are not equally
    # fast (measured 47ms/step vs 20ms/step between two installs here, i.e.
    # the difference between missing and meeting the 33ms budget). Set
    # FLYBRAIN_PYTHON to pin it.
    try:
        from NeuralPathways.flybrain_controller import _find_python_with_brian2
        log.meta("brain_interpreter", _find_python_with_brian2())
    except Exception as exc:
        log.meta("brain_interpreter", f"could not resolve: {exc!r}")
    log.meta("FLYBRAIN_PYTHON", os.environ.get("FLYBRAIN_PYTHON", "(unset)"))


def open_tello(log):
    try:
        from djitellopy import Tello
    except ImportError:
        print("djitellopy is not installed in this interpreter.\n"
              "    pip install djitellopy\n"
              f"(current interpreter: {sys.executable})", flush=True)
        log.raw("ABORTED: djitellopy not installed")
        raise SystemExit(1)

    tello = Tello()
    # wait_for_state=False matches the existing tello_test.py/tello_camera.py -
    # the state socket sometimes never reports on a fresh connect and the
    # default would block forever.
    tello.connect(wait_for_state=False)
    print("Connected (SDK mode). Waiting for state telemetry...", flush=True)
    # The handshake being acknowledged does NOT mean the state stream is up;
    # every get_*() that reads it raises TelloException until the first
    # packet lands.
    wait_for_state(tello, log)
    try:
        battery = tello.get_battery()
        print(f"Battery: {battery}%", flush=True)
    except Exception as exc:
        battery = "unknown"
        print(f"Battery: unknown ({exc})", flush=True)
    log.meta("battery_percent", battery)

    tello.streamoff()
    time.sleep(1)
    tello.streamon()
    print(f"Stream on - settling for {STREAM_SETTLE_SECONDS}s", flush=True)
    time.sleep(STREAM_SETTLE_SECONDS)
    return tello


def open_stream(tello, log):
    """Starts the background decoder and blocks until it delivers a real
    frame, rather than letting the loop run on the black placeholder.

    get_frame_read() itself opens the PyAV container (5s timeout) on the
    calling thread, so it can block or raise before returning."""
    try:
        frame_read = tello.get_frame_read()
    except Exception as exc:
        log.raw(f"ABORTED: could not open the video stream: {exc!r}")
        print(f"Could not open the video stream: {exc}\n"
              "Check you are on the Tello's wifi and that nothing else is "
              "holding UDP 11111 (another script, or a previous run that did "
              "not shut down).", flush=True)
        raise SystemExit(1)

    deadline = time.perf_counter() + FIRST_FRAME_TIMEOUT
    while time.perf_counter() < deadline:
        frame = frame_read.frame
        if not frame_not_ready(frame):
            log.meta("stream_frame_shape", f"{frame.shape[1]}x{frame.shape[0]}")
            print(f"first real frame: {frame.shape[1]}x{frame.shape[0]}", flush=True)
            return frame_read
        time.sleep(0.05)

    log.raw(f"ABORTED: no real frame within {FIRST_FRAME_TIMEOUT}s "
            f"(decoder only ever produced the placeholder)")
    print(f"No video frame within {FIRST_FRAME_TIMEOUT}s - the decoder never "
          "produced anything but its black placeholder. Try power-cycling the "
          "Tello and re-running; 'streamon' sometimes needs a fresh boot.",
          flush=True)
    raise SystemExit(1)


def read_attitude(tello):
    """Tello attitude + a couple of telemetry fields, best-effort.

    Reads the state dict once rather than calling get_pitch/get_roll/... -
    each of those re-reads the dict AND raises TelloException on a missing
    key, so the getter route means up to five raise/catch cycles per
    decision cycle whenever the state stream is down.

    'ok' reports whether any state arrived at all. It gets logged per cycle
    because it changes how the run must be read: with no state the
    orientation is identity every cycle, so the rotation term of the loom
    floor stays 0 and the floor sits at exactly LOOM_EXPANSION_FLOOR. On a
    level desk that is harmless - arguably ideal - but it must not be
    mistaken for "rotation was measured and found to be zero"."""
    try:
        state = tello.get_current_state() or {}
    except Exception:
        state = {}

    def num(key, default=0):
        try:
            return int(float(state[key]))
        except (KeyError, TypeError, ValueError):
            return default

    return {
        "ok": bool(state),
        "pitch": num("pitch"),
        "roll": num("roll"),
        "yaw": num("yaw"),
        # Logged for reference only - NOT fed to the loom floor. On a desk
        # it should read ~0; what it looks like here tells us how noisy it
        # will be when it does feed the floor in flight.
        "vgx": num("vgx"),
        "height": num("h"),
    }


def wait_for_state(tello, log, timeout=10.0):
    """Polls for the first state packet. connect(wait_for_state=False) - what
    tello_test.py/tello_camera.py use, and what this script inherited -
    returns as soon as the 'command' handshake is acknowledged, so the state
    dict can still be empty afterwards and any get_*() reading it raises
    TelloException. Not fatal here: a stationary level drone does not need
    attitude, so this warns and continues rather than aborting."""
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        try:
            if tello.get_current_state():
                waited = timeout - (deadline - time.perf_counter())
                log.meta("state_telemetry", f"available after {waited:.1f}s")
                return True
        except Exception:
            pass
        time.sleep(0.1)

    log.meta("state_telemetry", f"UNAVAILABLE after {timeout:.0f}s")
    log.raw("--- no attitude/battery telemetry: orientation is identity every")
    log.raw("--- cycle, so the loom floor's rotation term stays 0 and the floor")
    log.raw("--- sits at exactly LOOM_EXPANSION_FLOOR. Fine for a level desk,")
    log.raw("--- but it is a fallback, not a measurement.")
    print(f"WARNING: no state telemetry after {timeout:.0f}s - continuing with "
          "zero attitude (fine for a stationary level drone, but the log will "
          "show att_ok=0).", flush=True)
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--baseline", type=float, default=10.0)
    parser.add_argument("--fov", type=float, default=DEFAULT_VERTICAL_FOV,
                        help=f"camera VERTICAL fov in degrees (default {DEFAULT_VERTICAL_FOV}, "
                             f"derived from the {TELLO_DIAGONAL_FOV} diagonal spec at 4:3)")
    parser.add_argument("--no-video", action="store_true",
                        help="headless; disables SPACE swat marking")
    parser.add_argument("--log", default=str(ROOT / "Drone" / "flight_logs" / Path(__file__).with_suffix(".log").name))
    args = parser.parse_args()

    log = Logger(args.log)
    write_header(log, args)
    print(f"logging to {log.path}", flush=True)

    tello = open_tello(log)
    frame_read = open_stream(tello, log)

    looming = LoomingDetector(width=PROC_WIDTH, height=PROC_HEIGHT, fov=args.fov)
    print("starting the brain subprocess (brian2 build takes a few seconds)...", flush=True)
    # bounds=None: the boundary-containment logic is correctly skipped
    # throughout FlyBrainController when bounds is None, so none of it
    # needs stubbing for a desk test.
    controller = FlyBrainController(bounds=None)
    print("brain ready", flush=True)

    # Capture yaw/forward/escape/spike_counts for EVERY cycle, not just the
    # ones FlyBrainController's own _log_spikes writes (it only fires when a
    # DN actually spiked, and it does not return the values to the caller).
    # Wrapping the subprocess request is the one place both the input and
    # the full output are visible.
    last = {}
    inner_request = controller._brain.request

    def capturing_request(payload):
        result = inner_request(payload)
        last.clear()
        last.update(payload=payload, result=result)
        return result

    controller._brain.request = capturing_request

    prev_gray = None          # for grid flow (magnitude)
    prev_accepted = None      # raw gray of the last ACCEPTED frame, for dup detection
    prev_yaw = None
    accepted = 0
    duplicates = 0
    att_missing = 0
    marks = []
    stats = {"exp_c": [], "loom": [], "escape": [], "hz": [],
             "ms_cycle": [], "ms_brain": [], "ms_flow": []}
    baseline_exp = []
    escape_events = []
    state_counts = {}

    # Two clocks on purpose: t_abs (never reset) is what every logged
    # timestamp, swat marker and escape trigger uses, so the log has one
    # unambiguous timeline to align neuron responses against. phase_t0 only
    # drives the baseline/live deadlines.
    run_t0 = time.perf_counter()
    phase_t0 = run_t0
    last_accept_t = None
    phase = "warmup"
    log.header()

    try:
        while True:
            cycle_start = time.perf_counter()
            t_abs = cycle_start - run_t0
            phase_elapsed = cycle_start - phase_t0

            grab_start = time.perf_counter()
            frame = frame_read.frame
            if frame_not_ready(frame):
                time.sleep(0.005)
                continue
            # Tello frames arrive as RGB (same conversion tello_camera.py does)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            small = cv2.resize(frame_bgr, (PROC_WIDTH, PROC_HEIGHT), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            ms_grab = (time.perf_counter() - grab_start) * 1000

            # frame_read.frame hands back the same background buffer until
            # the next decode lands. Processing it twice produces zero flow
            # and therefore zero expansion - which would look exactly like
            # "the circuit never fired" while actually being a sampling
            # artifact. Skip duplicates and count them, so the log can tell
            # those two situations apart.
            if prev_accepted is not None and np.array_equal(gray, prev_accepted):
                duplicates += 1
                if not args.no_video:
                    if (cv2.waitKey(1) & 0xFF) == ord("q"):
                        break
                else:
                    time.sleep(0.002)
                continue
            prev_accepted = gray
            accepted += 1

            now = time.perf_counter()
            # dt between frames actually fed to the detector - NOT the
            # nominal loop period. LoomingDetector divides its per-cell
            # expansion by (frames_apart * dt), so feeding it a wrong dt
            # scales the expansion rate directly.
            dt = (now - last_accept_t) if last_accept_t else 1.0 / 30.0
            last_accept_t = now
            dt = max(dt, 1e-3)

            att = read_attitude(tello)
            # The Tello reports yaw clockwise-positive; this project is
            # counter-clockwise-positive. Convert once, here, exactly as
            # Drone/tello_drone.py's get_state() does - these scripts read
            # the attitude themselves rather than going through it, so the
            # conversion has to be applied in both places or the perception
            # tests and the flight path disagree about which way a turn went.
            yaw_deg = TELLO_YAW_SIGN * att["yaw"]
            quat = euler_deg_to_quat(att["roll"], att["pitch"], yaw_deg)
            yaw_rad = math.radians(yaw_deg)
            if prev_yaw is None:
                yaw_rate = 0.0
            else:
                d_yaw = (yaw_rad - prev_yaw + math.pi) % (2 * math.pi) - math.pi
                yaw_rate = d_yaw / dt
            prev_yaw = yaw_rad

            flow_start = time.perf_counter()
            expansion = looming.update(gray, quat, dt)
            grid = {"left": 0.0, "right": 0.0, "top": 0.0, "bottom": 0.0, "center": 0.0}
            if prev_gray is not None:
                raw_flow = compute_flow(prev_gray, gray)
                derotated = derotate_flow(raw_flow, yaw_rate, dt)
                grid = grid_flow_strengths(derotated)
            prev_gray = gray
            ms_flow = (time.perf_counter() - flow_start) * 1000

            if phase == "warmup":
                if accepted >= WARMUP_FRAMES:
                    # Drop the EMA/median history built from the settling
                    # exposure so the baseline phase starts clean.
                    looming.reset()
                    prev_gray = None
                    phase = "baseline"
                    phase_t0 = time.perf_counter()
                    log.raw(f"PHASE baseline starts t={t_abs:.2f} "
                            f"(warmup discarded {accepted} frames)")
                    print(f"--- BASELINE: hold still, {args.baseline:.0f}s of nothing ---", flush=True)
                continue
            if phase == "baseline" and phase_elapsed >= args.baseline:
                phase = "live"
                phase_t0 = time.perf_counter()
                log.raw(f"PHASE live starts t={t_abs:.2f} - SPACE marks a swat, Q quits")
                print("--- LIVE: swat at it. Press SPACE each time you swat, Q to stop ---",
                      flush=True)
            elif phase == "live" and phase_elapsed >= args.seconds:
                break

            flow = dict(grid)
            flow.update({f"expansion_{side}": value for side, value in expansion.items()})

            # Stationary drone: zero position/velocity. actual_vx is forced
            # to 0.0 rather than read from the Tello so the loom floor is at
            # its most sensitive and nothing self-motion-related is being
            # subtracted (att["vgx"] is logged separately for reference).
            state = {
                "position": (0.0, 0.0, 0.0),
                "orientation": quat,
                "yaw_degrees": float(yaw_deg),
                "actual_vx": 0.0,
            }

            brain_start = time.perf_counter()
            controller.decide(flow, state)
            ms_brain = (time.perf_counter() - brain_start) * 1000

            result = last.get("result", {})
            payload = last.get("payload", {})
            escape = result.get("escape", 0.0)
            loom_l = payload.get("loom_left", 0.0)
            loom_r = payload.get("loom_right", 0.0)
            spikes = ",".join(f"{k}={v}" for k, v in result.get("spike_counts", {}).items() if v) or "-"
            # Reconstructed rather than recomputed: calling _loom_floor here
            # would advance its _prev_tilt/_rotation_floor state a second
            # time per cycle and corrupt the real one.
            floor = (fbc.LOOM_EXPANSION_FLOOR + fbc.LOOM_EXPANSION_FLOOR_PER_MPS * 0.0
                     + controller._rotation_floor)

            ms_cycle = (time.perf_counter() - cycle_start) * 1000
            hz = 1.0 / dt

            state_counts[controller.state] = state_counts.get(controller.state, 0) + 1
            if controller.state == "ESCAPE" and (not escape_events or escape_events[-1][1] != "run"):
                escape_events.append((round(t_abs, 2), "run"))
            elif controller.state != "ESCAPE" and escape_events and escape_events[-1][1] == "run":
                escape_events[-1] = (escape_events[-1][0], "done")

            log.row([
                f"{t_abs:.3f}", phase, f"{dt * 1000:.1f}", duplicates,
                f"{expansion['left']:.3f}", f"{expansion['center']:.3f}", f"{expansion['right']:.3f}",
                f"{floor:.3f}", f"{loom_l:.3f}", f"{loom_r:.3f}",
                f"{escape:.3f}", f"{result.get('yaw', 0.0):.3f}", f"{result.get('forward', 0.0):.3f}",
                controller.state, spikes,
                f"{grid['left']:.3f}", f"{grid['center']:.3f}", f"{grid['right']:.3f}",
                att["pitch"], att["roll"], att["yaw"], int(att["ok"]),
                att["vgx"], att["height"],
                f"{gray.mean():.1f}",
                f"{ms_grab:.1f}", f"{ms_flow:.1f}", f"{ms_brain:.1f}", f"{ms_cycle:.1f}",
            ])

            stats["exp_c"].append(expansion["center"])
            stats["loom"].append(max(loom_l, loom_r))
            stats["escape"].append(escape)
            if not att["ok"]:
                att_missing += 1
            stats["hz"].append(hz)
            stats["ms_cycle"].append(ms_cycle)
            stats["ms_brain"].append(ms_brain)
            stats["ms_flow"].append(ms_flow)
            if phase == "baseline":
                baseline_exp.append(max(expansion.values()))

            if not args.no_video:
                view = small.copy()
                fired = controller.state == "ESCAPE"
                color = (0, 0, 255) if fired else (0, 255, 0)
                lines = [
                    f"{phase.upper()}  {t_abs:5.1f}s  {hz:4.1f}Hz  dup={duplicates}",
                    f"EXP L={expansion['left']:.2f} C={expansion['center']:.2f} R={expansion['right']:.2f}",
                    f"floor={floor:.2f}  loom L={loom_l:.2f} R={loom_r:.2f}",
                    f"escape={escape:.2f}  state={controller.state}",
                    f"spikes: {spikes}",
                    f"marks={len(marks)}   SPACE=swat  Q=quit",
                ]
                for i, line in enumerate(lines):
                    cv2.putText(view, line, (6, 16 + i * 16), cv2.FONT_HERSHEY_SIMPLEX,
                                0.42, color, 1, cv2.LINE_AA)
                if fired:
                    cv2.putText(view, "DNp01 ESCAPE", (60, PROC_HEIGHT - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imshow("Tello neuron test", view)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord(" "):
                    marks.append(round(t_abs, 2))
                    log.raw(f"MARK swat t={t_abs:.2f} phase={phase}")
                    print(f"  [swat marked at {t_abs:.2f}s]", flush=True)

    except KeyboardInterrupt:
        print("\ninterrupted", flush=True)
    finally:
        # --- summary: everything needed to judge the run without the raw rows
        log.raw("=" * 70)
        log.raw("SUMMARY")
        log.raw(f"accepted_frames: {accepted}   duplicate_frames_skipped: {duplicates}")
        log.raw(f"cycles_without_attitude: {att_missing}"
                + ("  (orientation was identity on those - the loom floor's "
                   "rotation term could not be measured)" if att_missing else ""))
        if stats["hz"]:
            log.raw(f"effective_fps  mean={statistics.fmean(stats['hz']):.1f} "
                    f"median={percentile(stats['hz'], 50):.1f} "
                    f"p5={percentile(stats['hz'], 5):.1f}")
            log.raw(f"cycle_ms       mean={statistics.fmean(stats['ms_cycle']):.1f} "
                    f"p95={percentile(stats['ms_cycle'], 95):.1f}")
            log.raw(f"flow_ms        mean={statistics.fmean(stats['ms_flow']):.1f} "
                    f"p95={percentile(stats['ms_flow'], 95):.1f}")
            brain_mean = statistics.fmean(stats["ms_brain"])
            log.raw(f"brain_ms       mean={brain_mean:.1f} "
                    f"p95={percentile(stats['ms_brain'], 95):.1f}")
            # 33ms is the 30Hz decision-loop budget TESTING.md holds
            # the circuit to. Over it usually means the brain is running
            # under a slower brian2 install than intended - see
            # brain_interpreter in the header above.
            if brain_mean > 33.0:
                log.raw(f"WARNING: brain_ms mean {brain_mean:.1f} exceeds the 33ms "
                        f"30Hz budget. Check 'brain_interpreter' in the header - "
                        f"pin the fast env with FLYBRAIN_PYTHON and re-run.")
                print(f"\nWARNING: brain step averaged {brain_mean:.1f}ms (budget 33ms) - "
                      f"the brain may be running under the wrong brian2 install.",
                      flush=True)
        if baseline_exp:
            log.raw(f"baseline_expansion (quiet, max over columns) "
                    f"mean={statistics.fmean(baseline_exp):.3f} "
                    f"p95={percentile(baseline_exp, 95):.3f} "
                    f"max={max(baseline_exp):.3f}")
            log.raw(f"  -> compare against LOOM_EXPANSION_FLOOR="
                    f"{fbc.LOOM_EXPANSION_FLOOR}: the floor must sit above this "
                    f"noise or the circuit fires at nothing")
        if stats["exp_c"]:
            log.raw(f"peak_expansion_center={max(stats['exp_c']):.3f}  "
                    f"peak_loom={max(stats['loom']):.3f}  "
                    f"peak_escape={max(stats['escape']):.3f}  "
                    f"(ESCAPE_STATE_THRESHOLD={fbc.ESCAPE_STATE_THRESHOLD})")
        log.raw(f"state_cycle_counts: {state_counts}")
        log.raw(f"swat_marks ({len(marks)}): {marks}")
        log.raw(f"escape_triggers ({len(escape_events)}): {[t for t, _ in escape_events]}")
        log.raw("END")
        log.close()

        try:
            controller.close()
        except Exception:
            pass
        try:
            tello.streamoff()
            tello.end()
        except Exception:
            pass
        cv2.destroyAllWindows()

        print(f"\nwrote {log.path}", flush=True)
        print(f"accepted={accepted} duplicates={duplicates} "
              f"marks={len(marks)} escape_triggers={len(escape_events)}", flush=True)
        if stats["loom"]:
            print(f"peak loom={max(stats['loom']):.2f} peak escape={max(stats['escape']):.2f}",
                  flush=True)


if __name__ == "__main__":
    main()
