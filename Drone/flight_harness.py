"""
Pieces shared by the real-Tello scripts in Drone/FlightTests/: camera
geometry, stream start-up, the plain-text log writer and attitude readout.

Two flavours of connect/stream helpers live here, as they did in the
scripts they came from:
  - open_tello / open_stream take a TelloLogger and record telemetry
    availability into it (the desk tests: tello_neuron_test.py,
    tello_dng02_test.py).
  - open_tello_for_flight / open_stream_for_flight take a print-like
    callable and warn on low battery (the flight tests:
    tello_escape_flight_test.py, tello_optomotor_flight_test.py).
Scripts import them under the names open_tello / open_stream, which is
also what Drone/Tests/test_tello_harness_smoke.py patches.
"""

import math
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

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

STREAM_SETTLE_SECONDS = 3.0   # fly_tello.py uses the same wait - the
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

ARM_GRACE_SECONDS = 2.0  # post-takeoff settle before the brain's output is
                          # trusted - the climb itself is a big, non-looming
                          # expansion transient (main.py's equivalent is
                          # HOVER_BEFORE_EXPLORE_CYCLES)
EMERGENCY_HOVER_SECONDS = 1.0  # how long 'q'/SPACE hovers before landing


class TelloLogger:
    """Writes one self-contained plain-text file: '#' metadata/marker/
    summary lines plus tab-separated per-cycle rows under a COLUMNS
    header. Flushed every line - the run may well end by yanking wifi or
    Ctrl-C, and a truncated log is still worth reading."""

    COLUMNS = []  # each script's subclass lists its own per-cycle fields

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
    # wait_for_state=False matches the existing tello_test.py/fly_tello.py -
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
    tello_test.py/fly_tello.py use, and what this script inherited -
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


def open_tello_for_flight(log):
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


def open_stream_for_flight(tello, log):
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
