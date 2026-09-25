"""
Adapter between main.py's autonomous-controller contract and the real
Fly-Brain connectome circuit in fly_brain_controller.py.

main.py only ever talks to an autonomous controller through
`decide(flow, state) -> command dict` (same shape as ReflexController and
ManualController - see reflex_controller.py's module docstring), plus a
`.state` string (shown on the HUD, and checked by SafetyLayer.apply()'s
already_avoiding argument) and a `.reset()` method. This class provides
exactly that surface, backed by fly_brain_controller.py's
FlyBrainController instead of reflex_controller.py's hand-written state
machine.

fly_brain_controller.py needs Brian2 (+pandas/pyarrow) to build and run
its connectome subnetwork - deliberately NOT installed into this
project's own venv (main.py's side has no business linking against a
spiking neural simulator just to fly a drone). So it runs as a separate,
persistent subprocess - built once at startup, then fed one
loom_left/loom_right request per decide() call over stdin/stdout JSON
lines - under whichever Python environment does have those installed
(see _find_python_with_brian2 below for how that's located). Everything
downstream of this file (main.py, SafetyLayer) stays identical whichever
controller is plugged in; this is the only file that needs to know
FlyBrainController - or Brian2 - exists at all.
"""

import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FLY_BRAIN_SCRIPT = REPO_ROOT / "fly_brain_controller.py"
SPIKE_LOG_PATH = REPO_ROOT / "flybrain_spikes.log"

# Looming input: image expansion rate (flow["expansion_*"], 1/s, from
# vision/optical_flow.LoomingDetector - roughly 2 / time-to-contact).
# Expansion at/below a floor maps to loom=0, floor + LOOM_EXPANSION_SPAN
# to loom=1; the circuit's Giant Fiber fires from loom ~0.25. The floor
# rises a little with speed (the ground ahead genuinely expands during
# forward flight) and with the drone's own rotation - an efference copy:
# LoomingDetector removes rotation geometrically, but Farneback's residual
# error still grows during fast maneuvers.
LOOM_EXPANSION_FLOOR = 0.8
LOOM_EXPANSION_FLOOR_PER_MPS = 0.5
LOOM_EXPANSION_FLOOR_PER_RAD = 25.0   # per radian of rotation since the last decide()
# LoomingDetector's temporal filtering makes its output lag rotation by a
# few cycles, so the rotation allowance decays instead of dropping the
# instant rotation stops - otherwise the tail of a fast turn read as looming.
LOOM_ROTATION_FLOOR_DECAY = 0.7
LOOM_EXPANSION_SPAN = 2.0

YAW_RATE_SCALE = 0.9          # rad/s at |yaw|=1.0 - comparable to
                                # safety_layer's EMERGENCY_TURN_RATE
CRUISE_FORWARD_SPEED = 1.0    # m/s at forward=1.0 - matches
                                # reflex_controller's CRUISE_SPEED
ESCAPE_STATE_THRESHOLD = 0.6  # escape signal above this -> "ESCAPE" state
AVOID_STATE_THRESHOLD = 0.05  # |yaw| above this -> "AVOID_LEFT"/"AVOID_RIGHT" state

# --- Escape maneuver. Like the real Giant Fiber jump, once triggered it's
# a committed, fixed motor program rather than something re-decided every
# cycle: a hard sideways dodge out of the approach path, while backing
# away. Just stopping doesn't work - an approaching object keeps coming -
# and backing straight up stays on its line of travel. Sideways alone was
# too slow for the sim drone (~1s to move 0.6m): it got hit by a second
# box in 3/3 test runs, vs 0/3 with the backward component. ---
ESCAPE_CYCLES = 30                # ~1s at the 30Hz decision loop - about what
                                  # the drone needs to get ~0.6m sideways
ESCAPE_STRAFE_SPEED = 2.0         # m/s - pybullet_drone's MAX_SPEED
ESCAPE_BACK_SPEED = 2.0           # m/s - pitch and roll saturate independently,
                                  # so backing off at full speed doesn't slow the
                                  # dodge sideways and buys extra time (0.12m ->
                                  # 0.42m miss distance vs 1.0 in testing)
ESCAPE_REFRACTORY_CYCLES = 30     # ~1s after a dodge before another can trigger
ESCAPE_SIDE_MIN_YAW = 0.3         # |DNp06 yaw| needed to trust it for dodge direction
ESCAPE_SIDE_MIN_FLOW_DIFF = 0.2   # else: dodge toward the lower-flow side if it's this clear
ESCAPE_DODGE_SIDE_M = 1.2         # roughly how far one dodge carries the drone
ESCAPE_DODGE_BACK_M = 1.2         # (from Testing/drone_step_response.py), padded
ESCAPE_BOUNDS_MARGIN = 0.3        # m inside the flight-area bounds a dodge must end

# --- Boundary containment - same behavior as
# controllers/reflex_controller.py's BOUNDARY_RETURN state (same
# constants), duplicated rather than imported: the neural circuit only
# ever sees looming, it has no notion of this course's flight-area edges,
# so it needs independent handling exactly like ReflexController already
# does, and each autonomous controller is meant to be self-contained/
# swappable (see reflex_controller.py's "Swap-in contract" note). ---
BOUNDARY_FORWARD_SPEED = 0.5
BOUNDARY_TURN_RATE = 0.35
BOUNDARY_TURN_GAIN = 1.2
BOUNDARY_RELEASE_MARGIN = 1.0


def _find_python_with_brian2():
    """Locates a Python interpreter with brian2/pandas/pyarrow installed
    to run fly_brain_controller.py under. Checked in order: an explicit
    FLYBRAIN_PYTHON override, the current interpreter (in case someone's
    default env does have brian2), a handful of common conda env
    locations, then `conda run -n brian2 which python`."""
    override = os.environ.get("FLYBRAIN_PYTHON")
    if override:
        return override

    try:
        import brian2  # noqa: F401
        return sys.executable
    except ImportError:
        pass

    for candidate in (
        Path.home() / "miniconda3" / "envs" / "brian2" / "bin" / "python",
        Path.home() / "anaconda3" / "envs" / "brian2" / "bin" / "python",
        Path("/opt/homebrew/Caskroom/miniconda/base/envs/brian2/bin/python"),
        Path("/opt/miniconda3/envs/brian2/bin/python"),
    ):
        if candidate.exists():
            return str(candidate)

    conda = shutil.which("conda")
    if conda:
        try:
            out = subprocess.run(
                [conda, "run", "-n", "brian2", "python", "-c", "import sys; print(sys.executable)"],
                capture_output=True, text=True, timeout=15,
            )
            path = out.stdout.strip()
            if path:
                return path
        except Exception:
            pass

    raise RuntimeError(
        "Can't find a Python with brian2/pandas/pyarrow installed to run "
        "fly_brain_controller.py under. Create one, e.g.:\n"
        "    conda create -n brian2 python=3.10 brian2 pandas pyarrow\n"
        "or point FLYBRAIN_PYTHON at an existing one's python executable."
    )


def _pitch_roll(q):
    x, y, z, w = q
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    return pitch, roll


class _FlyBrainProcess:
    """Owns the persistent fly_brain_controller.py subprocess (started
    once - it takes ~1-3s to load the connectome subgraph and build the
    Brian2 network, nowhere near fast enough to redo every decide() call)
    and its line-delimited JSON protocol."""

    def __init__(self):
        python = _find_python_with_brian2()
        self._proc = subprocess.Popen(
            [python, str(FLY_BRAIN_SCRIPT)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1,
        )
        ready_line = self._proc.stdout.readline()
        if "ready" not in ready_line:
            raise RuntimeError(
                f"fly_brain_controller.py failed to start:\n{self._proc.stderr.read()}"
            )

    def request(self, payload):
        if self._proc.poll() is not None:
            raise RuntimeError(f"fly_brain_controller.py exited:\n{self._proc.stderr.read()}")
        self._proc.stdin.write(json.dumps(payload) + "\n")
        self._proc.stdin.flush()
        line = self._proc.stdout.readline()
        if not line:
            raise RuntimeError(f"fly_brain_controller.py closed its output:\n{self._proc.stderr.read()}")
        result = json.loads(line)
        if "error" in result:
            raise RuntimeError(f"fly_brain_controller.py error: {result['error']}")
        return result

    def close(self):
        if self._proc.poll() is None:
            self._proc.terminate()


class FlyBrainController:
    """Drop-in replacement for ReflexController: same
    decide(flow, state) -> command dict contract, same .state/.reset()
    surface, so main.py can swap one for the other without changing
    anything else. See this module's docstring for how it's wired to the
    real connectome subnetwork, and fly_brain_controller.py for what that
    subnetwork actually is."""

    def __init__(self, bounds=None):
        self.bounds = bounds
        self.state = "CRUISE"
        self._escape_timer = 0
        self._escape_dir = 1          # +1 = dodge left, -1 = dodge right
        self.escape_direction = "LEFT"
        self._refractory_timer = 0
        self._last_tie_dir = -1
        self._prev_tilt = None
        self._rotation_floor = 0.0
        self._brain = _FlyBrainProcess()
        # Fresh log each run (not appended) - this is a debug tool for
        # "what did the circuit just do", not a long-lived history.
        self._log_file = open(SPIKE_LOG_PATH, "w")

    def reset(self):
        self.state = "CRUISE"
        self._escape_timer = 0
        self._refractory_timer = 0
        self._prev_tilt = None
        self._rotation_floor = 0.0
        self._brain.request({"reset": True})

    def close(self):
        self._brain.close()
        self._log_file.close()

    def decide(self, flow, state=None):
        position = state["position"] if state else (0.0, 0.0, 0.0)

        if self.bounds is not None:
            if self.state == "BOUNDARY_RETURN":
                if self._well_inside_bounds(position):
                    self.state = "CRUISE"
            elif self._outside_bounds(position):
                self.state = "BOUNDARY_RETURN"

        if self.state == "BOUNDARY_RETURN":
            return self._boundary_return_command(position, state)

        # Something looming in the center threatens both eyes at once,
        # so it feeds both loom_left and loom_right, not just whichever
        # side literally reads higher.
        floor = self._loom_floor(state)
        loom_left = self._loom(max(flow["expansion_left"], flow["expansion_center"]), floor)
        loom_right = self._loom(max(flow["expansion_right"], flow["expansion_center"]), floor)
        result = self._brain.request({"loom_left": loom_left, "loom_right": loom_right})
        yaw, forward, escape = result["yaw"], result["forward"], result["escape"]
        spike_counts = result.get("spike_counts", {})

        if self._escape_timer > 0:
            self._escape_timer -= 1
            if self._escape_timer == 0:
                self._refractory_timer = ESCAPE_REFRACTORY_CYCLES
        elif self._refractory_timer > 0:
            self._refractory_timer -= 1
        elif escape >= ESCAPE_STATE_THRESHOLD:
            self._escape_timer = ESCAPE_CYCLES
            self._escape_dir = self._pick_escape_dir(yaw, flow, state)

        if self._escape_timer > 0:
            self.state = "ESCAPE"
            self.escape_direction = "LEFT" if self._escape_dir > 0 else "RIGHT"
            forward_speed = -ESCAPE_BACK_SPEED
            strafe_speed = ESCAPE_STRAFE_SPEED * self._escape_dir
            yaw_rate = 0.0
        else:
            forward_speed = CRUISE_FORWARD_SPEED * forward * (1.0 - escape)
            strafe_speed = 0.0
            yaw_rate = YAW_RATE_SCALE * yaw
            if abs(yaw) > AVOID_STATE_THRESHOLD:
                self.state = "AVOID_LEFT" if yaw > 0 else "AVOID_RIGHT"
            else:
                self.state = "CRUISE"

        if any(spike_counts.values()):
            self._log_spikes(loom_left, loom_right, spike_counts, escape, forward_speed, yaw_rate)

        return self._command(forward_speed=forward_speed, yaw_rate=yaw_rate, strafe_speed=strafe_speed)

    def _loom_floor(self, state):
        """Expansion expected from the drone's own motion, which isn't looming."""
        if state is None:
            return LOOM_EXPANSION_FLOOR
        pitch, roll = _pitch_roll(state["orientation"])
        yaw = math.radians(state["yaw_degrees"])
        rotation = 0.0
        if self._prev_tilt is not None:
            prev_pitch, prev_roll, prev_yaw = self._prev_tilt
            d_yaw = (yaw - prev_yaw + math.pi) % (2 * math.pi) - math.pi
            rotation = abs(pitch - prev_pitch) + abs(roll - prev_roll) + abs(d_yaw)
        self._prev_tilt = (pitch, roll, yaw)
        self._rotation_floor = max(LOOM_EXPANSION_FLOOR_PER_RAD * rotation,
                                   self._rotation_floor * LOOM_ROTATION_FLOOR_DECAY)
        # Only forward motion makes the scene ahead expand; sideways or
        # backward motion (e.g. still coasting out of a dodge) doesn't.
        forward_speed = max(0.0, state["actual_vx"])
        return (LOOM_EXPANSION_FLOOR
                + LOOM_EXPANSION_FLOOR_PER_MPS * forward_speed
                + self._rotation_floor)

    @staticmethod
    def _loom(expansion, floor):
        return min(1.0, max(0.0, (expansion - floor) / LOOM_EXPANSION_SPAN))

    def _pick_escape_dir(self, yaw, flow, state):
        """+1 = dodge left, -1 = dodge right. A head-on approach drives both
        DNp06s about equally, so their difference is mostly noise then."""
        if abs(yaw) >= ESCAPE_SIDE_MIN_YAW:
            preferred = 1 if yaw > 0 else -1
        elif abs(flow["right"] - flow["left"]) >= ESCAPE_SIDE_MIN_FLOW_DIFF:
            preferred = 1 if flow["right"] > flow["left"] else -1
        else:
            self._last_tie_dir = -self._last_tie_dir
            preferred = self._last_tie_dir

        # The dodge moves sideways and backward blind (the camera only
        # faces forward), so don't pick the side that would end up past
        # the flight-area edge - that's where the perimeter walls are.
        if self.bounds is None or state is None:
            return preferred
        margins = {d: self._dodge_landing_margin(state, d) for d in (preferred, -preferred)}
        if margins[preferred] < ESCAPE_BOUNDS_MARGIN and margins[-preferred] > margins[preferred]:
            return -preferred
        return preferred

    def _dodge_landing_margin(self, state, direction):
        heading = math.radians(state["yaw_degrees"])
        back, side = -ESCAPE_DODGE_BACK_M, ESCAPE_DODGE_SIDE_M * direction
        x = state["position"][0] + back * math.cos(heading) - side * math.sin(heading)
        y = state["position"][1] + back * math.sin(heading) + side * math.cos(heading)
        b = self.bounds
        return min(x - b["min_x"], b["max_x"] - x, y - b["min_y"], b["max_y"] - y)

    # --- debug: neuron activity logging ---

    def _log_spikes(self, loom_left, loom_right, spike_counts, escape, forward_speed, yaw_rate):
        """Called only on cycles where at least one of the 6 output (DN)
        neurons actually spiked - so this is the "when it sees it
        approaching" trace: quiet during ordinary cruising, active exactly
        while the looming circuit is responding to something."""
        fired = ", ".join(f"{name}={n}" for name, n in spike_counts.items() if n)
        line = (
            f"[flybrain] loom(L={loom_left:.2f} R={loom_right:.2f}) spikes: {fired}  "
            f"-> state={self.state} escape={escape:.2f} "
            f"forward={forward_speed:.2f}m/s yaw_rate={yaw_rate:.2f}rad/s"
        )
        print(line, flush=True)
        self._log_file.write(line + "\n")
        self._log_file.flush()

    # --- command builder ---

    def _command(self, forward_speed, yaw_rate, strafe_speed=0.0):
        return {
            "forward_speed": forward_speed,
            "strafe_speed": strafe_speed,
            "yaw_rate": yaw_rate,
            "altitude_delta": 0.0,
            "hover": False,
            "land": False,
            "reset": False,
            "pressed_direction": "(flybrain)",
        }

    # --- boundary helpers (identical to reflex_controller.py's) ---

    def _boundary_return_command(self, position, state):
        yaw = math.radians(state["yaw_degrees"]) if state else 0.0
        dx = self.bounds["center_x"] - position[0]
        dy = self.bounds["center_y"] - position[1]
        target_heading = math.atan2(dy, dx)
        heading_error = (target_heading - yaw + math.pi) % (2 * math.pi) - math.pi
        yaw_rate = max(-BOUNDARY_TURN_RATE, min(BOUNDARY_TURN_RATE, heading_error * BOUNDARY_TURN_GAIN))
        return self._command(BOUNDARY_FORWARD_SPEED, yaw_rate)

    def _outside_bounds(self, position):
        x, y = position[0], position[1]
        b = self.bounds
        return x < b["min_x"] or x > b["max_x"] or y < b["min_y"] or y > b["max_y"]

    def _well_inside_bounds(self, position):
        x, y = position[0], position[1]
        b = self.bounds
        m = BOUNDARY_RELEASE_MARGIN
        return (b["min_x"] + m < x < b["max_x"] - m
                and b["min_y"] + m < y < b["max_y"] - m)
