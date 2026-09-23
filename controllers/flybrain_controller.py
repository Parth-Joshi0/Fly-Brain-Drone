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

# Looming normalization: optic-flow readings at/above this are treated as
# "as looming as the neural model was tuned for" (loom=1.0). Lined up
# with safety_layer.py's own calibration notes (flow climbs past
# ~1.5-2.4+ right up close to an obstacle).
LOOM_FLOW_MAX = 2.0

YAW_RATE_SCALE = 0.9          # rad/s at |yaw|=1.0 - comparable to
                                # safety_layer's EMERGENCY_TURN_RATE
CRUISE_FORWARD_SPEED = 1.0    # m/s at forward=1.0 - matches
                                # reflex_controller's CRUISE_SPEED
ESCAPE_STATE_THRESHOLD = 0.6  # escape signal above this -> "ESCAPE" state
AVOID_STATE_THRESHOLD = 0.05  # |yaw| above this -> "AVOID_LEFT"/"AVOID_RIGHT" state

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
        self._brain = _FlyBrainProcess()

    def reset(self):
        self.state = "CRUISE"
        self._brain.request({"reset": True})

    def close(self):
        self._brain.close()

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
        loom_left = min(1.0, max(flow["left"], flow["center"]) / LOOM_FLOW_MAX)
        loom_right = min(1.0, max(flow["right"], flow["center"]) / LOOM_FLOW_MAX)
        result = self._brain.request({"loom_left": loom_left, "loom_right": loom_right})
        yaw, forward, escape = result["yaw"], result["forward"], result["escape"]

        if escape >= ESCAPE_STATE_THRESHOLD:
            self.state = "ESCAPE"
        elif abs(yaw) > AVOID_STATE_THRESHOLD:
            self.state = "AVOID_LEFT" if yaw > 0 else "AVOID_RIGHT"
        else:
            self.state = "CRUISE"

        return self._command(
            forward_speed=CRUISE_FORWARD_SPEED * forward * (1.0 - escape),
            yaw_rate=YAW_RATE_SCALE * yaw,
        )

    # --- command builder ---

    def _command(self, forward_speed, yaw_rate):
        return {
            "forward_speed": forward_speed,
            "strafe_speed": 0.0,
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
