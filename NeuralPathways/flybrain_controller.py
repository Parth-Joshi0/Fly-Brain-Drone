"""
Adapter between main.py's autonomous-controller contract and the real
Fly-Brain connectome circuit in connectome_worker.py.

main.py only ever talks to an autonomous controller through
`decide(flow, state) -> command dict` (same shape as ReflexController and
ManualController - see Simulator/reflex_controller.py's module docstring), plus a
`.state` string (shown on the HUD, and checked by SafetyLayer.apply()'s
already_avoiding argument) and a `.reset()` method. This class provides
exactly that surface, backed by connectome_worker.py's
ConnectomeNetwork instead of Simulator/reflex_controller.py's hand-written state
machine.

connectome_worker.py needs Brian2 (+pandas/pyarrow) to build and run
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

Two behaviours live here, and only the first is on by default:

  * LOOMING / ESCAPE - always active. Maps LoomingDetector expansion to the
    circuit's loom inputs and turns DNp01/DNp03/DNp06 activity into a dodge.

  * DNg02 OPTOMOTOR + THRUST - only when constructed with optomotor=True.
    Maps residual optic flow onto DNg02's drive pool and turns the population's
    recruitment count into a yaw correction and a forward-speed adjustment.
    With the flag off the brain subprocess never builds the DNg02 half at all,
    so the escape path is byte-for-byte the network it has always been - which
    matters because the escape flight test is being tuned separately.

Escape stays strictly dominant either way: the ESCAPE branch of decide() is
untouched by the optomotor path, and every DNg02 term is additionally scaled by
(1 - escape), so two independent things have to fail before a dodge gets
watered down by a course correction.
"""

import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

from Simulator.boundary_math import (heading_rate_toward, outside_bounds, well_inside_bounds)

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
CONNECTOME_WORKER_SCRIPT = HERE / "connectome_worker.py"
SPIKE_LOG_PATH = REPO_ROOT / "flybrain_spikes.log"

# Looming input: image expansion rate (flow["expansion_*"], 1/s, from
# NeuralPathways/EscapeNeuron/optical_flow.LoomingDetector - roughly 2 / time-to-contact).
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
                                # Simulator/reflex_controller's CRUISE_SPEED
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
ESCAPE_SIDE_PUSH_M = 0.4          # push sideways/back only until moved this far,
ESCAPE_BACK_PUSH_M = 0.4          # then brake - momentum carries it ~1m total
ESCAPE_SIDE_MIN_YAW = 0.3         # |DNp06 yaw| needed to trust it for dodge direction
ESCAPE_SIDE_MIN_FLOW_DIFF = 0.2   # else: dodge toward the lower-flow side if it's this clear
ESCAPE_DODGE_SIDE_M = 1.2         # roughly how far one dodge carries the drone,
ESCAPE_DODGE_BACK_M = 1.2         # including the slide while braking, padded
ESCAPE_BOUNDS_MARGIN = 0.3        # m inside the flight-area bounds a dodge must end

# --- DNg02 optomotor course stabilization + graded thrust. Off unless the
# caller passes optomotor=True, and when it's off the brain subprocess doesn't
# even build the DNg02 half (see _FlyBrainProcess), so the escape path is the
# same 274-neuron network it has always been. ---
#
# Residual common-mode horizontal flow (px/frame, from
# NeuralPathways/EscapeNeuron/optical_flow.signed_hemifield_flow on already-derotated flow) below
# the floor reads as zero; floor + SPAN maps to a full request. The floor is
# doing the same job LOOM_EXPANSION_FLOOR does for looming - derotation leaves
# a real residual even with nothing moving, so without it the drone chases its
# own Farneback noise.
# 0.25, not 0.15: a clean props-off baseline (Drone/Tests/dng02_pan2.log) measured
# quiet |rotation| p95 0.085 / max 0.150, i.e. right on the old floor.
OPTOMOTOR_FLOW_FLOOR = 0.25
OPTOMOTOR_FLOW_SPAN = 1.5

# Common drive level the steering request swings around. Needs to be non-zero:
# the opponent channel works by pushing one side up and the other down, and
# there's nothing to push down from at zero. 0.5 sits in the middle of the
# measured recruitment curve, where the population has room both ways.
OPTOMOTOR_BASE_DRIVE = 0.5

# rad/s of yaw at |steer| = 1.0. Deliberately small - about a sixth of
# YAW_RATE_SCALE. This is the first closed feedback loop in this project that
# runs through the airframe, so the failure mode is a growing oscillation
# rather than a wrong-but-steady heading, and the gain is the thing that sets
# how fast that would grow. Raise it only after a flight test shows no
# oscillation at this value.
DNG02_YAW_GAIN = 0.15
DNG02_YAW_AUTHORITY = 0.4     # hard cap on the DNg02 contribution, rad/s

# Forward speed the population's recruitment count buys, m/s at thrust = 1.0.
# Modest next to CRUISE_FORWARD_SPEED (1.0) because this is added on top of it.
DNG02_THRUST_SPEED = 0.4

# Thrust is SET-POINT regulation, not proportional to flow, and that's a
# stability requirement rather than a stylistic choice. Namiki et al. measured
# DNg02 open loop: imposed wide-field motion in, higher wingbeat amplitude out.
# In closed loop the drone's own speed is what generates the flow, so wiring
# "more flow -> more thrust" would be positive feedback and it would accelerate
# until something else stopped it. Flies instead hold a preferred image
# velocity, which is negative feedback: thrust rises when flow is BELOW the
# set-point. So the population gets driven by the set-point error.
OPTOMOTOR_FLOW_SETPOINT = 1.2   # px/frame of translational flow to hold
OPTOMOTOR_SETPOINT_SPAN = 1.2   # error that maps to a full request

OPTOMOTOR_STATE_THRESHOLD = 0.2  # |steer| above this -> "OPTOMOTOR" state

# --- Boundary containment - same behavior as
# Simulator/reflex_controller.py's BOUNDARY_RETURN state, with its own
# copy of the tuning constants below: the neural circuit only ever sees
# looming, it has no notion of this course's flight-area edges, so it
# needs independent handling exactly like ReflexController already does,
# and each autonomous controller is meant to be self-contained/swappable
# (see Simulator/reflex_controller.py's "Swap-in contract" note). The stateless
# geometry itself (Simulator/boundary_math.py) is shared - only the
# tuning and state-machine behavior are kept independent. ---
BOUNDARY_FORWARD_SPEED = 0.5
BOUNDARY_TURN_RATE = 0.35
BOUNDARY_TURN_GAIN = 1.2
BOUNDARY_RELEASE_MARGIN = 1.0


def _find_python_with_brian2():
    """Locates a Python interpreter with brian2/pandas/pyarrow installed
    to run connectome_worker.py under. Checked in order: an explicit
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
        "connectome_worker.py under. Create one, e.g.:\n"
        "    conda create -n brian2 python=3.10 brian2 pandas pyarrow\n"
        "or point FLYBRAIN_PYTHON at an existing one's python executable."
    )


def _pitch_roll(q):
    x, y, z, w = q
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    return pitch, roll


class _FlyBrainProcess:
    """Owns the persistent connectome_worker.py subprocess (started
    once - it takes ~1-3s to load the connectome subgraph and build the
    Brian2 network, nowhere near fast enough to redo every decide() call)
    and its line-delimited JSON protocol."""

    def __init__(self, with_dng02=False):
        python = _find_python_with_brian2()
        # --dng02 makes connectome_worker.py build the DNg02 flight-motor
        # half as well. Without it the network is the original 274 neurons with
        # the original object graph, and therefore the same Poisson RNG stream -
        # so leaving optomotor off doesn't just leave the escape behaviour
        # statistically similar, it leaves it identical.
        argv = [python, str(CONNECTOME_WORKER_SCRIPT)] + (["--dng02"] if with_dng02 else [])
        self._proc = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1,
        )
        ready_line = self._proc.stdout.readline()
        if "ready" not in ready_line:
            raise RuntimeError(
                f"connectome_worker.py failed to start:\n{self._proc.stderr.read()}"
            )
        # The handshake carries the network's own constants and the DNg02 labels
        # in recruitment order. Worth keeping: this process cannot import
        # connectome_worker (no brian2 here on purpose), so this is the only
        # way a log can record what the circuit was ACTUALLY configured with
        # rather than what this side assumed.
        try:
            self.info = json.loads(ready_line)
        except ValueError:
            self.info = {}
        self.constants = self.info.get("constants", {})
        self.dng02_labels = self.info.get("dng02_labels", [])
        self.dng02_sides = self.info.get("dng02_sides", [])

    def request(self, payload):
        if self._proc.poll() is not None:
            raise RuntimeError(f"connectome_worker.py exited:\n{self._proc.stderr.read()}")
        self._proc.stdin.write(json.dumps(payload) + "\n")
        self._proc.stdin.flush()
        line = self._proc.stdout.readline()
        if not line:
            raise RuntimeError(f"connectome_worker.py closed its output:\n{self._proc.stderr.read()}")
        result = json.loads(line)
        if "error" in result:
            raise RuntimeError(f"connectome_worker.py error: {result['error']}")
        return result

    def close(self):
        if self._proc.poll() is None:
            self._proc.terminate()


class FlyBrainController:
    """Drop-in replacement for ReflexController: same
    decide(flow, state) -> command dict contract, same .state/.reset()
    surface, so main.py can swap one for the other without changing
    anything else. See this module's docstring for how it's wired to the
    real connectome subnetwork, and connectome_worker.py for what that
    subnetwork actually is."""

    def __init__(self, bounds=None, *, optomotor=False):
        self.bounds = bounds
        self.optomotor = optomotor
        self.state = "CRUISE"
        self._escape_timer = 0
        self._escape_dir = 1          # +1 = dodge left, -1 = dodge right
        self.escape_direction = "LEFT"
        self._refractory_timer = 0
        self._escape_origin = (0.0, 0.0, 0.0)
        self._escape_back_m = ESCAPE_DODGE_BACK_M
        self._returning_to_bounds = False
        self._last_tie_dir = -1
        self._prev_tilt = None
        self._rotation_floor = 0.0
        # Extra loom floor a caller sets while it is deliberately closing on
        # something (Simulator/banana_seek_controller.py, flying to and
        # eating the banana): that target growing in view isn't a threat.
        self.loom_floor_offset = 0.0
        self.dng02 = {"n_left": 0, "n_right": 0, "thrust": 0.0, "steer": 0.0, "counts": {}}
        self.escape_level = 0.0
        self._brain = _FlyBrainProcess(with_dng02=optomotor)
        # Fresh log each run (not appended) - this is a debug tool for
        # "what did the circuit just do", not a long-lived history.
        self._log_file = open(SPIKE_LOG_PATH, "w")

    def reset(self):
        self.state = "CRUISE"
        self._escape_timer = 0
        self._refractory_timer = 0
        self._escape_origin = (0.0, 0.0, 0.0)
        self._escape_back_m = ESCAPE_DODGE_BACK_M
        self._returning_to_bounds = False
        self._prev_tilt = None
        self._rotation_floor = 0.0
        self.escape_level = 0.0
        self._brain.request({"reset": True})

    def close(self):
        self._brain.close()
        self._log_file.close()

    def decide(self, flow, state=None):
        position = state["position"] if state else (0.0, 0.0, 0.0)

        if self.bounds is not None:
            if self._returning_to_bounds:
                if self._well_inside_bounds(position):
                    self._returning_to_bounds = False
            elif self._outside_bounds(position):
                self._returning_to_bounds = True

        # The brain runs even while returning to the flight area: dodges
        # can end just outside it, and skipping the brain there left the
        # drone blind to the next incoming object.
        # Something looming in the center threatens both eyes at once,
        # so it feeds both loom_left and loom_right, not just whichever
        # side literally reads higher.
        floor = self._loom_floor(state)
        loom_left = self._loom(max(flow["expansion_left"], flow["expansion_center"]), floor)
        loom_right = self._loom(max(flow["expansion_right"], flow["expansion_center"]), floor)
        payload = {"loom_left": loom_left, "loom_right": loom_right}
        if self.optomotor:
            payload.update(self._optomotor_drive(flow))
        result = self._brain.request(payload)
        yaw, forward, escape = result["yaw"], result["forward"], result["escape"]
        self.escape_level = escape
        spike_counts = result.get("spike_counts", {})
        self.dng02 = result.get("dng02", self.dng02)

        if self._escape_timer > 0:
            self._escape_timer -= 1
            if self._escape_timer == 0:
                self._refractory_timer = ESCAPE_REFRACTORY_CYCLES
        elif self._refractory_timer > 0:
            self._refractory_timer -= 1
        elif escape >= ESCAPE_STATE_THRESHOLD:
            self._escape_timer = ESCAPE_CYCLES
            # Back off too unless that alone would leave the flight area
            # (repeated dodges otherwise walk the drone out backwards).
            self._escape_back_m = ESCAPE_DODGE_BACK_M
            if state is not None and self.bounds is not None and \
                    self._dodge_landing_margin(state, 0, back_m=ESCAPE_DODGE_BACK_M) < ESCAPE_BOUNDS_MARGIN:
                self._escape_back_m = 0.0
            self._escape_dir = self._pick_escape_dir(yaw, flow, state)
            self._escape_origin = (position[0], position[1],
                                   math.radians(state["yaw_degrees"]) if state else 0.0)

        if self._escape_timer > 0:
            self.state = "ESCAPE"
            self.escape_direction = "LEFT" if self._escape_dir > 0 else "RIGHT"
            # Push only until the drone has moved PUSH_M away, then let it
            # brake for the rest of the program: it keeps sliding ~1m after
            # a full-speed push, so pushing the whole time carried it into
            # the perimeter walls.
            side_moved, back_moved = self._escape_displacement(position)
            strafe_speed = ESCAPE_STRAFE_SPEED * self._escape_dir if side_moved < ESCAPE_SIDE_PUSH_M else 0.0
            backing = self._escape_back_m > 0 and back_moved < ESCAPE_BACK_PUSH_M
            forward_speed = -ESCAPE_BACK_SPEED if backing else 0.0
            yaw_rate = 0.0
        elif self._returning_to_bounds:
            self.state = "BOUNDARY_RETURN"
            cmd = self._boundary_return_command(position, state)
            forward_speed, strafe_speed, yaw_rate = cmd["forward_speed"], 0.0, cmd["yaw_rate"]
        else:
            forward_speed = CRUISE_FORWARD_SPEED * forward * (1.0 - escape)
            strafe_speed = 0.0
            yaw_rate = YAW_RATE_SCALE * yaw
            steer = 0.0
            if self.optomotor:
                steer = self.dng02.get("steer", 0.0)
                thrust = self.dng02.get("thrust", 0.0)
                # Added to DNp06's yaw, but with the opposite sign convention
                # - see dng02_yaw_rate().
                yaw_rate += self.dng02_yaw_rate(escape)
                forward_speed += DNG02_THRUST_SPEED * thrust * (1.0 - escape)
            if abs(yaw) > AVOID_STATE_THRESHOLD:
                self.state = "AVOID_LEFT" if yaw > 0 else "AVOID_RIGHT"
            elif abs(steer) > OPTOMOTOR_STATE_THRESHOLD:
                self.state = "OPTOMOTOR"
            else:
                self.state = "CRUISE"

        if any(spike_counts.values()):
            self._log_spikes(loom_left, loom_right, spike_counts, escape, forward_speed, yaw_rate)

        return self._command(forward_speed=forward_speed, yaw_rate=yaw_rate, strafe_speed=strafe_speed)

    def dng02_yaw_rate(self, escape=None):
        """The DNg02 stabilizer's yaw correction from the last decide(),
        rad/s (0.0 unless optomotor=True). decide() adds it to cruise
        steering; Simulator/banana_seek_controller.py adds it to the food
        behaviour's own steering instead. escape defaults to the level the
        last decide() saw.

        MINUS, and it is the opposite sign to DNp06's yaw in decide().
        DNg02 activity tracks wingbeat amplitude in the CONTRALATERAL wing, so
        more right-side DNg02 means a bigger left wingbeat, which yaws the fly
        RIGHT - connectome_worker.py's steer is positive for exactly that
        case. This project's convention is positive yaw_rate = turn LEFT.
        Hence subtract. DNp06's yaw is added instead because that circuit
        steers AWAY from a looming object, which is already positive-is-left.
        Two opposite conventions, both deliberate."""
        if not self.optomotor:
            return 0.0
        if escape is None:
            escape = self.escape_level
        dng02_yaw = -DNG02_YAW_GAIN * self.dng02.get("steer", 0.0) * (1.0 - escape)
        return max(-DNG02_YAW_AUTHORITY, min(DNG02_YAW_AUTHORITY, dng02_yaw))

    @staticmethod
    def _optomotor_drive(flow):
        """Turns optic flow into a DNg02 drive request.

        Expects flow to carry NeuralPathways/EscapeNeuron/optical_flow.signed_hemifield_flow's
        "rotation" and "translation" keys, computed on flow that has already
        been through derotate_flow() - so "rotation" is the rotation the drone
        did NOT command, which is the only part worth correcting. Both are
        read with a default, so a caller that never computed them (which is
        every caller that existed before this) gets a zero request and the
        flight-motor path stays idle.

        Steering is opponent: one side's request goes up exactly as much as the
        other's goes down, around OPTOMOTOR_BASE_DRIVE. That is the pattern
        Namiki et al. recorded for a yaw stimulus - rightward motion raised the
        right DNg02 cells while simultaneously lowering the left ones.
        """
        rotation = flow.get("rotation", 0.0)
        translation = flow.get("translation", 0.0)

        # Steering: magnitude from how much uncommanded rotation there is,
        # direction from its sign. rotation > 0 means the scene is sliding
        # right, i.e. the drone is rotating LEFT without being asked to, so the
        # correction is to yaw right - which means favouring the RIGHT DNg02.
        magnitude = (abs(rotation) - OPTOMOTOR_FLOW_FLOOR) / OPTOMOTOR_FLOW_SPAN
        magnitude = max(0.0, min(1.0, magnitude))
        offset = magnitude if rotation > 0 else -magnitude

        # Thrust: set-point error, not raw flow. See OPTOMOTOR_FLOW_SETPOINT.
        error = (OPTOMOTOR_FLOW_SETPOINT - abs(translation)) / OPTOMOTOR_SETPOINT_SPAN
        drive_common = OPTOMOTOR_BASE_DRIVE + max(-1.0, min(1.0, error)) * (
            1.0 - OPTOMOTOR_BASE_DRIVE)

        # Steering gets its authority first, and thrust yields whatever is left.
        # The brain clips each side's request to +-1, so without this cap a
        # common drive near 1.0 pins BOTH sides at the ceiling and the opponent
        # channel stops doing anything at all - the two requests come out equal.
        # Measured on the first real desk run: translational flow on a stationary
        # drone is ~0.01 against a setpoint of 1.2, so the set-point error pinned
        # drive_common at 0.995 median and both sides saturated on 18% of cycles,
        # cutting the steering signal to roughly a third of what it manages with
        # headroom. That is not a desk-only problem: hover, slow flight, climbing
        # and any low-texture scene all sit below the flow setpoint, so steering
        # authority would collapse exactly when course-holding matters most.
        # Yielding thrust is the right way round - recruitment is already near
        # saturation up there, so the thrust cost is small.
        drive_common = min(drive_common, 1.0 - abs(offset))
        return {
            "drive_common": drive_common,
            "drive_left": -offset,
            "drive_right": offset,
        }

    def _loom_floor(self, state):
        """Expansion expected from the drone's own motion, which isn't looming."""
        if state is None:
            return LOOM_EXPANSION_FLOOR + self.loom_floor_offset
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
                + self._rotation_floor
                + self.loom_floor_offset)

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
        margins = {d: self._dodge_landing_margin(state, d, back_m=self._escape_back_m)
                   for d in (preferred, -preferred)}
        if margins[preferred] < ESCAPE_BOUNDS_MARGIN and margins[-preferred] > margins[preferred]:
            return -preferred
        return preferred

    def _escape_displacement(self, position):
        """How far the drone has moved since the dodge started, as (toward
        the dodge side, backward), both relative to its heading at the time."""
        x0, y0, heading = self._escape_origin
        dx, dy = position[0] - x0, position[1] - y0
        left = -dx * math.sin(heading) + dy * math.cos(heading)
        back = -(dx * math.cos(heading) + dy * math.sin(heading))
        return left * self._escape_dir, back

    def _dodge_landing_margin(self, state, direction, back_m):
        """Distance inside the flight-area bounds where a dodge toward
        `direction` (+1 left, -1 right, 0 none) that also backs off back_m
        would end up; negative = outside."""
        heading = math.radians(state["yaw_degrees"])
        back, side = -back_m, ESCAPE_DODGE_SIDE_M * direction
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

    # --- boundary helpers ---

    def _boundary_return_command(self, position, state):
        yaw_degrees = state["yaw_degrees"] if state else 0.0
        target = (self.bounds["center_x"], self.bounds["center_y"])
        yaw_rate = heading_rate_toward(
            position, yaw_degrees, target, BOUNDARY_TURN_RATE, BOUNDARY_TURN_GAIN
        )
        return self._command(BOUNDARY_FORWARD_SPEED, yaw_rate)

    def _outside_bounds(self, position):
        return outside_bounds(position, self.bounds)

    def _well_inside_bounds(self, position):
        return well_inside_bounds(position, self.bounds, BOUNDARY_RELEASE_MARGIN)
