"""
Table-top test of the DNg02 flight-motor circuit on the REAL Tello - the
perception half only, with the drone sitting still on a desk.

    Tello video -> signed_hemifield_flow -> DNg02 drive pool -> (log only)

THIS SCRIPT NEVER FLIES THE DRONE. It never calls takeoff(), land() or
send_rc_control(). Leave the propellers off. It is the same deal as
tello_neuron_test.py: answer the perception questions with zero flight risk
before anything is allowed to move.

There are two questions, and they need different experiments, hence --mode:

  --mode calibrate  MUST BE RUN FIRST, and it is a hard prerequisite for any
        optomotor flight. NeuralPathways/EscapeNeuron/optical_flow._PIXELS_PER_RADIAN is 75.0,
        "empirically calibrated for the default DroneCamera" - i.e. the
        SIMULATOR's camera, not this one. Geometrically the Tello at 320px
        across ~70.3 deg horizontal should be nearer 260 px/rad. Nothing has
        cared until now, because every existing user of derotate_flow() feeds
        the result into a magnitude, where a scale error is absorbed by
        hand-tuned thresholds downstream. The optomotor loop uses the signed
        RESIDUAL, and a too-small correction leaves part of the drone's own
        commanded yaw in the signal with the SAME SIGN as a real drift - which
        is positive feedback, i.e. the drone chasing its own turn. So: rotate
        the drone by hand through a range of speeds, and this regresses the
        measured horizontal flow against the yaw rate from the Tello's own
        attitude telemetry and prints the fitted px/rad and the R^2.

        It also re-checks Drone/tello_drone.py's TELLO_YAW_SIGN, the
        constant that converts the Tello's clockwise-positive yaw into this
        project's counter-clockwise-positive convention. That constant was
        measured by this mode (slope -86.6, R^2 0.888, on 2026-09-27) and is
        now applied before the fit, so a CORRECT value yields a POSITIVE
        slope. A negative slope here means the constant has gone wrong - after
        a firmware update, say - and needs flipping back.

  --mode pan        The population-code check. Sweep a textured card (a
        newspaper, a book cover - something with detail; a blank wall gives
        Farneback nothing to track) across the field of view, left to right
        and back. Press D when you start sweeping rightwards and A when you
        start sweeping leftwards - they latch, so press once per direction
        change rather than holding, and X clears the label between sweeps. What should happen:
          - still:      rotation ~0, nL = nR = 0, steer ~0
          - sweep right: rotation > 0, right DNg02 recruits more, steer > 0,
                         and the yaw the controller WOULD have commanded is
                         negative, i.e. turn right. See the sign note in
                         NeuralPathways/flybrain_controller.py's decide().
          - sweep left:  the mirror image.
        The HUD draws the population as two rows of cells in recruitment-ladder
        order, so you can watch them light up in order as you sweep harder -
        that is the population code, visible in real time.

Run (needs djitellopy + opencv + numpy; the brain runs as a subprocess under
whichever env has brian2 - see NeuralPathways/flybrain_controller.py):

    python Drone/FlightTests/tello_dng02_test.py --mode calibrate
    python Drone/FlightTests/tello_dng02_test.py --mode pan

    --seconds N     live phase length (default 60)
    --baseline N    quiet phase before it (default 10)
    --fov DEG       camera VERTICAL fov (default 55.6)
    --ppr N         pixels-per-radian for derotation (default: whatever
                    NeuralPathways/EscapeNeuron/optical_flow uses). Set this to what --mode
                    calibrate measured, then re-run --mode pan.
    --no-video      headless; disables the HUD and the marking keys
    --log PATH      default Drone/flight_logs/tello_dng02_test.log

Afterwards, score the run instead of trusting your eyes on the HUD:

    python Drone/FlightTests/tello_dng02_test.py --analyze            # scores --log's path

No drone and no brain needed - it reads the log back and reports whether the
population actually recruited in the committed ladder order, whether the
response was graded or all-or-nothing, whether the steering signal tracked
rotation, whether the baseline noise stayed under the steering floor, and the
sign, each as a PASS/FAIL. It parses the ladder out of the log's own header
rather than the repo, so a log handed over from another machine or another
commit still analyses.

Why this is not optional: the HUD updates 25 times a second and the eye
averages it, so "the cells lit up in roughly the right order" is not something
you can honestly judge by watching. The escape circuit could be scored by
counting discrete events against SPACE markers; DNg02 has no events, only the
shape of a distribution.

DO ONE DRY RUN WITH INTERNET FIRST (no Tello needed, it will just fail to
connect): brian2 compiles generated C++ on first use and you will be on the
Tello's wifi with no internet during the real test.

The harness helpers - the placeholder-frame guard, attitude reading,
connection handshake, the log format - come from Drone/flight_harness.py,
shared with tello_neuron_test.py, so a fix to either script's stream handling
benefits both. Only the columns and the per-cycle body differ. Fitting the
calibration and scoring a log (--analyze) live in dng02_analysis.py.
"""

import argparse
import math
import os
import platform
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

import NeuralPathways.flybrain_controller as fbc
from NeuralPathways.flybrain_controller import FlyBrainController
from NeuralPathways.EscapeNeuron.optical_flow import (compute_flow, derotate_flow, grid_flow_strengths,
                                 signed_hemifield_flow, LoomingDetector,
                                 _PIXELS_PER_RADIAN)
from Drone.tello_drone import TELLO_YAW_SIGN
from Drone.flight_harness import (DEFAULT_VERTICAL_FOV, PROC_HEIGHT, PROC_WIDTH, WARMUP_FRAMES,
                                  TelloLogger as BaseLogger, YawTracker, capture_brain_requests,
                                  frame_not_ready, git_commit, open_stream, open_tello, percentile,
                                  read_attitude)
# analyse is re-exported: Drone/Tests/test_tello_harness_smoke.py calls it from here
from Drone.FlightTests.dng02_analysis import analyse, report_calibration

# The network's own constants (MAX_DRIVE_RATE, the recruitment curves, ...) are
# NOT imported - connectome_worker needs brian2, which this interpreter
# deliberately does not have. They arrive from the subprocess's ready
# handshake instead, which is strictly better: the log then records what the
# network was actually built with rather than what this side assumed.
ADAPTER_CONSTANTS = ("OPTOMOTOR_FLOW_FLOOR", "OPTOMOTOR_FLOW_SPAN",
                     "OPTOMOTOR_BASE_DRIVE", "DNG02_YAW_GAIN",
                     "DNG02_YAW_AUTHORITY", "DNG02_THRUST_SPEED",
                     "OPTOMOTOR_FLOW_SETPOINT", "OPTOMOTOR_SETPOINT_SPAN",
                     "OPTOMOTOR_STATE_THRESHOLD")



class Logger(BaseLogger):
    COLUMNS = [
        "t_s", "phase", "dt_ms", "dup", "mark",
        "dx_l", "dx_r", "rotation", "dx_raw", "translation",
        "drive_common", "drive_l", "drive_r",
        "n_left", "n_right", "thrust", "steer", "state",
        "yaw_cmd", "escape", "recruited",
        "yaw_rate_meas", "pitch", "roll", "yaw", "att_ok",
        "mean_px", "ms_flow", "ms_brain", "ms_cycle",
    ]


def write_header(log, args, controller):
    log.meta("test", f"tello_dng02_test --mode {args.mode} - stationary Tello, "
                     "DNg02 flight-motor perception only, props OFF")
    log.meta("started", time.strftime("%Y-%m-%d %H:%M:%S"))
    log.meta("git", git_commit())
    log.meta("python", f"{platform.python_version()} on {platform.platform()}")
    log.meta("opencv", cv2.__version__)
    log.meta("proc_resolution", f"{PROC_WIDTH}x{PROC_HEIGHT}")
    log.meta("vertical_fov_deg", args.fov)
    log.meta("pixels_per_radian", f"{args.ppr} (module default {_PIXELS_PER_RADIAN})")
    log.meta("TELLO_YAW_SIGN", TELLO_YAW_SIGN)
    log.meta("baseline_seconds", args.baseline)
    log.meta("live_seconds", args.seconds)
    log.raw("--- network constants, as reported by the brain subprocess ---")
    for name, value in sorted(controller._brain.constants.items()):
        log.meta(name, value)
    log.meta("brain_with_dng02", controller._brain.info.get("with_dng02"))
    log.raw("--- flybrain_controller (adapter) constants in effect ---")
    for name in ADAPTER_CONSTANTS:
        log.meta(name, getattr(fbc, name))
    log.raw("--- the DNg02 population, in recruitment-ladder order ---")
    for i, label in enumerate(controller._brain.dng02_labels):
        log.raw(f"  {i:>2} {label}")
    log.raw("--- drone stationary and props OFF: nothing is commanded, the yaw")
    log.raw("--- column is what the controller WOULD have asked for")
    try:
        from NeuralPathways.flybrain_controller import _find_python_with_brian2
        log.meta("brain_interpreter", _find_python_with_brian2())
    except Exception as exc:
        log.meta("brain_interpreter", f"could not resolve: {exc!r}")
    log.meta("FLYBRAIN_PYTHON", os.environ.get("FLYBRAIN_PYTHON", "(unset)"))


def draw_population(view, labels, counts, left_mask, x0, y0):
    """Two rows of cells in ladder order, filled when that neuron spiked -
    the '#####.....' recruitment bar, as actual pixels."""
    cell, gap = 9, 2
    for row, want_left in ((0, True), (1, False)):
        col = 0
        for label, is_left in zip(labels, left_mask):
            if is_left != want_left:
                continue
            x = x0 + col * (cell + gap)
            y = y0 + row * (cell + gap)
            fired = counts.get(label, 0) > 0
            colour = (0, 255, 255) if fired else (70, 70, 70)
            cv2.rectangle(view, (x, y), (x + cell, y + cell), colour,
                          -1 if fired else 1)
            col += 1
        cv2.putText(view, "L" if want_left else "R", (x0 - 12, y0 + row * (cell + gap) + cell),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA)


class Run:
    """Everything accumulated over a run, for the summary at the end."""

    def __init__(self):
        self.accepted = 0
        self.duplicates = 0
        self.att_missing = 0
        self.marks = []
        self.cal = {"dyaw": [], "raw": [], "deg": []}
        self.stats = {"rotation": [], "n_total": [], "steer": [], "hz": [],
                      "ms_cycle": [], "ms_brain": [], "ms_flow": []}
        self.baseline_rotation = []
        self.by_mark = {"d": [], "a": []}
        self.state_counts = {}


def draw_hud(small, args, phase, t_abs, hz, run, hemi, payload, d, n_left, n_right,
             cmd, controller_state, yaw_rate, dt, mark_now, labels, counts, left_mask):
    view = small.copy()
    steer = d.get("steer", 0.0)
    colour = (0, 255, 255) if abs(steer) > fbc.OPTOMOTOR_STATE_THRESHOLD else (0, 255, 0)
    lines = [
        f"{phase.upper()} [{args.mode}] {t_abs:5.1f}s {hz:4.1f}Hz dup={run.duplicates}",
        f"dx L={hemi['left']:+.2f} R={hemi['right']:+.2f}  "
        f"rot={hemi['rotation']:+.2f} trans={hemi['translation']:+.2f}",
        f"drive c={payload.get('drive_common', 0.0):.2f} "
        f"L={payload.get('drive_left', 0.0):+.2f} R={payload.get('drive_right', 0.0):+.2f}",
        f"nL={n_left:>2} nR={n_right:>2} thrust={d.get('thrust', 0.0):.2f} "
        f"steer={steer:+.2f}",
        f"would command yaw={cmd['yaw_rate']:+.3f} rad/s  state={controller_state}",
        (f"yaw_rate={yaw_rate:+.2f} deg/frame={math.degrees(abs(yaw_rate * dt)):.1f} "
         f"samples={len(run.cal['dyaw'])}  Q=quit"
         if args.mode == "calibrate"
         else f"mark={mark_now}  D=sweep right  A=sweep left  X=clear  Q=quit"),
    ]
    for i, line in enumerate(lines):
        cv2.putText(view, line, (6, 14 + i * 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, colour, 1, cv2.LINE_AA)
    draw_population(view, labels, counts, left_mask, 20, PROC_HEIGHT - 34)
    if abs(steer) > fbc.OPTOMOTOR_STATE_THRESHOLD:
        cv2.putText(view, f"DNg02 STEER {'RIGHT' if steer > 0 else 'LEFT'}",
                    (150, PROC_HEIGHT - 44), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow("Tello DNg02 test", view)


def write_summary(log, args, run, labels):
    stats = run.stats
    log.raw("=" * 70)
    log.raw("SUMMARY")
    log.raw(f"mode: {args.mode}")
    log.raw(f"accepted_frames: {run.accepted}   duplicate_frames_skipped: {run.duplicates}")
    log.raw(f"cycles_without_attitude: {run.att_missing}")
    if stats["hz"]:
        log.raw(f"effective_fps  mean={statistics.fmean(stats['hz']):.1f} "
                f"p5={percentile(stats['hz'], 5):.1f}")
        log.raw(f"cycle_ms  mean={statistics.fmean(stats['ms_cycle']):.1f} "
                f"p95={percentile(stats['ms_cycle'], 95):.1f}")
        brain_mean = statistics.fmean(stats["ms_brain"])
        log.raw(f"brain_ms  mean={brain_mean:.1f} "
                f"p95={percentile(stats['ms_brain'], 95):.1f}")
        if brain_mean > 33.0:
            log.raw(f"WARNING: brain_ms mean {brain_mean:.1f} exceeds the 33ms 30Hz "
                    f"budget - check brain_interpreter in the header.")
    baseline_rotation = run.baseline_rotation
    if baseline_rotation:
        worst = max(abs(v) for v in baseline_rotation)
        log.raw(f"baseline |rotation| mean="
                f"{statistics.fmean(abs(v) for v in baseline_rotation):.3f} "
                f"p95={percentile([abs(v) for v in baseline_rotation], 95):.3f} "
                f"max={worst:.3f}")
        log.raw(f"  -> compare against OPTOMOTOR_FLOW_FLOOR="
                f"{fbc.OPTOMOTOR_FLOW_FLOOR}: the floor must sit above this "
                f"or the drone steers at its own Farneback noise")
        if worst > fbc.OPTOMOTOR_FLOW_FLOOR:
            log.raw(f"  WARNING: quiet-scene rotation reached {worst:.3f}, above the "
                    f"floor. Raise OPTOMOTOR_FLOW_FLOOR before flying.")
    log.raw(f"state_cycle_counts: {run.state_counts}")

    if args.mode == "calibrate":
        report_calibration(log, run.cal)
    else:
        write_sweep_summary(log, run, labels)
    log.raw("END")


def write_sweep_summary(log, run, labels):
    """--mode pan: the marked sweeps, and the sign check they add up to."""
    by_mark = run.by_mark
    for key, label in (("d", "sweep RIGHT"), ("a", "sweep LEFT")):
        rows = by_mark[key]
        if not rows:
            log.raw(f"{label}: no marked cycles")
            continue
        nl = statistics.fmean(r[0] for r in rows)
        nr = statistics.fmean(r[1] for r in rows)
        st = statistics.fmean(r[2] for r in rows)
        rot = statistics.fmean(r[3] for r in rows)
        yc = statistics.fmean(r[4] for r in rows)
        log.raw(f"{label}: {len(rows)} cycles  rotation={rot:+.3f}  "
                f"nL={nl:.2f} nR={nr:.2f}  steer={st:+.3f}  "
                f"would_command_yaw={yc:+.3f}")
    d_rows, a_rows = by_mark["d"], by_mark["a"]
    if d_rows and a_rows:
        d_steer = statistics.fmean(r[2] for r in d_rows)
        a_steer = statistics.fmean(r[2] for r in a_rows)
        ok = d_steer > a_steer
        log.raw(f"SIGN CHECK: steer(sweep right)={d_steer:+.3f} vs "
                f"steer(sweep left)={a_steer:+.3f} -> "
                f"{'CORRECT' if ok else 'INVERTED'}")
        log.raw("  Expected: sweeping right gives the HIGHER steer, because "
                "rightward motion raises the right DNg02 cells (Namiki et al. "
                "2022) and positive steer means 'the fly yaws right'.")
        if not ok:
            log.raw("  INVERTED: do NOT fly this. Either the sign of rotation in "
                    "signed_hemifield_flow or the drive assignment in "
                    "_optomotor_drive is backwards, and in closed loop that is "
                    "positive feedback.")
        print(f"\nSIGN CHECK: {'CORRECT' if ok else 'INVERTED - DO NOT FLY'} "
              f"(steer right={d_steer:+.3f}, left={a_steer:+.3f})", flush=True)
    stats = run.stats
    if stats["n_total"]:
        log.raw(f"peak_recruited={max(stats['n_total'])} of {len(labels)}  "
                f"peak_|steer|={max(abs(v) for v in stats['steer']):.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("calibrate", "pan"), default="pan")
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--baseline", type=float, default=10.0)
    parser.add_argument("--fov", type=float, default=DEFAULT_VERTICAL_FOV)
    parser.add_argument("--ppr", type=float, default=_PIXELS_PER_RADIAN,
                        help="pixels per radian for derotation; run --mode calibrate first")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--log", default=str(ROOT / "Drone" / "flight_logs" / Path(__file__).with_suffix(".log").name))
    parser.add_argument("--analyze", metavar="PATH", nargs="?", const="",
                        help="score a log written by an earlier run and exit; "
                             "no drone, no brain. Defaults to --log's path.")
    args = parser.parse_args()

    if args.analyze is not None:
        return analyse(args.analyze or args.log)

    log = Logger(args.log)
    print(f"logging to {log.path}", flush=True)

    tello = open_tello(log)
    frame_read = open_stream(tello, log)

    looming = LoomingDetector(width=PROC_WIDTH, height=PROC_HEIGHT, fov=args.fov)
    print("starting the brain subprocess (brian2 build takes a few seconds)...", flush=True)
    controller = FlyBrainController(bounds=None, optomotor=True)
    # Labels in recruitment-ladder order, straight from the network that built
    # them - no need to re-derive the sort from NeuralPathways/StabilizerNeuron/dng02_circuit_neurons.json here
    # and risk the two orders drifting apart.
    labels = controller._brain.dng02_labels
    left_mask = [side == "left" for side in controller._brain.dng02_sides]
    if not labels:
        raise SystemExit("the brain subprocess reported no DNg02 population - "
                         "it was started without --dng02, which means optomotor "
                         "did not reach _FlyBrainProcess")
    print(f"brain ready ({len(labels)} DNg02 cells)", flush=True)
    write_header(log, args, controller)

    last = capture_brain_requests(controller)
    yaw_tracker = YawTracker()
    run = Run()

    prev_gray = None
    prev_accepted = None
    mark_now = "-"

    run_t0 = phase_t0 = time.perf_counter()
    last_accept_t = None
    phase = "warmup"
    log.header()

    try:
        while True:
            cycle_start = time.perf_counter()
            t_abs = cycle_start - run_t0
            phase_elapsed = cycle_start - phase_t0

            frame = frame_read.frame
            if frame_not_ready(frame):
                time.sleep(0.005)
                continue
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            small = cv2.resize(frame_bgr, (PROC_WIDTH, PROC_HEIGHT), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

            if prev_accepted is not None and np.array_equal(gray, prev_accepted):
                run.duplicates += 1
                # Polled on EVERY raw iteration, not just accepted frames, so a
                # keypress can't be swallowed behind a run of duplicates.
                if not args.no_video and (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
                continue
            prev_accepted = gray
            run.accepted += 1

            now = time.perf_counter()
            dt = max((now - last_accept_t) if last_accept_t else 1.0 / 30.0, 1e-3)
            last_accept_t = now

            att = read_attitude(tello)
            yaw_deg, quat, yaw_rate = yaw_tracker.update(att, dt)

            flow_start = time.perf_counter()
            expansion = looming.update(gray, quat, dt)
            grid = {"left": 0.0, "right": 0.0, "top": 0.0, "bottom": 0.0, "center": 0.0}
            hemi = {"left": 0.0, "right": 0.0, "rotation": 0.0, "translation": 0.0}
            raw_hemi = dict(hemi)
            if prev_gray is not None:
                raw_flow = compute_flow(prev_gray, gray)
                # Raw (un-derotated) hemifield flow is what the calibration
                # regresses against yaw rate - derotating first would subtract
                # the very thing being measured.
                raw_hemi = signed_hemifield_flow(raw_flow)
                derotated = derotate_flow(raw_flow, yaw_rate, dt, pixels_per_radian=args.ppr)
                grid = grid_flow_strengths(derotated)
                hemi = signed_hemifield_flow(derotated)
            prev_gray = gray
            ms_flow = (time.perf_counter() - flow_start) * 1000

            if phase == "warmup":
                if run.accepted >= WARMUP_FRAMES:
                    looming.reset()
                    prev_gray = None
                    phase = "baseline"
                    phase_t0 = time.perf_counter()
                    log.raw(f"PHASE baseline starts t={t_abs:.2f} "
                            f"(warmup discarded {run.accepted} frames)")
                    print(f"--- BASELINE: hold still, {args.baseline:.0f}s of nothing ---",
                          flush=True)
                continue
            if phase == "baseline" and phase_elapsed >= args.baseline:
                phase = "live"
                phase_t0 = time.perf_counter()
                if args.mode == "calibrate":
                    log.raw(f"PHASE live starts t={t_abs:.2f} - rotate the drone by hand")
                    print("--- LIVE: rotate the drone left and right by hand, a range of\n"
                          "    speeds, keeping the scene textured. Q to stop ---", flush=True)
                else:
                    log.raw(f"PHASE live starts t={t_abs:.2f} - D=sweeping right, A=sweeping left")
                    print("--- LIVE: sweep a textured card across the view.\n"
                          "    Hold D while sweeping RIGHT, A while sweeping LEFT. Q to stop ---",
                          flush=True)
            elif phase == "live" and phase_elapsed >= args.seconds:
                break

            flow = dict(grid)
            flow.update({f"expansion_{side}": value for side, value in expansion.items()})
            flow.update({"rotation": hemi["rotation"], "translation": hemi["translation"]})

            state = {
                "position": (0.0, 0.0, 0.0),
                "orientation": quat,
                "yaw_degrees": float(yaw_deg),
                # Forced to 0.0, as in tello_neuron_test.py: the drone is not
                # moving, and this keeps the loom floor at its most sensitive.
                "actual_vx": 0.0,
            }

            brain_start = time.perf_counter()
            cmd = controller.decide(flow, state)
            ms_brain = (time.perf_counter() - brain_start) * 1000

            result = last.get("result", {})
            payload = last.get("payload", {})
            d = result.get("dng02", {})
            n_left = d.get("n_left", 0)
            n_right = d.get("n_right", 0)
            counts = d.get("counts", {})
            recruited = ",".join(k for k, v in counts.items() if v) or "-"

            ms_cycle = (time.perf_counter() - cycle_start) * 1000
            hz = 1.0 / dt
            run.state_counts[controller.state] = run.state_counts.get(controller.state, 0) + 1

            log.row([
                f"{t_abs:.3f}", phase, f"{dt * 1000:.1f}", run.duplicates, mark_now,
                f"{hemi['left']:.3f}", f"{hemi['right']:.3f}",
                f"{hemi['rotation']:.3f}", f"{raw_hemi['rotation']:.3f}",
                f"{hemi['translation']:.3f}",
                f"{payload.get('drive_common', 0.0):.3f}",
                f"{payload.get('drive_left', 0.0):.3f}",
                f"{payload.get('drive_right', 0.0):.3f}",
                n_left, n_right, f"{d.get('thrust', 0.0):.3f}", f"{d.get('steer', 0.0):+.3f}",
                controller.state, f"{cmd['yaw_rate']:+.3f}", f"{result.get('escape', 0.0):.3f}",
                recruited,
                f"{yaw_rate:+.3f}", att["pitch"], att["roll"], att["yaw"], int(att["ok"]),
                f"{gray.mean():.1f}",
                f"{ms_flow:.1f}", f"{ms_brain:.1f}", f"{ms_cycle:.1f}",
            ])

            stats = run.stats
            stats["rotation"].append(hemi["rotation"])
            stats["n_total"].append(n_left + n_right)
            stats["steer"].append(d.get("steer", 0.0))
            stats["hz"].append(hz)
            stats["ms_cycle"].append(ms_cycle)
            stats["ms_brain"].append(ms_brain)
            stats["ms_flow"].append(ms_flow)
            if not att["ok"]:
                run.att_missing += 1
            if phase == "baseline":
                run.baseline_rotation.append(hemi["rotation"])
            if phase == "live":
                if args.mode == "calibrate":
                    # derotate_flow assumes u ~= pixels_per_radian * yaw_rate *
                    # dt, so d_yaw is the regressor and the RAW (un-derotated)
                    # common-mode flow is the response. Every sample is kept;
                    # the filtering happens per window, not per frame.
                    run.cal["dyaw"].append(yaw_rate * dt)
                    run.cal["raw"].append(raw_hemi["rotation"])
                    run.cal["deg"].append(abs(math.degrees(yaw_rate * dt)))
                if mark_now in run.by_mark:
                    run.by_mark[mark_now].append((n_left, n_right, d.get("steer", 0.0),
                                                  hemi["rotation"], cmd["yaw_rate"]))

            if not args.no_video:
                draw_hud(small, args, phase, t_abs, hz, run, hemi, payload, d, n_left, n_right,
                         cmd, controller.state, yaw_rate, dt, mark_now, labels, counts, left_mask)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                # A LATCH, not a held key: a sweep spans many frames and all of
                # them want the same label, but OpenCV's key repeat under a held
                # key is inconsistent across platforms, so requiring it held
                # would silently label only a fraction of the cycles. D and A
                # latch the label on, X clears it.
                if key in (ord("d"), ord("a"), ord("x")):
                    mark_now = "-" if key == ord("x") else chr(key)
                    run.marks.append((round(t_abs, 2), mark_now))
                    log.raw(f"MARK {mark_now} t={t_abs:.2f}")

    except KeyboardInterrupt:
        print("\ninterrupted", flush=True)
    finally:
        write_summary(log, args, run, labels)
        log.close()

        for cleanup in (controller.close,
                        lambda: tello.streamoff(),
                        lambda: tello.end()):
            try:
                cleanup()
            except Exception:
                pass
        cv2.destroyAllWindows()
        print(f"\nwrote {log.path}", flush=True)
        print(f"accepted={run.accepted} duplicates={run.duplicates} marks={len(run.marks)}", flush=True)


if __name__ == "__main__":
    sys.exit(main() or 0)
