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

    python Drone/Tests/tello_dng02_test.py --mode calibrate
    python Drone/Tests/tello_dng02_test.py --mode pan

    --seconds N     live phase length (default 60)
    --baseline N    quiet phase before it (default 10)
    --fov DEG       camera VERTICAL fov (default 55.6)
    --ppr N         pixels-per-radian for derotation (default: whatever
                    NeuralPathways/EscapeNeuron/optical_flow uses). Set this to what --mode
                    calibrate measured, then re-run --mode pan.
    --no-video      headless; disables the HUD and the marking keys
    --log PATH      default Drone/flight_logs/tello_dng02_test.log

Afterwards, score the run instead of trusting your eyes on the HUD:

    python Drone/Tests/tello_dng02_test.py --analyze            # scores --log's path

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

The harness helpers - the placeholder-frame guard, duplicate-frame rejection,
attitude reading, connection handshake, the log format - are imported from
tello_neuron_test.py rather than copied, so a fix to either script's stream
handling benefits both. Only the columns and the per-cycle body differ.
"""

import argparse
import math
import os
import re
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
from Drone.Tests.tello_neuron_test import (DEFAULT_VERTICAL_FOV, FIRST_FRAME_TIMEOUT,
                                       PROC_HEIGHT, PROC_WIDTH, WARMUP_FRAMES,
                                       Logger as BaseLogger, euler_deg_to_quat,
                                       frame_not_ready, git_commit, open_stream,
                                       open_tello, percentile, read_attitude)

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

# --- calibration ---
#
# The fit accumulates flow and yaw over a WINDOW of frames rather than
# regressing frame by frame, and that is not a refinement - a per-frame fit is
# invalid on this hardware. The Tello streams attitude on its own socket at
# roughly 9Hz while frames arrive near 20-25Hz, so more than half of all frames
# report d_yaw exactly 0 and the rest report a jump that did not happen over
# the interval the flow was measured across. Measured on a real run: the same
# data fits R^2 0.38 per-frame and R^2 0.89 accumulated over 20 frames, with
# the slope moving from -21 to -87. Windowing also absorbs the Tello's 1-degree
# yaw quantisation, and it removes a subtler problem - filtering individual
# frames by their yaw rate selects on the regressor, which biases the fit.
CALIBRATE_WINDOWS = (1, 5, 10, 20, 30)

# A window has to contain at least this much real rotation to say anything
# about the slope. ~2 degrees, i.e. comfortably above the 1-degree quantisation.
CALIBRATE_MIN_WINDOW_RAD = 0.035

# Above this much rotation per frame, Farneback (compute_flow: levels=2,
# winsize=13) stops tracking the displacement and starts under-reporting it,
# which corrupts the fitted MAGNITUDE while often leaving the sign intact. At
# ~260 px/rad, 1.5 deg/frame is about 7px - comfortably inside range. A real
# run at 4 deg/frame median recovered only 3.25px of an implied 18px.
CALIBRATE_MAX_DEG_PER_FRAME = 2.0


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


def linear_fit(xs, ys):
    """Least-squares slope, intercept and R^2. R^2 is what decides whether the
    calibration is usable at all - a good-looking slope from a scatter that
    does not actually track is worse than no number."""
    if len(xs) < 3:
        return float("nan"), float("nan"), float("nan")
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    # Rotating at a dead-constant rate leaves almost no spread in the
    # regressor, and polyfit then emits a RankWarning and returns a slope
    # fitted to rounding noise. There is genuinely nothing to measure from one
    # operating point, so say so instead of printing a confident number.
    if float(np.std(x)) < 1e-9:
        return float("nan"), float("nan"), float("nan")
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(slope), float(intercept), r2


def windowed_fit(dyaw, raw, window, min_rad=CALIBRATE_MIN_WINDOW_RAD):
    """Accumulates d_yaw and raw flow over non-overlapping windows of `window`
    frames, then fits. See CALIBRATE_WINDOWS for why this is not optional.

    Returns (slope, intercept, r2, n_windows)."""
    xs, ys = [], []
    for i in range(0, len(dyaw) - window + 1, window):
        dy = sum(dyaw[i:i + window])
        fl = sum(raw[i:i + window])
        if abs(dy) >= min_rad:
            xs.append(dy)
            ys.append(fl)
    slope, intercept, r2 = linear_fit(xs, ys)
    return slope, intercept, r2, len(xs)


def report_calibration(log, cal):
    """Fits across every window size and reports the whole table, because the
    TREND is the diagnosis: a slope still growing at the largest window means
    the flow is being under-tracked, not that the fit has converged."""
    dyaw, raw, deg = cal["dyaw"], cal["raw"], cal["deg"]
    if len(dyaw) < 20:
        log.raw(f"calibration: only {len(dyaw)} samples - not enough to fit")
        print(f"\ncalibration: only {len(dyaw)} samples - not enough to fit", flush=True)
        return

    moving = [d for d in deg if d > 0]
    burst_deg = statistics.fmean(moving) if moving else 0.0
    # The rate that matters for whether Farneback can track is degrees per
    # FRAME, averaged over every frame - including the ones where telemetry
    # reported no change. Those zeros are a telemetry artifact, not the drone
    # holding still, so averaging only the non-zero frames overstates the speed
    # by however much the telemetry lags the camera.
    eff_deg = statistics.fmean(deg) if deg else 0.0
    log.raw(f"calibration samples: {len(dyaw)} frames, "
            f"{len(moving)} with any reported yaw change "
            f"({1 - len(moving) / max(len(deg), 1):.0%} report zero - telemetry is "
            f"slower than the camera, which is why the fit is windowed)")
    log.raw(f"rotation speed: {eff_deg:.2f} deg/frame effective "
            f"(want under {CALIBRATE_MAX_DEG_PER_FRAME}), "
            f"{burst_deg:.2f} per reported update, peak {max(deg) if deg else 0:.1f}")
    log.raw(f"raw |flow|: mean {statistics.fmean(abs(v) for v in raw):.2f} "
            f"max {max(abs(v) for v in raw):.2f} px/frame")

    log.raw("fit vs window size (window=1 is the per-frame fit, which is invalid "
            "on this hardware - see CALIBRATE_WINDOWS):")
    log.raw(f"  {'window':>7} {'n':>5} {'slope':>9} {'R^2':>7}")
    results = []
    for w in CALIBRATE_WINDOWS:
        slope, intercept, r2, n = windowed_fit(dyaw, raw, w)
        results.append((w, slope, r2, n))
        log.raw(f"  {w:>7} {n:>5} {slope:>9.1f} {r2:>7.3f}")

    # Best R^2 among windows with enough windows to mean anything - NOT simply
    # the largest window, which runs out of samples and starts fitting noise.
    usable = sorted((r for r in results if r[3] >= 8 and r[2] == r[2] and r[2] >= 0.5),
                    key=lambda r: r[2])
    print()
    for w, slope, r2, n in results:
        print(f"  window {w:>3}: n={n:>4} slope={slope:>8.1f} R^2={r2:.3f}", flush=True)

    if not usable:
        best_r2 = max((r2 for _, _, r2, _ in results if r2 == r2), default=float("nan"))
        log.raw(f"NOT USABLE: best R^2 across all window sizes is {best_r2:.3f}, below 0.5. "
                f"Do NOT fly the optomotor loop on this.")
        if eff_deg > CALIBRATE_MAX_DEG_PER_FRAME:
            log.raw(f"  Most likely cause: too fast. {eff_deg:.1f} deg/frame is past what "
                    f"Farneback tracks here - rotate at under "
                    f"{CALIBRATE_MAX_DEG_PER_FRAME} deg/frame and retry.")
        else:
            log.raw("  Rotation speed was in range, so look elsewhere: confirm att_ok=1 "
                    "throughout, that there is real texture in view, and that the drone "
                    "rotated about its own axis rather than being carried.")
        print("\nCALIBRATION NOT USABLE - do not fly the optomotor loop on this.", flush=True)
        return

    w, slope, r2, n = usable[-1]        # highest R^2
    log.raw(f"best fit: window {w}, n={n}, pixels_per_radian = {slope:.1f}, R^2 = {r2:.3f}")
    log.raw(f"  current _PIXELS_PER_RADIAN = {_PIXELS_PER_RADIAN}")

    # A slope still climbing at the largest window has not converged, which
    # means the magnitude is under-tracked even if the sign is solid.
    trend = [sl for _, sl, r2_, n_ in results if n_ >= 8 and r2_ == r2_ and r2_ >= 0.5]
    climbing = len(trend) >= 3 and abs(trend[-1]) > abs(trend[-3]) * 1.15
    if climbing or eff_deg > CALIBRATE_MAX_DEG_PER_FRAME:
        log.raw("  MAGNITUDE NOT TRUSTWORTHY: the slope is still growing with window "
                "size, which means the flow is being under-tracked rather than the fit "
                "having converged. The SIGN below is still meaningful; the number is a "
                "lower bound. Re-run slower before using it.")
        print("  MAGNITUDE not trustworthy (slope still growing) - re-run slower.", flush=True)

    # The regressor is yaw already converted by TELLO_YAW_SIGN, so a POSITIVE
    # slope now means that constant is right. It was measured at -1.0 on
    # 2026-09-27; this is the check that it still holds.
    if slope < 0:
        log.raw(f"  TELLO_YAW_SIGN IS WRONG: the slope is negative with "
                f"TELLO_YAW_SIGN={TELLO_YAW_SIGN} already applied, which means the "
                f"conversion is now backwards. Flip it in Drone/tello_drone.py and "
                f"re-run. Left as it is, the efference copy ADDS the drone's own rotation "
                f"instead of removing it, in the escape path as well as the optomotor one.")
        print(f"  TELLO_YAW_SIGN IS WRONG (slope {slope:.1f} with {TELLO_YAW_SIGN} applied) "
              f"- flip it and re-run.", flush=True)
    else:
        log.raw(f"  TELLO_YAW_SIGN={TELLO_YAW_SIGN} confirmed: positive slope, so the "
                f"conversion matches this project's positive-is-left convention.")
        log.raw(f"  -> re-run --mode pan with --ppr {slope:.0f}")
        print(f"  TELLO_YAW_SIGN={TELLO_YAW_SIGN} confirmed (slope {slope:+.1f})", flush=True)
        print(f"  -> re-run --mode pan with --ppr {slope:.0f}", flush=True)


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


# ---------------------------------------------------------------- analysis
#
# Scoring a DNg02 run is not like scoring an escape run. The escape test had a
# discrete event - DNp01 fired or it didn't - so you counted hits against SPACE
# markers. DNg02 has no event. What makes it a population code rather than 24
# noisy neurons is the ORDER cells come on in, and no amount of staring at the
# HUD settles that: the bar updates 25 times a second and the eye averages it.
# So this reads the log back and scores it.


def parse_log(path):
    """-> (meta, ladder, rows). Reads only the log, never the repo, so a file
    handed over from another machine or another commit still analyses - which is
    the whole point of the log being self-contained."""
    meta, ladder, rows, columns = {}, [], [], None
    ladder_re = re.compile(r"^#\s+(\d+)\s+(DNg02_\S+)\s*$")
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("#"):
                m = ladder_re.match(line)
                if m:
                    ladder.append(m.group(2))
                elif ": " in line:
                    key, _, value = line[1:].strip().partition(": ")
                    meta[key] = value
                continue
            fields = line.split("\t")
            if columns is None:
                columns = fields
                continue
            if len(fields) == len(columns):
                rows.append(dict(zip(columns, fields)))
    return meta, ladder, rows


def spearman(a, b):
    """Rank correlation, without pulling in scipy for one number."""
    def ranks(xs):
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        r = [0.0] * len(xs)
        i = 0
        while i < len(order):           # average ranks within ties
            j = i
            while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
                j += 1
            shared = (i + j) / 2.0
            for k in range(i, j + 1):
                r[order[k]] = shared
            i = j + 1
        return r

    if len(a) < 3:
        return float("nan")
    ra, rb = ranks(a), ranks(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = math.sqrt(sum((x - ma) ** 2 for x in ra))
    db = math.sqrt(sum((y - mb) ** 2 for y in rb))
    return num / (da * db) if da and db else float("nan")


def _bin_by(rows, key, value_keys, n_bins=5, min_rows=20):
    """Sorts rows by |key| and returns [(mean |key|, means), ...], where means
    holds both the signed mean of each value key and, under "<key>_abs", the
    mean of its MAGNITUDE.

    Both are needed, and using the wrong one quietly guts the measurement: a
    bin mixes leftward and rightward sweeps, so the signed mean of steer
    averages +0.4 and -0.4 to nearly zero no matter how strong the response is.
    The magnitude mean is what answers "did it respond more".

    Binned rather than correlated because these relationships saturate, and one
    correlation coefficient would hide that."""
    pairs = []
    for r in rows:
        try:
            x = abs(float(r[key]))
            vals = {k: float(r[k]) for k in value_keys}
        except (KeyError, ValueError):
            continue
        pairs.append((x, vals))
    if len(pairs) < min_rows:
        return []
    pairs.sort(key=lambda p: p[0])
    size = len(pairs) // n_bins
    out = []
    for b in range(n_bins):
        chunk = pairs[b * size:(b + 1) * size] if b < n_bins - 1 else pairs[b * size:]
        if not chunk:
            continue
        mean_x = sum(x for x, _ in chunk) / len(chunk)
        means = {}
        for k in value_keys:
            means[k] = sum(v[k] for _, v in chunk) / len(chunk)
            means[f"{k}_abs"] = sum(abs(v[k]) for _, v in chunk) / len(chunk)
        # Per-cycle asymmetry, which is not the same as the asymmetry of the
        # per-bin means, for the same cancellation reason.
        if {"n_left", "n_right"} <= set(value_keys):
            means["spread_abs"] = sum(abs(v["n_right"] - v["n_left"])
                                      for _, v in chunk) / len(chunk)
        out.append((mean_x, means))
    return out


def analyse(path):
    if not Path(path).is_file():
        print(f"no such log: {path}\n"
              f"Pass the path explicitly, or run the test first - it writes "
              f"{ROOT / 'Drone' / 'flight_logs' / Path(__file__).with_suffix('.log').name} by default.")
        return 1
    meta, ladder, rows = parse_log(path)
    live = [r for r in rows if r.get("phase") == "live"]
    base = [r for r in rows if r.get("phase") == "baseline"]
    mode = meta.get("test", "")
    print(f"log:      {path}")
    print(f"recorded: {meta.get('started', '?')}  git {meta.get('git', '?')}")
    print(f"rows:     {len(rows)} ({len(base)} baseline, {len(live)} live)")
    if not ladder:
        print("\nNo DNg02 ladder in the log header - was this written by an older "
              "version of this script?")
        return 1
    if not live:
        print("\nNo live-phase rows: the run never got past the baseline phase.")
        return 1

    verdicts = []

    def verdict(ok, message):
        print(f"  {'PASS' if ok else 'FAIL'}  {message}")
        verdicts.append(ok)

    # Per-cell activity, and the recruitment level each cell needs to come on.
    fired_at = {label: [] for label in ladder}
    totals = []
    for r in live:
        try:
            n_total = int(r["n_left"]) + int(r["n_right"])
        except (KeyError, ValueError):
            continue
        totals.append(n_total)
        for label in r.get("recruited", "-").split(","):
            if label in fired_at:
                fired_at[label].append(n_total)

    n_live = len(totals)
    if not n_live:
        print("\nNo usable recruitment rows.")
        return 1

    print(f"\n=== per-cell recruitment, in committed ladder order ===")
    print(f"{'#':>3} {'cell':>17} {'fired':>7} {'rate':>7} {'onset':>7}")
    freqs, onsets, positions = [], [], []
    for i, label in enumerate(ladder):
        hits = fired_at[label]
        rate = len(hits) / n_live
        # "onset" = the recruitment level this cell typically needs before it
        # joins in. A population code means early-ladder cells come on while
        # few others are active, and late ones only in a crowd.
        onset = percentile(hits, 25) if hits else float("nan")
        bar = "#" * int(round(24 * rate))
        print(f"{i:>3} {label:>17} {len(hits):>7} {rate:>6.1%} "
              f"{onset if hits else float('nan'):>7.1f}  {bar}")
        if hits:
            freqs.append(rate)
            onsets.append(onset)
            positions.append(i)

    silent = [l for l in ladder if not fired_at[l]]
    print(f"\nnever recruited: {len(silent)} of {len(ladder)}"
          + (f" -> {', '.join(silent)}" if silent else ""))

    print("\n=== is it a population code? ===")
    # Ladder position ascending = weaker drive, so a working code gives rate
    # DEcreasing with position (negative rho) and onset INcreasing (positive).
    rho_rate = spearman(positions, freqs)
    rho_onset = spearman(positions, onsets)
    print(f"  ladder position vs firing rate : rho = {rho_rate:+.3f}  "
          f"(want negative: cells earlier in the ladder fire more often)")
    print(f"  ladder position vs onset level : rho = {rho_onset:+.3f}  "
          f"(want positive: later cells need more of the population active)")
    verdict(rho_rate < -0.4,
            f"recruitment follows the committed ladder order (rho {rho_rate:+.3f})")
    verdict(len(silent) <= len(ladder) * 0.55,
            f"at least ~45% of the population recruits ({len(ladder) - len(silent)} "
            f"of {len(ladder)} cells)")

    print("\n=== is it graded, or all-or-nothing? ===")
    distinct = sorted(set(totals))
    at_zero = totals.count(0) / n_live
    at_peak = totals.count(max(totals)) / n_live
    print(f"  recruited count: min {min(totals)} median {percentile(totals, 50):.0f} "
          f"max {max(totals)}, {len(distinct)} distinct levels")
    print(f"  cycles at 0: {at_zero:.1%}   cycles at the peak: {at_peak:.1%}")
    print("  deciles: " + " ".join(f"{percentile(totals, q):.0f}" for q in range(10, 100, 10)))
    verdict(len(distinct) >= 5,
            f"the count takes many values, not a couple ({len(distinct)} distinct)")
    verdict(at_zero + at_peak < 0.8,
            f"not bimodal - only {at_zero + at_peak:.0%} of cycles sit at 0 or the peak")

    # The two channels have to be scored separately, and conflating them is an
    # easy mistake: rotation drives the OPPONENT channel, which redistributes
    # recruitment between the sides rather than adding to it, so total
    # recruitment is not what rotation should move. Total recruitment belongs to
    # the thrust channel, which rides on drive_common.
    print("\n=== steering channel: does |steer| track |rotation|? ===")
    rot_bins = _bin_by(live, "rotation", ("steer", "n_left", "n_right"))
    if rot_bins:
        steers = []
        for rot, vals in rot_bins:
            steer = vals["steer_abs"]
            spread = vals["spread_abs"]
            total = vals["n_left"] + vals["n_right"]
            steers.append(steer)
            print(f"  |rotation| ~{rot:5.2f}  ->  mean|steer| {steer:5.3f}   "
                  f"mean|nR-nL| {spread:4.2f}   total {total:5.2f} cells")
        verdict(steers[-1] > steers[0] + 0.05,
                f"stronger rotation gives a stronger steering signal "
                f"({steers[0]:.3f} -> {steers[-1]:.3f})")
        first_total = rot_bins[0][1]["n_left"] + rot_bins[0][1]["n_right"]
        last_total = rot_bins[-1][1]["n_left"] + rot_bins[-1][1]["n_right"]
        if last_total < first_total - 1.0:
            print(f"  note: total recruitment FALLS with rotation "
                  f"({first_total:.1f} -> {last_total:.1f}). That is expected, not a "
                  f"fault - a large opponent offset saturates one side and suppresses "
                  f"the other, so a hard turn request costs thrust.")
    else:
        print("  too few rows to bin")

    print("\n=== thrust channel: does recruitment track drive_common? ===")
    drive_bins = _bin_by(live, "drive_common", ("n_left", "n_right"))
    spread = (max(d for d, _ in drive_bins) - min(d for d, _ in drive_bins)
              if drive_bins else 0.0)
    if drive_bins and spread >= 0.15:
        totals_by_drive = []
        for drive, vals in drive_bins:
            total = vals["n_left"] + vals["n_right"]
            totals_by_drive.append(total)
            print(f"  drive_common ~{drive:5.2f}  ->  {total:5.2f} cells")
        verdict(totals_by_drive[-1] > totals_by_drive[0] + 0.5,
                f"more common drive recruits more cells "
                f"({totals_by_drive[0]:.2f} -> {totals_by_drive[-1]:.2f})")
    else:
        print(f"  drive_common barely varied (range {spread:.3f}) - not scored.")
        print("  Expected on a desk: drive_common comes from the TRANSLATIONAL flow")
        print("  set-point error, and sweeping a card sideways produces almost no")
        print("  translational flow. This channel only becomes testable in flight.")

    if base:
        print("\n=== baseline noise vs the steering floor ===")
        quiet = []
        for r in base:
            try:
                quiet.append(abs(float(r["rotation"])))
            except (KeyError, ValueError):
                pass
        if quiet:
            p95, worst = percentile(quiet, 95), max(quiet)
            floor = float(meta.get("OPTOMOTOR_FLOW_FLOOR", fbc.OPTOMOTOR_FLOW_FLOOR))
            print(f"  quiet |rotation|: mean {statistics.fmean(quiet):.3f} "
                  f"p95 {p95:.3f} max {worst:.3f}   floor {floor}")
            verdict(worst < floor,
                    f"floor sits above the noise (max {worst:.3f} < {floor})")
            if worst >= floor:
                print(f"        -> raise OPTOMOTOR_FLOW_FLOOR to about "
                      f"{2 * p95:.2f} before flying, or it steers at its own noise")

    # Sign, recomputed from the rows rather than trusting the summary line.
    marked = {"d": [], "a": []}
    for r in live:
        if r.get("mark") in marked:
            try:
                marked[r["mark"]].append(float(r["steer"]))
            except (KeyError, ValueError):
                pass
    if marked["d"] and marked["a"]:
        d_mean = statistics.fmean(marked["d"])
        a_mean = statistics.fmean(marked["a"])
        print("\n=== sign, recomputed from the rows ===")
        print(f"  sweep right: steer {d_mean:+.3f} over {len(marked['d'])} cycles")
        print(f"  sweep left : steer {a_mean:+.3f} over {len(marked['a'])} cycles")
        verdict(d_mean > a_mean,
                f"rightward sweeps give the higher steer ({d_mean:+.3f} > {a_mean:+.3f})")
    elif "calibrate" not in mode:
        print("\n=== sign ===\n  no direction marks in the log - press D/A while sweeping")

    if live:
        print("\n=== timing ===")
        for col, budget in (("ms_brain", 33.0), ("ms_cycle", None)):
            vals = []
            for r in live:
                try:
                    vals.append(float(r[col]))
                except (KeyError, ValueError):
                    pass
            if vals:
                print(f"  {col}: mean {statistics.fmean(vals):5.1f} "
                      f"p95 {percentile(vals, 95):5.1f}"
                      + (f"   budget {budget}" if budget else ""))
                if budget:
                    verdict(statistics.fmean(vals) < budget,
                            f"{col} within the {budget}ms budget")

    ok = all(verdicts)
    print(f"\n{'ALL CHECKS PASSED' if ok else f'{verdicts.count(False)} of {len(verdicts)} CHECKS FAILED'}"
          f"  ({len(verdicts)} checks)")
    if not ok:
        print("A failure here is a measurement, not a crash - read the section it "
              "came from before changing any constant.")
    return 0 if ok else 1


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

    # Same trick tello_neuron_test.py uses: wrap the subprocess request so both
    # the drive request going in and the full population readout coming back
    # are visible every cycle, not just when the adapter decides to log.
    last = {}
    inner_request = controller._brain.request

    def capturing_request(payload):
        result = inner_request(payload)
        last.clear()
        last.update(payload=payload, result=result)
        return result

    controller._brain.request = capturing_request

    prev_gray = None
    prev_accepted = None
    prev_yaw = None
    accepted = duplicates = att_missing = 0
    marks = []
    cal = {"dyaw": [], "raw": [], "deg": []}
    stats = {"rotation": [], "n_total": [], "steer": [], "hz": [],
             "ms_cycle": [], "ms_brain": [], "ms_flow": []}
    baseline_rotation = []
    by_mark = {"d": [], "a": []}
    state_counts = {}
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
                duplicates += 1
                # Polled on EVERY raw iteration, not just accepted frames, so a
                # keypress can't be swallowed behind a run of duplicates.
                if not args.no_video and (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
                continue
            prev_accepted = gray
            accepted += 1

            now = time.perf_counter()
            dt = max((now - last_accept_t) if last_accept_t else 1.0 / 30.0, 1e-3)
            last_accept_t = now

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
                if accepted >= WARMUP_FRAMES:
                    looming.reset()
                    prev_gray = None
                    phase = "baseline"
                    phase_t0 = time.perf_counter()
                    log.raw(f"PHASE baseline starts t={t_abs:.2f} "
                            f"(warmup discarded {accepted} frames)")
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
            state_counts[controller.state] = state_counts.get(controller.state, 0) + 1

            log.row([
                f"{t_abs:.3f}", phase, f"{dt * 1000:.1f}", duplicates, mark_now,
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

            stats["rotation"].append(hemi["rotation"])
            stats["n_total"].append(n_left + n_right)
            stats["steer"].append(d.get("steer", 0.0))
            stats["hz"].append(hz)
            stats["ms_cycle"].append(ms_cycle)
            stats["ms_brain"].append(ms_brain)
            stats["ms_flow"].append(ms_flow)
            if not att["ok"]:
                att_missing += 1
            if phase == "baseline":
                baseline_rotation.append(hemi["rotation"])
            if phase == "live":
                if args.mode == "calibrate":
                    # derotate_flow assumes u ~= pixels_per_radian * yaw_rate *
                    # dt, so d_yaw is the regressor and the RAW (un-derotated)
                    # common-mode flow is the response. Every sample is kept;
                    # the filtering happens per window, not per frame.
                    cal["dyaw"].append(yaw_rate * dt)
                    cal["raw"].append(raw_hemi["rotation"])
                    cal["deg"].append(abs(math.degrees(yaw_rate * dt)))
                if mark_now in by_mark:
                    by_mark[mark_now].append((n_left, n_right, d.get("steer", 0.0),
                                              hemi["rotation"], cmd["yaw_rate"]))

            if not args.no_video:
                view = small.copy()
                steer = d.get("steer", 0.0)
                colour = (0, 255, 255) if abs(steer) > fbc.OPTOMOTOR_STATE_THRESHOLD else (0, 255, 0)
                lines = [
                    f"{phase.upper()} [{args.mode}] {t_abs:5.1f}s {hz:4.1f}Hz dup={duplicates}",
                    f"dx L={hemi['left']:+.2f} R={hemi['right']:+.2f}  "
                    f"rot={hemi['rotation']:+.2f} trans={hemi['translation']:+.2f}",
                    f"drive c={payload.get('drive_common', 0.0):.2f} "
                    f"L={payload.get('drive_left', 0.0):+.2f} R={payload.get('drive_right', 0.0):+.2f}",
                    f"nL={n_left:>2} nR={n_right:>2} thrust={d.get('thrust', 0.0):.2f} "
                    f"steer={steer:+.2f}",
                    f"would command yaw={cmd['yaw_rate']:+.3f} rad/s  state={controller.state}",
                    (f"yaw_rate={yaw_rate:+.2f} deg/frame={math.degrees(abs(yaw_rate * dt)):.1f} "
                     f"samples={len(cal['dyaw'])}  Q=quit"
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
                    marks.append((round(t_abs, 2), mark_now))
                    log.raw(f"MARK {mark_now} t={t_abs:.2f}")

    except KeyboardInterrupt:
        print("\ninterrupted", flush=True)
    finally:
        log.raw("=" * 70)
        log.raw("SUMMARY")
        log.raw(f"mode: {args.mode}")
        log.raw(f"accepted_frames: {accepted}   duplicate_frames_skipped: {duplicates}")
        log.raw(f"cycles_without_attitude: {att_missing}")
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
        log.raw(f"state_cycle_counts: {state_counts}")

        if args.mode == "calibrate":
            report_calibration(log, cal)
        else:
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
            if stats["n_total"]:
                log.raw(f"peak_recruited={max(stats['n_total'])} of {len(labels)}  "
                        f"peak_|steer|={max(abs(v) for v in stats['steer']):.3f}")
        log.raw("END")
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
        print(f"accepted={accepted} duplicates={duplicates} marks={len(marks)}", flush=True)


if __name__ == "__main__":
    sys.exit(main() or 0)
