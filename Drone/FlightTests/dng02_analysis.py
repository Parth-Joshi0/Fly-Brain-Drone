"""
Offline half of tello_dng02_test.py: fitting --mode calibrate's pixels-per-
radian, and scoring a --mode pan log (--analyze). Reads logs and numbers
only - no drone, no brain, no cv2 window.
"""

import math
import re
import statistics
from pathlib import Path

import numpy as np

import NeuralPathways.flybrain_controller as fbc
from NeuralPathways.EscapeNeuron.optical_flow import _PIXELS_PER_RADIAN
from Drone.tello_drone import TELLO_YAW_SIGN
from Drone.flight_harness import ROOT, percentile

DEFAULT_LOG = ROOT / "Drone" / "flight_logs" / "tello_dng02_test.log"

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


class _Verdicts:
    """Prints each PASS/FAIL as it's decided and keeps the tally."""

    def __init__(self):
        self.results = []

    def __call__(self, ok, message):
        print(f"  {'PASS' if ok else 'FAIL'}  {message}")
        self.results.append(ok)


def _recruitment(ladder, live):
    """Per-cell activity, and the recruitment level each cell needs to come
    on. -> (fired_at {label: [total recruited each time it fired]}, totals)."""
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
    return fired_at, totals


def _score_ladder(ladder, fired_at, n_live, verdict):
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


def _score_graded(totals, n_live, verdict):
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

def _score_steering(live, verdict):
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


def _score_thrust(live, verdict):
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


def _score_baseline(base, meta, verdict):
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


def _score_sign(live, mode, verdict):
    """Sign, recomputed from the rows rather than trusting the summary line."""
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


def _score_timing(live, verdict):
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


def analyse(path):
    if not Path(path).is_file():
        print(f"no such log: {path}\n"
              f"Pass the path explicitly, or run the test first - it writes "
              f"{DEFAULT_LOG} by default.")
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

    verdict = _Verdicts()

    fired_at, totals = _recruitment(ladder, live)
    n_live = len(totals)
    if not n_live:
        print("\nNo usable recruitment rows.")
        return 1

    _score_ladder(ladder, fired_at, n_live, verdict)
    _score_graded(totals, n_live, verdict)
    _score_steering(live, verdict)
    _score_thrust(live, verdict)
    if base:
        _score_baseline(base, meta, verdict)
    _score_sign(live, mode, verdict)
    if live:
        _score_timing(live, verdict)

    verdicts = verdict.results
    ok = all(verdicts)
    print(f"\n{'ALL CHECKS PASSED' if ok else f'{verdicts.count(False)} of {len(verdicts)} CHECKS FAILED'}"
          f"  ({len(verdicts)} checks)")
    if not ok:
        print("A failure here is a measurement, not a crash - read the section it "
              "came from before changing any constant.")
    return 0 if ok else 1
