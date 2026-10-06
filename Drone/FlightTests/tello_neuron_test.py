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
    python Drone/FlightTests/tello_neuron_test.py

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
from NeuralPathways.EscapeNeuron.optical_flow import (LoomingDetector, compute_flow, derotate_flow,
                                 grid_flow_strengths)
from Drone.flight_harness import (DEFAULT_VERTICAL_FOV, PROC_HEIGHT, PROC_WIDTH, TELLO_DIAGONAL_FOV,
                                  WARMUP_FRAMES, TelloLogger, YawTracker, capture_brain_requests,
                                  frame_not_ready, git_commit, open_stream, open_tello, percentile,
                                  read_attitude)


class Logger(TelloLogger):
    COLUMNS = [
        "t_s", "phase", "dt_ms", "dup",
        "exp_l", "exp_c", "exp_r",
        "floor", "loom_l", "loom_r",
        "escape", "brain_yaw", "brain_fwd", "state", "spikes",
        "flow_l", "flow_c", "flow_r",
        "pitch", "roll", "yaw", "att_ok", "tello_vgx", "tello_h", "mean_px",
        "ms_grab", "ms_flow", "ms_brain", "ms_cycle",
    ]


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


class Run:
    """Everything accumulated over a run, for the summary at the end."""

    def __init__(self):
        self.accepted = 0
        self.duplicates = 0
        self.att_missing = 0
        self.marks = []
        self.stats = {"exp_c": [], "loom": [], "escape": [], "hz": [],
                      "ms_cycle": [], "ms_brain": [], "ms_flow": []}
        self.baseline_exp = []
        self.escape_events = []
        self.state_counts = {}

    def note_state(self, state, t_abs):
        self.state_counts[state] = self.state_counts.get(state, 0) + 1
        if state == "ESCAPE" and (not self.escape_events or self.escape_events[-1][1] != "run"):
            self.escape_events.append((round(t_abs, 2), "run"))
        elif state != "ESCAPE" and self.escape_events and self.escape_events[-1][1] == "run":
            self.escape_events[-1] = (self.escape_events[-1][0], "done")


def draw_hud(small, phase, t_abs, hz, run, expansion, floor, loom_l, loom_r, escape, state, spikes):
    view = small.copy()
    fired = state == "ESCAPE"
    color = (0, 0, 255) if fired else (0, 255, 0)
    lines = [
        f"{phase.upper()}  {t_abs:5.1f}s  {hz:4.1f}Hz  dup={run.duplicates}",
        f"EXP L={expansion['left']:.2f} C={expansion['center']:.2f} R={expansion['right']:.2f}",
        f"floor={floor:.2f}  loom L={loom_l:.2f} R={loom_r:.2f}",
        f"escape={escape:.2f}  state={state}",
        f"spikes: {spikes}",
        f"marks={len(run.marks)}   SPACE=swat  Q=quit",
    ]
    for i, line in enumerate(lines):
        cv2.putText(view, line, (6, 16 + i * 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, color, 1, cv2.LINE_AA)
    if fired:
        cv2.putText(view, "DNp01 ESCAPE", (60, PROC_HEIGHT - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow("Tello neuron test", view)


def write_summary(log, run):
    """Everything needed to judge the run without the raw rows."""
    stats = run.stats
    log.raw("=" * 70)
    log.raw("SUMMARY")
    log.raw(f"accepted_frames: {run.accepted}   duplicate_frames_skipped: {run.duplicates}")
    log.raw(f"cycles_without_attitude: {run.att_missing}"
            + ("  (orientation was identity on those - the loom floor's "
               "rotation term could not be measured)" if run.att_missing else ""))
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
        # 33ms is the 30Hz decision-loop budget NeuralPathways/README.md
        # holds the circuit to. Over it usually means the brain is running
        # under a slower brian2 install than intended - see
        # brain_interpreter in the header above.
        if brain_mean > 33.0:
            log.raw(f"WARNING: brain_ms mean {brain_mean:.1f} exceeds the 33ms "
                    f"30Hz budget. Check 'brain_interpreter' in the header - "
                    f"pin the fast env with FLYBRAIN_PYTHON and re-run.")
            print(f"\nWARNING: brain step averaged {brain_mean:.1f}ms (budget 33ms) - "
                  f"the brain may be running under the wrong brian2 install.",
                  flush=True)
    if run.baseline_exp:
        log.raw(f"baseline_expansion (quiet, max over columns) "
                f"mean={statistics.fmean(run.baseline_exp):.3f} "
                f"p95={percentile(run.baseline_exp, 95):.3f} "
                f"max={max(run.baseline_exp):.3f}")
        log.raw(f"  -> compare against LOOM_EXPANSION_FLOOR="
                f"{fbc.LOOM_EXPANSION_FLOOR}: the floor must sit above this "
                f"noise or the circuit fires at nothing")
    if stats["exp_c"]:
        log.raw(f"peak_expansion_center={max(stats['exp_c']):.3f}  "
                f"peak_loom={max(stats['loom']):.3f}  "
                f"peak_escape={max(stats['escape']):.3f}  "
                f"(ESCAPE_STATE_THRESHOLD={fbc.ESCAPE_STATE_THRESHOLD})")
    log.raw(f"state_cycle_counts: {run.state_counts}")
    log.raw(f"swat_marks ({len(run.marks)}): {run.marks}")
    log.raw(f"escape_triggers ({len(run.escape_events)}): {[t for t, _ in run.escape_events]}")
    log.raw("END")


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

    last = capture_brain_requests(controller)
    yaw_tracker = YawTracker()
    run = Run()

    prev_gray = None          # for grid flow (magnitude)
    prev_accepted = None      # raw gray of the last ACCEPTED frame, for dup detection

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
            # Tello frames arrive as RGB (same conversion fly_tello.py does)
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
                run.duplicates += 1
                if not args.no_video:
                    if (cv2.waitKey(1) & 0xFF) == ord("q"):
                        break
                else:
                    time.sleep(0.002)
                continue
            prev_accepted = gray
            run.accepted += 1

            now = time.perf_counter()
            # dt between frames actually fed to the detector - NOT the
            # nominal loop period. LoomingDetector divides its per-cell
            # expansion by (frames_apart * dt), so feeding it a wrong dt
            # scales the expansion rate directly.
            dt = (now - last_accept_t) if last_accept_t else 1.0 / 30.0
            last_accept_t = now
            dt = max(dt, 1e-3)

            att = read_attitude(tello)
            yaw_deg, quat, yaw_rate = yaw_tracker.update(att, dt)

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
                if run.accepted >= WARMUP_FRAMES:
                    # Drop the EMA/median history built from the settling
                    # exposure so the baseline phase starts clean.
                    looming.reset()
                    prev_gray = None
                    phase = "baseline"
                    phase_t0 = time.perf_counter()
                    log.raw(f"PHASE baseline starts t={t_abs:.2f} "
                            f"(warmup discarded {run.accepted} frames)")
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

            run.note_state(controller.state, t_abs)

            log.row([
                f"{t_abs:.3f}", phase, f"{dt * 1000:.1f}", run.duplicates,
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

            stats = run.stats
            stats["exp_c"].append(expansion["center"])
            stats["loom"].append(max(loom_l, loom_r))
            stats["escape"].append(escape)
            if not att["ok"]:
                run.att_missing += 1
            stats["hz"].append(hz)
            stats["ms_cycle"].append(ms_cycle)
            stats["ms_brain"].append(ms_brain)
            stats["ms_flow"].append(ms_flow)
            if phase == "baseline":
                run.baseline_exp.append(max(expansion.values()))

            if not args.no_video:
                draw_hud(small, phase, t_abs, hz, run, expansion, floor, loom_l, loom_r,
                         escape, controller.state, spikes)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord(" "):
                    run.marks.append(round(t_abs, 2))
                    log.raw(f"MARK swat t={t_abs:.2f} phase={phase}")
                    print(f"  [swat marked at {t_abs:.2f}s]", flush=True)

    except KeyboardInterrupt:
        print("\ninterrupted", flush=True)
    finally:
        write_summary(log, run)
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
        print(f"accepted={run.accepted} duplicates={run.duplicates} "
              f"marks={len(run.marks)} escape_triggers={len(run.escape_events)}", flush=True)
        if run.stats["loom"]:
            print(f"peak loom={max(run.stats['loom']):.2f} peak escape={max(run.stats['escape']):.2f}",
                  flush=True)


if __name__ == "__main__":
    main()
