"""
First FLYING test of the DNg02 optomotor loop on the Tello. Yaw only.

    Tello video -> derotate -> signed_hemifield_flow -> DNg02 population ->
    steer -> yaw command -> real RC

The drone takes off and hovers. The ONLY thing the brain is allowed to command
is yaw, from the DNg02 opponent channel. Forward/strafe stay zero (thrust is
logged but not flown), DNp06 avoidance yaw is dropped, and an ESCAPE is turned
into a plain hover here - so exactly one loop is closed through the airframe.

The experiment is the textbook optomotor response: hold a big textured sheet
(newspaper, patterned blanket) close in front of the drone so it fills most of
the view, then slide it sideways. The drone should turn WITH the sheet (sheet
moves right -> drone yaws right). Stop the sheet and it should stop turning.

Why pixels_per_radian stays at derotate_flow's default (75): the measured
flow-per-radian on the Tello is 106-184 at flight yaw rates (Farneback
under-tracks - see NeuralPathways/DNG02_SESSION_NOTES.txt 6.2). 75 is below all of it, so the
drone's own rotation is always UNDER-cancelled, the residual keeps the sign of
the real rotation, and the loop opposes it: a damper. Over-cancelling (e.g. the
geometric 228) flips that residual and is the positive-feedback case.

--dry-run skips takeoff and never sends RC, so this doubles as the props-off
steering check: hold the drone and turn it by hand, and the HUD's yaw command
should push back against the turn.

Keys (video window focused):
    l         land immediately
    q / SPACE hover, then land shortly after

    python Drone/Tests/tello_optomotor_flight_test.py [--seconds 20] [--dry-run] [--yaw-gain G] [--log PATH]
"""

import argparse
import math
import sys
import time
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from NeuralPathways.flybrain_controller import (DNG02_YAW_AUTHORITY, DNG02_YAW_GAIN,
                                             FlyBrainController)
from Drone.tello_drone import TelloDrone
from Drone.Tests.tello_escape_flight_test import (ARM_GRACE_SECONDS, DEFAULT_VERTICAL_FOV,
                                              EMERGENCY_HOVER_SECONDS, EMPTY_CMD, PROC_HEIGHT,
                                              PROC_WIDTH, apply_command, frame_not_ready,
                                              open_stream, open_tello)
from NeuralPathways.EscapeNeuron.optical_flow import (LoomingDetector, compute_flow, derotate_flow,
                                 grid_flow_strengths, signed_hemifield_flow)

WARMUP_FRAMES = 15

# Tello attitude arrives at ~9 Hz against 20-25 Hz video, so a per-frame yaw
# difference is zero most frames and a spike on the rest. Differencing over a
# short window gives derotation a usable rate instead of that staircase.
YAW_RATE_WINDOW_S = 0.25

# Auto-hover-and-land if the drone is spinning faster than the loop can ever
# command (DNG02_YAW_AUTHORITY), sustained - i.e. something is wrong.
RUNAWAY_YAW_RATE = 1.2      # rad/s
RUNAWAY_SECONDS = 0.5


class WindowedYawRate:
    def __init__(self, window_s=YAW_RATE_WINDOW_S):
        self.window_s = window_s
        self._hist = deque()

    def update(self, t, yaw_deg):
        self._hist.append((t, math.radians(yaw_deg)))
        while len(self._hist) > 2 and t - self._hist[1][0] >= self.window_s:
            self._hist.popleft()
        t0, y0 = self._hist[0]
        if t - t0 <= 1e-3:
            return 0.0
        d = (self._hist[-1][1] - y0 + math.pi) % (2 * math.pi) - math.pi
        return d / (t - t0)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seconds", type=float, default=20.0, help="hard auto-land time after arming")
    parser.add_argument("--fov", type=float, default=DEFAULT_VERTICAL_FOV)
    parser.add_argument("--dry-run", action="store_true", help="props off: no takeoff, no RC")
    parser.add_argument("--no-video", action="store_true")
    # Desk runs measured |steer| ~0.1, which at the module gain is ~1% yaw RC -
    # below what the Tello visibly responds to. Step this up flight by flight;
    # DNG02_YAW_AUTHORITY still caps the result.
    parser.add_argument("--yaw-gain", type=float, default=DNG02_YAW_GAIN)
    parser.add_argument("--log", default=str(ROOT / "Drone" / "flight_logs" / Path(__file__).with_suffix(".log").name))
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
    log("starting the brain subprocess with DNg02 (optomotor=True)...")
    brain = FlyBrainController(bounds=None, optomotor=True)
    log(f"brain ready  yaw_gain={args.yaw_gain}  yaw_authority={DNG02_YAW_AUTHORITY} rad/s  "
        f"dry_run={args.dry_run}")
    drone = TelloDrone(tello, frame_read, PROC_WIDTH, PROC_HEIGHT)
    yaw_meter = WindowedYawRate()

    log_file.write("ROWS t,rotation,translation,yaw_rate_meas,steer,n_left,n_right,thrust,"
                   "yaw_cmd,brain_state\n")
    rows = []
    prev_gray = prev_accepted = last_accept_t = None
    accepted = 0
    phase, armed_t0 = "warmup", None
    abort_mode = abort_deadline = None
    runaway_since = None
    escapes_suppressed = 0

    try:
        if args.dry_run:
            log("DRY RUN - not taking off. Turn the drone by hand; yaw_cmd should oppose.")
        else:
            log("Taking off...")
            drone.takeoff()
            log(f"Airborne (state={drone.state}).")

        while True:
            key = 255 if args.no_video else cv2.waitKey(1) & 0xFF
            if abort_mode is None:
                if key == ord("l"):
                    log("ABORT: 'l' pressed - landing now")
                    abort_mode, abort_deadline = "land_now", time.perf_counter()
                elif key in (ord("q"), ord(" ")):
                    log("ABORT: emergency key - hover, landing shortly")
                    abort_mode = "hover_then_land"
                    abort_deadline = time.perf_counter() + EMERGENCY_HOVER_SECONDS

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
            yaw_rate = yaw_meter.update(now, state["yaw_degrees"])
            expansion = looming.update(gray, state["orientation"], dt)
            flow = {"left": 0.0, "right": 0.0, "top": 0.0, "bottom": 0.0, "center": 0.0,
                    "rotation": 0.0, "translation": 0.0}
            if prev_gray is not None:
                derotated = derotate_flow(compute_flow(prev_gray, gray), yaw_rate, dt)
                flow = grid_flow_strengths(derotated)
                hemi = signed_hemifield_flow(derotated)
                # Only these two - hemi's own left/right would clobber the grid's.
                flow["rotation"], flow["translation"] = hemi["rotation"], hemi["translation"]
            prev_gray = gray
            flow.update({f"expansion_{side}": v for side, v in expansion.items()})

            if phase == "warmup" and accepted >= WARMUP_FRAMES:
                looming.reset()
                prev_gray = None
                phase, armed_t0 = "arming", time.perf_counter()
                log(f"warmup done - hovering {ARM_GRACE_SECONDS:.1f}s before the loop closes")

            t_armed = 0.0 if armed_t0 is None else time.perf_counter() - armed_t0
            live = phase in ("arming", "armed") and t_armed >= ARM_GRACE_SECONDS
            if phase == "arming" and live:
                phase = "armed"
                prev_gray = None
                log("ARMED - optomotor yaw loop closed")

            if abs(yaw_rate) > RUNAWAY_YAW_RATE and live and abort_mode is None and not args.dry_run:
                runaway_since = runaway_since or now
                if now - runaway_since >= RUNAWAY_SECONDS:
                    log(f"ABORT: yaw runaway ({yaw_rate:+.2f} rad/s for {RUNAWAY_SECONDS}s) - "
                        f"hover then land")
                    abort_mode = "hover_then_land"
                    abort_deadline = now + EMERGENCY_HOVER_SECONDS
            else:
                runaway_since = None

            final = dict(EMPTY_CMD)
            final["hover"] = True
            yaw_cmd, d = 0.0, brain.dng02
            if abort_mode is not None:
                if time.perf_counter() >= abort_deadline:
                    log(f"abort ({abort_mode}): landing now")
                    break
            elif live:
                brain.decide(flow, state)
                d = brain.dng02
                if brain.state == "ESCAPE":
                    escapes_suppressed += 1     # hover - this test isolates DNg02
                else:
                    # Same term decide() adds, without DNp06's yaw on top of it.
                    yaw_cmd = -args.yaw_gain * d.get("steer", 0.0)
                    yaw_cmd = max(-DNG02_YAW_AUTHORITY, min(DNG02_YAW_AUTHORITY, yaw_cmd))
                    final["hover"] = False
                    final["yaw_rate"] = yaw_cmd
                row = (round(t_armed, 3), round(flow["rotation"], 4), round(flow["translation"], 4),
                       round(yaw_rate, 4), round(d.get("steer", 0.0), 4), d.get("n_left", 0),
                       d.get("n_right", 0), round(d.get("thrust", 0.0), 3), round(yaw_cmd, 4),
                       brain.state)
                rows.append(row)
                log_file.write(",".join(str(v) for v in row) + "\n")

            apply_command(drone, final)
            drone.step()

            if not args.no_video:
                view = small.copy()
                turn = "RIGHT" if yaw_cmd < 0 else "LEFT" if yaw_cmd > 0 else "-"
                lines = [
                    f"{phase.upper()}{' DRY' if args.dry_run else ''}  t={t_armed:5.1f}/{args.seconds:.0f}s",
                    f"rotation={flow['rotation']:+.2f}px  yaw_meas={yaw_rate:+.2f}rad/s",
                    f"DNg02 L={d.get('n_left', 0):2d} R={d.get('n_right', 0):2d}  "
                    f"steer={d.get('steer', 0.0):+.2f}",
                    f"yaw_cmd={yaw_cmd:+.3f} ({turn})  state={brain.state if live else '-'}",
                    "l=land  q/SPACE=hover+land",
                ]
                for i, line in enumerate(lines):
                    cv2.putText(view, line, (6, 16 + i * 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                                (0, 255, 0), 1, cv2.LINE_AA)
                cv2.imshow("Tello optomotor flight test", view)

            if phase == "armed" and t_armed >= args.seconds:
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
        for fn in (brain.close, tello.streamoff, tello.end):
            try:
                fn()
            except Exception:
                pass
        if not args.no_video:
            cv2.destroyAllWindows()

        log("=" * 60)
        log("SUMMARY")
        log(f"accepted_frames={accepted}  live_cycles={len(rows)}  "
            f"escapes_suppressed={escapes_suppressed}")
        active = [r for r in rows if r[8] != 0.0]
        if len(active) >= 5:
            rot = np.array([r[1] for r in active])
            cmd = np.array([r[8] for r in active])
            # Negative feedback means the command opposes the residual
            # rotation: rotation > 0 -> yaw right (negative). So want < 0.
            corr = float(np.corrcoef(rot, cmd)[0, 1]) if rot.std() > 0 and cmd.std() > 0 else float("nan")
            log(f"steering cycles={len(active)}  corr(rotation, yaw_cmd)={corr:+.3f} (want negative)  "
                f"max|yaw_cmd|={np.abs(cmd).max():.3f}")
        else:
            log(f"steering cycles={len(active)} - too few to score")
        log("END")
        log_file.close()
        print(f"\nwrote {log_path}", flush=True)


if __name__ == "__main__":
    main()
