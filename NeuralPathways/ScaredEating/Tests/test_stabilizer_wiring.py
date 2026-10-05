"""Locks down the DNg02 stabilizer as wired into the banana-eating drone brain.

test_optomotor_sign.py covers the adapter (flow dict -> decide() -> yaw_rate).
This covers the part the Tello actually runs, from camera pictures to the rc
yaw that fly_tello.py sends:

    pictures -> FearBrain(stabilize=True): Farneback, efference-copy
             derotation, signed_hemifield_flow -> brain subprocess (DNg02)
             -> dng02_yaw_rc -> ScaredEatingBrain adds it to the behaviour's yaw

Synthetic textured pictures slide sideways at a known rate, a fake Tello
answers telemetry, and the clock is faked so every picture is exactly
FRAME_DT apart. Checks:

  - still scene: no correction to speak of
  - scene sliding right (drone rotating LEFT, uncommanded) -> yaw right (+rc)
  - scene sliding left -> yaw left (-rc)
  - the same leftward slide while the behaviour COMMANDS that right turn
    (efference copy) -> correction mostly gone, i.e. DNg02 doesn't fight it
  - ScaredEatingBrain.step(): sent yaw = behaviour's yaw + DNg02's

Runs in the normal env - the brain is a subprocess, as always:
    python NeuralPathways/ScaredEating/Tests/test_stabilizer_wiring.py
"""
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

import NeuralPathways.EscapeNeuron.fear_brain as fb
from NeuralPathways.EscapeNeuron.fear_brain import FearBrain, EFFERENCE_PIXELS_PER_RADIAN
from Drone.tello_drone import RC_YAW_RATE_AT_100

FRAME_DT = 0.05     # s between pictures (20/s, about what the Tello loop manages)
SETTLE = 25         # pictures for the DNg02 EMAs to reach the new operating point
MEASURE = 25        # pictures averaged after that
TURN_RC = 40        # feeding_behaviour's SCAN_TURN_SPEED
# |rc| a correction must clear to count as clearly signed. The spiking
# population wobbles the median by ~+-1 run to run, and left-going
# corrections come out weaker than right-going ones (~-5 vs +9 here - the
# same population asymmetry test_optomotor_sign.py prints), so this checks
# the sign, not a strength.
SIGNED = 3

failures = []


def check(ok, message):
    print(f"  {'PASS' if ok else 'FAIL'}  {message}")
    if not ok:
        failures.append(message)


class FakeClock:
    """Stands in for fear_brain's `time` module: time() is ours to step."""

    def __init__(self):
        self.t = 1000.0

    def time(self):
        return self.t

    def perf_counter(self):
        return time.perf_counter()


class FakeTello:
    def get_current_state(self):
        return {"pitch": 0, "roll": 0, "yaw": 0, "h": 80}


class Scene:
    """A smooth texture panned horizontally; shift_px > 0 slides it RIGHT."""

    def __init__(self):
        rng = np.random.default_rng(3)
        noise = rng.integers(0, 255, (240, 2400), dtype=np.uint8)
        field = cv2.GaussianBlur(noise, (0, 0), 2.5)
        field = cv2.normalize(field, None, 0, 255, cv2.NORM_MINMAX)
        self.field = cv2.cvtColor(field, cv2.COLOR_GRAY2BGR)
        self.x = 1000.0

    def next(self, shift_px):
        # Content moving right = the crop window moving left
        self.x -= shift_px
        x0 = int(round(self.x))
        return self.field[:, x0:x0 + fb.PROC_WIDTH].copy()


def run(fear, clock, scene, shift_px, intended_yaw=0, pictures=SETTLE + MEASURE, residual=False):
    """Median dng02_yaw_rc over the last MEASURE pictures (and, with
    residual=True, the median rotation DNg02 was shown)."""
    out, rotation = [], []
    for _ in range(pictures):
        clock.t += FRAME_DT
        fear.record_command(0, 0, 0, intended_yaw + fear.dng02_yaw_rc, intended_yaw=intended_yaw)
        fear.update(scene.next(shift_px))
        out.append(fear.dng02_yaw_rc)
        rotation.append(fear.dng02_rotation)
    rc = statistics.median(out[-MEASURE:])
    return (rc, statistics.median(rotation[-MEASURE:])) if residual else rc


def main():
    clock = FakeClock()
    fb.time = clock

    print("starting the brain subprocess with DNg02...")
    fear = FearBrain(FakeTello(), None, stabilize=True)
    fear.start()
    scene = Scene()

    # Arm (warmup frames + grace period) on a still scene
    run(fear, clock, scene, 0.0, pictures=int(fb.ARM_GRACE_SECONDS / FRAME_DT) + fb.WARMUP_FRAMES + 2)
    check(fear.armed, "armed after warmup + grace period")

    print("\nstill scene")
    still = run(fear, clock, scene, 0.0)
    check(abs(still) <= 3, f"no real correction: median dng02_yaw_rc {still:+} (|.| <= 3)")

    print("\nuncommanded rotation")
    right = run(fear, clock, scene, +2.0)
    check(right > SIGNED, f"scene sliding right (drone turning left) -> yaw right: {right:+} (> +{SIGNED})")
    run(fear, clock, scene, 0.0)
    left = run(fear, clock, scene, -2.0)
    check(left < -SIGNED, f"scene sliding left (drone turning right) -> yaw left: {left:+} (< -{SIGNED})")

    print("\nefference copy: the same right turn, but commanded")
    run(fear, clock, scene, 0.0)
    # Exactly the flow a commanded TURN_RC right turn should cause
    turn_rate = TURN_RC / 100 * RC_YAW_RATE_AT_100
    turn_shift = -EFFERENCE_PIXELS_PER_RADIAN * turn_rate * FRAME_DT
    uncommanded, rot_uncommanded = run(fear, clock, scene, turn_shift, residual=True)
    run(fear, clock, scene, 0.0)
    commanded, rot_commanded = run(fear, clock, scene, turn_shift, intended_yaw=TURN_RC, residual=True)
    print(f"  (scene shift {turn_shift:+.1f} px/picture; rotation DNg02 sees: "
          f"uncommanded {rot_uncommanded:+.2f} px, commanded {rot_commanded:+.2f} px)")
    check(uncommanded < -SIGNED, f"uncommanded, DNg02 fights it: {uncommanded:+} (< -{SIGNED})")
    # Judged on the rotation DNg02 is SHOWN, not on its noisy spiking output.
    # Not ~0: Farneback under-reads this texture's 4.2 px slide by ~17%, so
    # subtracting the full expected slide leaves ~+0.7 px - the "true flow per
    # radian below EFFERENCE_PIXELS_PER_RADIAN" case, where the drone turns a
    # little faster than asked. That's the safe direction; what must never
    # happen is DNg02 fighting the turn.
    check(abs(rot_commanded) <= 0.25 * abs(rot_uncommanded),
          f"efference copy removes >= 75% of the commanded turn's image motion "
          f"({abs(rot_commanded) / max(1e-6, abs(rot_uncommanded)):.0%} left)")
    check(-3 <= commanded <= 8,
          f"commanded, DNg02 doesn't fight it and adds at most 20% of the "
          f"{TURN_RC} rc turn: {commanded:+} (-3..+8)")

    fear.close()

    print("\nScaredEatingBrain.step(): sent yaw = behaviour + DNg02")
    from NeuralPathways.ScaredEating import scared_eating_brain as seb

    class NoBananas:
        def detect(self, frame):
            return []

    brain = seb.ScaredEatingBrain(NoBananas(), FakeTello(), None, scared=True, stabilize=True)
    brain.start()
    mismatches, corrected = 0, 0
    for _ in range(int(fb.ARM_GRACE_SECONDS / FRAME_DT) + fb.WARMUP_FRAMES + 60):
        clock.t += FRAME_DT
        cmd = brain.step(scene.next(+2.0))
        expected = max(-100, min(100, brain.fear.intended_yaw_rc + brain.fear.dng02_yaw_rc))
        mismatches += cmd.yaw != expected
        corrected += brain.fear.dng02_yaw_rc != 0
    print(f"  final state {brain.behaviour.state}")
    brain.close()
    check(mismatches == 0, f"every sent yaw is behaviour + DNg02 ({mismatches} mismatches)")
    check(corrected > 0, f"DNg02 corrected while the behaviour was running ({corrected} pictures)")

    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s)")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
