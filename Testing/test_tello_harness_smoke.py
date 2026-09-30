"""Runs the real main loops of the Tello scripts against a FAKE drone, so the
code paths that need hardware get exercised without any.

Why this exists: the per-cycle body and the cv2 HUD of tello_dng02_test.py and
tello_neuron_test.py are only reachable with a Tello attached and a video
window open. Everything else in Testing/ can be run at a desk, so those few
hundred lines were the one part of the project whose first execution was
always on real hardware, mid-session, with a charged battery burning. A stale
dict key in one HUD f-string - invisible to import, invisible to the analyser,
invisible to --no-video - crashed a calibration run one cycle after warmup and
cost the whole session. This test would have caught it in under a minute.

It is a SMOKE test, not a correctness test: it asserts the loops run, log, and
shut down cleanly in every mode, including the video path. It says nothing
about whether the numbers are right - that is what test_dng02_circuit.py,
test_optomotor_sign.py and the --analyze pass are for.

    python Testing/test_tello_harness_smoke.py

No drone, no video window (cv2's display calls are stubbed), but a real brain
subprocess and real Farneback on real frames.
"""
import math
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

failures = []


def check(ok, message):
    print(f"  {'PASS' if ok else 'FAIL'}  {message}")
    if not ok:
        failures.append(message)


class FakeFrameRead:
    """Textured frames that move, so Farneback has something real to track."""

    def __init__(self, width=960, height=720):
        self.w, self.h = width, height
        rng = np.random.default_rng(7)
        # A big noise field, panned across, gives genuine optic flow.
        self._field = rng.integers(0, 255, (height * 2, width * 2, 3), dtype=np.uint8)
        self._i = 0

    @property
    def frame(self):
        self._i += 1
        # Pan diagonally at a few px per frame; every frame differs, so the
        # duplicate-frame guard does not reject them all.
        ox, oy = (self._i * 3) % self.w, (self._i * 2) % self.h
        return self._field[oy:oy + self.h, ox:ox + self.w].copy()


class FakeTello:
    """Only the handful of calls the harness makes. Yaw advances so calibrate
    mode has attitude to regress against."""

    def __init__(self):
        self._n = 0

    def get_current_state(self):
        self._n += 1
        return {"pitch": 0, "roll": 0, "yaw": (self._n // 2) % 360,
                "vgx": 0, "h": 80, "tof": 80, "bat": 90}

    def get_battery(self):
        return 90

    def streamoff(self):
        pass

    def streamon(self):
        pass

    def end(self):
        pass

    # Present so an accidental flight call would be loud rather than silent.
    def takeoff(self):
        raise AssertionError("smoke test must never call takeoff()")

    def land(self):
        raise AssertionError("smoke test must never call land()")

    def send_rc_control(self, *a):
        raise AssertionError("smoke test must never call send_rc_control()")


def stub_cv2_display():
    """Neutralise the window calls so the HUD code itself still runs. That is
    the point - the bug this test exists for lived inside an f-string that only
    the video path evaluates."""
    cv2.imshow = lambda *a, **k: None
    cv2.waitKey = lambda *a, **k: 255        # 255 = "no key pressed"
    cv2.destroyAllWindows = lambda *a, **k: None
    cv2.namedWindow = lambda *a, **k: None


def run_script(module, argv, label):
    """Runs a script's real main() with the drone and display faked out."""
    module.open_tello = lambda log: FakeTello()
    module.open_stream = lambda tello, log: FakeFrameRead()
    module.WARMUP_FRAMES = 3
    old_argv = sys.argv
    sys.argv = argv
    try:
        rc = module.main()
        check(rc in (None, 0), f"{label}: main() returned cleanly (rc={rc})")
        return True
    except SystemExit as exc:
        check(exc.code in (None, 0), f"{label}: exited {exc.code}")
        return exc.code in (None, 0)
    except Exception as exc:
        check(False, f"{label}: raised {type(exc).__name__}: {exc}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        sys.argv = old_argv


def main():
    stub_cv2_display()
    tmp = Path(tempfile.mkdtemp(prefix="tello_smoke_"))
    print(f"logs -> {tmp}\n")

    import Testing.tello_dng02_test as dng
    import Testing.tello_neuron_test as neu
    import Testing.tello_optomotor_flight_test as opto

    # Every mode x the video path, because the HUD is mode-dependent and is
    # exactly where the untested code lives.
    cases = [
        (dng, "pan, video", ["x", "--mode", "pan", "--seconds", "1.5", "--baseline", "0.5",
                             "--log", str(tmp / "pan.log")]),
        (dng, "pan, headless", ["x", "--mode", "pan", "--seconds", "1.5", "--baseline", "0.5",
                                "--no-video", "--log", str(tmp / "pan_nv.log")]),
        (dng, "calibrate, video", ["x", "--mode", "calibrate", "--seconds", "1.5",
                                   "--baseline", "0.5", "--log", str(tmp / "cal.log")]),
        (dng, "calibrate, headless", ["x", "--mode", "calibrate", "--seconds", "1.5",
                                      "--baseline", "0.5", "--no-video",
                                      "--log", str(tmp / "cal_nv.log")]),
        (neu, "neuron_test, video", ["x", "--seconds", "1.5", "--baseline", "0.5",
                                     "--log", str(tmp / "neu.log")]),
        # --dry-run only: FakeTello raises on takeoff/land/RC, which is what
        # proves dry-run really never touches the motors.
        (opto, "optomotor flight, dry-run video", ["x", "--dry-run", "--seconds", "4",
                                                   "--log", str(tmp / "opto.log")]),
    ]
    for module, label, argv in cases:
        print(f"--- {label} ---")
        run_script(module, argv, label)

    # A log with rows, a header and a summary is the real evidence the loop ran
    # rather than falling over quietly somewhere.
    print("\n--- logs are complete ---")
    for name in ("pan.log", "pan_nv.log", "cal.log", "cal_nv.log", "neu.log", "opto.log"):
        path = tmp / name
        if not path.is_file():
            check(False, f"{name}: not written")
            continue
        text = path.read_text()
        rows = [l for l in text.splitlines() if l[:1].isdigit()]
        check("SUMMARY" in text and "END" in text and len(rows) >= 3,
              f"{name}: {len(rows)} data rows, header + summary present")

    # And the analyser has to read back what the loop just wrote - the two
    # drifted apart once already when a column was added.
    print("\n--- the analyser reads its own output ---")
    try:
        rc = dng.analyse(str(tmp / "pan.log"))
        check(rc in (0, 1), f"analyse() ran on a fresh log (verdict rc={rc})")
    except Exception as exc:
        check(False, f"analyse() raised {type(exc).__name__}: {exc}")

    print(f"\n{'ALL CHECKS PASSED' if not failures else f'{len(failures)} CHECK(S) FAILED:'}")
    for f in failures:
        print(f"  - {f}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
