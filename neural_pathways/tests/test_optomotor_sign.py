"""Locks down the sign of the DNg02 optomotor loop, and that escape still wins.

Why this exists as its own test: the DNg02 steering term is the first closed
feedback loop in this project that runs through the airframe, and it uses the
OPPOSITE sign convention to DNp06 two lines away in the same function (see
neural_pathways/flybrain_controller.py's decide()). If that sign is ever flipped -
by a refactor, by a "fix" to one of the two conventions, by a change to
signed_hemifield_flow's definition of positive - the drone does not fly a bit
worse. It yaws harder into its own drift until it hits something. That failure
is invisible in the offline circuit test, which never sees a yaw command at all,
and expensive to discover in the air.

So this drives the whole adapter chain with synthetic flow and asserts on the
commanded yaw, which is the thing that actually reaches the motors:

    synthetic flow -> _optomotor_drive -> brain subprocess -> DNg02 readout
                   -> decide() -> yaw_rate

Runs in the normal (non-brian2) env - the brain is a subprocess, as always:
    python neural_pathways/tests/test_optomotor_sign.py

The loop it is checking, stated once: positive yaw_rate means turn LEFT in this
project. Turning left sweeps the scene rightwards across the image, so
signed_hemifield_flow's "rotation" goes positive. derotate_flow has already
subtracted the yaw the drone COMMANDED, so a positive residual means it is
rotating left more than it asked to. The correction is to yaw right, i.e. a
NEGATIVE yaw_rate. Negative feedback. If any link in that chain inverts, the
same drift produces a positive yaw_rate and the loop diverges.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import neural_pathways.flybrain_controller as fbc
from neural_pathways.flybrain_controller import FlyBrainController

SETTLE = 30          # cycles for the DNg02 EMAs to reach the new operating point
QUIET_FLOW = 0.1     # plain magnitude in every grid cell: "nothing interesting"

failures = []


def check(ok, message):
    print(f"  {'PASS' if ok else 'FAIL'}  {message}")
    if not ok:
        failures.append(message)


def make_flow(rotation=0.0, translation=0.0, expansion=0.0):
    """A flow dict shaped like the one main.py builds, with the two
    signed_hemifield_flow keys the optomotor path reads."""
    flow = {f"{row}_{col}": QUIET_FLOW
            for row in ("top", "center", "bottom")
            for col in ("left", "center", "right")}
    flow.update({"left": QUIET_FLOW, "right": QUIET_FLOW, "top": QUIET_FLOW,
                 "bottom": QUIET_FLOW, "center": QUIET_FLOW,
                 "expansion_left": expansion, "expansion_center": expansion,
                 "expansion_right": expansion,
                 "rotation": rotation, "translation": translation})
    return flow


STATE = {"position": (0.0, 0.0, 1.0), "orientation": (0.0, 0.0, 0.0, 1.0),
         "yaw_degrees": 0.0, "actual_vx": 0.0}


def settle(controller, flow, cycles=SETTLE):
    for _ in range(cycles):
        cmd = controller.decide(dict(flow), dict(STATE))
    return cmd


def main():
    print(f"DNG02_YAW_GAIN={fbc.DNG02_YAW_GAIN}  "
          f"OPTOMOTOR_FLOW_FLOOR={fbc.OPTOMOTOR_FLOW_FLOOR}  "
          f"OPTOMOTOR_FLOW_SPAN={fbc.OPTOMOTOR_FLOW_SPAN}")

    # --- the drive mapping alone, no network involved. Pure arithmetic, so it
    # pins the sign of the half of the chain that has no stochastic element. ---
    print("\n_optomotor_drive mapping (no spiking involved):")
    drive = FlyBrainController._optomotor_drive   # a staticmethod; no subprocess needed
    for rotation, expect_right_favoured in ((2.0, True), (-2.0, False)):
        d = drive(make_flow(rotation=rotation))
        favoured = d["drive_right"] > d["drive_left"]
        check(favoured is expect_right_favoured,
              f"rotation {rotation:+.1f} -> drive L={d['drive_left']:+.2f} "
              f"R={d['drive_right']:+.2f}: "
              f"{'right' if favoured else 'left'} DNg02 favoured")
    below = drive(make_flow(rotation=fbc.OPTOMOTOR_FLOW_FLOOR * 0.5))
    check(below["drive_left"] == 0.0 and below["drive_right"] == 0.0,
          f"rotation below the floor asks for no steering "
          f"(L={below['drive_left']:+.2f} R={below['drive_right']:+.2f})")

    print("\nstarting the brain subprocess...")
    c = FlyBrainController(bounds=None, optomotor=True)
    try:
        # --- the whole chain, ending on the commanded yaw ---
        print("\nfull chain, synthetic residual rotation -> commanded yaw:")
        results = {}
        for label, rotation in (("still", 0.0), ("scene->right", 2.0), ("scene->left", -2.0)):
            cmd = settle(c, make_flow(rotation=rotation,
                                      translation=fbc.OPTOMOTOR_FLOW_SETPOINT))
            d = c.dng02
            results[label] = (d["steer"], cmd["yaw_rate"])
            print(f"    {label:>12}: nL={d['n_left']:>2} nR={d['n_right']:>2} "
                  f"steer={d['steer']:+.3f} -> yaw_rate={cmd['yaw_rate']:+.4f} "
                  f"({'RIGHT' if cmd['yaw_rate'] < 0 else 'LEFT ' if cmd['yaw_rate'] > 0 else 'none '})")

        right_steer, right_yaw = results["scene->right"]
        left_steer, left_yaw = results["scene->left"]
        check(right_steer > left_steer,
              f"scene sliding right gives the higher steer "
              f"({right_steer:+.3f} vs {left_steer:+.3f}) - matches Namiki et al. 2022, "
              f"rightward motion raising the right DNg02 cells")
        # THE load-bearing assertion. Rotating left uncommanded must produce a
        # right-turn command, or the loop is positive feedback.
        check(right_yaw < 0,
              f"rotating LEFT uncommanded -> commands a RIGHT turn "
              f"(yaw_rate={right_yaw:+.4f}) = NEGATIVE feedback")
        check(left_yaw > 0,
              f"rotating RIGHT uncommanded -> commands a LEFT turn "
              f"(yaw_rate={left_yaw:+.4f}) = NEGATIVE feedback")
        check(abs(right_yaw) <= fbc.DNG02_YAW_AUTHORITY + 1e-9
              and abs(left_yaw) <= fbc.DNG02_YAW_AUTHORITY + 1e-9,
              f"the DNg02 yaw contribution stays inside its authority cap "
              f"({fbc.DNG02_YAW_AUTHORITY} rad/s)")

        # --- escape has to win. A hard loom plus a hard steering request must
        # still produce the dodge, not a compromise between the two. ---
        print("\nescape dominance (hard loom AND a hard steering request):")
        cmd = settle(c, make_flow(rotation=4.0, translation=0.0,
                                  expansion=fbc.LOOM_EXPANSION_FLOOR
                                  + fbc.LOOM_EXPANSION_SPAN), cycles=20)
        print(f"    state={c.state} yaw_rate={cmd['yaw_rate']:+.4f} "
              f"strafe={cmd['strafe_speed']:+.2f} fwd={cmd['forward_speed']:+.2f}")
        check(c.state == "ESCAPE", f"state is ESCAPE, not OPTOMOTOR (got {c.state})")
        check(cmd["yaw_rate"] == 0.0,
              f"the escape maneuver's yaw is untouched by the optomotor term "
              f"(yaw_rate={cmd['yaw_rate']:+.4f}, should be exactly 0)")
        check(abs(cmd["strafe_speed"]) > 0.0,
              f"the dodge still strafes ({cmd['strafe_speed']:+.2f} m/s)")
    finally:
        c.close()

    # --- and with the flag off, none of this exists ---
    print("\noptomotor=False leaves the old behaviour alone:")
    c = FlyBrainController(bounds=None)
    try:
        cmd = settle(c, make_flow(rotation=2.0), cycles=10)
        check(c._brain.info.get("with_dng02") is False,
              "the brain subprocess did not build the DNg02 half at all")
        check(cmd["yaw_rate"] == 0.0,
              f"residual rotation is ignored entirely (yaw_rate={cmd['yaw_rate']:+.4f})")
        check(c.state in ("CRUISE", "AVOID_LEFT", "AVOID_RIGHT"),
              f"state never becomes OPTOMOTOR (got {c.state})")
    finally:
        c.close()

    print(f"\n{'ALL CHECKS PASSED' if not failures else f'{len(failures)} CHECK(S) FAILED:'}")
    for f in failures:
        print(f"  - {f}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
