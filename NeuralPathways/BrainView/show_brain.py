"""
Runs any of this repo's flight scripts with the live brain diagram
(brain_diagram.py) open next to it - without editing the script.

    python NeuralPathways/BrainView/show_brain.py main.py
    python NeuralPathways/BrainView/show_brain.py Drone/Tests/tello_escape_flight_test.py --seconds 30
    python NeuralPathways/BrainView/show_brain.py Drone/Tests/tello_optomotor_flight_test.py --dry-run
    python NeuralPathways/BrainView/show_brain.py Drone/tello_camera.py

Everything after the script path is passed to that script untouched.

How: before the script runs, NeuralPathways.flybrain_controller's
FlyBrainController is swapped for a subclass whose only addition is calling
brain_diagram.attach() on each new instance. Every script here looks that name
up through its module at import time (main.py does it inside main()), so it
picks up the subclass automatically. The script then
runs as __main__, exactly as `python <script>` would run it.
"""

import argparse
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from NeuralPathways.BrainView.brain_diagram import DEFAULT_MAX_FPS, WINDOW_NAME, attach  # noqa: E402


def install(max_fps=DEFAULT_MAX_FPS, window=WINDOW_NAME):
    """Swaps in the watching subclass. Importing flybrain_controller needs
    no brian2 (the brain itself runs as a subprocess), so this is safe from
    the main venv."""
    import NeuralPathways.flybrain_controller as fbc

    class FlyBrainController(fbc.FlyBrainController):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            attach(self, window=window, max_fps=max_fps)

    fbc.FlyBrainController = FlyBrainController


def main():
    parser = argparse.ArgumentParser(
        description="Run a flight script with the live fly-brain diagram open.",
        usage="%(prog)s [--fps N] SCRIPT [script args...]",
    )
    parser.add_argument("--fps", type=float, default=DEFAULT_MAX_FPS,
                        help=f"max diagram redraws per second (default {DEFAULT_MAX_FPS:g}; spikes are "
                             "recorded every brain step regardless, only the repaint is capped)")
    parser.add_argument("script", help="path to the script to run, e.g. main.py")
    parser.add_argument("script_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    script = Path(args.script).resolve()
    if not script.exists():
        parser.error(f"no such script: {args.script}")

    install(max_fps=args.fps)

    # Same sys.argv / sys.path[0] the script would see under `python <script>`.
    sys.argv = [str(script)] + args.script_args
    sys.path.insert(0, str(script.parent))
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
