# BrainView

A live window ("Fly Brain Activity") showing the fly brain while the drone flies. The brain is drawn head-on as a cloud of its real neurons: every FlyWire soma (139k), coloured by region (teal optic lobes, blue-grey central brain, green sensory). Each time one of the 418 cells `connectome_worker.py` simulates spikes, it glows at its real position and fades over about 0.3 s. These cells are LC4/LPLC2, DNp01/03/06, DNg02 and its drive pool, and faint yellow dots mark where they sit. One header line shows the controller state and the spikes in the last brain step. Below the map, **NEURONS FIRING THIS TICK** lists each simulated cell type (LC4, LPLC2, DNp01, DNp03, DNp06, DNg02, drive pool) in a left column and a right column. Each entry shows how many of its cells spiked out of how many are simulated (for example `12/47`), with a bar for that fraction and a peak-hold tick. A **KEY** says what each type does and what the colours mean.

The layout follows the brain view HUD in [blendi-remade/fly-brain-minecraft](https://github.com/blendi-remade/fly-brain-minecraft), cut down to stay cheap next to the simulator (about 1-3 ms per repaint; the static text is drawn once at startup).

## Running it

```bash
python NeuralPathways/BrainView/show_brain.py main.py                 # PyBullet sim
python NeuralPathways/BrainView/show_brain.py Drone/fly_tello.py   # any other script works the same way
```

Everything after the script path goes to that script untouched. `--fps N` (before the script path) caps repaints; the default is 10. Spikes are recorded every brain step regardless.

`Drone/fly_tello.py` also has it built in: `python Drone/fly_tello.py --fly --scared --show-brain` (needs `--scared` and/or `--stabilize`, since those are what start the fly brain).

From your own code:

```python
from NeuralPathways.BrainView.brain_diagram import attach
attach(controller)          # controller = FlyBrainController(...)
```

## Files

| File | What it does |
|---|---|
| `brain_diagram.py` | The window. `attach()` wraps the controller's `_brain.request` (to read `spiked`) and `decide()` (to repaint), passing everything through unchanged. |
| `show_brain.py` | Runs any flight script with the diagram attached, without editing it. |
| `build_brain_atlas.py` | Offline: bakes the brain image from Schlegel et al. 2024's FlyWire annotation table (downloaded once to `../Data/`, 31 MB) and records each simulated cell's pixel, region and cell type. |
| `brain_atlas.npz` | Its output (45 KB, committed). Rebuild with `python NeuralPathways/BrainView/build_brain_atlas.py`. |

The spikes are real. `connectome_worker.py`'s `step()` returns `spiked`, every spike in the network that step as local indices. Its ready handshake sends `neuron_ids`, which maps those indices to root ids.

## Test

```bash
python NeuralPathways/BrainView/Tests/test_brain_diagram.py           # headless, no brian2
python NeuralPathways/BrainView/Tests/test_brain_diagram.py --show    # watch it animate
```

It runs a fake brain through a quiet phase, a left-eye loom and a reset. It checks that `attach()` is transparent, that the left LC4/LPLC2 somata glow while the right ones don't and the neuron list agrees, that a reset clears both, and that a repaint is cheap.

The window never calls `cv2.waitKey()` itself. It repaints on the host script's own `waitKey(1)`, so the script's keys (such as `l` = land) still work.
