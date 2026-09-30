# BrainView

A live diagram of which parts of the fly brain are working while the drone flies, instead of a line of HUD text naming the neurons that fired. It opens as its own window ("Fly Brain Activity") next to whatever the flight script already shows.

It draws a frontal view of the brain. The compound eyes are on the outside, then the optic lobes with the LC4/LPLC2 looming detectors, then the central brain with the DNp01/DNp03/DNp06 descending neurons, the DNg02 population and its drive pool, and the FoodNeuron feeding state. Descending output leaves through the neck into a motor panel. A 4-second activity raster runs alongside. Header chips show which pathway is active right now (ESCAPE, STABILIZER, FEEDING), and a pathway that isn't running in the current script is marked `off`.

Nothing outside this folder is modified. The diagram attaches to a running `FlyBrainController` / `FoodOrbitBehaviour` by wrapping two of its instance methods at runtime, and passes every argument and result through unchanged.

## Files

| File | What it does |
|---|---|
| `brain_diagram.py` | The diagram itself (`BrainDiagram`), the per-cycle activity it draws from (`BrainActivity`), and the hooks: `attach(controller)` for a `NeuralPathways/flybrain_controller.py` `FlyBrainController`, `attach_food(behaviour)` for a `FoodNeuron/food_orbit.py` `FoodOrbitBehaviour`. Cell counts, sides and the DNg02 ladder come from `../looming_circuit_neurons.json` and `../StabilizerNeuron/dng02_circuit_neurons.json`. The network constants come from the brain subprocess's own `ready` handshake. |
| `show_brain.py` | Runs any existing flight script with the diagram attached, without editing that script. |

## Running it

Put `show_brain.py` in front of the script you would normally run. Everything after the script path goes to that script untouched.

```bash
python NeuralPathways/BrainView/show_brain.py main.py                                          # PyBullet sim
python NeuralPathways/BrainView/show_brain.py Drone/Tests/tello_neuron_test.py                 # real Tello, props off
python NeuralPathways/BrainView/show_brain.py Drone/Tests/tello_escape_flight_test.py --seconds 30
python NeuralPathways/BrainView/show_brain.py Drone/Tests/tello_optomotor_flight_test.py --dry-run
python NeuralPathways/BrainView/show_brain.py Drone/tello_camera.py                            # banana feeding
```

`--fps N` (before the script path) caps how often the diagram is redrawn. The default is 15, which is every other 30 Hz decision cycle. A redraw costs about 6 ms, and on the real drone the brain step alone already uses about 20 of the 33 ms budget. Activity is still recorded every cycle, so the raster never skips a step. Only the repaint is rate-limited.

From your own code, one line does the same:

```python
from NeuralPathways.BrainView.brain_diagram import attach
attach(controller)          # controller = FlyBrainController(...)
```

## What each part shows

| Part | Source | Measured or derived |
|---|---|---|
| DNp01 / DNp03 / DNp06 nodes, spike numbers | `spike_counts` from `fly_brain_controller.py`'s `step()` | **Real spikes**, per cell, for the last 20 ms window |
| DNg02 cells, `L n/13 R n/11 recruited` | `dng02.counts` / `n_left` / `n_right` | **Real spikes**, drawn in ladder order with the strongest input nearest the midline, so recruitment grows outward |
| LC4 / LPLC2 dots, `loom`, `Hz / cell` | the `loom_left` / `loom_right` sent to the brain | Input **drive rate** (`loom × MAX_POI_RATE`). The network doesn't report per-input spikes, so the flicker is *sampled* at that rate: statistically what the model does, not a recording |
| DNg02 drive-pool dots (green excitatory, purple inhibitory) | the `drive_*` request sent to the brain | Each driver's rate, computed with the same rule as `step()` |
| Eyes, `exp` | `flow["expansion_*"]` | max(side, center), the expansion that side's loom is built from |
| ESCAPE / TURN / FORWARD / THRUST / STEER bars | `escape`, `yaw`, `forward`, `dng02.thrust`, `dng02.steer` | Circuit output. TURN and STEER are both drawn screen-left = turn left, even though DNp06 and DNg02 use opposite signs (see `../README.md`) |
| brain command + drone glyph | what `decide()` returned | Before `main.py`'s `NEURON_TEST_MODE` hover and before `SafetyLayer`, both of which can still override it |
| state (top right) | `controller.state` / `escape_direction` | |
| `brain step … ms` | wall time of each subprocess round trip | |

## Tests (`Tests/`)

### `test_brain_diagram.py`: does the diagram draw the right thing?

Headless: no drone, no simulator, no brian2. It drives `attach()` on a fake brain with the same surface as `FlyBrainController` through quiet → left-eye loom → escape → DNg02 steering → reset, and checks the following:
- the hooks are transparent (identical commands and brain results with and without the diagram)
- the right regions light up, and only then
- reset clears the raster
- the optomotor-off and food-only layouts draw
- the drawing cost stays under the budget

```bash
python NeuralPathways/BrainView/Tests/test_brain_diagram.py
python NeuralPathways/BrainView/Tests/test_brain_diagram.py --out snapshots/   # save PNGs of each phase
python NeuralPathways/BrainView/Tests/test_brain_diagram.py --show             # watch it animate
```

The spike counts in this test are synthetic, so it checks the drawing and the hooks, not the circuit. `NeuralPathways/Tests/test_brain_circuit.py` checks the circuit.

## Notes

- The window never calls `cv2.waitKey()` itself, because that would swallow the host script's keys (`l` = land on the real drone). It repaints on the host loop's own `waitKey(1)`, which `main.py` and every `Drone/` script already call each cycle. HighGUI key events are shared between windows, so those keys still work while the diagram has focus. With a script's `--no-video` flag there is no `waitKey`, so the window won't repaint.
- `main.py` only calls `decide()` once the drone is autonomous and past its take-off hover. Until then, and in MANUAL mode, the diagram holds its last frame.
