# BrainView

A live diagram of which parts of the fly brain are working while the drone flies, instead of a line of HUD text naming the neurons that fired. It opens as its own window ("Fly Brain Activity") next to whatever the flight script already shows.

The layout follows the brain view HUD in [blendi-remade/fly-brain-minecraft](https://github.com/blendi-remade/fly-brain-minecraft) (`BrainViewHud.java`). The left column is the brain itself, drawn as a cloud of its real neurons. Every FlyWire neuron's soma (139k) is projected head-on (**frontal**) and from above (**dorsal**). Each pixel is coloured by the region most of its somata belong to (teal optic lobes, blue-grey central brain, green sensory, and so on), with brightness set by log(density). Each time one of the simulated cells spikes, it glows white-yellow at its real soma position and fades over about 0.3 s. Below the maps are spike counts per region for this tick, each with a peak-hold tick, and a spikes/tick chart of the last 60 brain steps.

The right column is the circuit readout, styled like that repo's Neuroscope HUD:
- chips for which pathway is active right now (ESCAPE, STABILIZER, FEEDING); a pathway that isn't running in the current script is marked `off`
- the MOTOR bars the circuit produced
- the firing rate of each simulated population, with a peak-hold tick
- the FoodNeuron feeding state
- the brain command
- a key to the populations

Only the 418 neurons `fly_brain_controller.py` actually simulates can light up: LC4/LPLC2, DNp01/03/06, DNg02 and its drive pool. The faint yellow dots mark where they sit. The rest of the cloud is anatomy, there to show where in the brain the circuit lives. FlyWire covers the brain only, so unlike the Minecraft HUD there is no nerve cord half; the dorsal view of the brain takes its place.

The diagram attaches to a running `FlyBrainController` / `FoodOrbitBehaviour` by wrapping two of its instance methods at runtime, and passes every argument and result through unchanged. The only change outside this folder is that `fly_brain_controller.py`'s `step()` also returns `spiked` (every spike in the network this window, as local indices) and its `ready` handshake carries `neuron_ids`, so the map can show recorded spikes for every cell, not just the DNs.

## Files

| File | What it does |
|---|---|
| `brain_diagram.py` | The diagram itself (`BrainDiagram`), the per-cycle activity it draws from (`BrainActivity`: heat maps, region and population counts), and the hooks: `attach(controller)` for a `NeuralPathways/flybrain_controller.py` `FlyBrainController`, `attach_food(behaviour)` for a `FoodNeuron/food_orbit.py` `FoodOrbitBehaviour`. Populations come from `../looming_circuit_neurons.json` and `../StabilizerNeuron/dng02_circuit_neurons.json`. The network constants and the index → root id map come from the brain subprocess's own `ready` handshake. |
| `build_brain_atlas.py` | Offline: projects every neuron in Schlegel et al. 2024's FlyWire annotation table (`../Data/flywire_neuron_annotations_630.tsv`, downloaded once, 31 MB) into the two maps and records the pixel each simulated circuit neuron sits at. Needs pandas, not brian2. |
| `brain_atlas.npz` | Its output (74 KB, committed): somata per pixel and dominant region for each view, plus the circuit neurons' pixels. Rebuild with `python NeuralPathways/BrainView/build_brain_atlas.py` only to change the map size or the region grouping. |
| `show_brain.py` | Runs any existing flight script with the diagram attached, without editing that script. |

## Running it

Put `show_brain.py` in front of the script you would normally run. Everything after the script path goes to that script untouched.

```bash
python NeuralPathways/BrainView/show_brain.py main.py                                          # PyBullet sim
python NeuralPathways/BrainView/show_brain.py Drone/Tests/tello_neuron_test.py                 # real Tello, props off
python NeuralPathways/BrainView/show_brain.py Drone/Tests/tello_escape_flight_test.py --seconds 30
python NeuralPathways/BrainView/show_brain.py Drone/Tests/tello_optomotor_flight_test.py --dry-run
python NeuralPathways/BrainView/show_brain.py Drone/tello_camera.py                            # banana feeding, dry run
python NeuralPathways/BrainView/show_brain.py Drone/tello_camera.py --fly --scared             # feeding + Giant Fiber scare
```

With `Drone/tello_camera.py` the FoodNeuron state machine is what flies the Tello. The brain (only with `--scared`, via `EscapeNeuron/fear_brain.py`) decides *when* to get scared, and its own dodge command is thrown away. So for that script the diagram shows the feeding state (SEARCH / APPROACH / FEED / SCARED / WAIT / DONE / LAND) as the headline state, and the RC command food_orbit.py actually sent in place of the brain command. The DNp01/03/06 somata still glow from the real spikes, including the extra catch-up brain steps `FearBrain` runs per camera frame. Without `--scared` no connectome runs, so only FEEDING is lit.

`--fps N` (before the script path) caps how often the diagram is redrawn. The default is 15, which is every other 30 Hz decision cycle. A redraw costs about 7-9 ms, and on the real drone the brain step alone already uses about 20 of the 33 ms budget. Activity is still recorded every cycle: every spike is splatted into the map and counted, and the spikes/tick chart never skips a step. Only the repaint is rate-limited.

From your own code, one line does the same:

```python
from NeuralPathways.BrainView.brain_diagram import attach
attach(controller)          # controller = FlyBrainController(...)
```

## What each part shows

| Part | Source | Measured or derived |
|---|---|---|
| map glow, region bars, spikes/tick, POPULATIONS Hz | `spiked` from `fly_brain_controller.py`'s `step()`, placed via the handshake's `neuron_ids` and `brain_atlas.npz` | **Real spikes** of every simulated cell for the last 20 ms window. The footnote says `glow: recorded spikes of the simulated cells` |
| ... with a brain that doesn't send `spiked` | `spike_counts` and `dng02.counts`, plus the `loom_*` / `drive_*` request | DN and DNg02 spikes are still real. LC4/LPLC2 and drive-pool spikes are *sampled* at the Poisson rate each cell is driven at (`loom × MAX_POI_RATE`, and the drive rule in `step()`): statistically what the model does, not a recording. The footnote says so |
| the cloud | `brain_atlas.npz` | Anatomy only: soma positions (or a point on the neuron when there's no soma), coloured by FlyWire super-class. Visual projection neurons such as LC4/LPLC2 count as optic lobe, split by side. The ~2,000 ascending neurons, whose somata are in the nerve cord, sit at the neck cut, which is the brown patch at the bottom of the frontal view and the strip at the back of the dorsal view |
| region bars | spikes this tick per region, with peak-hold (×0.97 per step) | Only regions that contain simulated cells can be non-zero: optic lobe L/R (LC4/LPLC2), central brain and ascending (drive pool), descending (DNp01/03/06, DNg02) |
| ESCAPE / TURN / FORWARD / THRUST / STEER bars | `escape`, `yaw`, `forward`, `dng02.thrust`, `dng02.steer` | Circuit output. TURN and STEER are both drawn screen-left = turn left, even though DNp06 and DNg02 use opposite signs (see `../README.md`) |
| brain command + drone glyph | what `decide()` returned | Before `main.py`'s `NEURON_TEST_MODE` hover and before `SafetyLayer`, both of which can still override it. Under `tello_camera.py` this panel shows the FoodNeuron RC command that was sent instead |
| state (header) | `controller.state` / `escape_direction`, or the FoodNeuron state when that is flying the drone | |
| `RT x.xx×` | simulated time per wall-clock time of the last subprocess round trip | Red below 0.9×, meaning the brain is falling behind real time |

## Tests (`Tests/`)

### `test_brain_diagram.py`: does the diagram draw the right thing?

Headless: no drone, no simulator, no brian2. It drives `attach()` on a fake brain with the same surface as `FlyBrainController` through quiet → left-eye loom → escape → DNg02 steering → reset, and checks the following:
- the hooks are transparent (identical commands and brain results with and without the diagram)
- the atlas places all 418 simulated neurons inside both maps
- the right somata glow (left LC4/LPLC2 on left loom, the left Giant Fiber on escape, DNg02 on steering) and only then, and the region bars agree. This runs once with a brain that sends `spiked` and once with one that doesn't (the sampled fallback)
- reset clears the glow and the history
- the optomotor-off and food-only layouts draw
- the drawing cost stays under the budget

```bash
python NeuralPathways/BrainView/Tests/test_brain_diagram.py
python NeuralPathways/BrainView/Tests/test_brain_diagram.py --out snapshots/   # save PNGs of each phase
python NeuralPathways/BrainView/Tests/test_brain_diagram.py --show             # watch it animate
```

The spikes in this test are synthetic, so it checks the drawing and the hooks, not the circuit. `NeuralPathways/Tests/test_brain_circuit.py` checks the circuit, including that `spiked` agrees with `spike_counts` for the 6 DNs on every step.

## Notes

- The window never calls `cv2.waitKey()` itself, because that would swallow the host script's keys (`l` = land on the real drone). It repaints on the host loop's own `waitKey(1)`, which `main.py` and every `Drone/` script already call each cycle. HighGUI key events are shared between windows, so those keys still work while the diagram has focus. With a script's `--no-video` flag there is no `waitKey`, so the window won't repaint.
- `main.py` only calls `decide()` once the drone is autonomous and past its take-off hover. Until then, and in MANUAL mode, the diagram holds its last frame.
