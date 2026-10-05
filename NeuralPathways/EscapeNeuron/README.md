# EscapeNeuron

The drone's "eyes" for the escape/looming-detection pathway: dense optical flow plus the expansion signal that feeds the Giant Fiber circuit in `NeuralPathways/`.

## Files

| File | What it does |
|---|---|
| `optical_flow.py` | Dense (Farneback) optical flow between consecutive camera frames, split into a 3x3 grid (TOP/CENTER/BOTTOM × LEFT/CENTER/RIGHT). Exposes `compute_flow`, `derotate_flow` (cancels the drone's own commanded rotation out of the flow), `signed_hemifield_flow` (residual rotation/translation for the DNg02 optomotor path), `grid_flow_strengths`, `LoomingDetector` (the actual escape-trigger expansion signal), and `FlowVisualizer` (debug overlay). |

## Tests (`Tests/`)

### `calibrate_looming.py`: is the vision signal right?

Flies scripted maneuvers in the simulator (no brain, no SafetyLayer) and runs the real `LoomingDetector` over the recorded frames:

- **Obstacle approaches** (`hover_obstacle`, `fly_obstacle`, `fly_box`): prints measured expansion next to the ideal 2 / time-to-contact. Should track that from about 1.4 s out.
- **Maneuvers with nothing approaching** (`yaw_hover`, `strafe_dodge`, `start_stop`, `fly_open`): prints the background noise distribution (median, p90, p98, max).

```bash
python NeuralPathways/EscapeNeuron/Tests/calibrate_looming.py                  # all scenarios
python NeuralPathways/EscapeNeuron/Tests/calibrate_looming.py hover_obstacle   # just one
```

Re-run and re-tune the `LOOM_EXPANSION_*` constants in `NeuralPathways/flybrain_controller.py` whenever the camera, `LoomingDetector`, or the drone's flight dynamics change. The loom floor needs to sit above the background noise while approach readings still clear it early enough to dodge. Writes its recordings to a new temp directory each run, not into the repo.
