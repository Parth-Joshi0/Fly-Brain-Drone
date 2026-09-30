# NeuralPathways

The real Fly-Brain connectome, run live as the drone's autonomous controller. Two circuits share one Brian2 network:

- **Escape** (always on) — `EscapeNeuron/`'s looming detector feeds LC4/LPLC2 → DNp01/DNp03/DNp06, the Giant Fiber dodge. This is what makes the drone swat-reactive.
- **DNg02 flight-motor / stabilizer** (opt-in, `optomotor=True`) — a population of 24 descending neurons whose *recruitment count* sets thrust and steering from residual optic flow, replacing what a built-in gyroscope would otherwise do. Its own data, generator script, session notes and circuit-level tests live in `StabilizerNeuron/` — see that folder's README (`FoodNeuron/`'s sibling pathway for feeding is unrelated and doesn't touch this circuit either).

Escape and DNg02 live in the same file pair (`fly_brain_controller.py` / `flybrain_controller.py`) because they share one Brian2 `Network`/`NeuronGroup`/`Synapses` — see `fly_brain_controller.py`'s docstring for why (a measured perf argument, not a shortcut). That's the one piece of DNg02 that couldn't move into `StabilizerNeuron/`: everything else DNg02-specific that doesn't touch the shared network did.

## Files

| File | What it does |
|---|---|
| `fly_brain_controller.py` | The actual Brian2 connectome: builds the LIF network from real FlyWire v630 synapses, runs it as a persistent subprocess (`conda run -n brian2`), and speaks stdin/stdout JSON lines. Never imported directly by `main.py`. |
| `flybrain_controller.py` | The adapter `main.py` actually imports (`FlyBrainController`). Same `decide(flow, state)` / `.state` / `.reset()` contract as `Simulator/reflex_controller.py`, backed by `fly_brain_controller.py` over a subprocess so the main venv never needs Brian2 installed. |
| `looming_circuit_neurons.json` | The 274 neurons making up the LC4/LPLC2 → DNp01/DNp03/DNp06 escape circuit. |
| `Data/` | The real FlyWire v630 connectome data both circuits are built from (rescued out of the upstream Fly-Brain clone, which is otherwise not vendored here — see the root README for credit). `2023_03_23_connectivity_630_final.parquet` and `...completeness_630_final.csv` are git-tracked; `flywire_neuron_annotations_630.tsv` is a gitignored 31 MB cache. |
| `StabilizerNeuron/` | Everything DNg02-specific that's independent of the shared network: `dng02_circuit_neurons.json`, `build_dng02_circuit.py`, and its own `Tests/`. See its README. |

## Environments

Two Python environments are involved here:

| Environment | Needs | Used by |
|---|---|---|
| **brian2** (conda env) | `brian2`, `pandas`, `pyarrow` | `Tests/test_brain_circuit.py`, `StabilizerNeuron/build_dng02_circuit.py`, `StabilizerNeuron/Tests/test_dng02_circuit.py`, and the `fly_brain_controller.py` subprocess itself |
| everything else (e.g. `venv/`) | none of the above | `flybrain_controller.py` and its callers — the brain runs as a subprocess, so this side never links against Brian2 |

`flybrain_controller.py` finds the brian2 env on its own (common conda locations, or `conda run -n brian2`). If it can't, point `FLYBRAIN_PYTHON` at that env's python:

```bash
export FLYBRAIN_PYTHON=/path/to/envs/brian2/bin/python
```

## Tests (`Tests/`)

### `test_brain_circuit.py`: are the neurons firing?

Drives the LC4/LPLC2 → DNp01/DNp03/DNp06 circuit (`fly_brain_controller.py`) directly with fixed looming inputs, no simulator involved. Prints the synapse wiring into each output neuron, then for each stimulus (none, weak, strong, left-only, right-only): spike counts over 1 s, peak escape signal and whether it crosses the ESCAPE threshold (0.6), mean turn signal, and compute time per 20 ms brain step (must stay under ~33 ms for the 30 Hz control loop).

```bash
conda run -n brian2 python NeuralPathways/Tests/test_brain_circuit.py
```

**Expected:** silent with no stimulus. From loom ≈0.25 the Giant Fiber (DNp01) fires and ESCAPE = YES. Left-only loom → negative turn (turn right, away from it); right-only → positive.

DNg02-specific circuit tests (`test_dng02_circuit.py`, `test_optomotor_sign.py`) live in `StabilizerNeuron/Tests/` — see that folder's README.

## Notes

- `main.py` is deliberately **not** wired for optomotor yet — it builds its `flow` dict without `rotation`/`translation` keys, so `optomotor=True` there today would silently get a zero drive request.
- DNg02 steering uses the **opposite** sign convention to DNp06: DNg02 activity tracks wingbeat amplitude in the *contralateral* wing (more right-side DNg02 yaws right), whereas DNp06 steers *away* from looming. `flybrain_controller.py`'s `decide()` subtracts the DNg02 term where it adds the DNp06 one, two lines apart. Both are commented — don't "fix" either.
- The scripts that start the brain overwrite `flybrain_spikes.log` in the repo root, just as `main.py` does. It's gitignored.
