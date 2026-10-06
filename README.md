# Fruity Fly Drone

> **Note:** This project is still being actively developed and is bound to
> change. The repo will be further cleaned up later.

A drone piloted by an actual fly brain. The autonomous controller is a
[Brian2](https://brian2.readthedocs.io/) spiking network built from real
**FlyWire v630** connectome data — not a neural-net metaphor, the measured
synapses of a real fly's escape and flight-stabilization circuits, wired up
to fly a quadcopter. The same controller flies a PyBullet simulator or a
real DJI Tello behind one shared interface, and a second, independent
pipeline (YOLOv8 + a trained ripeness classifier) lets the drone find and
"eat" a banana.

## What it does

- **Escape reflex** — the fly's own Giant Fiber circuit (LC4/LPLC2 →
  DNp01/DNp03/DNp06) watches optical-flow expansion for something looming
  (a swatting hand, an obstacle) and fires a hard dodge, exactly like the
  biological pathway it's copied from.
- **DNg02 flight-motor / stabilizer** *(opt-in, in progress)* — a
  24-neuron population whose recruitment count sets thrust and steering
  from residual optic flow, standing in for a drone's built-in gyroscope.
  On the real Tello it holds the heading in every state of the banana
  behaviour (`fly_tello.py --stabilize`), with an efference copy so it
  doesn't fight the drone's own turns.
- **Banana seek-and-eat** — a two-stage vision model (YOLOv8n to find a
  banana, a trained MobileNetV2 head to judge ripeness) drives a
  scan-approach-feed behavior on the real drone.
- **Safety net** — an obstacle-avoidance layer sits between any controller
  (neural, hand-written reflex FSM, or manual keyboard input) and the
  motors, and can override a command outright if something gets too close.

Currently flying: a real, front-camera-only Tello running the banana-seek
behavior with the escape reflex as a background dodge (swat detection
validated 5/5 on a desk test with zero false positives).

## Architecture

Every layer above "what is actually flying" is written once and shared by
both the simulator and the real drone, through two small interfaces:

```
 Camera → Optical Flow → Controller.decide(flow, state) → SafetyLayer → DroneInterface
                              ^                                              ^
                     FlyBrainController                              TelloDrone (real)
                     or ReflexController                        PyBulletDrone (simulated)
```

- `DroneInterface` (`Drone/drone_interface.py`) — `takeoff`, `land`,
  `move_forward`, `get_state`, ... — implemented by `Drone/tello_drone.py`
  (real Tello) and `Simulator/pybullet_drone.py` (simulated).
- Every controller shares one `decide(flow, state)` contract, so
  `NeuralPathways/flybrain_controller.py` (the real connectome) and
  `Controllers/reflex_controller.py` (a hand-written state-machine
  alternative) are drop-in swaps for each other.

This is why `main.py` (simulator entry point) doesn't change when the
underlying drone does — only the interface implementation does.

## Components

| Folder | What it is |
|---|---|
| [`NeuralPathways/`](NeuralPathways/README.md) | The real fly connectome as a live Brian2 controller — escape circuit + the in-progress DNg02 flight-motor/stabilizer circuit. |
| [`Simulator/`](Simulator/README.md) | PyBullet-backed virtual drone, camera, and obstacle course used for development and headless evaluation. |
| [`Drone/`](Drone/README.md) | The real DJI Tello: flight interface, the live banana-eating flight script, and the real-hardware test suite. |
| [`Controllers/`](Controllers/README.md) | The non-neural controllers (reflex FSM, banana seek, manual), the shared command dict, and the `SafetyLayer` override that sits between any controller and the motors. |
| [`BananaModel/`](BananaModel/README.md) | YOLOv8 + MobileNetV2 pipeline that finds bananas and judges ripeness (93.6% test accuracy). |
| [`Website/`](Website/README.md) | Static whitepaper site (plain HTML/CSS/JS, Vite for dev/build) documenting the project. |
| [`main.py`](main.py) | Interactive simulator entry point — visual PyBullet run with manual/autonomous mode switching and debug overlays. |

## Getting started

### Simulator (no hardware needed)

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python main.py
```

Opens a PyBullet GUI window; starts in autonomous mode (takes off, hovers,
then finds a banana and eats it, with the fly brain's escape and DNg02
circuits running - the `USE_BANANA` / `USE_OPTOMOTOR` / `USE_FLYBRAIN`
flags at the top of `main.py` switch each off). Press `M` to switch to
manual keyboard control, or left-click in the window to throw a test
obstacle at the drone and watch the escape reflex fire. See [`Simulator/README.md`](Simulator/README.md) for the full
test/evaluation suite.

### Real drone

Needs `djitellopy` (included in `requirements.txt`) and a Tello on wifi.
Start with the desk tests — nothing here calls `takeoff()`:

```bash
python Drone/FlightTests/tello_battery_test.py       # connectivity + battery
python Drone/FlightTests/tello_neuron_test.py        # props off: does the brain see a swat?
```

then the flight tests, in order of how much trust they require. See
[`Drone/README.md`](Drone/README.md) for the full progression and safety
notes before flying anything.

### The connectome controller (`USE_FLYBRAIN = True` in `main.py`)

The Brian2 network runs in its own conda environment (`brian2`,
`pandas`, `pyarrow`) as a subprocess, so the main `venv` never needs to
link against it:

```bash
conda create -n brian2 python=3.x brian2 pandas pyarrow
export FLYBRAIN_PYTHON=/path/to/envs/brian2/bin/python   # if not auto-detected
```

See [`NeuralPathways/README.md`](NeuralPathways/README.md) for the two
environments involved and how they talk to each other.

## Credits

The escape and flight-motor circuits are built from real
[FlyWire](https://flywire.ai/) v630 connectome data (`NeuralPathways/Data/`)
— synapse-level connectivity and neuron annotations for an actual fly
brain, reconstructed by the FlyWire community. The connectome files here
were rescued out of an earlier, unvendored clone of
[philshiu/Drosophila_brain_model](https://github.com/philshiu/Drosophila_brain_model)
(MIT, © 2023 Philip Shiu and Nico Spiller) — the official code for Shiu et
al. below — which this project grew out of. The real-time Brian2 worker
(`NeuralPathways/connectome_worker.py`) reuses that model's LIF constants
and Poisson-drive scheme from its `model.py`, scoped down from the whole
brain to the escape and flight-motor circuits.

## References

- **Shiu, P. K., Sterne, G. R., Spiller, N., et al.** (2023). *A leaky
  integrate-and-fire computational model based on the connectome of the
  entire adult Drosophila brain reveals insights into sensorimotor
  processing.* bioRxiv.
  [doi:10.1101/2023.05.02.539144](https://www.biorxiv.org/content/10.1101/2023.05.02.539144v1)
  — the whole-brain LIF connectome model this project's Brian2 controller
  is based on. Later published as *A Drosophila computational brain model
  reveals sensorimotor processing*, Nature 634, 210–219 (2024).
- **[philshiu/Drosophila_brain_model](https://github.com/philshiu/Drosophila_brain_model)**
  — the code accompanying Shiu et al.; the origin of the connectome data
  files and the LIF model the Brian2 subprocess is adapted from.
- **[blendi-remade/fly-brain-minecraft](https://github.com/blendi-remade/fly-brain-minecraft)**
  — a Minecraft mod that runs a simulated fly nervous system inside a mob.
  The original inspiration for putting a fly brain in control of something,
  and an occasional reference while building this.
