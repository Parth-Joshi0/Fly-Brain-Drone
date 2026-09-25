"""
Real-time surrogate for the Fly-Brain connectome model (Fly-Brain/model.py),
scoped down to one real circuit instead of the whole ~130k-neuron brain:

    LC4 + LPLC2                       looming-sensitive visual projection
    (268 neurons, both eyes)          neurons - the fly's "something is
                                       approaching fast" detectors.
        |
        | 7,973 real FlyWire synapses (this repo's own v630 connectome -
        | see looming_circuit_neurons.json for exactly which 274 neurons
        | and Fly-Brain/2023_03_23_connectivity_630_final.parquet for the
        | synapse weights actually used)
        v
    DNp01, DNp03, DNp06 (6 neurons)   descending neurons that carry the
                                       looming response out of the brain.

Why not just run Fly-Brain/model.py's full connectome? That's built for
offline experiments (run_exp() batches 30 x 1-second trials across all
~130k neurons and takes minutes) - nothing about it is meant to answer
"what should the drone do about the last 20ms" fast enough for a live
30Hz control loop. This module builds ONE small Brian2 network *once*,
using the real synapses among those 274 neurons (same LIF equations and
constants as Fly-Brain/model.py's default_params), and then just keeps
advancing that live network in short windows, one per step() call -
lightweight enough to run in real time, but still a real (if narrow)
slice of the actual connectome rather than a hand-built toy.

Why these specific neurons:
    LC4, LPLC2  - canonical looming-sensitive visual projection neurons;
                  known to feed the giant fiber escape circuit.
    DNp01       - the "Giant Fiber", the fly's escape-jump command neuron.
    DNp06       - descending neuron for evasive FLIGHT TURNS driven by
                  LPLC2 input (Current Biology 2022, "A visuomotor circuit
                  for evasive flight turns in Drosophila").
    DNp03       - receives direct LPLC1/LC4 looming input on its dendrites
                  and synapses onto flight steering motor neurons in the
                  VNC.
These are documented cell-type names, cross-referenced from
flyconnectome/flywire_annotations against this repo's own v630 completeness
list - see looming_circuit_neurons.json's "source" field.

Wiring check (done against the real synapse data, not assumed): every
direct LC4/LPLC2 -> DN synapse in this circuit is ipsilateral (right-eye
looming drives right-side DNp01/03/06, left drives left) - see the "source"
comment in looming_circuit_neurons.json. Combined with DNp06's documented
role in turning AWAY from a looming stimulus, that means: stronger RIGHT
DNp06 activity should produce a LEFT turn. yaw's sign below is built to
match that, and also matches this project's own yaw convention (positive
yaw_rate = turn left, see controllers/safety_layer.py's _steer_toward).

Contract - this is the only thing controllers/flybrain_controller.py (the
main-loop-facing adapter) needs to know about:

    step(loom_left, loom_right) -> {"yaw": float, "forward": float, "escape": float}

    loom_left / loom_right: 0..1, how strongly something is looming in the
    left/right visual field over the upcoming control tick (0 = nothing,
    1 = as close/fast as the model was tuned for).

    yaw:     -1..1, signed turn urgency (positive = turn left, matching
             this project's convention).
    forward: 0..1, how much forward drive the looming response leaves
             room for (1 = no looming response at all, drops as the
             circuit gets more active - it decelerates before it turns,
             not just after).
    escape:  0..1, Giant Fiber (DNp01) drive - this project's caller
             treats a high value as "hard stop / evasive", not a literal
             jump takeoff.

Run standalone (under the `brian2` conda env - `conda env create -f
Fly-Brain/environment.yml`, or whatever env has brian2+pandas+pyarrow
installed) as a persistent stdin/stdout JSON-lines server; see __main__
below and controllers/flybrain_controller.py for how the main venv (which
does NOT have brian2 installed, deliberately - the PyBullet side of this
project has no business linking against a spiking neural simulator) talks
to it as a subprocess.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from brian2 import NeuronGroup, Synapses, PoissonGroup, SpikeMonitor, Network
from brian2 import mV, ms, Hz, second

HERE = Path(__file__).resolve().parent
NEURON_IDS_PATH = HERE / "looming_circuit_neurons.json"
PATH_CON = HERE / "Fly-Brain" / "2023_03_23_connectivity_630_final.parquet"

# Same LIF constants as Fly-Brain/model.py's default_params - see that
# file for the citations (Kakaria & de Bivort 2017, Jürgensen et al,
# Lazar et al, Paul et al). Duplicated rather than imported because
# model.py also pulls in joblib/the full run_exp() batch machinery this
# module has no use for.
V_0 = -52 * mV
V_RST = -52 * mV
V_TH = -45 * mV
T_MBR = 20 * ms
TAU = 5 * ms
T_RFC = 2.2 * ms
T_DLY = 1.8 * ms
W_SYN = 0.275 * mV
F_POI = 250          # Poisson synapse scaling factor - model.py's poi()

# loom=1.0 -> this many Hz of (individually supra-threshold, per model.py's
# own poi() weighting) Poisson drive onto EACH of the 268 input neurons.
# NOT model.py's r_poi default (150-250Hz): at that strength, with 268
# input neurons and 7,973 real synapses converging on just 6 output
# neurons, even loom=0.05 (from ordinary, obstacle-free cruising flow -
# see LOOM_FLOW_MAX below) already saturated escape past 0.4. 30Hz is
# picked empirically (swept 10/20/30/50Hz against controllers/
# safety_layer.py's own flow calibration notes - cruising flow ~0.1-0.8,
# AVOID/DANGER territory ~1.2+, CRITICAL ~2.2+) so the circuit stays
# quiet during normal cruising and only meaningfully engages once flow
# actually reaches that DANGER-ish range.
MAX_POI_RATE = 30 * Hz

STEP_DT = 20 * ms     # brain time simulated per step() call
RATE_SMOOTHING = 0.25  # EMA factor for the output spike-rate readout, see
                        # its use in __init__/step() below

# Saturation points for turning smoothed output spike-rate differences/
# sums into the 0..1 / -1..1 ranges the caller expects. Tuned by hand
# together with MAX_POI_RATE above against a loom_left/loom_right sweep,
# not derived from a calibration experiment - tunables, not physiological
# constants.
ESCAPE_SAT_HZ = 42.0
YAW_SAT_HZ = 60.0
FORWARD_SAT_HZ = 30.0

EQS = '''
dv/dt = (V_0 - v + g) / T_MBR : volt (unless refractory)
dg/dt = -g / TAU               : volt (unless refractory)
rfc                             : second
'''
EQ_TH = 'v > V_TH'
EQ_RST = 'v = V_RST; g = 0*mV'


def _load_circuit():
    with open(NEURON_IDS_PATH) as f:
        circuit = json.load(f)

    inputs = sorted(circuit["input_neurons"], key=lambda n: n["side"])   # 'left' block, then 'right' block
    outputs = circuit["output_neurons"]

    n_left = sum(1 for n in inputs if n["side"] == "left")
    n_right = len(inputs) - n_left
    all_neurons = inputs + outputs
    root_ids = [n["root_id"] for n in all_neurons]
    root_to_local = {rid: i for i, rid in enumerate(root_ids)}

    return {
        "inputs": inputs,
        "outputs": outputs,
        "n_left": n_left,
        "n_right": n_right,
        "n_total": len(all_neurons),
        "root_to_local": root_to_local,
    }


def _load_synapses(root_to_local):
    con = pd.read_parquet(
        PATH_CON,
        columns=["Presynaptic_ID", "Postsynaptic_ID", "Excitatory x Connectivity"],
    )
    pool_ids = set(root_to_local)
    sub = con[con["Presynaptic_ID"].isin(pool_ids) & con["Postsynaptic_ID"].isin(pool_ids)]
    i_pre = sub["Presynaptic_ID"].map(root_to_local).to_numpy()
    i_post = sub["Postsynaptic_ID"].map(root_to_local).to_numpy()
    weights = sub["Excitatory x Connectivity"].to_numpy()
    return i_pre, i_post, weights


class FlyBrainController:
    def __init__(self, step_dt=STEP_DT):
        self.step_dt = step_dt
        circuit = _load_circuit()
        self._outputs = circuit["outputs"]
        n_left, n_right = circuit["n_left"], circuit["n_right"]
        n_in = n_left + n_right
        n_total = circuit["n_total"]

        neu = NeuronGroup(
            N=n_total, model=EQS, method="linear",
            threshold=EQ_TH, reset=EQ_RST, refractory="rfc", name="looming_circuit",
        )
        neu.v = V_0
        neu.g = 0 * mV
        neu.rfc = T_RFC
        neu.rfc[:n_in] = 0 * ms   # matches model.py's poi(): Poisson-driven neurons get no refractory period

        syn = Synapses(neu, neu, "w : volt", on_pre="g += w", delay=T_DLY, name="looming_circuit_synapses")
        i_pre, i_post, weights = _load_synapses(circuit["root_to_local"])
        syn.connect(i=i_pre, j=i_post)
        syn.w = weights * W_SYN

        # One Poisson source per input neuron (independent spike trains,
        # like model.py's poi() gives each stimulated neuron), one-to-one
        # connected - PoissonGroup.rates is a plain settable state
        # variable, unlike PoissonInput.rate (fixed at construction), so
        # this is what lets step() drive a *different* rate each call on
        # the same live network. Left/right blocks are contiguous by
        # construction (_load_circuit sorts 'left' before 'right'), so
        # updating one hemifield's rate is just a slice assignment.
        self._n_left = n_left
        self._n_in = n_in
        self._poisson = PoissonGroup(n_in, rates=0 * Hz)
        poisson_syn = Synapses(self._poisson, neu, model="w : volt", on_pre="v += w", name="poisson_drive")
        poisson_syn.connect(j="i")
        poisson_syn.w = W_SYN * F_POI

        spk_mon = SpikeMonitor(neu[n_in:])  # only watching the 6 output (DN) neurons
        self._spk_mon = spk_mon
        self._prev_counts = np.zeros(n_total - n_in, dtype=int)
        # Each of the 6 output neurons only ever fires 0-4 times in one
        # 20ms window (refractory period alone caps it well under 500Hz,
        # and these are single real neurons, not a population average) -
        # read literally, that's a jumpy 0/50/100/150Hz staircase.
        # Smoothing the rate estimate (same fix, same reasoning, as
        # controllers/safety_layer.py's optic-flow smoothing) turns that
        # into a usable continuous signal instead of the drone's yaw
        # visibly stepping between a handful of discrete values.
        self._smoothed_rate_hz = np.zeros(n_total - n_in)

        self.net = Network(neu, syn, self._poisson, poisson_syn, spk_mon)

    def reset(self):
        """Clears accumulated membrane potential/spike state without
        rebuilding the network (synapses/connectivity are unaffected -
        those never change), for a fresh takeoff/reset in the sim."""
        self.net["looming_circuit"].v = V_0
        self.net["looming_circuit"].g = 0 * mV
        # spk_mon.count is a cumulative, read-only Brian2 array - rather
        # than clearing it, just re-baseline what "since last read" means
        # so the next step()'s delta is measured from here.
        self._prev_counts = np.asarray(self._spk_mon.count).copy()
        self._smoothed_rate_hz[:] = 0
        self._poisson.rates = 0 * Hz

    def step(self, loom_left, loom_right):
        loom_left = min(1.0, max(0.0, loom_left))
        loom_right = min(1.0, max(0.0, loom_right))
        self._poisson.rates[:self._n_left] = loom_left * MAX_POI_RATE
        self._poisson.rates[self._n_left:self._n_in] = loom_right * MAX_POI_RATE

        self.net.run(self.step_dt)

        counts = np.asarray(self._spk_mon.count)
        delta = counts - self._prev_counts
        self._prev_counts = counts
        instant_rate_hz = delta / (self.step_dt / second)
        a = RATE_SMOOTHING
        self._smoothed_rate_hz = a * instant_rate_hz + (1 - a) * self._smoothed_rate_hz
        rate_hz = self._smoothed_rate_hz

        rate_by_id = {n["root_id"]: rate_hz[i] for i, n in enumerate(self._outputs)}

        def rate_of(cell_type, side):
            for n in self._outputs:
                if n["cell_type"] == cell_type and n["side"] == side:
                    return rate_by_id[n["root_id"]]
            return 0.0

        escape_rate = rate_of("DNp01", "left") + rate_of("DNp01", "right")
        escape = min(1.0, escape_rate / (2 * ESCAPE_SAT_HZ))

        yaw_rate = rate_of("DNp06", "right") - rate_of("DNp06", "left")
        yaw = max(-1.0, min(1.0, yaw_rate / YAW_SAT_HZ))

        forward_drive = rate_of("DNp03", "left") + rate_of("DNp03", "right") \
            + rate_of("DNp06", "left") + rate_of("DNp06", "right")
        forward = max(0.0, 1.0 - forward_drive / (2 * FORWARD_SAT_HZ))

        # Raw (unsmoothed) spike count for each of the 6 output DN neurons
        # in this exact 20ms window - literally "did it fire, how many
        # times" - as opposed to rate_hz above, which is an EMA-smoothed
        # rate meant for smooth flight control, not for showing someone
        # the actual firing events.
        spike_counts = {
            f"{n['cell_type']}_{n['side']}": int(delta[i])
            for i, n in enumerate(self._outputs)
        }

        return {
            "yaw": float(yaw), "forward": float(forward), "escape": float(escape),
            "spike_counts": spike_counts,
        }


def _serve_stdio():
    """JSON-lines server: one {"loom_left": .., "loom_right": ..} request
    per line on stdin, one step() result per line on stdout. Kept dumb on
    purpose - controllers/flybrain_controller.py owns retries/timeouts."""
    controller = FlyBrainController()
    print(json.dumps({"ready": True}), flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            if req.get("reset"):
                controller.reset()
                result = {"yaw": 0.0, "forward": 1.0, "escape": 0.0}
            else:
                result = controller.step(req["loom_left"], req["loom_right"])
        except Exception as exc:  # keep the server alive, report the error inline
            result = {"error": str(exc)}
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    _serve_stdio()
