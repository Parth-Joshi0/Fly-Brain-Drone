"""
Real-time surrogate for the upstream Fly-Brain connectome model (model.py in
the original FlyWire Fly-Brain project, credited in the repo README; not
vendored here beyond the two data files under NeuralPathways/Data/),
scoped down to real circuits instead of the whole ~130k-neuron brain. Two of
them now share one network:

  1. LOOMING / ESCAPE - "something is about to hit me"

    LC4 + LPLC2                       looming-sensitive visual projection
    (268 neurons, both eyes)          neurons - the fly's "something is
                                       approaching fast" detectors.
        |
        | 7,973 real FlyWire synapses (this repo's own v630 connectome -
        | see looming_circuit_neurons.json for exactly which 274 neurons
        | and data/2023_03_23_connectivity_630_final.parquet for the
        | synapse weights actually used)
        v
    DNp01, DNp03, DNp06 (6 neurons)   descending neurons that carry the
                                       looming response out of the brain.

  2. FLIGHT MOTOR - "how hard, and how symmetrically, to fly"

    120 central drive neurons          posterior slope / inferior bridge /
    (PS*, IB*, CL*, LAL*, AN_multi*)   clamp / LAL / ascending-from-VNC
        |                              cells - DNg02's real presynaptic
        | ~2,900 real FlyWire synapses  partners in this connectome.
        v
    DNg02 x 24                         descending neurons that set wingbeat
                                       amplitude - a POPULATION, not a pair.

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
    DNg02       - at least 15 near-identical cell PAIRS that set wingbeat
                  amplitude. Namiki et al. 2022 ("A population of
                  descending neurons that regulates the flight motor of
                  Drosophila", Current Biology 32(5):1189-1196) showed that
                  recruiting more pairs raises wingbeat amplitude roughly
                  linearly (~2.77 deg/pair), that they respond to wide-field
                  visual motion in flight, and that the two sides act
                  independently - symmetric for thrust, opponent for yaw.
These are documented cell-type names, cross-referenced from
flyconnectome/flywire_annotations against this repo's own v630 completeness
list - see looming_circuit_neurons.json's and StabilizerNeuron/dng02_circuit_neurons.json's
"source" fields.

Why DNg02's readout is different in kind from the others: DNp01/03/06 are one
neuron per side, so all you can read out is one cell's firing rate, and a
single real neuron only fires 0-4 times per 20ms window - a jumpy
0/50/100/150Hz staircase that has to be EMA-smoothed into something a drone
can fly on (see RATE_SMOOTHING). DNg02 is a population with a strongly graded
input gradient (375 down to 0, see StabilizerNeuron/dng02_circuit_neurons.json's recruitment
ladder), so its cells recruit in roughly weight order as drive rises, and the
natural readout is simply HOW MANY fired - an analog signal that needs no
smoothing trick to exist. That's the population code Namiki et al. describe.

The one modelling assumption in this file: DNg02 has essentially no visual
input in this connectome (137 of 12,848 incoming synapses come from visual
projection neurons, NONE from the HS/VS/H2 wide-field tangential cells that
carry optic flow, and - checked directly - NONE AT ALL from the 268 LC4/LPLC2
cells above; the only synapses the looming circuit makes into DNg02 are 7 from
DNp01/03/06 themselves). Its input is central. So the drive pool driven by optic flow below is central cells that
the connectome does not say carry a motion signal. That pathway is a
published open question, not something this file is glossing over - Schnell
2026 (Trends in Neurosciences 49:98-110) states that DNg02's "connectivity
with LPTCs or other visually responsive neurons remains to be established".
Everything downstream of that injection point is measured: which driver
excites or inhibits which DNg02, and how strongly, all comes from the
parquet. See StabilizerNeuron/build_dng02_circuit.py.

Why not just run the upstream Fly-Brain project's full connectome? That's built for
offline experiments (run_exp() batches 30 x 1-second trials across all
~130k neurons and takes minutes) - nothing about it is meant to answer
"what should the drone do about the last 20ms" fast enough for a live
30Hz control loop. This module builds ONE small Brian2 network *once*,
using the real synapses among those 418 neurons (same LIF equations and
constants as that upstream model's default_params), and then just keeps
advancing that live network in short windows, one per step() call -
lightweight enough to run in real time, but still a real (if narrow)
slice of the actual connectome rather than a hand-built toy.

Both circuits live in ONE NeuronGroup, ONE Synapses, ONE PoissonGroup and ONE
Network on purpose, and the reason is measured rather than assumed. Brian2's
cost is dominated by per-OBJECT overhead, not by neuron or synapse count: on
this machine, adding the DNg02 circuit as a second PoissonGroup + Synapses pair
costs 20.33 ms/step, while folding it into the existing PoissonGroup as extra
indices costs 17.40 ms - a 2.9 ms difference for identical neurons and synapses,
against a 33 ms budget for the 30Hz control loop. The extra 144 neurons and
~2,900 synapses themselves are nearly free. Hence the single resized group and
the explicit index arrays below rather than the tidier connect(j="i").

Merging also means the real synapses BETWEEN the two circuits are included
rather than discarded: 52 from the looming circuit into the DNg02 pool, and 131
back the other way. That second number needs care, so it was checked cell by
cell - every one of those return synapses onto DNp01 is INHIBITORY (4 drivers,
net weight -22), and no excitatory driver touches the Giant Fiber at all. So
driving this pool cannot make DNp01 fire; if anything it raises the escape
threshold slightly. That is real connectome wiring rather than a bug, but it
does mean escape is not perfectly independent of the flight-motor path, which
is why NeuralPathways/StabilizerNeuron/Tests/test_dng02_circuit.py sweeps loom x drive together instead of
just checking each alone.

The DNg02 half is built only when asked for (with_dng02=True, which
NeuralPathways/flybrain_controller.py passes as "--dng02" only when its optomotor
flag is on). Without it this module constructs literally the 274-neuron network
it always did, with the same object graph and therefore the same Poisson RNG
stream - so the escape path is not merely statistically similar to the
pre-DNg02 behaviour, it is identical. That matters because the escape flight
test is being tuned independently and must not move underneath it.

Wiring check (done against the real synapse data, not assumed): every
direct LC4/LPLC2 -> DN synapse in this circuit is ipsilateral (right-eye
looming drives right-side DNp01/03/06, left drives left) - see the "source"
comment in looming_circuit_neurons.json. Combined with DNp06's documented
role in turning AWAY from a looming stimulus, that means: stronger RIGHT
DNp06 activity should produce a LEFT turn. yaw's sign below is built to
match that, and also matches this project's own yaw convention (positive
yaw_rate = turn left, see safety_layer.py's _steer_toward).

NOTE that DNg02's steer sign below is the OPPOSITE convention, for a real
biological reason - see the comment on the steer readout in step().

Contract - this is the only thing NeuralPathways/flybrain_controller.py (the
main-loop-facing adapter) needs to know about:

    step(loom_left, loom_right, drive_common, drive_left, drive_right)
        -> {"yaw": float, "forward": float, "escape": float,
            "spike_counts": {...}, "dng02": {...}}

    loom_left / loom_right: 0..1, how strongly something is looming in the
    left/right visual field over the upcoming control tick (0 = nothing,
    1 = as close/fast as the model was tuned for).

    drive_common: -1..1, requested change in OVERALL DNg02 recruitment -
    the symmetric/thrust channel. 0 leaves the flight-motor path idle.
    drive_left / drive_right: -1..1 each, requested change in that side's
    recruitment on top of drive_common - the opponent/steering channel
    (pass drive_left = -drive_right for a pure yaw request).

    yaw:     -1..1, signed turn urgency (positive = turn left, matching
             this project's convention).
    forward: 0..1, how much forward drive the looming response leaves
             room for (1 = no looming response at all, drops as the
             circuit gets more active - it decelerates before it turns,
             not just after).
    escape:  0..1, Giant Fiber (DNp01) drive - this project's caller
             treats a high value as "hard stop / evasive", not a literal
             jump takeoff.
    dng02:   {"n_left": int, "n_right": int, "thrust": 0..1,
              "steer": -1..1, "counts": {label: int}} - the population
             readout. See step() for what each one means.

Run standalone (under a `brian2` conda env - any env with
brian2+pandas+pyarrow installed) as a persistent stdin/stdout JSON-lines
server; see __main__ below and NeuralPathways/flybrain_controller.py for
how the main venv (which
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
DNG02_IDS_PATH = HERE / "StabilizerNeuron" / "dng02_circuit_neurons.json"
PATH_CON = HERE / "Data" / "2023_03_23_connectivity_630_final.parquet"

# Same LIF constants as the upstream Fly-Brain model.py's default_params - see that
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
# picked empirically (swept 10/20/30/50Hz against
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

# --- DNg02 flight-motor path ---
#
# Peak Poisson rate onto a drive-pool neuron, same role MAX_POI_RATE plays
# for the looming inputs. Started at the same 30Hz;
# NeuralPathways/StabilizerNeuron/Tests/test_dng02_circuit.py's recruitment sweep is what this gets tuned against
# (the target is a graded curve across the 24 cells, not an all-or-nothing
# step - if the whole population switches on at once there is no population
# code left to read).
# 150Hz, swept with exactly the sweep NeuralPathways/StabilizerNeuron/Tests/test_dng02_circuit.py runs. Mean
# cells recruited (of 24) at requests 0/0.2/0.4/0.6/0.8/1.0:
#     30Hz  ->  0.1  0.3  0.9  2.1  3.8  5.4    too compressed, 22% of the range
#     60Hz  ->  0.4  0.6  2.3  4.8  7.7  8.9
#    150Hz  ->  0.9  2.0  5.8 10.3 12.2 13.0    graded across the widest span
#    250Hz  ->  1.3  3.4  8.7 12.5 13.5 14.2    saturates by 0.6, top end wasted
#    400Hz  ->  1.7  4.0 10.5 13.6 14.1 14.9
# 150Hz is the pick: recruitment stays monotone and spread out over the whole
# request range instead of hitting the ceiling early, which is the whole point
# of reading a population count rather than a rate.
MAX_DRIVE_RATE = 150 * Hz

# Resting fraction of MAX_DRIVE_RATE that drive-pool neurons sit at when the
# request is zero but the path IS engaged. This is not decoration: 73 of the
# 120 drivers are inhibitory, so "recruit more DNg02" has to be expressible
# as those cells firing LESS, which is impossible from a floor of zero. It is
# also what lets the model reproduce Namiki et al.'s actual observation -
# rightward motion raised the right DNg02 cells and simultaneously LOWERED
# the left ones, a decrease that no amount of added excitation can produce.
# A request of exactly 0.0 on every channel bypasses this entirely and holds
# the pool at 0Hz, which is what keeps the escape path untouched by default.
DRIVE_REST_FRACTION = 0.5
DRIVE_GAIN = 0.5      # how far a +-1.0 request swings a driver off its rest rate

DNG02_SMOOTHING = 0.25  # EMA on thrust/steer, same reasoning as RATE_SMOOTHING.
                        # The raw per-cell counts stay unsmoothed for display.

# How many recruited cells count as "full thrust". NOT len(dng02): the
# population never all fires at once - 13.1 of 24 at the strongest request, and
# the bottom 8 cells of the ladder (excitatory weight 11 and below) never
# recruit at all, so the effective population is ~16 cells. Dividing by 24
# would cap thrust at ~0.54 and throw away half the usable range. Measured by
# NeuralPathways/StabilizerNeuron/Tests/test_dng02_circuit.py, which re-checks it rather than trusting it.
DNG02_RECRUIT_SAT = 13.1

# Measured recruitment curves per side, used to undo the population's built-in
# left/right asymmetry before taking a difference. The right-side DNg02 cells
# receive substantially more input than the left (1,857 vs 1,119 total
# excitatory weight from the drive pool - StabilizerNeuron/build_dng02_circuit.py prints the
# ratio), so at ANY symmetric request the right side recruits more cells.
#
# A single per-side scale factor does not fix this, because the two sides'
# curves differ in shape rather than just gain - scaling leaves a standing bias
# at intermediate drive, which on a drone reads as a constant slow yaw. So each
# side's count is mapped back through its OWN measured curve onto the common
# 0..1 request axis, and the difference is taken there; symmetric drive then
# gives zero at every level, not just at the one point a scale factor was fit
# to. Regenerated as a paste-ready tuple by NeuralPathways/StabilizerNeuron/Tests/test_dng02_circuit.py.
DNG02_DRIVE_LEVELS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
DNG02_LEFT_RECRUITMENT = (0.3, 0.9, 1.9, 3.9, 5.4, 5.8)
DNG02_RIGHT_RECRUITMENT = (0.6, 1.4, 3.9, 5.7, 6.7, 7.3)

EQS = '''
dv/dt = (V_0 - v + g) / T_MBR : volt (unless refractory)
dg/dt = -g / TAU               : volt (unless refractory)
rfc                             : second
'''
EQ_TH = 'v > V_TH'
EQ_RST = 'v = V_RST; g = 0*mV'


def _load_circuit(with_dng02=False):
    """Builds the neuron index. Layout, which several slices below depend on:

        [0, n_left)                     LC4/LPLC2, left eye
        [n_left, n_in)                  LC4/LPLC2, right eye
        [n_in, n_in + 6)                DNp01/03/06 x left/right
      --- only when with_dng02 ---
        [dng02_at, dng02_at + 24)       DNg02
        [drive_at, n_total)             DNg02's drive pool

    The looming block stays first and its indices never move, so with
    with_dng02=False this returns exactly the 274-neuron circuit this module
    started as, and everything downstream of it behaves identically.
    """
    with open(NEURON_IDS_PATH) as f:
        circuit = json.load(f)

    inputs = sorted(circuit["input_neurons"], key=lambda n: n["side"])   # 'left' block, then 'right' block
    outputs = circuit["output_neurons"]

    dng02, drivers = [], []
    if with_dng02:
        with open(DNG02_IDS_PATH) as f:
            dng02_circuit = json.load(f)
        # Highest excitatory input weight first, so the local index order IS
        # the predicted recruitment order - which makes a recruitment readout
        # or a HUD bar just a prefix of this list.
        dng02 = sorted(dng02_circuit["output_neurons"], key=lambda n: -n["exc_weight"])
        drivers = dng02_circuit["drive_neurons"]

        # DNg02 subtypes repeat (6x DNg02_a, 7x DNg02_b, ...), so cell_type
        # plus side is not unique. Number them within each type+side so every
        # cell has a stable label to log and display under.
        seen = {}
        for n in dng02:
            key = (n["cell_type"], n["side"])
            seen[key] = seen.get(key, 0) + 1
            n["label"] = f"{n['cell_type']}_{n['side']}_{seen[key]}"

    n_left = sum(1 for n in inputs if n["side"] == "left")
    n_right = len(inputs) - n_left
    all_neurons = inputs + outputs + dng02 + drivers
    root_ids = [n["root_id"] for n in all_neurons]
    root_to_local = {rid: i for i, rid in enumerate(root_ids)}

    n_in = n_left + n_right
    return {
        "inputs": inputs,
        "outputs": outputs,
        "dng02": dng02,
        "drivers": drivers,
        "n_left": n_left,
        "n_right": n_right,
        "n_in": n_in,
        "dng02_at": n_in + len(outputs),
        "drive_at": n_in + len(outputs) + len(dng02),
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
    def __init__(self, step_dt=STEP_DT, with_dng02=False):
        self.step_dt = step_dt
        self.with_dng02 = with_dng02
        circuit = _load_circuit(with_dng02)
        self._outputs = circuit["outputs"]
        self._dng02 = circuit["dng02"]
        self._drivers = circuit["drivers"]
        n_left, n_right = circuit["n_left"], circuit["n_right"]
        n_in = circuit["n_in"]
        n_total = circuit["n_total"]
        dng02_at, drive_at = circuit["dng02_at"], circuit["drive_at"]
        n_drive = n_total - drive_at

        neu = NeuronGroup(
            N=n_total, model=EQS, method="linear",
            threshold=EQ_TH, reset=EQ_RST, refractory="rfc", name="looming_circuit",
        )
        neu.v = V_0
        neu.g = 0 * mV
        neu.rfc = T_RFC
        neu.rfc[:n_in] = 0 * ms   # matches model.py's poi(): Poisson-driven neurons get no refractory period
        if with_dng02:
            neu.rfc[drive_at:] = 0 * ms   # ...and so do the DNg02 drive-pool neurons, same reason

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
        # ONE Poisson group covering the looming inputs and (when present) the
        # DNg02 drive pool, because Brian2's cost is per-object: a second
        # group + Synapses pair measured 2.9ms/step more than widening this one
        # (20.33 vs 17.40ms) for exactly the same neurons and synapses. The two
        # driven blocks are not contiguous - the 6 DNs and 24 DNg02 sit between
        # them - so the connect has to be an explicit index pair rather than
        # the tidier connect(j="i").
        self._n_left = n_left
        self._n_in = n_in
        self._drive_at = drive_at
        self._n_drive = n_drive
        self._poisson = PoissonGroup(n_in + n_drive, rates=0 * Hz)
        poisson_syn = Synapses(self._poisson, neu, model="w : volt", on_pre="v += w", name="poisson_drive")
        if n_drive:
            poisson_syn.connect(
                i=np.arange(n_in + n_drive),
                j=np.concatenate([np.arange(n_in), np.arange(drive_at, n_total)]),
            )
        else:
            # Kept as the original one-to-one form in the no-DNg02 case so the
            # 274-neuron network is the same object graph it has always been.
            poisson_syn.connect(j="i")
        poisson_syn.w = W_SYN * F_POI

        # Per-driver constants for the drive rule in step(), all straight from
        # the connectome (StabilizerNeuron/build_dng02_circuit.py measured them): the sign of
        # its net effect on DNg02, and how its influence splits across the two
        # halves of the population.
        self._drive_sign = np.array([d["sign"] for d in self._drivers], dtype=float)
        w_l = np.array([d["w_left"] for d in self._drivers], dtype=float)
        w_r = np.array([d["w_right"] for d in self._drivers], dtype=float)
        total = np.maximum(w_l + w_r, 1e-9)
        self._drive_frac_left = w_l / total
        self._drive_frac_right = w_r / total

        # record=False: only the cumulative per-neuron .count is needed, and
        # the two readout groups (the 6 DNs, the 24 DNg02) are not contiguous,
        # so watching the whole group and indexing absolutely is both cheaper
        # and simpler than two subgroup monitors.
        spk_mon = SpikeMonitor(neu, record=False)
        self._spk_mon = spk_mon
        self._prev_counts = np.zeros(n_total, dtype=int)
        self._dn_idx = np.arange(n_in, n_in + len(self._outputs))
        self._dng02_idx = np.arange(dng02_at, dng02_at + len(self._dng02))
        self._dng_left_idx = np.array(
            [dng02_at + i for i, n in enumerate(self._dng02) if n["side"] == "left"])
        self._dng_right_idx = np.array(
            [dng02_at + i for i, n in enumerate(self._dng02) if n["side"] == "right"])

        # Each of the 6 output neurons only ever fires 0-4 times in one
        # 20ms window (refractory period alone caps it well under 500Hz,
        # and these are single real neurons, not a population average) -
        # read literally, that's a jumpy 0/50/100/150Hz staircase.
        # Smoothing the rate estimate (same fix, same reasoning, as
        # safety_layer.py's optic-flow smoothing) turns that
        # into a usable continuous signal instead of the drone's yaw
        # visibly stepping between a handful of discrete values.
        self._smoothed_rate_hz = np.zeros(len(self._outputs))
        self._smoothed_n_left = 0.0
        self._smoothed_n_right = 0.0

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
        self._smoothed_n_left = 0.0
        self._smoothed_n_right = 0.0
        self._poisson.rates = 0 * Hz

    def step(self, loom_left=0.0, loom_right=0.0,
             drive_common=0.0, drive_left=0.0, drive_right=0.0):
        loom_left = min(1.0, max(0.0, loom_left))
        loom_right = min(1.0, max(0.0, loom_right))
        self._poisson.rates[:self._n_left] = loom_left * MAX_POI_RATE
        self._poisson.rates[self._n_left:self._n_in] = loom_right * MAX_POI_RATE

        # Requested change in each side's recruitment. All zero means the
        # caller isn't using the flight-motor path at all, and the pool is held
        # silent - which matters because 131 real synapses run from this pool
        # back into the looming circuit (see the module docstring).
        if self._n_drive:
            if drive_common == 0.0 and drive_left == 0.0 and drive_right == 0.0:
                self._poisson.rates[self._n_in:] = 0 * Hz
            else:
                req_left = min(1.0, max(-1.0, drive_common + drive_left))
                req_right = min(1.0, max(-1.0, drive_common + drive_right))
                # Each driver gets the request weighted by how its own influence
                # splits across the two halves of the population, then flipped
                # if it's inhibitory - an inhibitory cell has to fire LESS for
                # its targets to recruit MORE.
                req = self._drive_frac_left * req_left + self._drive_frac_right * req_right
                # Named drive_rates, not rates: Brian2 resolves names against
                # the caller's locals too, and a local called `rates` shadows
                # the PoissonGroup's own state variable and spams a warning.
                drive_rates = np.clip(
                    DRIVE_REST_FRACTION + DRIVE_GAIN * (self._drive_sign * req), 0.0, 1.0)
                self._poisson.rates[self._n_in:] = drive_rates * MAX_DRIVE_RATE

        self.net.run(self.step_dt)

        counts = np.asarray(self._spk_mon.count)
        delta = counts - self._prev_counts
        self._prev_counts = counts
        dn_delta = delta[self._dn_idx]
        instant_rate_hz = dn_delta / (self.step_dt / second)
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
            f"{n['cell_type']}_{n['side']}": int(dn_delta[i])
            for i, n in enumerate(self._outputs)
        }

        return {
            "yaw": float(yaw), "forward": float(forward), "escape": float(escape),
            "spike_counts": spike_counts,
            "dng02": self._read_dng02(delta),
        }

    def _read_dng02(self, delta):
        """The population readout: how MANY DNg02 fired, not how fast one did.

        n_left/n_right are recruitment counts - the quantity Namiki et al.
        2022 found sets wingbeat amplitude, roughly linearly at ~2.77 deg per
        pair recruited, which is why thrust below is a plain linear fraction
        of the population rather than anything fitted.

        steer's sign is the OPPOSITE convention to DNp06's yaw above, and the
        reason is biological rather than arbitrary: DNg02 activity correlates
        positively with wingbeat amplitude in the CONTRALATERAL wing. More
        right-side DNg02 therefore means a bigger left wingbeat, which yaws
        the fly RIGHT - and Namiki et al. saw exactly that pairing, rightward
        motion raising the right cells and lowering the left. DNp06 goes the
        other way because it steers AWAY from a looming object. So a positive
        steer here means "turn right", while a positive yaw above means "turn
        left"; NeuralPathways/flybrain_controller.py is where that gets
        reconciled with this project's positive-is-left convention, and it
        subtracts rather than adds for precisely this reason.
        """
        if not self._dng02:
            return {"n_left": 0, "n_right": 0, "thrust": 0.0, "steer": 0.0, "counts": {}}

        n_left = int((delta[self._dng_left_idx] > 0).sum())
        n_right = int((delta[self._dng_right_idx] > 0).sum())

        # Smooth the two COUNTS, then derive thrust/steer from the smoothed
        # values - not the other way round. A ratio of two small integers
        # flaps between a handful of discrete values (2/3 vs 3/3 is a 33% jump);
        # a ratio of two smoothed counts does not.
        a = DNG02_SMOOTHING
        self._smoothed_n_left = a * n_left + (1 - a) * self._smoothed_n_left
        self._smoothed_n_right = a * n_right + (1 - a) * self._smoothed_n_right

        thrust = min(1.0, (self._smoothed_n_left + self._smoothed_n_right) / DNG02_RECRUIT_SAT)

        # Map each side's count back through its own measured recruitment curve
        # onto the common request axis before differencing, so the population's
        # intrinsic right-side bias cancels at every drive level. See the
        # comment on DNG02_LEFT_RECRUITMENT for why scaling isn't enough.
        act_left = float(np.interp(self._smoothed_n_left,
                                   DNG02_LEFT_RECRUITMENT, DNG02_DRIVE_LEVELS))
        act_right = float(np.interp(self._smoothed_n_right,
                                    DNG02_RIGHT_RECRUITMENT, DNG02_DRIVE_LEVELS))
        steer = max(-1.0, min(1.0, act_right - act_left))

        return {
            "n_left": n_left,
            "n_right": n_right,
            "thrust": float(thrust),
            "steer": float(steer),
            # Raw per-cell counts, in recruitment (input-weight) order, for
            # the same "what actually fired" purpose spike_counts serves.
            "counts": {n["label"]: int(delta[self._dng02_idx[i]])
                       for i, n in enumerate(self._dng02)},
        }


def _serve_stdio():
    """JSON-lines server: one {"loom_left": .., "loom_right": ..} request
    per line on stdin, one step() result per line on stdout. Kept dumb on
    purpose - NeuralPathways/flybrain_controller.py owns retries/timeouts.
    Every key is optional and defaults to 0.0, so an older caller that only
    sends loom_left/loom_right gets exactly the behaviour it always did."""
    controller = FlyBrainController(with_dng02="--dng02" in sys.argv)
    # The ready line carries the constants actually in effect in THIS process.
    # The caller lives in a different interpreter and cannot import this module
    # (no brian2 there, deliberately), so without this a test log could only
    # record what the parent guessed the network was configured with.
    print(json.dumps({
        "ready": True,
        "with_dng02": controller.with_dng02,
        "constants": {
            "MAX_POI_RATE": float(MAX_POI_RATE / Hz),
            "MAX_DRIVE_RATE": float(MAX_DRIVE_RATE / Hz),
            "STEP_DT_MS": float(STEP_DT / ms),
            "RATE_SMOOTHING": RATE_SMOOTHING,
            "ESCAPE_SAT_HZ": ESCAPE_SAT_HZ,
            "YAW_SAT_HZ": YAW_SAT_HZ,
            "FORWARD_SAT_HZ": FORWARD_SAT_HZ,
            "DRIVE_REST_FRACTION": DRIVE_REST_FRACTION,
            "DRIVE_GAIN": DRIVE_GAIN,
            "DNG02_SMOOTHING": DNG02_SMOOTHING,
            "DNG02_RECRUIT_SAT": DNG02_RECRUIT_SAT,
            "DNG02_DRIVE_LEVELS": list(DNG02_DRIVE_LEVELS),
            "DNG02_LEFT_RECRUITMENT": list(DNG02_LEFT_RECRUITMENT),
            "DNG02_RIGHT_RECRUITMENT": list(DNG02_RIGHT_RECRUITMENT),
        },
        # Ladder order, so a caller can label the population readout without
        # re-deriving the sort from the JSON.
        "dng02_labels": [n["label"] for n in controller._dng02],
        "dng02_sides": [n["side"] for n in controller._dng02],
    }), flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            if req.get("reset"):
                controller.reset()
                result = {
                    "yaw": 0.0, "forward": 1.0, "escape": 0.0,
                    "dng02": {"n_left": 0, "n_right": 0, "thrust": 0.0,
                              "steer": 0.0, "counts": {}},
                }
            else:
                result = controller.step(
                    req.get("loom_left", 0.0), req.get("loom_right", 0.0),
                    req.get("drive_common", 0.0),
                    req.get("drive_left", 0.0), req.get("drive_right", 0.0),
                )
        except Exception as exc:  # keep the server alive, report the error inline
            result = {"error": str(exc)}
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    _serve_stdio()
