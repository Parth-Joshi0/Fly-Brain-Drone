"""
Builds dng02_circuit_neurons.json - the neuron set for the DNg02 flight-motor
circuit, the population-coded counterpart to looming_circuit_neurons.json's
looming/escape circuit.

Why DNg02: unlike DNp01/DNp03/DNp06 (one neuron per side, so their readout is
one cell's spike rate and has to be heavily smoothed to be usable), DNg02 is a
POPULATION - at least 15 near-identical cell pairs. Namiki et al. 2022
("A population of descending neurons that regulates the flight motor of
Drosophila", Current Biology 32(5):1189-1196) showed that optogenetically
recruiting more DNg02 pairs raises wingbeat amplitude roughly linearly
(~2.77 deg per pair), i.e. the fly encodes flight power in *how many* of them
are firing. That makes the natural readout a recruitment count rather than a
rate, which is what this circuit is for.

Why a build script at all: looming_circuit_neurons.json was produced by hand
and only records its provenance in a "source" string, so nobody can re-derive
it or change a threshold and see what happens. This script is the fix for that
pattern - every selection rule below is a constant at module top, and running
it prints the resulting circuit so the output is self-checking.

    conda run -n brian2 python build_dng02_circuit.py

What the connectome actually says (all of this is printed by --verbose, and the
numbers below came from running exactly this analysis):

  * 25 neurons in FlyWire v630 are typed DNg02, as subtypes DNg02_a..DNg02_h,
    13 left / 12 right, predominantly cholinergic. 24 of them appear in this
    repo's own data/2023_03_23_completeness_630_final.csv; the 25th is
    dropped, the same cross-referencing rule looming_circuit_neurons.json's
    "source" field describes.

  * Their excitatory input weight is strongly graded across the population -
    727, 684, 632, 602, 590, 560, 351, 310, 272, 251, 179, 130, 117, 101, 95,
    95, 44, 31, 28, 21, 16, 13, 9, 1, 0 - a ~700x monotone spread, with left
    and right cells interleaved almost evenly. That gradient IS the population
    code: under a common drive the cells recruit in roughly that order, so the
    number recruited is a monotone function of drive strength.

  * DNg02 has no usable visual input in this connectome. Of 12,848 incoming
    synapses, only 137 (1%) come from visual projection neurons (MTe11, LPLC4,
    LC36, LC46, LTe64), ZERO come from the HS/VS/H2 wide-field tangential
    cells you would expect to carry optic flow, and only 8 come from the 268
    LC4/LPLC2 cells already in looming_circuit_neurons.json. Its input is
    central: posterior slope (PS005/PS008/PS109/PS180/PS181/PS182), inferior
    bridge (IB008/IB010/IB025), clamp (CL216/CL336/CL171/CL155), LAL197/LAL200,
    plus ascending neurons from the ventral nerve cord (AN_multi_*).

    This is a real gap in the published picture rather than a bug here:
    Namiki et al. found DNg02 responds to wide-field visual motion during
    flight, but the pathway is still unknown, and Schnell 2026 (Trends in
    Neurosciences 49:98-110, "Flexible circuits for visually guided flight
    control in Drosophila") states plainly that DNg02's "connectivity with
    LPTCs or other visually responsive neurons remains to be established".

    So fly_brain_controller.py drives these central partners with optic flow.
    That is an explicit modelling assumption - the connectome does not say
    these cells carry a motion signal - and it is the one assumption in this
    circuit. Everything downstream of it (which cell excites or inhibits which
    DNg02, and how strongly) is measured, not assumed, which is why this file
    records each driver's sign and its left/right weight split.

  * The drive pool is only weakly lateralized as a whole (ipsilateral/
    contralateral weight ratio ~1.1), so splitting it by hemifield would
    produce a fixed right-side bias rather than a flow-dependent left/right
    difference. Lateralized SUBSETS do exist though, and the largest single
    one is ipsilateral inhibition (211 cells, weight 1,710; PS117b 0.97,
    PS137 1.00, PS008 0.78) - which is the mechanism Namiki et al.'s own
    result needs, since "rightward motion elicited an increase in activity of
    the right DNg02 cells and a simultaneous DECREASE in activity of the left
    DNg02 cells" cannot come from excitation alone. Hence sign and w_left/
    w_right per driver below, instead of a plain hemifield split.
"""

import argparse
import csv
import io
import json
import sys
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PATH_CON = HERE / "Data" / "2023_03_23_connectivity_630_final.parquet"
PATH_COMPLETENESS = HERE / "Data" / "2023_03_23_completeness_630_final.csv"
OUT_PATH = HERE / "dng02_circuit_neurons.json"

# Schlegel et al. 2024's FlyWire annotation table - the same v630 cell-type
# source looming_circuit_neurons.json cites. 31MB, so it's cached next to the
# rest of the connectome data (gitignored - it's an external artifact, not ours).
ANNOTATIONS_URL = (
    "https://raw.githubusercontent.com/flyconnectome/flywire_annotations/main/"
    "supplemental_files/Supplemental_file1_neuron_annotations.tsv"
)
ANNOTATIONS_CACHE = HERE / "Data" / "flywire_neuron_annotations_630.tsv"

# --- selection rules ---

CELL_TYPE_PREFIX = "DNg02"

# How many presynaptic partners to keep as the optic-flow drive pool. Ranked by
# total |weight| onto the DNg02 population, so this is "the cells that most
# determine DNg02's activity". 120 keeps ~77% of DNg02's total excitatory input
# weight for ~11.1k synapses in the merged network (vs 7,973 today); the cost
# is why this is a constant - see fly_brain_controller.py's step budget and
# NeuralPathways/Tests/test_dng02_circuit.py, which asserts ms/step stays under 33.
# Coverage/cost at other sizes, measured: 40 -> 56% / 9.1k, 60 -> 64% / ~9.5k,
# 80 -> 70% / 10.0k, 160 -> 82% / 12.2k.
DRIVE_POOL_SIZE = 120

# A driver has to be in this repo's own completeness list (same cross-reference
# rule as the looming circuit) AND have a known left/right side, since the
# whole point of keeping w_left/w_right is to know which hemisphere it acts on.
# Requiring a side drops the unsided/untyped cells but keeps 81.6% of the
# excitatory weight, so it's cheap.
REQUIRE_KNOWN_SIDE = True

# Ignore partners connected by almost nothing - below this they're noise in the
# reconstruction rather than a pathway, and they'd crowd out real drivers in the
# ranking. (The pool is weight-ranked anyway, so this mainly documents intent.)
MIN_DRIVER_WEIGHT = 5.0

# ipsi_frac thresholds for the human-readable "group" label only. Nothing in
# fly_brain_controller.py reads these - it uses the continuous w_left/w_right
# split directly, which is strictly more faithful. They exist so the JSON and
# the HUD can say "this is an ipsilateral inhibitor" in words.
IPSI_GROUP_THRESHOLD = 0.75
CONTRA_GROUP_THRESHOLD = 0.25


def load_annotations(path=ANNOTATIONS_CACHE, url=ANNOTATIONS_URL):
    """root_id -> (cell_type, side, super_class, top_nt), from the cached TSV
    (downloaded on first run)."""
    if not path.exists():
        print(f"downloading {url}\n         -> {path} (31MB, once)", file=sys.stderr)
        path.write_bytes(urllib.request.urlopen(url).read())

    meta = {}
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if not row["root_id"]:
                continue
            meta[int(row["root_id"])] = (
                row["cell_type"] or row["hemibrain_type"] or "",
                row["side"] or "",
                row["super_class"] or "",
                row["top_nt"] or "",
            )
    return meta


def load_completeness(path=PATH_COMPLETENESS):
    ids = pd.read_csv(path).iloc[:, 0]
    return set(ids.astype("int64"))


def group_label(ipsi_frac):
    if ipsi_frac >= IPSI_GROUP_THRESHOLD:
        return "ipsi"
    if ipsi_frac <= CONTRA_GROUP_THRESHOLD:
        return "contra"
    return "bilateral"


def build(verbose=False):
    meta = load_annotations()
    complete = load_completeness()

    # --- the DNg02 population itself ---
    all_dng02 = [r for r, m in meta.items() if m[0].startswith(CELL_TYPE_PREFIX)]
    dng02 = sorted(r for r in all_dng02 if r in complete)
    dropped = sorted(set(all_dng02) - set(dng02))
    if not dng02:
        raise RuntimeError(f"no {CELL_TYPE_PREFIX} neurons found in the annotations")

    con = pd.read_parquet(
        PATH_CON,
        columns=["Presynaptic_ID", "Postsynaptic_ID", "Connectivity", "Excitatory x Connectivity"],
    )
    inc = con[con["Postsynaptic_ID"].isin(dng02)].copy()
    inc["post_side"] = inc["Postsynaptic_ID"].map(lambda r: meta[r][1])

    # --- rank candidate drivers by how much they weigh on the population ---
    w = inc["Excitatory x Connectivity"]
    per_pre = inc.assign(
        abs_w=w.abs(),
        w_left=np.where(inc["post_side"] == "left", w.abs(), 0.0),
        w_right=np.where(inc["post_side"] == "right", w.abs(), 0.0),
        ipsi_w=np.where(
            inc["Presynaptic_ID"].map(lambda r: meta.get(r, ("", ""))[1]) == inc["post_side"],
            w.abs(), 0.0,
        ),
        signed=w,
    ).groupby("Presynaptic_ID").agg(
        abs_w=("abs_w", "sum"), w_left=("w_left", "sum"), w_right=("w_right", "sum"),
        ipsi_w=("ipsi_w", "sum"), signed=("signed", "sum"),
    )

    eligible = per_pre[
        per_pre.index.isin(complete)
        & (per_pre["abs_w"] >= MIN_DRIVER_WEIGHT)
    ]
    if REQUIRE_KNOWN_SIDE:
        eligible = eligible[[meta.get(r, ("", ""))[1] in ("left", "right") for r in eligible.index]]
    eligible = eligible.sort_values("abs_w", ascending=False)
    pool = eligible.head(DRIVE_POOL_SIZE)

    drive_neurons = []
    for root_id, row in pool.iterrows():
        cell_type, side, super_class, top_nt = meta[root_id]
        drive_neurons.append({
            "root_id": int(root_id),
            "cell_type": cell_type or "?",
            "side": side,
            "super_class": super_class,
            "top_nt": top_nt,
            # +1 excitatory / -1 inhibitory, from the parquet's own signed
            # "Excitatory x Connectivity" - not assumed from neurotransmitter.
            "sign": 1 if row["signed"] > 0 else -1,
            # |weight| this cell puts on the left / right half of the DNg02
            # population. fly_brain_controller.py uses the ratio to decide how
            # much of a left-vs-right steering signal should reach it.
            "w_left": round(float(row["w_left"]), 2),
            "w_right": round(float(row["w_right"]), 2),
            "ipsi_frac": round(float(row["ipsi_w"] / row["abs_w"]), 3),
            "group": group_label(float(row["ipsi_w"] / row["abs_w"])),
        })

    # --- the recruitment ladder: excitatory weight each DNg02 gets FROM THE
    # POOL (not from the whole brain), because that's what actually drives it
    # in the model. Sorted descending = predicted recruitment order. ---
    pool_ids = set(pool.index)
    from_pool = inc[inc["Presynaptic_ID"].isin(pool_ids)]
    exc = from_pool[from_pool["Excitatory x Connectivity"] > 0]
    inh = from_pool[from_pool["Excitatory x Connectivity"] < 0]
    exc_w = exc.groupby("Postsynaptic_ID")["Excitatory x Connectivity"].sum()
    inh_w = inh.groupby("Postsynaptic_ID")["Excitatory x Connectivity"].sum()

    output_neurons = []
    for root_id in dng02:
        cell_type, side, super_class, top_nt = meta[root_id]
        output_neurons.append({
            "root_id": int(root_id),
            "cell_type": cell_type,
            "side": side,
            "top_nt": top_nt,
            "exc_weight": round(float(exc_w.get(root_id, 0.0)), 2),
            "inh_weight": round(float(inh_w.get(root_id, 0.0)), 2),
        })
    output_neurons.sort(key=lambda n: -n["exc_weight"])

    circuit = {
        "source": (
            "DNg02 flight-motor circuit. Cell types from "
            "flyconnectome/flywire_annotations Supplemental_file1_neuron_annotations.tsv "
            "(Schlegel et al. 2024, FlyWire v630), cross-referenced against this repo's "
            "data/2023_03_23_completeness_630_final.csv; synapse weights from this "
            "repo's data/2023_03_23_connectivity_630_final.parquet "
            f"('Excitatory x Connectivity'). Generated by build_dng02_circuit.py - re-run it "
            f"to regenerate. output_neurons: all {len(all_dng02)} v630 neurons typed "
            f"{CELL_TYPE_PREFIX}*, minus {len(dropped)} absent from the completeness list "
            f"(dropped: {dropped}), leaving {len(dng02)}. drive_neurons: their presynaptic "
            f"partners in the parquet, restricted to the completeness list with a known "
            f"left/right side and total |weight| >= {MIN_DRIVER_WEIGHT}, ranked by total "
            f"|weight| onto the population, top {DRIVE_POOL_SIZE}. DNg02 has essentially no "
            "visual input in this connectome (137 of 12,848 incoming synapses from visual "
            "projection neurons, none from HS/VS/H2 tangential cells, 8 from the LC4/LPLC2 "
            "pool in looming_circuit_neurons.json), so these drivers are central neurons "
            "(posterior slope / inferior bridge / clamp / LAL / ascending). Driving them "
            "with optic flow is this circuit's one modelling assumption - the pathway from "
            "wide-field motion to DNg02 is a published open question (Schnell 2026, Trends "
            "in Neurosciences 49:98-110). Each driver's sign and w_left/w_right split are "
            "measured from the parquet, not assumed."
        ),
        "output_neurons": output_neurons,
        "drive_neurons": drive_neurons,
    }

    if verbose:
        report(circuit, meta, inc, all_dng02, dropped, eligible, per_pre)
    return circuit


def report(circuit, meta, inc, all_dng02, dropped, eligible, per_pre):
    out, drv = circuit["output_neurons"], circuit["drive_neurons"]
    print(f"\nDNg02 population: {len(all_dng02)} typed in v630, {len(out)} kept "
          f"({len(dropped)} not in the completeness list)")
    print("  sides:", dict(Counter(n["side"] for n in out)))
    print("  subtypes:", dict(sorted(Counter(n["cell_type"] for n in out).items())))
    print("  transmitters:", dict(Counter(n["top_nt"] or "?" for n in out)))

    total = inc["Connectivity"].sum()
    vis = inc[inc["Presynaptic_ID"].map(lambda r: meta.get(r, ("", "", ""))[2])
             .isin(["visual_projection", "visual_centrifugal", "optic"])]["Connectivity"].sum()
    e = inc[inc["Excitatory x Connectivity"] > 0]["Connectivity"].sum()
    print(f"\nwhole-brain input to the population: {total:.0f} synapses, "
          f"{100 * e / total:.0f}% excitatory, {vis:.0f} ({100 * vis / total:.1f}%) visual")

    print(f"\ndrive pool: {len(drv)} of {len(eligible)} eligible partners, "
          f"{dict(Counter(n['side'] for n in drv))}")
    print("  by sign x laterality:",
          dict(Counter(f"{'exc' if n['sign'] > 0 else 'inh'}/{n['group']}" for n in drv)))
    print("  strongest 12:")
    for n in drv[:12]:
        print(f"    {n['cell_type']:>12} {n['side']:>5}  "
              f"{'exc' if n['sign'] > 0 else 'inh'}  "
              f"L{n['w_left']:>7.1f} R{n['w_right']:>7.1f}  ipsi {n['ipsi_frac']:.2f}")

    print("\nrecruitment ladder (excitatory weight from the pool, = predicted order):")
    for n in out:
        bar = "#" * int(round(20 * n["exc_weight"] / max(out[0]["exc_weight"], 1)))
        print(f"    {n['cell_type']:>9} {n['side']:>5}  exc {n['exc_weight']:>7.1f}  "
              f"inh {n['inh_weight']:>8.1f}  {bar}")
    ladder = [n["exc_weight"] for n in out]
    print(f"  spread: {max(ladder):.0f} -> {min(ladder):.0f}, "
          f"{sum(1 for v in ladder if v > 0)}/{len(ladder)} cells receive any drive")
    nl = sum(n["exc_weight"] for n in out if n["side"] == "left")
    nr = sum(n["exc_weight"] for n in out if n["side"] == "right")
    print(f"  left total {nl:.0f} vs right total {nr:.0f} (ratio {nl / max(nr, 1):.2f}) - "
          f"the intrinsic bias the readout's calibrated baselines cancel")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("-o", "--out", type=Path, default=OUT_PATH)
    ap.add_argument("-q", "--quiet", action="store_true", help="skip the circuit report")
    args = ap.parse_args()

    circuit = build(verbose=not args.quiet)
    args.out.write_text(json.dumps(circuit, indent=1) + "\n")
    print(f"\nwrote {args.out} "
          f"({len(circuit['output_neurons'])} DNg02 + {len(circuit['drive_neurons'])} drivers)")


if __name__ == "__main__":
    main()
