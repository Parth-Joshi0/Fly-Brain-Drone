"""
Builds brain_atlas.npz - the finished picture of the fly brain that
brain_diagram.py lights up, made out of the real neurons rather than drawn.

Same idea as the BrainViewHud in blendi-remade/fly-brain-minecraft: every
neuron's soma is projected head-on (frontal view), each pixel is coloured by
the region most of its somata belong to and brightened by log(how many there
are), so the optic lobes and central brain appear on their own. The fly's
LEFT is on the viewer's LEFT, like the rest of this project's left/right
readouts.

Source: Schlegel et al. 2024's FlyWire annotation table, the same file
StabilizerNeuron/build_dng02_circuit.py takes cell types from and caches under
Data/. Coordinates are FlyWire voxels (4 x 4 x 40 nm); x grows toward the
fly's right, y ventrally. Neurons without a reconstructed soma use pos_x/y,
a point on the neuron itself.

Everything is baked here so the live diagram only has to copy one image and
draw the spiking cells on top. Saved:

    background    H x W x 3 BGR image, with the simulated cells marked in
                  faint yellow and L / R labels
    circuit_ids   root ids of the 418 cells connectome_worker.py simulates
    circuit_px    their (x, y) pixel on the background
    circuit_region   which of REGIONS each one is in
    circuit_type     TYPES index * 2 + (0 left / 1 right) for each one
    type_names       TYPES, the rows of the diagram's neuron list
    region_names / region_colors   REGIONS, for the diagram's bars and legend

    python NeuralPathways/BrainView/build_brain_atlas.py

Needs pandas (like the other build scripts), not brian2.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PATHWAYS = HERE.parent
sys.path.insert(0, str(PATHWAYS.parent))

from NeuralPathways.StabilizerNeuron.build_dng02_circuit import (  # noqa: E402
    ANNOTATIONS_CACHE, ANNOTATIONS_URL, load_annotations)

LOOMING_IDS_PATH = PATHWAYS / "looming_circuit_neurons.json"
DNG02_IDS_PATH = PATHWAYS / "StabilizerNeuron" / "dng02_circuit_neurons.json"
OUT_PATH = HERE / "brain_atlas.npz"

MAP_W = 440                # px - drawn 1:1 by brain_diagram.py
MARGIN = 6
CLIP_PERCENTILE = 0.05     # ignore the outermost 0.05% of somata when framing
VOXEL_NM = np.array([4.0, 4.0])

# Regions and their BGR colours (fly-brain-minecraft's palette). The diagram
# uses the same list for its spike bars and legend, so it is saved too.
REGIONS = (
    ("optic lobe L", (176, 184, 63)),
    ("optic lobe R", (216, 224, 111)),
    ("central brain", (230, 176, 157)),
    ("descending", (255, 155, 91)),
    ("ascending", (106, 154, 200)),
    ("motor", (64, 160, 255)),
    ("sensory", (110, 214, 85)),
    ("other", (138, 138, 138)),
)
SUPER_CLASS_REGION = {
    "central": 2, "endocrine": 2, "descending": 3, "ascending": 4, "motor": 5,
    "sensory": 6, "sensory_ascending": 6,
}
# Visual projection neurons such as LC4/LPLC2 sit in the optic lobe, so they
# count with it - which also makes a one-sided loom readable off the bars.
OPTIC_CLASSES = {"optic", "visual_projection", "visual_centrifugal"}
MARKER = (102, 224, 255)   # faint yellow where the simulated cells sit
# Neuron types the diagram lists, in display order. Each simulated cell is
# saved as type * 2 + (0 left / 1 right).
TYPES = ("LC4", "LPLC2", "DNp01", "DNp03", "DNp06", "DNg02", "drive pool")


def circuit_cells():
    """root id -> type * 2 + side for every simulated cell, sorted by root id."""
    with open(LOOMING_IDS_PATH) as f:
        looming = json.load(f)
    with open(DNG02_IDS_PATH) as f:
        dng02 = json.load(f)
    groups = ((looming["input_neurons"] + looming["output_neurons"], lambda n: n["cell_type"]),
              (dng02["output_neurons"], lambda n: "DNg02"),        # DNg02_a, _b, ... -> one row
              (dng02["drive_neurons"], lambda n: "drive pool"))
    cells = {n["root_id"]: TYPES.index(name(n)) * 2 + (n["side"] == "right")
             for group, name in groups for n in group}
    return dict(sorted(cells.items()))


def build():
    if not ANNOTATIONS_CACHE.exists():
        load_annotations(ANNOTATIONS_CACHE, ANNOTATIONS_URL)       # downloads + caches it
    a = pd.read_csv(ANNOTATIONS_CACHE, sep="\t", low_memory=False,
                    usecols=["root_id", "pos_x", "pos_y", "soma_x", "soma_y", "super_class", "side"])
    xy = a[["soma_x", "soma_y"]].to_numpy(float)
    xy = np.where(np.isnan(xy), a[["pos_x", "pos_y"]].to_numpy(float), xy) * VOXEL_NM / 1000.0   # um
    ok = ~np.isnan(xy).any(axis=1)
    a, xy = a[ok], xy[ok]

    # Fly's left on the viewer's left - measured, not assumed.
    if xy[(a["side"] == "left").to_numpy(), 0].mean() > xy[(a["side"] == "right").to_numpy(), 0].mean():
        xy[:, 0] = -xy[:, 0]
    lo = np.percentile(xy, CLIP_PERCENTILE, axis=0)
    hi = np.percentile(xy, 100 - CLIP_PERCENTILE, axis=0)
    scale = (hi[0] - lo[0]) / (MAP_W - 2 * MARGIN)            # um per px
    h = int(round((hi[1] - lo[1]) / scale)) + 2 * MARGIN
    px = np.round((xy - lo) / scale).astype(int) + MARGIN

    # Per pixel: how many somata, and the region most of them are in.
    sc, side = a["super_class"].fillna("").to_numpy(), a["side"].fillna("").to_numpy()
    cls = np.array([(0 if sd == "left" else 1 if sd == "right" else 7) if c in OPTIC_CLASSES
                    else SUPER_CLASS_REGION.get(c, 7) for c, sd in zip(sc, side)])
    inside = (px[:, 0] >= 0) & (px[:, 0] < MAP_W) & (px[:, 1] >= 0) & (px[:, 1] < h)
    flat = px[inside, 1] * MAP_W + px[inside, 0]
    per_class = np.zeros((len(REGIONS), h * MAP_W), dtype=np.int32)
    np.add.at(per_class, (cls[inside], flat), 1)
    count = per_class.sum(axis=0)
    palette = np.array([color for _, color in REGIONS], dtype=np.float32)

    # The reference's brightness: floor + gain x log-density, normalised to the
    # 99.5th percentile (the ascending neurons pinned at the neck cut would
    # otherwise set the scale for everything else).
    lit = count > 0
    ref = max(2.0, float(np.percentile(count[lit], 99.5)))
    bright = 0.24 + 0.56 * np.clip(np.log1p(count) / np.log1p(ref), 0.0, 1.0)
    img = np.zeros((h * MAP_W, 3), dtype=np.float32)
    img[lit] = palette[per_class[:, lit].argmax(axis=0)] * bright[lit, None]
    img = img.reshape(h, MAP_W, 3)

    cells = circuit_cells()
    want = np.array(list(cells), dtype=np.int64)
    index = {rid: i for i, rid in enumerate(a["root_id"].to_numpy(np.int64))}
    missing = [rid for rid in want if rid not in index]
    if missing:
        raise SystemExit(f"{len(missing)} circuit neurons have no position in the table: {missing[:5]}")
    rows = [index[rid] for rid in want]
    cpx = np.clip(px[rows], 0, [MAP_W - 1, h - 1])
    img[cpx[:, 1], cpx[:, 0]] = img[cpx[:, 1], cpx[:, 0]] * 0.4 + np.array(MARKER) * 0.6

    img = img.astype(np.uint8)
    for text, x in (("L", 4), ("R", MAP_W - 13)):
        cv2.putText(img, text, (x, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (156, 147, 138), 1, cv2.LINE_AA)

    np.savez_compressed(OUT_PATH, background=img, circuit_ids=want, circuit_px=cpx.astype(np.int16),
                        circuit_region=cls[rows].astype(np.int8),
                        circuit_type=np.array(list(cells.values()), dtype=np.int8),
                        type_names=np.array(TYPES),
                        region_names=np.array([n for n, _ in REGIONS]),
                        region_colors=np.array([c for _, c in REGIONS], dtype=np.uint8))
    print(f"{len(a)} neurons -> {MAP_W}x{h} px; {len(want)} circuit neurons placed")
    print(f"wrote {OUT_PATH} ({OUT_PATH.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    build()
