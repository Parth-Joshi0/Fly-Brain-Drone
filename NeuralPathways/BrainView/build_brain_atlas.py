"""
Builds brain_atlas.npz - the picture of the whole fly brain that
brain_diagram.py lights up, made out of the real neurons rather than drawn.

Same idea as the BrainViewHud in blendi-remade/fly-brain-minecraft
(src/client/java/com/fruitfly/client/hud/BrainViewHud.java): every neuron's
soma is projected onto a 2-D image, each pixel is coloured by the region most
of its neurons belong to and brightened by log(how many there are), so the
optic lobes, central brain and their outline appear on their own. Two
projections are baked, stacked in the diagram:

    frontal   looking at the head-on face of the brain (x across, y down)
    dorsal    looking down from above, anterior at the top (x across, z down)

Both keep the fly's LEFT on the viewer's LEFT, like the rest of this project's
left/right readouts (loom_left, DNp01_left, ...), rather than the mirror-image
"facing the fly" convention - so a left-eye loom lights up the left of the
screen in both views.

Source: Schlegel et al. 2024's FlyWire annotation table
(flyconnectome/flywire_annotations Supplemental_file1_neuron_annotations.tsv),
the same file StabilizerNeuron/build_dng02_circuit.py takes cell types from and
caches under Data/. Coordinates are FlyWire voxels (4 x 4 x 40 nm); measured
from the table itself (ORNs enter at z ~800, Kenyon cell somata sit at z ~4800
and low y), x grows toward the fly's right, y ventrally, z posteriorly.

The background uses every annotated neuron (139k), not only the 106k of them
whose root ids are in this repo's v630 completeness list - it is anatomy, and
dropping a fifth of the cells would just thin the picture. The circuit neurons
fly_brain_controller.py actually simulates (looming_circuit_neurons.json +
StabilizerNeuron/dng02_circuit_neurons.json, 418 cells) are all present and
get their own pixel coordinates, which is where their spikes are drawn. The
few without a reconstructed soma fall back to the table's pos_x/y/z, a point
on the neuron itself.

    python NeuralPathways/BrainView/build_brain_atlas.py

Output is a few hundred KB and committed; re-run only to change MAP_W or the
region grouping. Needs pandas (like the other build scripts), not brian2.
"""

import json
import sys
from pathlib import Path

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

# Display width of each map in pixels. The atlas is baked at exactly the size
# brain_diagram.py draws it, so one soma lands on one screen pixel and nothing
# is resampled.
MAP_W = 440
MARGIN = 6                 # px of empty border around the brain
CLIP_PERCENTILE = 0.05     # ignore the outermost 0.05% of somata when fitting
                           # the frame (a handful of mis-placed points would
                           # otherwise shrink the whole brain)
VOXEL_NM = np.array([4.0, 4.0, 40.0])

# Region rows, in the order brain_diagram.py lists them. The reference HUD's
# list adapted to FlyWire, which is brain-only: there is no nerve cord, so its
# "nerve cord" row becomes "ascending" (cells whose somata are in the VNC but
# whose axons reach the brain).
REGION_NAMES = ("optic lobe L", "optic lobe R", "central brain", "descending",
                "ascending", "motor", "sensory", "other")
OPTIC_CLASSES = {"optic", "visual_projection", "visual_centrifugal"}
CENTRAL_CLASSES = {"central", "endocrine"}


def region_of(super_class, side):
    """Region row for one neuron. Visual projection neurons (LC4, LPLC2, ...)
    are counted with the optic lobe of their side - their somata sit there,
    and it is what makes a one-sided loom readable straight off the bars."""
    if super_class in OPTIC_CLASSES:
        if side == "left":
            return 0
        if side == "right":
            return 1
        return 7
    if super_class in CENTRAL_CLASSES:
        return 2
    if super_class == "descending":
        return 3
    if super_class == "ascending":
        return 4
    if super_class == "motor":
        return 5
    if super_class in ("sensory", "sensory_ascending"):
        return 6
    return 7


def load_positions(path=ANNOTATIONS_CACHE):
    if not path.exists():
        load_annotations(path, ANNOTATIONS_URL)       # downloads + caches it
    cols = ["root_id", "pos_x", "pos_y", "pos_z", "soma_x", "soma_y", "soma_z", "super_class", "side"]
    a = pd.read_csv(path, sep="\t", usecols=cols, low_memory=False)
    soma = a[["soma_x", "soma_y", "soma_z"]].to_numpy(dtype=float)
    pos = a[["pos_x", "pos_y", "pos_z"]].to_numpy(dtype=float)
    has_soma = ~np.isnan(soma).any(axis=1)
    xyz = np.where(has_soma[:, None], soma, pos) * VOXEL_NM / 1000.0      # -> um
    ok = ~np.isnan(xyz).any(axis=1)
    region = np.array([region_of(sc, sd) for sc, sd in zip(a["super_class"].fillna(""), a["side"].fillna(""))],
                      dtype=np.int8)
    return (a["root_id"].to_numpy(dtype=np.int64)[ok], xyz[ok], region[ok], has_soma[ok],
            a["side"].fillna("").to_numpy()[ok])


def circuit_ids():
    with open(LOOMING_IDS_PATH) as f:
        looming = json.load(f)
    with open(DNG02_IDS_PATH) as f:
        dng02 = json.load(f)
    cells = (looming["input_neurons"] + looming["output_neurons"]
             + dng02["output_neurons"] + dng02["drive_neurons"])
    return np.array(sorted({n["root_id"] for n in cells}), dtype=np.int64)


def build():
    ids, xyz, region, has_soma, side = load_positions()
    print(f"{len(ids)} neurons with a position ({has_soma.sum()} somata, rest pos_*)")

    # Fly's left on the viewer's left. Measured, not assumed.
    left_x = xyz[side == "left", 0].mean()
    right_x = xyz[side == "right", 0].mean()
    x_sign = 1.0 if left_x < right_x else -1.0
    u = x_sign * xyz[:, 0]

    lo = np.percentile(xyz, CLIP_PERCENTILE, axis=0)
    hi = np.percentile(xyz, 100 - CLIP_PERCENTILE, axis=0)
    u_lo, u_hi = sorted((x_sign * lo[0], x_sign * hi[0]))
    um_per_px = (u_hi - u_lo) / (MAP_W - 2 * MARGIN)

    # (name, vertical coordinate). y grows ventrally and z posteriorly, so
    # plain increasing v puts dorsal / anterior at the top of the image.
    views = (("frontal", 1), ("dorsal", 2))
    out = {
        "region_names": np.array(REGION_NAMES),
        "views": np.array([v for v, _ in views]),
        "um_per_px": np.float32(um_per_px),
    }

    want = circuit_ids()
    index = {rid: i for i, rid in enumerate(ids)}
    missing = [rid for rid in want if rid not in index]
    if missing:
        raise SystemExit(f"{len(missing)} circuit neurons have no position in the table: {missing[:5]}")
    rows = np.array([index[rid] for rid in want])
    out["circuit_ids"] = want
    out["circuit_region"] = region[rows]
    out["circuit_has_soma"] = has_soma[rows]

    for name, axis in views:
        v = xyz[:, axis]
        h = int(round((hi[axis] - lo[axis]) / um_per_px)) + 2 * MARGIN
        px = np.round((u - u_lo) / um_per_px).astype(int) + MARGIN
        py = np.round((v - lo[axis]) / um_per_px).astype(int) + MARGIN
        # Outliers past the clipped frame are left out of the background
        # (clamping them stacks them into a line along the edge)...
        inside = (px >= 0) & (px < MAP_W) & (py >= 0) & (py < h)
        flat = py[inside] * MAP_W + px[inside]
        count = np.bincount(flat, minlength=h * MAP_W)
        per_region = np.zeros((len(REGION_NAMES), h * MAP_W), dtype=np.int32)
        np.add.at(per_region, (region[inside], flat), 1)
        # ...but a circuit neuron always gets a pixel, its spikes have to land somewhere.
        px, py = np.clip(px, 0, MAP_W - 1), np.clip(py, 0, h - 1)
        dominant = np.where(count > 0, per_region.argmax(axis=0), -1)
        out[f"count_{name}"] = count.reshape(h, MAP_W).astype(np.uint16)
        out[f"region_{name}"] = dominant.reshape(h, MAP_W).astype(np.int8)
        out[f"circuit_px_{name}"] = np.stack([px[rows], py[rows]], axis=1).astype(np.int16)
        print(f"  {name}: {MAP_W}x{h} px, {int((count > 0).sum())} lit pixels, max {count.max()} somata/pixel")

    np.savez_compressed(OUT_PATH, **out)
    print(f"{len(want)} circuit neurons placed ({int(has_soma[rows].sum())} at their soma, "
          f"{int((~has_soma[rows]).sum())} at pos_*), {um_per_px:.2f} um/px")
    print(f"wrote {OUT_PATH} ({OUT_PATH.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    build()
