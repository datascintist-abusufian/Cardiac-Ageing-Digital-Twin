"""Nuclear morphology features from segmented nuclei (e.g. DAPI images).

Classical, transparent baseline. A learned nuclear phenotyping model (the NucAgeAI
concept) can later replace `nuclear_features` if it returns the same feature names.

Input: a 2D mask, either binary (nuclei = True) or already labelled (0 = background,
1..N = nuclei). Segmentation itself (Cellpose, StarDist, thresholding) is done upstream.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull, QhullError

NUCLEAR_FEATURES = ("area_mean", "area_cv", "circularity_mean", "solidity_mean")


def _perimeter(obj: np.ndarray) -> float:
    """Perimeter estimate: count exposed pixel edges, corrected for the staircase bias."""
    padded = np.pad(obj, 1)
    edges = 0
    for axis in (0, 1):
        edges += np.count_nonzero(np.diff(padded.astype(np.int8), axis=axis))
    return edges * (np.pi / 4)          # mean projection correction for random orientation


def per_nucleus(mask: np.ndarray, min_area: int = 20) -> list[dict]:
    mask = np.asarray(mask)
    labels = mask if mask.dtype.kind in "iu" and mask.max() > 1 else ndimage.label(mask > 0)[0]
    out = []
    for i, sl in enumerate(ndimage.find_objects(labels), start=1):
        if sl is None:
            continue
        obj = labels[sl] == i
        area = int(obj.sum())
        if area < min_area:
            continue
        perim = _perimeter(obj)
        ys, xs = np.nonzero(obj)
        pts = np.column_stack([ys, xs]).astype(float)
        # Hull of pixel corners so that solidity of a perfect blob is ~1.
        corners = np.concatenate([pts + d for d in ((-.5, -.5), (-.5, .5), (.5, -.5), (.5, .5))])
        try:
            hull_area = ConvexHull(corners).volume
        except QhullError:
            hull_area = float(area)
        out.append({
            "area": area,
            "circularity": float(min(1.0, 4 * np.pi * area / perim ** 2)),
            "solidity": float(min(1.0, area / hull_area)),
        })
    return out


def nuclear_features(mask: np.ndarray, min_area: int = 20) -> dict:
    """Population summary: size, size variability, roundness and convexity."""
    nuc = per_nucleus(mask, min_area)
    if not nuc:
        raise ValueError("no nuclei found in mask")
    area = np.array([n["area"] for n in nuc], dtype=float)
    return {
        "area_mean": float(area.mean()),
        "area_cv": float(area.std() / area.mean()),
        "circularity_mean": float(np.mean([n["circularity"] for n in nuc])),
        "solidity_mean": float(np.mean([n["solidity"] for n in nuc])),
        "n_nuclei": len(nuc),
    }
