"""Turn measured features into the Ageing State Vector (ASV).

Each ASV score places a sample between two reference populations:

* **young reference** (required): e.g. early-culture wells;
* **aged anchor** (recommended): a positive control such as late-culture or
  induced-senescence wells (e.g. doxorubicin).

With an aged anchor, each feature is scaled to its position between the two means
(0 = young mean, 1 = aged-anchor mean) and the score is the clipped average:

    p_f   = (x_f - mean_young_f) / (mean_aged_f - mean_young_f)
    score = clip(mean_f(p_f), 0, 1)

Without an anchor, a z-score fallback is used:

    z_f   = direction_f * (x_f - mean_young_f) / sd_young_f
    score = clip(mean_f(z_f) / Z_FULL, 0, 1)

Directions are the expected changes with ageing. Per-feature z-scores are always
returned so each score can be explained. Senescence *markers* are never used here.

When several samples (videos, fields of view, traces) are given for one condition, the
ASV is their mean and the per-score sd is their spread, which feeds the uncertainty
propagation in `mapping.map_with_uncertainty`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .ageing_state import AgeingStateVector

Z_FULL = 4.0

# score -> {feature: direction (+1 = increases with ageing, -1 = decreases)}
SCORE_DEFINITIONS: dict[str, dict[str, int]] = {
    "motion_dysfunction": {
        "interval_cv": +1,            # less regular rhythm
        "amplitude_cv": +1,           # less consistent contractions
        "contraction_velocity": -1,   # slower contraction
        "relaxation_velocity": -1,    # slower relaxation
    },
    "nuclear_irregularity": {
        "area_mean": +1,              # nuclear enlargement (senescence-associated)
        "area_cv": +1,
        "circularity_mean": -1,       # less round
        "solidity_mean": -1,          # more folded / lobulated outline
    },
    "calcium_decay_delay": {
        "decay_tau_s": +1,            # slower Ca2+ re-uptake
        "cd50_s": +1,
    },
}


@dataclass
class YoungReference:
    """Mean and sd of each feature in young (reference) samples."""

    mean: dict = field(default_factory=dict)
    sd: dict = field(default_factory=dict)
    aged_mean: dict = field(default_factory=dict)   # optional positive-control anchor

    @classmethod
    def fit(cls, samples: list[dict], aged_anchor: list[dict] | None = None) -> "YoungReference":
        if len(samples) < 3:
            raise ValueError("need at least 3 young reference samples")
        ref = cls()
        for feats in SCORE_DEFINITIONS.values():
            for f in feats:
                vals = _finite(samples, f)
                if len(vals) < 3:
                    continue
                ref.mean[f] = float(vals.mean())
                ref.sd[f] = float(max(vals.std(ddof=1), 1e-6 * max(abs(vals.mean()), 1e-9)))
                if aged_anchor:
                    avals = _finite(aged_anchor, f)
                    # keep the anchor only if it moves in the expected ageing direction
                    if len(avals) and (avals.mean() - ref.mean[f]) * _direction(f) > 0:
                        ref.aged_mean[f] = float(avals.mean())
        return ref

    def to_dict(self) -> dict:
        return {"mean": self.mean, "sd": self.sd, "aged_anchor_mean": self.aged_mean}


def _finite(samples: list[dict], f: str) -> np.ndarray:
    return np.array([s[f] for s in samples if f in s and np.isfinite(s[f])], dtype=float)


def _direction(f: str) -> int:
    for feats in SCORE_DEFINITIONS.values():
        if f in feats:
            return feats[f]
    raise KeyError(f)


def explain(features: dict, ref: YoungReference) -> dict:
    """Per-feature directed z-scores, grouped by ASV score."""
    out = {}
    for score, feats in SCORE_DEFINITIONS.items():
        out[score] = {}
        for f, direction in feats.items():
            if f in features and f in ref.mean and np.isfinite(features[f]):
                out[score][f] = round(direction * (features[f] - ref.mean[f]) / ref.sd[f], 3)
    return out


def score_sample(features: dict, ref: YoungReference) -> dict:
    """ASV scores for one sample's merged feature dict (motion + nuclear + calcium)."""
    scores = {}
    for score, feats in SCORE_DEFINITIONS.items():
        parts = []
        for f, direction in feats.items():
            if f not in features or f not in ref.mean or not np.isfinite(features[f]):
                continue
            if f in ref.aged_mean:
                parts.append((features[f] - ref.mean[f]) / (ref.aged_mean[f] - ref.mean[f]))
            else:
                parts.append(direction * (features[f] - ref.mean[f]) / ref.sd[f] / Z_FULL)
        scores[score] = float(np.clip(np.mean(parts), 0, 1)) if parts else 0.0
    return scores


def build_asv(samples: list[dict], ref: YoungReference,
              senescence_confidence: float | None = None) -> tuple[AgeingStateVector, dict]:
    """Combine several samples of one condition into an ASV (mean, with sd).

    `senescence_confidence` is passed through untouched: it comes from separate
    marker staining and is kept for validation only.
    """
    per = [score_sample(s, ref) for s in samples]
    names = list(SCORE_DEFINITIONS)
    mean = {n: float(np.mean([p[n] for p in per])) for n in names}
    sd = {n: float(np.std([p[n] for p in per], ddof=1)) if len(per) > 1 else 0.0 for n in names}
    asv = AgeingStateVector(**{n: round(mean[n], 4) for n in names},
                            senescence_confidence=senescence_confidence,
                            sd={n: round(sd[n], 4) for n in names})
    return asv, {"per_sample_scores": per}
