"""Rule-based mapping: Ageing State Vector -> tissue parameters (roadmap phase 2).

Each tissue parameter is a clipped linear function of the three driver scores:

    param = baseline + sum_k  weight[param][k] * score_k      (then clipped to bounds)

Directions follow the ageing-heart literature (slower conduction, longer action
potentials, slower Ca2+ decay, less contractile reserve, more fibrosis).

IMPORTANT: the weights and bounds below are *placeholder assumptions* chosen to give
plausible ranges. They are not fitted to data and must be calibrated against paired
cell / tissue measurements (roadmap phase 5) before any result is interpreted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ageing_state import DRIVER_FIELDS, AgeingStateVector


@dataclass(frozen=True)
class Rule:
    baseline: float
    weights: dict          # score name -> weight
    bounds: tuple          # (low, high)
    rationale: str


RULES: dict[str, Rule] = {
    "conduction_velocity_scale": Rule(
        1.0,
        {"motion_dysfunction": -0.15, "nuclear_irregularity": -0.10, "calcium_decay_delay": -0.10},
        (0.5, 1.0),
        "Reduced / lateralised connexin-43 and fibrosis slow conduction.",
    ),
    "apd_scale": Rule(
        1.0,
        {"motion_dysfunction": 0.05, "nuclear_irregularity": 0.05, "calcium_decay_delay": 0.20},
        (1.0, 1.4),
        "Altered repolarising K+ currents and Ca2+ handling prolong the action potential.",
    ),
    "calcium_decay_scale": Rule(
        1.0,
        {"calcium_decay_delay": 0.50},
        (1.0, 1.8),
        "Lower SERCA2a activity slows Ca2+ re-uptake.",
    ),
    "contractility_scale": Rule(
        1.0,
        {"motion_dysfunction": -0.30, "calcium_decay_delay": -0.10},
        (0.5, 1.0),
        "Altered Ca2+ cycling and myocyte loss reduce contractile reserve.",
    ),
    "fibrosis_fraction": Rule(
        0.02,
        {"motion_dysfunction": 0.08, "nuclear_irregularity": 0.15, "calcium_decay_delay": 0.02},
        (0.0, 0.35),
        "Senescent cells and fibroblast activation increase collagen deposition.",
    ),
    "fibrosis_heterogeneity": Rule(
        0.10,
        {"motion_dysfunction": 0.30, "nuclear_irregularity": 0.40},
        (0.0, 1.0),
        "Ageing fibrosis becomes patchier, not just more abundant.",
    ),
}


def _apply(rule: Rule, scores: dict) -> float:
    value = rule.baseline + sum(w * scores[name] for name, w in rule.weights.items())
    return float(np.clip(value, *rule.bounds))


def map_to_tissue(asv: AgeingStateVector) -> dict:
    """Map an ASV to tissue parameters (point estimate)."""
    scores = asv.drivers()
    return {name: round(_apply(rule, scores), 4) for name, rule in RULES.items()}


def map_with_uncertainty(asv: AgeingStateVector, n: int = 2000, seed: int = 0) -> dict:
    """Propagate per-score uncertainty (asv.sd) by Monte Carlo.

    Returns {param: {"mean", "ci95": [low, high]}}. Scores without an sd are fixed.
    """
    rng = np.random.default_rng(seed)
    base = asv.drivers()
    samples = {
        name: np.clip(rng.normal(base[name], float(asv.sd.get(name, 0.0)), n), 0.0, 1.0)
        for name in DRIVER_FIELDS
    }
    out = {}
    for pname, rule in RULES.items():
        vals = rule.baseline + sum(w * samples[s] for s, w in rule.weights.items())
        vals = np.clip(vals, *rule.bounds)
        lo, hi = np.percentile(vals, [2.5, 97.5])
        out[pname] = {"mean": round(float(vals.mean()), 4),
                      "ci95": [round(float(lo), 4), round(float(hi), 4)]}
    return out


def describe_rules() -> str:
    """Human-readable table of the current mapping assumptions."""
    lines = ["parameter | baseline | weights | bounds | rationale", "---|---|---|---|---"]
    for name, r in RULES.items():
        w = ", ".join(f"{k}: {v:+.2f}" for k, v in r.weights.items())
        lines.append(f"{name} | {r.baseline} | {w} | {r.bounds} | {r.rationale}")
    return "\n".join(lines)
