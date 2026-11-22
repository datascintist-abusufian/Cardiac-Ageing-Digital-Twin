"""Literature calibration of the ageing -> tissue mapping (roadmap phase 5a).

The placeholder rules in `mapping.RULES` were chosen only to give plausible directions.
Here each tissue parameter is anchored to published young-vs-old measurements, so that
an Ageing State Vector of 1 on every score (the aged positive-control level) maps to the
change seen in old animals, with a literature range rather than a single number.

Calibration keeps *which* scores drive each parameter (the relative weights of the
placeholder rules) and rescales them so that all-ones reaches the literature target.

Caveats: values come from different species (rat, rabbit, mouse), tissues and methods;
they describe old animals, not iPSC-derived cardiomyocytes. Ranges are the spread
between studies, not formal confidence intervals. `fibrosis_heterogeneity` has no
quantitative source and keeps its placeholder rule.
"""

from __future__ import annotations

import numpy as np

from .mapping import RULES, Rule

# --- Literature table -------------------------------------------------------------
# young / old values as reported; "target" = old/young ratio (or absolute fraction for
# fibrosis) used for calibration, with a [low, high] range across studies.
SOURCES = {
    "cooper2012": ("Cooper LL, Odening KE, et al. Electromechanical and structural alterations in the "
                   "aging rabbit heart and aorta. Am J Physiol Heart Circ Physiol 2012;302:H1625-35. "
                   "doi:10.1152/ajpheart.00960.2011"),
    "howlett2022": ("Howlett LA. The age-related reduction in adrenergic response as a limitation of "
                    "cardiac function in old age. PhD thesis, University of Leeds, 2022. "
                    "https://etheses.whiterose.ac.uk/32273/"),
    "ajp2009": ("Intracellular Ca2+ decline in aged and adult rat hearts (Table 1). Am J Physiol Heart "
                "Circ Physiol 2009;297(5):H1594-H1605. doi:10.1152/ajpheart.00579.2009"),
    "jp2017": ("Ca2+ transients in adult vs aged mouse ventricular myocytes (decay rates not affected). "
               "J Physiol 2017;595(12):3721-3742. doi:10.1113/JP274134"),
    "scirep2020": ("Ageing in C57BL/6 mouse ventricular myocytes: contraction amplitudes similar with age. "
                   "Sci Rep 2020;10:10052. doi:10.1038/s41598-020-66903-z"),
}

LITERATURE = {
    "conduction_velocity_scale": {
        "target": 0.90, "range": (0.80, 1.00),
        "evidence": [
            ("cooper2012", "rabbit LV epicardium, 6-9 mo vs 3.5-6 yr: transverse CV 0.348 -> 0.280 m/s "
                           "(-20%); longitudinal 0.767 -> 0.828 (no slowing)"),
            ("cooper2012", "isotropic-equivalent sqrt(CVL x CVT): 0.517 -> 0.481 (-7%)"),
        ],
    },
    "apd_scale": {
        "target": 1.15, "range": (1.00, 1.29),
        "evidence": [
            ("howlett2022", "rat ventricular myocytes, 3 mo vs 22-23 mo: APD90 +29%"),
            ("cooper2012", "rabbit LV optical APD: 210 vs 206 ms (no change)"),
        ],
    },
    "calcium_decay_scale": {
        "target": 1.08, "range": (1.00, 1.17),
        "evidence": [
            ("ajp2009", "rat LV, Ca2+ decline tau: adult 71-73 ms vs aged 74-84 ms (+2% to +17%; mean +8%)"),
            ("jp2017", "aged mouse myocytes: decay rate not affected"),
        ],
    },
    "contractility_scale": {
        "target": 1.00, "range": (0.95, 1.00),
        "evidence": [
            ("scirep2020", "mouse ~12 vs ~24 mo: cell-shortening amplitude similar"),
            ("cooper2012", "rabbit ejection fraction 65.3% vs 64.2% (unchanged)"),
        ],
    },
    "fibrosis_fraction": {
        "young": 0.028, "target": 0.070, "range": (0.050, 0.110),
        "evidence": [
            ("cooper2012", "rabbit LV + septum interstitial fibrosis 2.82% -> 6.97%; total 5.60% -> 11.01%"),
        ],
    },
}


def _rescaled(name: str, target: float, baseline: float | None = None) -> Rule:
    """Placeholder rule with weights rescaled so all-ones scores reach `target`."""
    base_rule = RULES[name]
    baseline = base_rule.baseline if baseline is None else baseline
    total = sum(base_rule.weights.values())
    delta = target - baseline
    weights = {k: round(w * delta / total, 6) for k, w in base_rule.weights.items()} if total else {}
    lo, hi = sorted((baseline, target))
    return Rule(baseline, weights, (round(lo, 6), round(hi, 6)),
                base_rule.rationale + " [literature-calibrated]")


def calibrated_rules(targets: dict | None = None) -> dict[str, Rule]:
    """Rule set anchored to the literature (central targets unless `targets` overrides)."""
    targets = targets or {}
    rules = {}
    for name, rule in RULES.items():
        if name not in LITERATURE:
            rules[name] = rule                       # no quantitative source: keep placeholder
            continue
        lit = LITERATURE[name]
        rules[name] = _rescaled(name, targets.get(name, lit["target"]), lit.get("young"))
    return rules


def sample_rules(rng: np.random.Generator) -> dict[str, Rule]:
    """One rule set with each target drawn uniformly from its literature range."""
    return calibrated_rules({n: rng.uniform(*lit["range"]) for n, lit in LITERATURE.items()})


def literature_table() -> list[dict]:
    """Rows for reports: parameter, target, range, evidence (with full references)."""
    rows = []
    for name, lit in LITERATURE.items():
        rows.append({
            "parameter": name,
            "target_at_asv_1": lit["target"],
            "range": list(lit["range"]),
            "young": lit.get("young", RULES[name].baseline),
            "evidence": [{"source": SOURCES[k], "finding": f} for k, f in lit["evidence"]],
        })
    return rows


LITERATURE_RULES = calibrated_rules()
RULESETS = {"placeholder": RULES, "literature_v1": LITERATURE_RULES}
