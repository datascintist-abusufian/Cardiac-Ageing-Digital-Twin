"""Phase 5a: literature-calibrated mapping and what it means for re-entry vulnerability.

1. Compare the placeholder and literature-calibrated tissue parameters at ASV = 1.
2. Re-run the phase-4 S1-S2 vulnerable-window test with the literature mapping:
   young vs aged (central literature values), and
3. propagate literature uncertainty: draw N aged rule sets with every target sampled
   uniformly from its literature range, and record the change in window width and
   window centre for each draw.

Run from the repository root (about 5 minutes; --quick for about 2 minutes;
--replot redraws the figure from an existing outputs/calibration_report.json):

    python examples/calibration_uncertainty.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from cadt import AgeingStateVector, map_to_tissue  # noqa: E402
from cadt.calibration import LITERATURE, LITERATURE_RULES, literature_table, sample_rules  # noqa: E402
from cadt.mapping import RULES  # noqa: E402
from cadt.reentry import vulnerable_window  # noqa: E402

OUT = ROOT / "outputs"
AGED = AgeingStateVector(1.0, 1.0, 1.0)      # aged positive-control level
YOUNG = AgeingStateVector(0.0, 0.0, 0.0)


def centre(w: dict) -> float:
    c, p = np.array(w["couplings"]), np.array(w["probability"])
    return float((c * p).sum() / p.sum()) if p.sum() else float("nan")


def main(quick: bool = False, replot: bool = False) -> None:
    OUT.mkdir(exist_ok=True)
    if replot and (OUT / "calibration_report.json").exists():
        rep = json.loads((OUT / "calibration_report.json").read_text(encoding="utf-8"))
        w = rep["windows"]
        plot(w["young"], w["aged_central"], rep["uncertainty_draws"], w["young_reference_for_draws"],
             OUT / "calibration_uncertainty.png")
        print(json.dumps(rep["summary"], indent=2))
        return
    couplings = np.arange(20, 58, 4 if quick else 2)
    seeds = (0,) if quick else (0, 1, 2)
    n_draws = 4 if quick else 10

    young_t = map_to_tissue(YOUNG, LITERATURE_RULES)
    aged_t = map_to_tissue(AGED, LITERATURE_RULES)
    young_w = vulnerable_window(young_t, couplings, seeds=seeds)
    aged_w = vulnerable_window(aged_t, couplings, seeds=seeds)
    print(f"literature young: width {young_w['width']}  range {young_w['earliest']}-{young_w['latest']}")
    print(f"literature aged : width {aged_w['width']}  range {aged_w['earliest']}-{aged_w['latest']}")

    # Draws use 2 fibrosis seeds; compare them with a young baseline run on the same seeds.
    young_ref = vulnerable_window(young_t, couplings, seeds=seeds[:2])
    rng = np.random.default_rng(2026)
    draws = []
    for i in range(n_draws):
        rules = sample_rules(rng)
        t = map_to_tissue(AGED, rules)
        w = vulnerable_window(t, couplings, seeds=seeds[:2])
        d = {"tissue": {k: round(v, 4) for k, v in t.items()}, "width": w["width"],
             "centre": round(centre(w), 2), "wavelength": w["wavelength"]}
        draws.append(d)
        print(f"draw {i + 1:2d}: cv {t['conduction_velocity_scale']:.2f} apd {t['apd_scale']:.2f} "
              f"fib {t['fibrosis_fraction']:.3f} -> width {w['width']:.1f} centre {d['centre']:.1f}")

    dw = np.array([d["width"] - young_ref["width"] for d in draws])
    dc = np.array([d["centre"] - centre(young_ref) for d in draws])
    summary = {
        "young_width": young_w["width"], "aged_central_width": aged_w["width"],
        "aged_central_width_change": round(aged_w["width"] - young_w["width"], 2),
        "aged_central_centre_shift": round(centre(aged_w) - centre(young_w), 2),
        "draws_young_reference_width": young_ref["width"],
        "draws_width_change": {"median": float(np.median(dw)), "min": float(dw.min()), "max": float(dw.max()),
                               "fraction_wider": float((dw > 0).mean()),
                               "fraction_narrower": float((dw < 0).mean())},
        "draws_centre_shift": {"median": float(np.median(dc)), "min": float(dc.min()), "max": float(dc.max())},
    }
    print("\nSummary:", json.dumps(summary, indent=2))

    report = {"note": "literature-calibrated mapping (phase 5a); model units; not a clinical risk estimate",
              "literature": literature_table(),
              "tissue_at_asv_1": {"placeholder": map_to_tissue(AGED), "literature_v1": aged_t},
              "windows": {"young": young_w, "aged_central": aged_w, "young_reference_for_draws": young_ref},
              "uncertainty_draws": draws, "summary": summary,
              "protocol": {"couplings": couplings.tolist(), "seeds": list(seeds), "draws": n_draws}}
    (OUT / "calibration_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    plot(young_w, aged_w, draws, young_ref, OUT / "calibration_uncertainty.png")
    print(f"Saved {OUT / 'calibration_uncertainty.png'} and calibration_report.json")


def plot(young_w: dict, aged_w: dict, draws: list, young_ref: dict, path: Path) -> None:
    fig = plt.figure(figsize=(13, 4.6), facecolor="white")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1, 0.8], wspace=0.45)

    # A: placeholder vs literature at ASV = 1, as % change from young
    ax = fig.add_subplot(gs[0])
    params = list(LITERATURE)
    labels = ["Conduction velocity", "APD", "Ca²⁺ decay", "Contractility", "Fibrosis (pp)"]
    ph = map_to_tissue(AGED)
    for i, p in enumerate(params):
        lit = LITERATURE[p]
        if p == "fibrosis_fraction":
            base = lit["young"]
            lo, hi, mid = [(x - base) * 100 for x in (*lit["range"], lit["target"])]
            phv = (ph[p] - RULES[p].baseline) * 100
        else:
            lo, hi, mid = [(x - 1) * 100 for x in (*lit["range"], lit["target"])]
            phv = (ph[p] - 1) * 100
        ax.plot([lo, hi], [i, i], color="#08519c", lw=6, alpha=0.35, solid_capstyle="butt")
        ax.plot(mid, i, "o", color="#08519c", ms=7, label="Literature (range, central)" if i == 0 else None)
        ax.plot(phv, i, "x", color="#b4361f", ms=8, mew=2, label="Placeholder" if i == 0 else None)
    ax.axvline(0, color="#999", lw=0.8)
    ax.set_yticks(range(len(params)), labels)
    ax.invert_yaxis()
    ax.set_xlabel("Change at ASV = 1 vs young (%; fibrosis in percentage points)")
    ax.set_title("A  Calibration: literature vs placeholder", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)

    # B: vulnerable window, literature young vs aged
    ax = fig.add_subplot(gs[1])
    for w, name, col in [(young_w, "Young", "#9ecae1"), (aged_w, "Aged (literature central)", "#08306b")]:
        ax.plot(w["couplings"], w["probability"], "-o", ms=3, color=col, label=name)
        ax.fill_between(w["couplings"], w["probability"], color=col, alpha=0.15)
    ax.set_xlabel("S2 coupling interval (model units)")
    ax.set_ylabel("P(sustained re-entry)")
    ax.set_ylim(-0.03, 1.05)
    ax.set_title("B  Vulnerable window (calibrated)", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
    ax.spines[["top", "right"]].set_visible(False)

    # C: uncertainty draws
    ax = fig.add_subplot(gs[2])
    dw = [d["width"] - young_ref["width"] for d in draws]
    dc = [d["centre"] - centre(young_ref) for d in draws]
    pts, counts = np.unique(np.round(np.c_[dc, dw], 2), axis=0, return_counts=True)
    ax.scatter(pts[:, 0], pts[:, 1], s=36 * counts, color="#4292c6", edgecolor="black", lw=0.5, zorder=3)
    for (x, y), k in zip(pts, counts):
        if k > 1:
            ax.annotate(f"×{k}", (x, y), xytext=(6, 4), textcoords="offset points", fontsize=8)
    ax.axhline(0, color="#999", lw=0.8)
    ax.axvline(0, color="#999", lw=0.8)
    ax.set_xlabel("Window centre shift (aged − young)")
    ax.set_ylabel("Window width change (aged − young)")
    ax.set_title(f"C  Literature uncertainty ({len(draws)} draws)", loc="left", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Phase 5a: literature-calibrated ageing mapping and re-entry vulnerability "
                 "(model units; mixed-species sources)", fontsize=11)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main(quick="--quick" in sys.argv, replot="--replot" in sys.argv)
