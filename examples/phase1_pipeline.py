"""Phase 1 end to end: raw cell data -> features -> Ageing State Vector -> tissue -> simulation.

Uses SYNTHETIC recordings (cadt.synthetic) so it runs without any real data.
To use real data, replace the `load_*` helpers with your own loaders that return:
    video  : np.ndarray (T, H, W) brightfield frames + fps
    mask   : np.ndarray (H, W) labelled or binary nuclear segmentation
    trace  : np.ndarray (T,) calcium fluorescence + fps

Run from the repository root:

    python examples/phase1_pipeline.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from cadt import map_to_tissue, save_state, simulate  # noqa: E402
from cadt.calcium import calcium_features  # noqa: E402
from cadt.motion import motion_features  # noqa: E402
from cadt.nuclei import nuclear_features  # noqa: E402
from cadt.scoring import YoungReference, build_asv, explain  # noqa: E402
from cadt import synthetic  # noqa: E402

OUT = ROOT / "outputs"
VIDEO_FPS, CA_FPS = 30.0, 100.0


def load_sample(ageing: float, seed: int) -> dict:
    """One 'well': video + nuclear mask + calcium trace -> merged feature dict."""
    feats = {}
    feats.update(motion_features(synthetic.beating_video(ageing, fps=VIDEO_FPS, seed=seed), VIDEO_FPS))
    feats.update(nuclear_features(synthetic.nuclear_mask(ageing, seed=seed)))
    feats.update(calcium_features(synthetic.calcium_trace(ageing, fps=CA_FPS, seed=seed), CA_FPS))
    return feats


def main() -> None:
    OUT.mkdir(exist_ok=True)

    # 1. References (separate seeds from the test wells):
    #    young = early culture; aged anchor = positive control (e.g. induced senescence).
    reference = YoungReference.fit(
        [load_sample(0.0, seed) for seed in range(100, 110)],
        aged_anchor=[load_sample(1.0, seed) for seed in range(200, 210)],
    )

    # 2. Test conditions: 5 wells each, e.g. culture day 30 / 60 / 90.
    conditions = {"day-30 (synthetic)": 0.0, "day-60 (synthetic)": 0.35, "day-90 (synthetic)": 0.7}
    report = {"note": "synthetic data, for pipeline demonstration only",
              "young_reference": reference.to_dict(), "conditions": {}}
    asvs = {}
    for name, level in conditions.items():
        samples = [load_sample(level, seed) for seed in range(5)]
        asv, _ = build_asv(samples, reference)
        asvs[name] = asv
        mean_feats = {k: float(np.nanmean([s[k] for s in samples])) for k in samples[0]}
        report["conditions"][name] = {
            "ageing_state_vector": asv.to_dict(),
            "feature_z_scores": explain(mean_feats, reference),
            "tissue_parameters": map_to_tissue(asv),
        }
        print(f"\n{name}")
        for k, v in asv.drivers().items():
            print(f"  {k:22s} {v:.2f} ± {asv.sd[k]:.2f}")

    # 3. Aged condition -> tissue -> simulation.
    aged = asvs["day-90 (synthetic)"]
    tissue = map_to_tissue(aged)
    young_m = simulate(map_to_tissue(asvs["day-30 (synthetic)"]))["metrics"]
    aged_m = simulate(tissue)["metrics"]
    print("\nTissue simulation, day-90 vs day-30 (synthetic):")
    print(f"  conduction velocity {100 * (aged_m['conduction_velocity'] / young_m['conduction_velocity'] - 1):+.1f}%")
    print(f"  action potential    {100 * (aged_m['apd_median'] / young_m['apd_median'] - 1):+.1f}%")
    report["simulation"] = {"day-30": young_m, "day-90": aged_m}

    (OUT / "phase1_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    save_state(OUT / "phase1_aged_state.json", aged, tissue, source="synthetic phase-1 pipeline")
    plot(asvs, OUT / "phase1_asv.png")
    print(f"\nSaved {OUT / 'phase1_asv.png'}, phase1_report.json, phase1_aged_state.json")


def plot(asvs: dict, path: Path) -> None:
    scores = ["motion_dysfunction", "nuclear_irregularity", "calcium_decay_delay"]
    labels = ["Motion\ndysfunction", "Nuclear\nirregularity", "Calcium\ndecay delay"]
    colors = ["#9ecae1", "#4292c6", "#08519c"]
    x = np.arange(len(scores))
    w = 0.26
    fig, ax = plt.subplots(figsize=(7.5, 4.2), facecolor="white")
    for i, (name, asv) in enumerate(asvs.items()):
        vals = [getattr(asv, s) for s in scores]
        errs = [asv.sd[s] for s in scores]
        ax.bar(x + (i - 1) * w, vals, w, yerr=errs, capsize=3, label=name.split(" ")[0],
               color=colors[i], edgecolor="black", linewidth=0.5)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Ageing score (0 = young-like, 1 = most aged)")
    ax.set_title("Phase 1: Ageing State Vector from raw cell data\n(synthetic data; mean ± sd of 5 wells; 1 = positive-control level)",
                 fontsize=10)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
