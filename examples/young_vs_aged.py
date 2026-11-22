"""Young vs aged heart tissue: from Ageing State Vector to activation maps.

Run from the repository root:

    python examples/young_vs_aged.py

Writes:
    outputs/young_vs_aged.png     activation-time maps + fibrosis
    outputs/aged_state.json       ageing-state file (interface for the organ viewer)
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from cadt import (AgeingStateVector, load_state, map_to_tissue,  # noqa: E402
                  map_with_uncertainty, save_state, simulate)

OUT = ROOT / "outputs"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    aged_asv, _ = load_state(ROOT / "examples" / "aged_state_example.json")
    cases = {"Young": AgeingStateVector.young(), "Aged": aged_asv}

    results = {}
    for name, asv in cases.items():
        tissue = map_to_tissue(asv)
        results[name] = (tissue, simulate(tissue))
        m = results[name][1]["metrics"]
        print(f"\n{name}")
        print("  tissue :", tissue)
        print("  metrics:", m)

    y, a = results["Young"][1]["metrics"], results["Aged"][1]["metrics"]
    print("\nAged vs young:")
    print(f"  conduction velocity  {100 * (a['conduction_velocity'] / y['conduction_velocity'] - 1):+.1f}%")
    print(f"  action potential dur {100 * (a['apd_median'] / y['apd_median'] - 1):+.1f}%")
    print(f"  total activation time {100 * (a['total_activation_time'] / y['total_activation_time'] - 1):+.1f}%")

    print("\nAged tissue parameters with uncertainty (Monte Carlo, 95% interval):")
    for k, v in map_with_uncertainty(aged_asv).items():
        print(f"  {k:28s} {v['mean']:.3f}  [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}]")

    save_state(OUT / "aged_state.json", aged_asv, results["Aged"][0],
               source="example (illustrative scores)")
    plot(results, OUT / "young_vs_aged.png")
    print(f"\nSaved {OUT / 'young_vs_aged.png'} and {OUT / 'aged_state.json'}")


def plot(results: dict, path: Path) -> None:
    vmax = max(np.nanmax(r[1]["activation_time"]) for r in results.values())
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4.6),
                             facecolor="white", constrained_layout=True)
    for ax, (name, (tissue, res)) in zip(axes, results.items()):
        act = np.ma.masked_invalid(res["activation_time"])
        im = ax.imshow(act, cmap="viridis", vmin=0, vmax=vmax, origin="lower")
        ax.contour(res["activation_time"], levels=np.arange(10, vmax, 10),
                   colors="white", linewidths=0.6, origin="lower")
        fib = np.ma.masked_where(~res["fibrosis"], res["fibrosis"])
        ax.imshow(fib, cmap="Greys", vmin=0, vmax=1.2, origin="lower", interpolation="nearest")
        m = res["metrics"]
        ax.set_title(f"{name}\nCV {m['conduction_velocity']:.2f}  |  APD {m['apd_median']:.1f}  |  "
                     f"fibrosis {100 * tissue['fibrosis_fraction']:.0f}%", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes, shrink=0.85, label="activation time (dimensionless)")
    fig.suptitle("Plane wave from the left edge: young vs aged tissue (grey = fibrosis)")
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
