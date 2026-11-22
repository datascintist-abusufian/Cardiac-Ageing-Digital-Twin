# Cardiac Ageing Digital Twin

[![tests](https://github.com/datascintist-abusufian/Cardiac-Ageing-Digital-Twin/actions/workflows/ci.yml/badge.svg)](https://github.com/datascintist-abusufian/Cardiac-Ageing-Digital-Twin/actions/workflows/ci.yml)

**[▶ Open the live 3D heart viewer](https://datascintist-abusufian.github.io/Cardiac-Ageing-Digital-Twin/viewer/)**: move the age / senescence slider and watch conduction slow and fibrosis appear.

> A multiscale digital-twin concept that links **cardiomyocyte ageing**, measured with AI on
> human iPSC-derived cardiomyocytes, to **tissue and whole-heart behaviour**.
> Concept and design by **Md Abu Sufian**, PhD researcher, University of East London
> ([abusufian.dev](https://abusufian.dev)).

---

## Quick start

```bash
git clone https://github.com/datascintist-abusufian/Cardiac-Ageing-Digital-Twin.git
cd Cardiac-Ageing-Digital-Twin
pip install -r requirements.txt
python examples/young_vs_aged.py     # young vs aged simulation + figure
python examples/phase1_pipeline.py   # raw cell data -> ageing scores -> tissue
pytest -q                            # 16 tests
# 3D viewer: open viewer/index.html through any local web server, e.g.
python -m http.server 8000           # then visit http://localhost:8000/viewer/
```

```python
from cadt import AgeingStateVector, map_to_tissue, simulate

asv = AgeingStateVector(motion_dysfunction=0.7, nuclear_irregularity=0.6, calcium_decay_delay=0.65)
tissue = map_to_tissue(asv)          # ageing scores -> tissue parameters
result = simulate(tissue)            # 2D tissue simulation
print(result["metrics"])             # conduction velocity, APD, activation time, ...
```

### First result: young vs aged tissue

![Young vs aged activation maps](outputs/young_vs_aged.png)

| Measure (illustrative example) | Young | Aged | Change |
|---|---|---|---|
| Conduction velocity | 1.24 | 0.87 | −30% |
| Action potential duration | 22.9 | 27.6 | +21% |
| Time to activate the whole tissue | 95.1 | 141.5 | +49% |
| Fibrosis | 2% | 18% | — |

Units are dimensionless. The aged scores come from `examples/aged_state_example.json` and are
**illustrative, not measured**. The mapping weights are placeholder assumptions until
calibrated (roadmap phase 5).

### Phase 1 result: ageing scores from raw cell data

![Ageing State Vector from raw cell data](outputs/phase1_asv.png)

The phase-1 pipeline turns three kinds of raw recording into the three ageing scores:

| Recording | Features measured | Ageing score |
|---|---|---|
| Brightfield beating video | rhythm irregularity, contraction / relaxation speed, beat-to-beat amplitude variation | `motion_dysfunction` |
| Segmented nuclei (e.g. DAPI) | nuclear size and size spread, circularity, solidity | `nuclear_irregularity` |
| Calcium fluorescence trace | decay time constant (tau), time to 50 % decay | `calcium_decay_delay` |

Each score places a condition between a **young reference** (0) and an **aged positive
control** such as induced-senescence wells (1). Every score can be traced back to per-feature
z-scores (`outputs/phase1_report.json`). Senescence markers are never used to build the
scores, so they stay available as an independent check.

The figure uses **synthetic recordings** (`cadt/synthetic.py`) so the whole pipeline runs
without data. The extractors are classical, transparent baselines; learned models
(CardioFlowFormer for motion, NucAgeAI for nuclei) can replace them as long as they return
the same feature names.

### Phase 3: interactive 3D heart viewer

**[Open the live viewer](https://datascintist-abusufian.github.io/Cardiac-Ageing-Digital-Twin/viewer/)** (runs in the browser, nothing to install).

- **Age / senescence slider**, or set the three ageing scores separately.
- **Load pipeline output**: reads `outputs/phase1_aged_state.json` written by the phase-1 pipeline; **Open JSON…** loads any ageing-state file.
- **Beating** view: the activation wave spreads from apical sites, routes around fibrosis (grey), and the heart contracts; relaxation slows as calcium decay slows.
- **Activation map** view: colour shows when each point is activated; compare young and aged at the same scale.
- Live tables of tissue parameters and whole-heart results (total activation time, effective conduction, APD, fibrotic surface).
- Shareable states through the URL, e.g. [`?age=0.8&mode=map`](https://datascintist-abusufian.github.io/Cardiac-Ageing-Digital-Twin/viewer/?age=0.8&mode=map).

The viewer uses the same mapping rules as `cadt/mapping.py`; a test (`tests/test_viewer.py`) fails if
the two copies ever differ. The geometry is a generic, simplified biventricular shape, not
patient-specific.

### Repository layout

| Path | What it does |
|---|---|
| `cadt/ageing_state.py` | Ageing State Vector: validation, JSON load / save (`ageing-state/v0.1`) |
| `cadt/mapping.py` | Rule-based ageing → tissue mapping, with Monte Carlo uncertainty |
| `cadt/tissue_sim.py` | 2D Aliev–Panfilov tissue model with patchy fibrosis |
| `cadt/motion.py` | Beating-video features: rhythm, amplitude, contraction / relaxation speed |
| `cadt/nuclei.py` | Nuclear morphology from segmentation masks |
| `cadt/calcium.py` | Calcium-transient decay (tau, CD50) |
| `cadt/scoring.py` | Features → Ageing State Vector (young reference + positive-control anchor) |
| `cadt/synthetic.py` | Synthetic videos, nuclei and calcium traces for demos and tests |
| `examples/young_vs_aged.py` | Ageing scores → tissue → simulation demo |
| `examples/phase1_pipeline.py` | Raw cell data → features → ageing scores → tissue → simulation |
| `viewer/index.html` | Interactive 3D heart viewer (three.js), live on GitHub Pages |
| `tests/` | 16 tests: feature recovery, directions of change, bounds, JSON round-trip, simulation, viewer rules |

---

## 1. Motivation

Most heart digital twins and teaching models show an *adult, healthy* heart, plus a set of
arrhythmias such as atrial fibrillation (AF) and ventricular tachycardia (VT). Many of these
rhythm problems get more common with age, but standard models do not show **why**: the
cell-level changes that build up over decades and slowly change how the whole organ
conducts, contracts and fails.

This project builds a **cellular-ageing digital twin**: a multiscale model where an *ageing
state* measured or predicted at the level of single cardiomyocytes is carried up through
tissue properties into an interactive 3D heart.

**Core question:** *If we know how old and how senescent a population of cardiomyocytes
behaves, what should the heart built from them look like, conduct like and beat like?*

---

## 2. Multiscale architecture

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'primaryColor': '#ffffff', 'primaryBorderColor': '#333333', 'primaryTextColor': '#000000', 'lineColor': '#333333', 'clusterBkg': '#ffffff', 'clusterBorder': '#333333', 'background': '#ffffff'}}}%%
flowchart TB
    subgraph L1["Layer 1 — Cell (iPSC-derived cardiomyocytes)"]
        A1["Time-lapse brightfield video<br/>(beating motion)"]
        A2["Nuclear imaging<br/>(DAPI / morphology)"]
        A3["Calcium transients &<br/>field potentials (MEA)"]
        A4["Senescence markers<br/>p16, p21, SA-β-gal, γH2AX"]
    end

    subgraph AI["AI phenotyping"]
        B1["Motion encoder<br/>(CardioFlowFormer concept)"]
        B2["Nuclear phenotyping<br/>(NucAgeAI concept)"]
        B3["Fusion → Ageing State Vector (ASV)"]
    end

    subgraph L2["Layer 2 — Tissue"]
        C1["Conduction velocity"]
        C2["Action potential duration"]
        C3["Ca²⁺ handling / contractility"]
        C4["Fibrosis fraction & heterogeneity"]
    end

    subgraph L3["Layer 3 — Organ (interactive 3D heart)"]
        D1["Electrical propagation view"]
        D2["Remodelling & fibrosis overlays"]
        D3["Arrhythmia susceptibility"]
    end

    A1 --> B1
    A2 --> B2
    A3 --> B3
    A4 -. independent validation .-> B3
    B1 --> B3
    B2 --> B3
    B3 --> C1 & C2 & C3 & C4
    C1 & C2 --> D1
    C4 --> D2
    C1 & C2 & C4 --> D3

    classDef white fill:#ffffff,stroke:#333333,stroke-width:1px,color:#000000
    class A1,A2,A3,A4,B1,B2,B3,C1,C2,C3,C4,D1,D2,D3 white
    style L1 fill:#ffffff,stroke:#333333,color:#000000
    style AI fill:#ffffff,stroke:#333333,color:#000000
    style L2 fill:#ffffff,stroke:#333333,color:#000000
    style L3 fill:#ffffff,stroke:#333333,color:#000000
```

### Layer 1 — Cell
Human iPSC-derived cardiomyocytes (iPSC-CMs) imaged over time give a longitudinal view of
cellular ageing. Ageing can be modelled in vitro through long-term culture, induced
senescence (e.g. doxorubicin or irradiation) or progeroid models. Each readout is a
*phenotype*, not a label: the goal is to measure how cells behave, not just to tag them
"young" or "old".

### AI phenotyping → Ageing State Vector
Two model families turn raw images into numbers:

| Input | Model concept | What it extracts |
|---|---|---|
| Beating-motion video | **CardioFlowFormer** — spatiotemporal Transformer over optical-flow / motion signatures | Beat rate variability, contraction / relaxation velocity, synchrony, irregular beating |
| Nuclear images | **NucAgeAI** — hierarchy-aware nuclear phenotyping | Nuclear size, shape irregularity, texture / chromatin features, phenotype clusters |
| Ca²⁺ / MEA traces | Signal features (classical or learned) | Transient decay time, APD / field-potential duration proxies |

These are fused into an **Ageing State Vector (ASV)**: a small, interpretable set of
continuous scores. Senescence markers are kept **out of training** and used as an
*independent* check, so the twin is not just learning to copy a stain.

### Layer 2 — Tissue
The ASV is mapped onto tissue-level parameters that are known to shift with cardiac ageing:

| Tissue parameter | Typical direction with ageing | Main cellular driver |
|---|---|---|
| Conduction velocity | ↓ | Reduced / lateralised connexin-43, more fibrosis |
| Action potential duration | ↑ | Changes in repolarising K⁺ currents and Ca²⁺ handling |
| Ca²⁺ transient decay | Slower | Reduced SERCA2a activity |
| Contractile reserve | ↓ | Altered Ca²⁺ cycling, myocyte loss / hypertrophy |
| Fibrosis fraction & patchiness | ↑ | Fibroblast activation, collagen build-up |

The first version uses a **transparent, rule-based mapping** (documented coefficients, with
ranges taken from the literature). A learned mapping is only added once paired cell and
tissue data exist.

### Layer 3 — Organ (interactive 3D heart)
Tissue parameters drive an interactive 3D heart model (planned: a Python / web viewer, with an
option to export to a game engine for teaching). Ageing is shown through a small set of
organ-level controls: propagation speed of the activation wave, repolarisation timing,
regional remodelling, and fibrosis overlays. A learner or researcher can:

- move an **"age / senescence" slider** and watch conduction slow down and fibrotic patches appear;
- compare a *young* heart and an *aged* heart side by side running the same rhythm;
- see why aged tissue is more prone to **atrial fibrillation** and re-entry.

---

## 3. Interface contract (proposal)

A simple JSON file links the AI pipeline (Python) to the 3D organ viewer, so the two can be
developed separately:

```json
{
  "schema": "ageing-state/v0.1",
  "source": "iPSC-CM imaging",
  "timepoint_days": 60,
  "ageing_state_vector": {
    "motion_dysfunction": 0.42,
    "nuclear_irregularity": 0.37,
    "calcium_decay_delay": 0.30,
    "senescence_confidence": 0.55
  },
  "tissue_parameters": {
    "conduction_velocity_scale": 0.80,
    "apd_scale": 1.15,
    "fibrosis_fraction": 0.12,
    "fibrosis_heterogeneity": 0.40
  },
  "uncertainty": { "method": "ensemble", "ci95_width": 0.10 }
}
```

*All numbers above are illustrative placeholders, not results.*

The organ viewer reads this file and sets its matching visual and simulation parameters.

---

## 4. Validation plan

1. **Cell level:** ASV scores should go up with culture time and with induced senescence, and
   should agree with senescence markers that the models never saw during training.
2. **Tissue level:** predicted conduction / APD changes are checked against MEA or optical
   mapping data from the same cell lines, where available.
3. **Organ level (teaching):** clinicians and educators review whether the "aged heart"
   view matches textbook and clinical understanding of the ageing heart.
4. **Explainability:** every ASV score can be traced back to example cells / frames
   (attention maps, saliency, nearest-neighbour examples), so it is not a black box.

---

## 5. Known limitations

- **iPSC-CMs are immature** (closer to fetal than adult cardiomyocytes); in-vitro ageing is
  a model of ageing, not the real thing.
- Moving from a 2D cell culture to a 3D organ needs assumptions; the mapping in Layer 2 is a
  hypothesis to be tested, not an established law.
- The organ twin stays **generic**, not patient-specific, unless imaging data such as cardiac
  MRI are added later.
- **For education and research only — not for clinical use.**

---

## 6. Roadmap

| Phase | Goal | Status |
|---|---|---|
| 0 | This concept document; agree the JSON interface | ✅ Done |
| 1 | Python pipeline producing an ASV from iPSC-CM motion + nuclear images | 🟡 Classical pipeline done on synthetic data; real recordings and learned models next |
| 2 | Rule-based ASV → tissue parameter mapping with literature ranges | ✅ First version (placeholder weights) |
| 3 | Interactive 3D heart viewer with an "age / senescence" slider driven by the JSON file | ✅ [Live viewer](https://datascintist-abusufian.github.io/Cardiac-Ageing-Digital-Twin/viewer/) |
| 4 | Young-vs-aged comparison view with AF / VT re-entry scenarios | 🟡 Young-vs-aged comparison done (2D and 3D); re-entry scenarios next |
| 5 | Validation against held-out senescence markers and electrophysiology data | ⏳ Planned |
| 6 | Optional: patient-specific geometry from cardiac MRI segmentation | ⏳ Planned |

---

## 7. Author

**Concept and design:** Md Abu Sufian, PhD researcher, University of East London
([abusufian.dev](https://abusufian.dev))

---

© 2026 Md Abu Sufian. All rights reserved.
