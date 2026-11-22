"""Ageing State Vector (ASV): the cell-level input to the digital twin.

The ASV holds a few interpretable scores in [0, 1], produced by AI phenotyping of
iPSC-derived cardiomyocytes (motion, nuclear morphology, calcium handling).

`senescence_confidence` is kept separate on purpose: it comes from senescence
markers (p16, p21, SA-beta-gal, gammaH2AX) and is used only to *check* the twin,
never to drive it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

SCHEMA_VERSION = "ageing-state/v0.1"

# Scores that drive the tissue mapping.
DRIVER_FIELDS = ("motion_dysfunction", "nuclear_irregularity", "calcium_decay_delay")


def _check_unit(name: str, value: float) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}")
    return value


@dataclass
class AgeingStateVector:
    """Cell-level ageing scores, each in [0, 1] (0 = young-like, 1 = most aged)."""

    motion_dysfunction: float = 0.0
    nuclear_irregularity: float = 0.0
    calcium_decay_delay: float = 0.0
    # Validation only: never used by the mapping.
    senescence_confidence: Optional[float] = None
    # Optional per-score standard deviations, for uncertainty propagation.
    sd: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in DRIVER_FIELDS:
            setattr(self, name, _check_unit(name, getattr(self, name)))
        if self.senescence_confidence is not None:
            self.senescence_confidence = _check_unit(
                "senescence_confidence", self.senescence_confidence
            )
        for key, value in self.sd.items():
            if key not in DRIVER_FIELDS:
                raise ValueError(f"sd given for unknown score '{key}'")
            if float(value) < 0:
                raise ValueError(f"sd for {key} must be >= 0")

    def drivers(self) -> dict:
        return {name: getattr(self, name) for name in DRIVER_FIELDS}

    @classmethod
    def young(cls) -> "AgeingStateVector":
        return cls(0.0, 0.0, 0.0)

    @classmethod
    def from_dict(cls, data: dict) -> "AgeingStateVector":
        asv = data.get("ageing_state_vector", data)
        return cls(
            motion_dysfunction=asv.get("motion_dysfunction", 0.0),
            nuclear_irregularity=asv.get("nuclear_irregularity", 0.0),
            calcium_decay_delay=asv.get("calcium_decay_delay", 0.0),
            senescence_confidence=asv.get("senescence_confidence"),
            sd=asv.get("sd", {}),
        )

    def to_dict(self) -> dict:
        out = asdict(self)
        if not out["sd"]:
            out.pop("sd")
        return out


def load_state(path: str | Path) -> tuple[AgeingStateVector, dict]:
    """Load an ageing-state JSON file. Returns (ASV, full document)."""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    schema = doc.get("schema")
    if schema is not None and schema != SCHEMA_VERSION:
        raise ValueError(f"Unsupported schema '{schema}', expected '{SCHEMA_VERSION}'")
    return AgeingStateVector.from_dict(doc), doc


def save_state(path: str | Path, asv: AgeingStateVector, tissue: dict | None = None,
               **meta) -> None:
    """Write an ageing-state JSON file (the interface read by the organ viewer)."""
    doc = {"schema": SCHEMA_VERSION, **meta, "ageing_state_vector": asv.to_dict()}
    if tissue is not None:
        doc["tissue_parameters"] = tissue
    Path(path).write_text(json.dumps(doc, indent=2), encoding="utf-8")
