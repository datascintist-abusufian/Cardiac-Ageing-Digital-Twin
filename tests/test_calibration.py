import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cadt import AgeingStateVector, map_to_tissue  # noqa: E402
from cadt.calibration import LITERATURE, LITERATURE_RULES, SOURCES, sample_rules  # noqa: E402
from cadt.mapping import RULES  # noqa: E402

ONES = AgeingStateVector(1.0, 1.0, 1.0)
ZEROS = AgeingStateVector(0.0, 0.0, 0.0)


def test_aged_positive_control_hits_literature_targets():
    t = map_to_tissue(ONES, LITERATURE_RULES)
    for name, lit in LITERATURE.items():
        assert t[name] == pytest.approx(lit["target"], abs=1e-3)


def test_young_is_baseline():
    t = map_to_tissue(ZEROS, LITERATURE_RULES)
    assert t["conduction_velocity_scale"] == 1.0
    assert t["fibrosis_fraction"] == pytest.approx(LITERATURE["fibrosis_fraction"]["young"])


def test_sampled_rules_stay_in_literature_ranges():
    rng = np.random.default_rng(0)
    for _ in range(20):
        t = map_to_tissue(ONES, sample_rules(rng))
        for name, lit in LITERATURE.items():
            lo, hi = lit["range"]
            assert lo - 1e-6 <= t[name] <= hi + 1e-6


def test_every_claim_has_a_source():
    for lit in LITERATURE.values():
        assert lit["evidence"]
        for key, finding in lit["evidence"]:
            assert key in SOURCES and finding


def test_placeholder_mapping_unchanged_by_default():
    assert map_to_tissue(ONES) == map_to_tissue(ONES, RULES)
