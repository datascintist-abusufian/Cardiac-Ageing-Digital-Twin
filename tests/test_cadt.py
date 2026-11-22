import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cadt import (AgeingStateVector, load_state, map_to_tissue,  # noqa: E402
                  map_with_uncertainty, save_state, simulate)
from cadt.mapping import RULES  # noqa: E402

YOUNG = AgeingStateVector.young()
AGED = AgeingStateVector(0.8, 0.8, 0.8)


def test_scores_must_be_in_unit_range():
    with pytest.raises(ValueError):
        AgeingStateVector(motion_dysfunction=1.5)
    with pytest.raises(ValueError):
        AgeingStateVector(senescence_confidence=-0.1)


def test_young_state_gives_baseline_tissue():
    t = map_to_tissue(YOUNG)
    assert t["conduction_velocity_scale"] == 1.0
    assert t["apd_scale"] == 1.0
    assert t["contractility_scale"] == 1.0


def test_ageing_moves_parameters_in_expected_direction():
    y, a = map_to_tissue(YOUNG), map_to_tissue(AGED)
    assert a["conduction_velocity_scale"] < y["conduction_velocity_scale"]
    assert a["apd_scale"] > y["apd_scale"]
    assert a["calcium_decay_scale"] > y["calcium_decay_scale"]
    assert a["contractility_scale"] < y["contractility_scale"]
    assert a["fibrosis_fraction"] > y["fibrosis_fraction"]


def test_outputs_stay_within_bounds():
    worst = AgeingStateVector(1.0, 1.0, 1.0)
    for name, value in map_to_tissue(worst).items():
        lo, hi = RULES[name].bounds
        assert lo <= value <= hi


def test_senescence_markers_do_not_drive_mapping():
    a = AgeingStateVector(0.5, 0.5, 0.5, senescence_confidence=0.0)
    b = AgeingStateVector(0.5, 0.5, 0.5, senescence_confidence=1.0)
    assert map_to_tissue(a) == map_to_tissue(b)


def test_uncertainty_interval_contains_point_estimate():
    asv = AgeingStateVector(0.5, 0.5, 0.5, sd={"motion_dysfunction": 0.1})
    point = map_to_tissue(asv)
    ci = map_with_uncertainty(asv, n=500)
    for name, v in ci.items():
        assert v["ci95"][0] <= point[name] + 1e-9
        assert point[name] - 1e-9 <= v["ci95"][1]


def test_json_round_trip(tmp_path):
    asv = AgeingStateVector(0.3, 0.2, 0.1, senescence_confidence=0.4)
    path = tmp_path / "state.json"
    save_state(path, asv, map_to_tissue(asv), timepoint_days=30)
    loaded, doc = load_state(path)
    assert loaded.drivers() == asv.drivers()
    assert doc["timepoint_days"] == 30
    assert "tissue_parameters" in json.loads(path.read_text())


def test_aged_tissue_conducts_slower_with_longer_apd():
    y = simulate(map_to_tissue(YOUNG), n=60, t_end=90)["metrics"]
    a = simulate(map_to_tissue(AGED), n=60, t_end=90)["metrics"]
    assert a["conduction_velocity"] < y["conduction_velocity"]
    assert a["apd_median"] > y["apd_median"]
    assert a["fibrosis_fraction_actual"] > y["fibrosis_fraction_actual"]
