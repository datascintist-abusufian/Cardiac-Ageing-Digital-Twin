import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cadt import synthetic  # noqa: E402
from cadt.calcium import calcium_features  # noqa: E402
from cadt.motion import motion_features  # noqa: E402
from cadt.nuclei import nuclear_features  # noqa: E402
from cadt.scoring import YoungReference, build_asv, score_sample  # noqa: E402


def sample(ageing, seed):
    f = {}
    f.update(motion_features(synthetic.beating_video(ageing, duration_s=8, seed=seed), 30))
    f.update(nuclear_features(synthetic.nuclear_mask(ageing, n=16, size=200, seed=seed)))
    f.update(calcium_features(synthetic.calcium_trace(ageing, seed=seed), 100))
    return f


def test_motion_recovers_beat_rate():
    m = motion_features(synthetic.beating_video(0.0, duration_s=10, seed=3), 30)
    assert m["beat_rate_hz"] == pytest.approx(1.0, abs=0.15)


def test_motion_gets_worse_with_ageing():
    y = motion_features(synthetic.beating_video(0.0, seed=1), 30)
    a = motion_features(synthetic.beating_video(1.0, seed=1), 30)
    assert a["interval_cv"] > y["interval_cv"]
    assert a["contraction_velocity"] < y["contraction_velocity"]
    assert a["relaxation_velocity"] < y["relaxation_velocity"]


def test_circle_is_round_and_convex():
    yy, xx = np.mgrid[0:80, 0:80]
    mask = (yy - 40) ** 2 + (xx - 40) ** 2 <= 20 ** 2
    f = nuclear_features(mask)
    assert f["n_nuclei"] == 1
    assert f["circularity_mean"] > 0.9
    assert f["solidity_mean"] > 0.9


def test_nuclei_enlarge_and_lose_shape_with_ageing():
    y = nuclear_features(synthetic.nuclear_mask(0.0, seed=2))
    a = nuclear_features(synthetic.nuclear_mask(1.0, seed=2))
    assert a["area_mean"] > y["area_mean"]
    assert a["circularity_mean"] < y["circularity_mean"]


def test_calcium_recovers_decay_constant():
    trace = synthetic.calcium_trace(0.0, seed=4)   # true tau about 0.12 s
    assert calcium_features(trace, 100)["decay_tau_s"] == pytest.approx(0.12, rel=0.15)


def test_scores_are_graded_between_references():
    ref = YoungReference.fit([sample(0.0, s) for s in range(100, 104)],
                             aged_anchor=[sample(1.0, s) for s in range(200, 204)])
    # average of 3 wells per condition, as in the pipeline (single wells are noisy)
    young, mid, aged = (build_asv([sample(level, s) for s in range(3)], ref)[0].drivers()
                        for level in (0.0, 0.5, 1.0))
    for k in young:
        assert young[k] < mid[k] < aged[k] + 1e-9
        assert 0.0 <= young[k] <= 1.0 and 0.0 <= aged[k] <= 1.0
    assert score_sample(sample(0.0, 1), ref)["nuclear_irregularity"] < 0.2


def test_build_asv_reports_spread_and_keeps_markers_separate():
    ref = YoungReference.fit([sample(0.0, s) for s in range(100, 104)])
    asv, details = build_asv([sample(0.5, s) for s in range(3)], ref, senescence_confidence=0.6)
    assert set(asv.sd) == {"motion_dysfunction", "nuclear_irregularity", "calcium_decay_delay"}
    assert asv.senescence_confidence == 0.6
    assert len(details["per_sample_scores"]) == 3
