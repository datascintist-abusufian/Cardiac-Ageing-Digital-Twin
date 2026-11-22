import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cadt import AgeingStateVector, map_to_tissue  # noqa: E402
from cadt.reentry import s1s2, wavelength  # noqa: E402

YOUNG = map_to_tissue(AgeingStateVector(0, 0, 0))
AGED = map_to_tissue(AgeingStateVector(0.8, 0.8, 0.8))


def test_premature_beat_in_window_starts_sustained_reentry():
    r = s1s2(YOUNG, coupling=30, snapshot_times=(10,))
    assert r["sustained"]
    assert 10 in r["snapshots"]


def test_too_early_or_too_late_beat_does_not_reenter():
    assert not s1s2(YOUNG, coupling=20)["sustained"]   # S2 falls in refractory tissue
    assert not s1s2(YOUNG, coupling=70)["sustained"]   # tissue recovered: wave just spreads out


def test_ageing_shortens_wavelength():
    assert wavelength(AGED) < wavelength(YOUNG)
