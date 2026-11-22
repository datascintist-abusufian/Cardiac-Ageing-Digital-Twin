"""Calcium-transient features (e.g. Fluo-4 / GCaMP fluorescence traces).

Input: 1D trace (F or dF/F0) sampled at `fps` frames per second.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

CALCIUM_FEATURES = ("decay_tau_s", "cd50_s", "ca_amplitude")


def calcium_features(trace: np.ndarray, fps: float, min_interval_s: float = 0.3) -> dict:
    """Decay time constant (tau), time from peak to 50 % decay (CD50) and amplitude.

    tau is estimated per transient by a log-linear fit of the decay between 80 % and
    30 % of the peak above baseline, then averaged across transients.
    """
    y = np.asarray(trace, dtype=float)
    base = np.percentile(y, 5)
    y = y - base
    prom = 0.3 * (np.percentile(y, 99) - np.percentile(y, 5))
    peaks, _ = find_peaks(y, prominence=max(prom, 1e-9),
                          distance=max(1, int(min_interval_s * fps)))
    taus, cd50s, amps = [], [], []
    bounds = list(peaks[1:]) + [len(y)]
    for p, nxt in zip(peaks, bounds):
        seg = y[p:nxt]
        amp = seg[0]
        if amp <= 0 or len(seg) < 4:
            continue
        below50 = np.nonzero(seg <= 0.5 * amp)[0]
        if len(below50):
            cd50s.append(below50[0] / fps)
        win = np.nonzero((seg <= 0.8 * amp) & (seg >= 0.3 * amp))[0]
        if len(win) >= 3:
            # first contiguous run only (avoid the next upstroke)
            run = win[: np.argmax(np.diff(win) > 1) + 1] if np.any(np.diff(win) > 1) else win
            if len(run) >= 3:
                slope = np.polyfit(run / fps, np.log(seg[run]), 1)[0]
                if slope < 0:
                    taus.append(-1.0 / slope)
        amps.append(amp)
    nan = float("nan")
    return {
        "decay_tau_s": float(np.mean(taus)) if taus else nan,
        "cd50_s": float(np.mean(cd50s)) if cd50s else nan,
        "ca_amplitude": float(np.mean(amps)) if amps else nan,
        "n_transients": len(amps),
    }
