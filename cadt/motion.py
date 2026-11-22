"""Beating-motion features from brightfield video of iPSC-derived cardiomyocytes.

Classical, transparent baseline. A learned motion encoder (the CardioFlowFormer concept)
can later replace `motion_features` as long as it returns the same feature names.

Input: video as a NumPy array of shape (T, H, W), any numeric type, plus frames per second.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

MOTION_FEATURES = (
    "beat_rate_hz", "interval_cv", "amplitude", "amplitude_cv",
    "contraction_velocity", "relaxation_velocity",
)


def deformation_signal(video: np.ndarray, smooth_frames: int = 3) -> np.ndarray:
    """Per-frame deformation relative to the resting state.

    The resting frame is the per-pixel median over time (cells spend most of each
    cycle at rest). The signal is the mean absolute difference from that frame,
    normalised by the frame's intensity spread, then lightly smoothed.
    """
    video = np.asarray(video, dtype=float)
    if video.ndim != 3 or video.shape[0] < 10:
        raise ValueError("video must have shape (T, H, W) with T >= 10")
    rest = np.median(video, axis=0)
    scale = rest.std() + 1e-9
    d = np.abs(video - rest).mean(axis=(1, 2)) / scale
    if smooth_frames > 1:
        d = uniform_filter1d(d, smooth_frames)
    return d - np.percentile(d, 10)


def motion_features(video: np.ndarray, fps: float, min_interval_s: float = 0.3) -> dict:
    """Beat rate, rhythm irregularity, amplitude and contraction / relaxation speed."""
    d = deformation_signal(video)
    prom = 0.3 * (np.percentile(d, 99) - np.percentile(d, 5))
    peaks, _ = find_peaks(d, prominence=max(prom, 1e-9),
                          distance=max(1, int(min_interval_s * fps)))
    duration = len(d) / fps
    nan = float("nan")
    if len(peaks) < 3:
        return {"beat_rate_hz": len(peaks) / duration, "interval_cv": nan, "amplitude": nan,
                "amplitude_cv": nan, "contraction_velocity": nan, "relaxation_velocity": nan,
                "n_beats": int(len(peaks))}

    intervals = np.diff(peaks) / fps
    amps = d[peaks]
    vel = np.gradient(d) * fps
    contr, relax = [], []
    half = max(2, int(0.25 * np.median(np.diff(peaks))))
    for p in peaks:
        lo, hi = max(0, p - half), min(len(d), p + half + 1)
        contr.append(vel[lo:p + 1].max())
        relax.append(-vel[p:hi].min())
    return {
        "beat_rate_hz": float(len(peaks) / duration),
        "interval_cv": float(intervals.std() / intervals.mean()),
        "amplitude": float(amps.mean()),
        "amplitude_cv": float(amps.std() / (amps.mean() + 1e-12)),
        "contraction_velocity": float(np.mean(contr)),
        "relaxation_velocity": float(np.mean(relax)),
        "n_beats": int(len(peaks)),
    }
