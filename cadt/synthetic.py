"""Synthetic iPSC-cardiomyocyte data for demos and tests.

Everything here is *simulated* so the pipeline can be run end to end before real
recordings are added. `ageing` in [0, 1] controls how aged the synthetic cells look;
the effects follow the expected directions (weaker, slower, less regular beating;
larger, less regular nuclei; slower Ca2+ decay). None of it is real data.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates


def _beat_times(duration, rate_hz, irregularity, rng):
    times, t = [], 0.3
    while t < duration - 0.5:
        times.append(t)
        t += (1.0 / rate_hz) * max(0.4, 1 + irregularity * rng.standard_normal())
    return np.array(times)


def beating_video(ageing: float = 0.0, duration_s: float = 10.0, fps: float = 30.0,
                  size: int = 64, seed: int = 0) -> np.ndarray:
    """Textured cell sheet that contracts radially on each beat. Shape (T, size, size)."""
    rng = np.random.default_rng(seed)
    texture = gaussian_filter(rng.standard_normal((size, size)), 1.5)
    rate = 1.0 * (1 - 0.15 * ageing)
    beats = _beat_times(duration_s, rate, 0.04 + 0.30 * ageing, rng)
    rise = 0.08 + 0.10 * ageing            # contraction time (s)
    decay = 0.15 + 0.20 * ageing           # relaxation time constant (s)
    t = np.arange(int(duration_s * fps)) / fps
    d = np.zeros_like(t)
    for b in beats:
        amp = (1 - 0.4 * ageing) * max(0.2, 1 + (0.05 + 0.3 * ageing) * rng.standard_normal())
        dt = t - b
        up = (dt >= 0) & (dt < rise)
        down = dt >= rise
        d[up] = np.maximum(d[up], amp * np.sin(0.5 * np.pi * dt[up] / rise))
        d[down] = np.maximum(d[down], amp * np.exp(-(dt[down] - rise) / decay))
    yy, xx = np.mgrid[0:size, 0:size].astype(float)
    c = (size - 1) / 2
    frames = []
    for di in d:
        s = 1 + 0.06 * di                   # sample farther out = contraction toward centre
        coords = [c + (yy - c) * s, c + (xx - c) * s]
        frames.append(map_coordinates(texture, coords, order=1, mode="reflect"))
    video = np.array(frames) + 0.01 * rng.standard_normal((len(t), size, size))
    return video


def nuclear_mask(ageing: float = 0.0, n: int = 36, size: int = 300, seed: int = 0) -> np.ndarray:
    """Labelled mask of nuclei on a jittered grid. Aged nuclei are larger and lobulated."""
    rng = np.random.default_rng(seed)
    labels = np.zeros((size, size), dtype=np.int32)
    g = int(np.ceil(np.sqrt(n)))
    cell = size / g
    yy, xx = np.mgrid[0:size, 0:size].astype(float)
    for k in range(n):
        cy = (k // g + 0.5) * cell + rng.uniform(-3, 3)
        cx = (k % g + 0.5) * cell + rng.uniform(-3, 3)
        r0 = 9 * (1 + 0.45 * ageing) * rng.uniform(1 - 0.08 - 0.2 * ageing, 1 + 0.08 + 0.2 * ageing)
        ecc = rng.uniform(1.0, 1.25)
        rot = rng.uniform(0, np.pi)
        dy, dx = yy - cy, xx - cx
        u = dx * np.cos(rot) + dy * np.sin(rot)
        v = -dx * np.sin(rot) + dy * np.cos(rot)
        theta = np.arctan2(v, u)
        rad = np.hypot(u / ecc, v)
        bumps = sum(rng.uniform(0, 0.12 * ageing) * np.cos(h * theta + rng.uniform(0, 2 * np.pi))
                    for h in (3, 4, 5, 7))
        inside = rad <= r0 * (1 + bumps)
        labels[inside & (labels == 0)] = k + 1
    return labels


def calcium_trace(ageing: float = 0.0, duration_s: float = 10.0, fps: float = 100.0,
                  seed: int = 0) -> np.ndarray:
    """Fluorescence trace: fast upstroke, exponential decay. Aged = slower decay."""
    rng = np.random.default_rng(seed)
    tau = 0.12 * (1 + 0.8 * ageing) * rng.uniform(0.93, 1.07)
    beats = _beat_times(duration_s, 1.0 * (1 - 0.15 * ageing), 0.04 + 0.2 * ageing, rng)
    t = np.arange(int(duration_s * fps)) / fps
    y = np.ones_like(t)
    for b in beats:
        dt = t - b
        rise = (dt >= 0) & (dt < 0.03)
        fall = dt >= 0.03
        y[rise] += dt[rise] / 0.03
        y[fall] += np.exp(-(dt[fall] - 0.03) / tau)
    return y + 0.01 * rng.standard_normal(len(t))
