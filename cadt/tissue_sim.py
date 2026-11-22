"""2D cardiac tissue simulation driven by tissue parameters (roadmap phase 2 demo).

Model: Aliev-Panfilov (1996) two-variable excitable model on a square grid
(monodomain, explicit Euler, no-flux boundaries):

    du/dt = div(D grad u) - k u (u - a)(u - 1) - u v
    dv/dt = eps(u, v) * (-v - k u (u - a - 1)),   eps = eps0 + mu1 v / (u + mu2)

Units are dimensionless. How tissue parameters enter:
  * conduction_velocity_scale -> D = D0 * scale**2   (CV grows with sqrt(D))
  * apd_scale                 -> eps0 and mu1 both divided by apd_scale (slower recovery;
                                 in single-cell tests APD scales almost 1:1 with apd_scale)
  * fibrosis_fraction / fibrosis_heterogeneity -> non-conducting cells; no current
    flows across faces touching a fibrotic cell. Higher heterogeneity = larger patches.

This is a teaching / hypothesis-testing model, not a validated clinical simulator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class APParams:
    k: float = 8.0
    a: float = 0.15
    eps0: float = 0.002
    mu1: float = 0.2
    mu2: float = 0.3
    D0: float = 1.0


def fibrosis_mask(n: int, fraction: float, heterogeneity: float, seed: int = 0) -> np.ndarray:
    """Boolean mask, True = fibrotic (non-conducting). Exactly `fraction` of cells.

    Patch size is set by smoothing random noise; sigma grows with heterogeneity.
    The left stimulus columns are always kept healthy.
    """
    mask = np.zeros((n, n), dtype=bool)
    if fraction <= 0:
        return mask
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((n, n))
    sigma = 0.5 + 4.0 * heterogeneity          # grid cells
    kx = np.fft.fftfreq(n)[:, None]
    ky = np.fft.fftfreq(n)[None, :]
    smooth = np.real(np.fft.ifft2(np.fft.fft2(noise) *
                                  np.exp(-2 * (np.pi * sigma) ** 2 * (kx ** 2 + ky ** 2))))
    smooth[:, :5] = -np.inf                     # protect stimulus region
    thr = np.quantile(smooth[:, 5:], 1.0 - fraction * n / (n - 5))
    mask = smooth > thr
    return mask


def simulate(tissue: dict, n: int = 120, t_end: float = 160.0, dt: float = 0.05,
             p: APParams | None = None, seed: int = 0, snapshot_times=(25.0, 50.0, 75.0)):
    """Plane wave stimulated from the left edge.

    Returns dict with activation-time map, fibrosis mask, voltage snapshots and
    summary metrics (conduction velocity, APD, fraction of tissue activated).
    """
    p = p or APParams()
    cv_scale = float(tissue.get("conduction_velocity_scale", 1.0))
    apd_scale = float(tissue.get("apd_scale", 1.0))
    D = p.D0 * cv_scale ** 2
    eps0 = p.eps0 / apd_scale
    mu1 = p.mu1 / apd_scale
    if dt > 0.2 / max(D, 1e-9):
        raise ValueError("dt too large for stability; reduce dt")

    fib = fibrosis_mask(n, float(tissue.get("fibrosis_fraction", 0.0)),
                        float(tissue.get("fibrosis_heterogeneity", 0.0)), seed)
    healthy = ~fib
    gx = D * (healthy[:, 1:] & healthy[:, :-1])   # face conductances
    gy = D * (healthy[1:, :] & healthy[:-1, :])

    u = np.zeros((n, n))
    v = np.zeros((n, n))
    act = np.full((n, n), np.nan)
    upstroke_at = np.full((n, n), np.nan)
    apd = np.full((n, n), np.nan)
    snaps = {}
    stim_cols = slice(0, 3)
    act[:, stim_cols] = np.where(healthy[:, stim_cols], 0.0, np.nan)
    upstroke_at[:, stim_cols] = act[:, stim_cols]
    steps = int(round(t_end / dt))
    snap_steps = {int(round(t / dt)): t for t in snapshot_times}

    for step in range(steps):
        t = step * dt
        if t < 1.0:
            u[:, stim_cols] = np.where(healthy[:, stim_cols], 1.0, 0.0)
        fx = gx * (u[:, 1:] - u[:, :-1])
        fy = gy * (u[1:, :] - u[:-1, :])
        div = np.zeros_like(u)
        div[:, :-1] += fx
        div[:, 1:] -= fx
        div[:-1, :] += fy
        div[1:, :] -= fy
        eps = eps0 + mu1 * v / (u + p.mu2)
        du = div - p.k * u * (u - p.a) * (u - 1.0) - u * v
        dv = eps * (-v - p.k * u * (u - p.a - 1.0))
        u_new = np.where(healthy, u + dt * du, 0.0)
        v = np.where(healthy, v + dt * dv, 0.0)

        up = (u_new >= 0.5) & (u < 0.5) & np.isnan(act)
        act[up] = t + dt
        upstroke_at[up] = t + dt
        down = (u_new < 0.5) & (u >= 0.5) & ~np.isnan(upstroke_at) & np.isnan(apd)
        apd[down] = t + dt - upstroke_at[down]
        u = u_new
        if step + 1 in snap_steps:
            snaps[snap_steps[step + 1]] = u.copy()

    return {
        "activation_time": act,
        "fibrosis": fib,
        "snapshots": snaps,
        "metrics": _metrics(act, apd, healthy),
    }


def _metrics(act: np.ndarray, apd: np.ndarray, healthy: np.ndarray) -> dict:
    n = act.shape[0]
    x0, x1 = int(0.15 * n), int(0.85 * n)
    # Effective CV: distance between two columns / median arrival-time difference.
    t0 = np.nanmedian(act[:, x0][healthy[:, x0]]) if healthy[:, x0].any() else np.nan
    t1 = np.nanmedian(act[:, x1][healthy[:, x1]]) if healthy[:, x1].any() else np.nan
    cv = (x1 - x0) / (t1 - t0) if np.isfinite(t0) and np.isfinite(t1) and t1 > t0 else np.nan
    activated = np.isfinite(act) & healthy
    return {
        "conduction_velocity": round(float(cv), 4) if np.isfinite(cv) else None,
        "apd_median": round(float(np.nanmedian(apd)), 2) if np.isfinite(apd).any() else None,
        "total_activation_time": round(float(np.nanmax(act)), 2),
        "fraction_activated": round(float(activated.sum() / healthy.sum()), 4),
        "fibrosis_fraction_actual": round(float(1 - healthy.mean()), 4),
    }
