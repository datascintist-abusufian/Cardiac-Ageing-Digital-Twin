"""Re-entry vulnerability: S1-S2 premature-stimulus protocol (roadmap phase 4).

Protocol (2D Aliev-Panfilov tissue, same model and parameter mapping as `tissue_sim`):

1. S1: plane wave from the left edge.
2. S2: a small circular premature stimulus at `site`, delivered `coupling` time units
   after the S1 wave activates that site.
3. If S2 falls in the S1 wave's recovering tail, it can block in one direction and
   travel in the other, curling into re-entry (a rotor / figure-of-eight). Re-entry is
   called *sustained* if excitation is still present `observe` time units after S2,
   long after any non-re-entrant wave would have left the tissue.

The **vulnerable window** is the range of coupling intervals that induce sustained
re-entry; a wider window means a premature beat is more likely to start an arrhythmia.

Units are dimensionless model units. This is a mechanistic teaching / hypothesis model,
not a clinical arrhythmia-risk predictor.
"""

from __future__ import annotations

import numpy as np

from .tissue_sim import APParams, fibrosis_mask

CV0 = 1.24      # young plane-wave conduction velocity (grid units / time unit), see tissue_sim
APD0 = 22.9     # young action potential duration (time units)


def wavelength(tissue: dict) -> float:
    """Approximate excitation wavelength = CV x APD (grid units). Shorter favours re-entry."""
    return CV0 * tissue.get("conduction_velocity_scale", 1.0) * APD0 * tissue.get("apd_scale", 1.0)


def s1s2(tissue: dict, coupling: float, n: int = 100, site=(0.5, 0.5), radius: int = 5,
         seed: int = 0, dt: float = 0.05, observe: float = 200.0,
         snapshot_times=(), p: APParams | None = None) -> dict:
    """Run one S1-S2 trial. Returns sustained flag, timing and optional snapshots.

    `snapshot_times` are times *after S2* at which the voltage field is stored.
    """
    p = p or APParams()
    D = p.D0 * float(tissue.get("conduction_velocity_scale", 1.0)) ** 2
    apd_scale = float(tissue.get("apd_scale", 1.0))
    eps0, mu1 = p.eps0 / apd_scale, p.mu1 / apd_scale
    fib = fibrosis_mask(n, float(tissue.get("fibrosis_fraction", 0.0)),
                        float(tissue.get("fibrosis_heterogeneity", 0.0)), seed)
    yy, xx = np.mgrid[0:n, 0:n]
    s2 = (yy - site[1] * (n - 1)) ** 2 + (xx - site[0] * (n - 1)) ** 2 <= radius ** 2
    fib = fib & ~s2                       # premature beat is delivered to viable tissue
    h = ~fib
    gx = D * (h[:, 1:] & h[:, :-1])
    gy = D * (h[1:, :] & h[:-1, :])

    u = np.zeros((n, n))
    v = np.zeros((n, n))
    t_site = t_s2 = None
    last_active = 0.0
    snaps = {}
    pending = sorted(snapshot_times)
    k = 0
    t_limit = 3 * n / (CV0 * max(D, 1e-6) ** 0.5) + coupling + observe   # safety bound
    while True:
        t = k * dt
        if t < 1.0:
            u[:, :3] = np.where(h[:, :3], 1.0, 0.0)
        if t_site is None and (u[s2] > 0.5).mean() > 0.5:
            t_site, t_s2 = t, t + coupling
        if t_s2 is not None and t_s2 <= t < t_s2 + 1.0:
            u[s2] = 1.0
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
        u = np.where(h, u + dt * du, 0.0)
        v = np.where(h, v + dt * dv, 0.0)
        k += 1
        if (u > 0.5).any():
            last_active = t
        if t_s2 is not None:
            while pending and t >= t_s2 + pending[0]:
                snaps[pending.pop(0)] = u.copy()
            if t >= t_s2 + observe:
                break
            if t > t_s2 + 5 and t - last_active > 5 and not pending:
                break                                   # tissue silent: no re-entry
        if t > t_limit:
            break
    sustained = t_s2 is not None and last_active >= t_s2 + observe - 1.0
    return {
        "sustained": bool(sustained),
        "activity_after_s2": None if t_s2 is None else round(float(last_active - t_s2), 2),
        "t_site": t_site,
        "coupling": coupling,
        "snapshots": snaps,
        "fibrosis": fib,
    }


def vulnerable_window(tissue: dict, couplings, seeds=(0, 1), **kw) -> dict:
    """Probability of sustained re-entry at each coupling interval, and window width.

    width = sum over couplings of P(re-entry) x coupling step (time units), i.e. the
    expected length of the vulnerable window.
    """
    couplings = np.asarray(list(couplings), dtype=float)
    prob = np.array([np.mean([s1s2(tissue, c, seed=s, **kw)["sustained"] for s in seeds])
                     for c in couplings])
    step = float(np.median(np.diff(couplings))) if len(couplings) > 1 else 0.0
    hit = couplings[prob > 0]
    return {
        "couplings": couplings.round(3).tolist(),
        "probability": prob.round(3).tolist(),
        "width": round(float(prob.sum() * step), 2),
        "earliest": None if not len(hit) else float(hit.min()),
        "latest": None if not len(hit) else float(hit.max()),
        "wavelength": round(wavelength(tissue), 2),
    }
