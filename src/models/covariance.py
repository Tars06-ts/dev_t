"""
src/models/covariance.py - Shrinkage covariance estimation.

With 24 names and a 126-day window the sample covariance has T/N ~ 5. At that ratio
the matrix is too noisy to invert honestly: the smallest eigenvalues are close to pure
estimation error, and any optimiser that inverts it loads onto exactly those
directions. Shrinkage is a precondition here, not a refinement.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

__all__ = ["ledoit_wolf_cov"]


_EPS = 1e-12


def ledoit_wolf_cov(window_returns: pd.DataFrame, shrink: Optional[float] = None) -> pd.DataFrame:
    """
    Sample covariance shrunk toward a constant-correlation target (Ledoit & Wolf 2004).

    The target keeps each asset's own variance but replaces every pairwise correlation
    with the cross-sectional average. That is the right target for an equity-style
    universe, where the dominant structure really is "everything is correlated about
    rho-bar with the market" and the individual pair estimates are mostly noise.

    `shrink=None` uses a simple, stable intensity based on the sample ratio N/T rather
    than the full analytic formula; with N=24 and T=126 it lands around 0.2-0.4, and
    the sizing results are not sensitive to it at that scale.
    """
    X = window_returns.dropna(axis=1, how="all").dropna()
    n_obs, n_assets = X.shape
    if n_assets == 0 or n_obs < 10:
        return pd.DataFrame(dtype=float)

    S = X.cov().to_numpy()
    sd = np.sqrt(np.clip(np.diag(S), _EPS, None))

    R = S / np.outer(sd, sd)
    np.fill_diagonal(R, 1.0)
    iu = np.triu_indices(n_assets, k=1)
    rho_bar = float(np.nanmean(R[iu])) if len(iu[0]) else 0.0

    target_R = np.full((n_assets, n_assets), rho_bar)
    np.fill_diagonal(target_R, 1.0)
    F = target_R * np.outer(sd, sd)

    if shrink is None:
        shrink = float(np.clip(n_assets / max(n_obs, 1), 0.0, 1.0))

    Sigma = shrink * F + (1.0 - shrink) * S
    # Guarantee positive definiteness before anything tries to invert it.
    Sigma = 0.5 * (Sigma + Sigma.T)
    eig = np.linalg.eigvalsh(Sigma)
    if eig.min() <= _EPS:
        Sigma += np.eye(n_assets) * (abs(eig.min()) + 1e-10)

    return pd.DataFrame(Sigma, index=X.columns, columns=X.columns)
