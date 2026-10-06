"""
src/utils/math_helpers.py - Statistics shared across the signal and portfolio layers.

Two families live here:

* **Cross-sectional transforms** (`cs_winsorize`, `cs_zscore`, `cs_rank`) -- row-wise
  operations that turn a raw score into a comparable ranking. Being row-wise, they are
  trivially point-in-time.
* **Inference under autocorrelation** (`newey_west_tstat`, `block_bootstrap_sharpe`) --
  overlapping forward returns and volatility clustering both make naive iid standard
  errors far too narrow. Nothing in this project reports a t-statistic or a confidence
  interval that has not been through one of these.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

__all__ = [
    "TRADING_DAYS",
    "SEED",
    "cs_winsorize",
    "cs_zscore",
    "cs_rank",
    "newey_west_tstat",
    "two_sided_p",
    "block_bootstrap_sharpe",
]


TRADING_DAYS = 252


SEED = 42


def cs_winsorize(df: pd.DataFrame, lower: float = 0.05, upper: float = 0.95) -> pd.DataFrame:
    """Clip each date's cross-section to its own quantiles. With a 24-name universe a
    single outlier is 4% of the book, so trimming the tails before standardising keeps
    one asset from dictating the whole weight vector."""
    lo = df.quantile(lower, axis=1)
    hi = df.quantile(upper, axis=1)
    return df.clip(lower=lo, upper=hi, axis=0)


def cs_zscore(df: pd.DataFrame, min_names: int = 5) -> pd.DataFrame:
    """Demean and scale each date's cross-section. Dates with too few live names are
    blanked rather than standardised against noise."""
    mu = df.mean(axis=1)
    sd = df.std(axis=1, ddof=1)
    z = df.sub(mu, axis=0).div(sd.replace(0.0, np.nan), axis=0)
    return z.where(df.notna().sum(axis=1) >= min_names)


def cs_rank(df: pd.DataFrame, min_names: int = 5) -> pd.DataFrame:
    """
    Cross-sectional rank, rescaled to sum to exactly zero on every date.

    Ranks 1..n are mapped to (rank - (n+1)/2) / n, giving a symmetric spread of
    +/- (n-1)/(2n) with mean exactly 0. Note that the obvious `rank(pct=True) - 0.5`
    does *not* do this -- percentile ranks average to (n+1)/2n, so it would leave a
    small permanent long tilt (about +2% gross for a 24-name universe) baked into
    every weight vector. Rank is the outlier-proof alternative to the z-score and is
    what the IC is computed on.
    """
    n = df.notna().sum(axis=1)
    r = df.rank(axis=1, na_option="keep")
    centred = r.sub((n + 1) / 2.0, axis=0).div(n.replace(0, np.nan), axis=0)
    return centred.where(n >= min_names)


def newey_west_tstat(x: pd.Series, lags: Optional[int] = None) -> Dict[str, float]:
    """
    t-statistic for the mean of `x`, with a Newey-West HAC variance.

    Overlapping h-day forward returns make consecutive observations mechanically
    correlated; an iid standard error would overstate significance by roughly
    sqrt(h). Bartlett weights over `lags` periods remove that.
    """
    v = pd.Series(x).dropna().to_numpy(dtype=float)
    n = v.size
    if n < 10:
        return {"mean": np.nan, "se": np.nan, "tstat": np.nan, "n": n, "lags": 0}

    if lags is None:
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))   # Newey-West (1994)
    lags = max(0, min(lags, n - 1))

    e = v - v.mean()
    s = float(e @ e) / n
    for k in range(1, lags + 1):
        g = float(e[k:] @ e[:-k]) / n
        s += 2.0 * (1.0 - k / (lags + 1.0)) * g
    s = max(s, 1e-24)

    se = np.sqrt(s / n)
    return {"mean": float(v.mean()), "se": float(se), "tstat": float(v.mean() / se),
            "n": n, "lags": lags}


def two_sided_p(t: float, dof: int) -> float:
    if not np.isfinite(t) or dof < 1:
        return np.nan
    from scipy import stats

    return float(2.0 * stats.t.sf(abs(t), dof))


def block_bootstrap_sharpe(
    r: np.ndarray, n_boot: int = 2000, block: int = 21, seed: int = SEED
) -> Dict[str, float]:
    """
    Circular block bootstrap CI for the annualised Sharpe ratio.

    Blocks of ~1 month preserve volatility clustering and return autocorrelation,
    which an iid bootstrap would destroy and thereby produce an interval that is
    far too tight.
    """
    n = r.size
    if n < 3 * block:
        return {"sharpe_ci_low": np.nan, "sharpe_ci_high": np.nan}
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(n_boot, n_blocks))
    offs = np.arange(block)
    idx = (starts[:, :, None] + offs[None, None, :]).reshape(n_boot, -1)[:, :n] % n
    samp = r[idx]
    mu = samp.mean(axis=1)
    sd = samp.std(axis=1, ddof=1)
    sr = np.where(sd > 0, mu / sd, np.nan) * np.sqrt(TRADING_DAYS)
    lo, hi = np.nanpercentile(sr, [2.5, 97.5])
    return {"sharpe_ci_low": float(lo), "sharpe_ci_high": float(hi)}
