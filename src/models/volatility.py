"""
src/models/volatility.py - Volatility Engine API.

Four causal estimators of daily volatility, plus the dispatcher the sizing layer
calls. The estimate on row t uses rows <= t only.

Why the blend is the default: EWMA close-to-close is unbiased but noisy (one
observation per day); Parkinson's range estimator extracts roughly 5x more information
per observation but is biased low because it ignores overnight gaps. Averaging in
variance space keeps most of Parkinson's efficiency while the close-to-close term
restores the gap risk it drops.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.utils.data import Panel

__all__ = [
    "VolConfig",
    "rolling_vol",
    "ewma_vol",
    "parkinson_vol",
    "blended_vol",
    "estimate_vol",
]


@dataclass
class VolConfig:
    """Volatility estimator settings."""

    method: str = "blended"        # rolling | ewma | parkinson | blended
    window: int = 63               # lookback for rolling / Parkinson
    halflife: int = 42             # EWMA halflife in sessions
    blend_weight: float = 0.5      # weight on Parkinson inside "blended"
    min_periods: int = 20
    floor: float = 0.002           # daily vol floor, ~3% annualised
    cap: float = 0.25              # daily vol cap, ~400% annualised


def rolling_vol(returns: pd.DataFrame, window: int = 63, min_periods: int = 20) -> pd.DataFrame:
    """
    Equal-weighted close-to-close standard deviation.

    The transparent baseline. Its weakness is that it is a box filter: a shock enters
    and leaves the estimate abruptly `window` days later, producing the characteristic
    plateau-then-cliff in estimated risk.
    """
    return returns.rolling(window, min_periods=min_periods).std(ddof=1)


def ewma_vol(returns: pd.DataFrame, halflife: int = 42, min_periods: int = 20) -> pd.DataFrame:
    """
    Exponentially weighted volatility (RiskMetrics).

    This is the IGARCH(1,1) special case with omega = 0 and alpha + beta = 1, so it
    inherits GARCH's volatility-clustering response without any per-asset parameter
    fitting -- which matters here because the OOS period is a different sample and
    refitting 24 GARCH models on a rolling basis would be both slow and prone to
    unstable parameter estimates. Shocks decay geometrically instead of falling off
    a cliff, which is the empirically right shape (Engle 1982; Bollerslev 1986).
    """
    return returns.ewm(halflife=halflife, min_periods=min_periods, adjust=True).std(bias=False)


def parkinson_vol(
    high: pd.DataFrame, low: pd.DataFrame, window: int = 63, min_periods: int = 20
) -> pd.DataFrame:
    """
    Parkinson (1980) range estimator: sigma^2 = mean[ ln(H/L)^2 ] / (4 ln 2).

    Uses the intraday range rather than one closing price per day, so it extracts
    roughly 5x more information per observation than close-to-close. That efficiency
    is why it is in the blend. Its known bias is downward -- discrete trading means
    the observed range understates the true continuous-time range, and it ignores
    overnight gaps entirely, which on this dataset is a material omission.
    """
    lr2 = (np.log(high / low) ** 2) / (4.0 * np.log(2.0))
    return np.sqrt(lr2.rolling(window, min_periods=min_periods).mean())


def blended_vol(
    returns: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    cfg: VolConfig,
) -> pd.DataFrame:
    """
    Variance-space blend of EWMA close-to-close and Parkinson range vol.

    Rationale: the two estimators have complementary errors. EWMA close-to-close is
    unbiased but noisy (one observation per day); Parkinson is far more efficient but
    biased low because it misses overnight gaps. Averaging in variance space keeps
    most of Parkinson's efficiency while the close-to-close term restores the gap risk
    it drops. `blend_weight` is the weight on Parkinson.
    """
    v_ewma = ewma_vol(returns, cfg.halflife, cfg.min_periods) ** 2
    v_park = parkinson_vol(high, low, cfg.window, cfg.min_periods) ** 2
    w = float(np.clip(cfg.blend_weight, 0.0, 1.0))
    blended = w * v_park.add(0.0) + (1.0 - w) * v_ewma
    # Where one estimator is unavailable, fall back to the other rather than to NaN.
    blended = blended.where(v_park.notna() & v_ewma.notna(), v_ewma.fillna(v_park))
    return np.sqrt(blended)


def estimate_vol(panel: Panel, cfg: Optional[VolConfig] = None) -> pd.DataFrame:
    """
    Daily volatility per (date, ticker) using the configured estimator, then floored
    and capped.

    Estimation runs on `panel.returns_est`, which masks forward-filled sessions, so a
    stale zero cannot depress the estimate and inflate that name's position size --
    exactly the failure mode that makes a vol-scaled book quietly concentrate into
    whichever asset stopped printing.
    """
    cfg = cfg or VolConfig()
    r = panel.returns_est

    if cfg.method == "rolling":
        vol = rolling_vol(r, cfg.window, cfg.min_periods)
    elif cfg.method == "ewma":
        vol = ewma_vol(r, cfg.halflife, cfg.min_periods)
    elif cfg.method == "parkinson":
        vol = parkinson_vol(panel.high, panel.low, cfg.window, cfg.min_periods)
    elif cfg.method == "blended":
        vol = blended_vol(r, panel.high, panel.low, cfg)
    else:
        raise ValueError(f"Unknown vol method {cfg.method!r}")

    return vol.clip(lower=cfg.floor, upper=cfg.cap)
