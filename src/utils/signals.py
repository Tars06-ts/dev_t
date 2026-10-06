"""
src/utils/signals.py - Cross-sectional ranking signals (feature extraction).

Each builder takes a `utils.data.Panel` and returns a (date x ticker) score frame.
The contract every builder honours is that row `t` is a function of panel rows
`<= t` only, and `assert_causal` enforces that contract by recomputing each signal on
a truncated panel and demanding an exact match. The look-ahead guard is a test that
fails loudly, not a convention.

The statistics that *evaluate* these signals -- information coefficients,
Fama-MacBeth regressions, spread portfolios, concentration profiles -- live in
`research/signal_eval.py`, since they are research output rather than anything the
live strategy executes.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence

import numpy as np
import pandas as pd

from .data import Panel
from .math_helpers import cs_rank, cs_winsorize, cs_zscore

__all__ = [
    "momentum_12_1",
    "vol_scaled_momentum",
    "short_term_reversal",
    "SIGNAL_LIBRARY",
    "build_signals",
    "assert_causal",
    "forward_returns",
]


def momentum_12_1(panel: Panel, lookback: int = 252, skip: int = 21) -> pd.DataFrame:
    """
    Variant A -- classic cross-sectional momentum (Jegadeesh & Titman 1993;
    Asness, Moskowitz & Pedersen 2013).

    Score = cumulative log return from t-lookback to t-skip.

    Economic rationale: under-reaction to slow-diffusing information, plus
    institutional flow persistence, makes the 12-month trend of an asset
    positively autocorrelated in the cross-section. The most recent month is
    skipped because it carries the opposite sign -- short-horizon reversal from
    bid-ask bounce and liquidity provision -- which otherwise contaminates the
    medium-term signal.

    Causal by construction: `rolling(...).sum()` on row t spans rows t-k..t, and
    the `.shift(skip)` moves the window strictly further into the past.
    """
    lr = panel.log_returns.fillna(0.0).where(panel.close.notna())
    cum = lr.rolling(lookback - skip, min_periods=int(0.8 * (lookback - skip))).sum()
    return cum.shift(skip).where(panel.close.notna())


def vol_scaled_momentum(
    panel: Panel,
    lookback: int = 252,
    skip: int = 21,
    vol_window: int = 126,
    min_vol: float = 1e-4,
) -> pd.DataFrame:
    """
    Variant B -- volatility-scaled ("risk-adjusted") momentum.

    Score = 12-1 momentum / realised volatility over the trailing `vol_window`.

    Rationale: raw momentum loads mechanically on volatility -- a high-vol name
    posts a large trailing return from noise alone, so a raw-momentum ranking is
    partly a low-information bet on the volatility factor. Dividing by realised
    vol converts the score into a trailing information ratio, which is both the
    quantity the sizing layer actually wants and, per Moskowitz, Ooi & Pedersen
    (2012), a more stable predictor out of sample. This is the variant whose
    ranking is economically aligned with vol-adjusted position sizing.

    Volatility is estimated on `returns_est`, which masks forward-filled sessions
    so that a stale zero does not depress the denominator.
    """
    mom = momentum_12_1(panel, lookback=lookback, skip=skip)
    vol = (
        panel.returns_est.rolling(vol_window, min_periods=int(0.6 * vol_window))
        .std(ddof=1)
        .shift(skip)
    )
    return mom / vol.clip(lower=min_vol)


def short_term_reversal(panel: Panel, lookback: int = 21) -> pd.DataFrame:
    """
    One-month reversal (Jegadeesh 1990). Score = minus the trailing one-month return.

    Entered the study as a falsification control for `momentum_12_1` -- it trades
    exactly the window momentum skips -- but on the in-sample Track A panel
    (2016-09 to 2022-09, 24 names) it is the only one of the three with statistical
    support, so it is now the lead candidate:

      * mean IC at h=21 = +0.071, Newey-West t = +3.64 (momentum: +0.026, t = +0.87)
      * Fama-MacBeth lambda t = +3.41 (momentum: +0.43)
      * positive in all three sub-periods and *stronger* excluding Feb-Jun 2020
        (t = +4.04), so it is not an artefact of the COVID crash-and-rebound
      * lookback sweep 5/10/21/42/63/126d decays smoothly, peaking at 21-42d --
        a plateau, not a knife-edge, which is what distinguishes a real effect
        from a fitted parameter. 21d is kept because it is the canonical a-priori
        specification, not because the sweep selected it.
      * cross-sectionally orthogonal to both momentum variants (rank corr ~0.00)
        and not a disguised volatility bet (corr with trailing vol = -0.04)

    The economics are the standard ones: short-horizon liquidity provision and
    over-reaction to idiosyncratic news. The caveat is cost -- rank autocorrelation
    is 0.70 against momentum's 0.96, implying roughly 3x the turnover, so the
    sizing layer has to earn this edge back net of commission.
    """
    lr = panel.log_returns.fillna(0.0).where(panel.close.notna())
    return -lr.rolling(lookback, min_periods=int(0.8 * lookback)).sum()


SIGNAL_LIBRARY: Dict[str, Callable[[Panel], pd.DataFrame]] = {
    "MOM_12_1": momentum_12_1,
    "VOLADJ_MOM_12_1": vol_scaled_momentum,
    "ST_REV_21": short_term_reversal,
}


def build_signals(
    panel: Panel,
    names: Optional[Sequence[str]] = None,
    standardise: str = "rank",
    smooth=1,
) -> Dict[str, pd.DataFrame]:
    """
    Build the named signals and put them on a common scale.

    `standardise`:
      "rank"   -> cross-sectional rank in [-0.5, 0.5]   (outlier-proof; default)
      "zscore" -> winsorised cross-sectional z-score    (keeps score magnitudes)
      "raw"    -> untransformed builder output

    `smooth` accepts an int, or a sequence of ints for **parameter ensembling**:
    the signal is built at every listed window, ranked separately, and the ranks are
    averaged. This is the principled alternative to picking the best window from a
    sweep. Searching for an argmax and keeping it imports selection bias -- the
    winner's score is inflated by however many settings were tried. Averaging over
    the plausible range removes the selection step entirely, so the result is an
    honest estimate rather than a maximum. It also diversifies estimation noise
    across windows, in the same way smoothing diversifies it across days.

    The expectation is that an ensemble performs near the *average* of its members,
    not the best of them -- and that it degrades far less out of sample, because
    there is no fitted choice to decay.

    A single int behaves exactly as before: average the raw score over the trailing
    `smooth` sessions before standardising. A single noisy session can otherwise flip a name several ranks and
    trigger a round trip that carries no information. Smoothing always reduces
    turnover; the open question is whether it also destroys the edge, so it is off by
    default (`smooth=1`) and has to earn its place in the sweep. The trailing window
    is causal, so this does not weaken the look-ahead guarantee.
    """
    names = list(names) if names is not None else list(SIGNAL_LIBRARY)
    windows = [int(smooth)] if isinstance(smooth, (int, np.integer)) else [int(k) for k in smooth]
    if not windows:
        windows = [1]

    def _standardise(frame: pd.DataFrame) -> pd.DataFrame:
        if standardise == "rank":
            return cs_rank(frame)
        if standardise == "zscore":
            return cs_zscore(cs_winsorize(frame))
        if standardise == "raw":
            return frame
        raise ValueError(f"standardise must be rank|zscore|raw, got {standardise!r}")

    out: Dict[str, pd.DataFrame] = {}
    for name in names:
        if name not in SIGNAL_LIBRARY:
            raise KeyError(f"Unknown signal {name!r}; available: {sorted(SIGNAL_LIBRARY)}")
        base = SIGNAL_LIBRARY[name](panel)
        members = []
        for k in windows:
            raw = base.rolling(k, min_periods=1).mean() if k > 1 else base
            members.append(_standardise(raw.where(panel.tradable)))
        if len(members) == 1:
            out[name] = members[0]
        else:
            # Average the *ranks*, not the raw scores: ranks are already on a common
            # scale, so no member can dominate the blend through its units.
            stacked = pd.concat(members).groupby(level=0).mean()
            out[name] = _standardise(stacked) if standardise == "rank" else stacked
    return out


def assert_causal(
    panel: Panel,
    names: Optional[Sequence[str]] = None,
    n_checks: int = 6,
    standardise: str = "rank",
    tol: float = 1e-10,
    smooth: int = 1,
) -> pd.DataFrame:
    """
    Look-ahead guard.

    Rebuild each signal on `panel.up_to(d)` for several interior dates `d` and compare
    the final row against the same date's row from the full-sample build. Any builder
    that touches a future observation -- a global mean, a centred window, a reverse
    shift -- produces a mismatch here and fails loudly.
    """
    names = list(names) if names is not None else list(SIGNAL_LIBRARY)
    full = build_signals(panel, names, standardise=standardise, smooth=smooth)

    dates = panel.dates
    probes = dates[np.linspace(int(0.55 * len(dates)), len(dates) - 1, n_checks).astype(int)]

    rows = []
    for d in probes:
        trunc = build_signals(panel.up_to(d), names, standardise=standardise, smooth=smooth)
        for name in names:
            a = full[name].loc[d]
            b = trunc[name].loc[d]
            both = a.notna() & b.notna()
            diff = float((a[both] - b[both]).abs().max()) if both.any() else 0.0
            mism = int((a.notna() != b.notna()).sum())
            rows.append(
                {"date": d, "signal": name, "max_abs_diff": diff, "nan_mismatch": mism}
            )
            if not (diff <= tol and mism == 0):
                raise AssertionError(
                    f"LOOK-AHEAD in {name!r} at {d.date()}: "
                    f"max|diff|={diff:.3e}, nan mismatches={mism}"
                )
    return pd.DataFrame(rows)


def forward_returns(panel: Panel, horizon: int = 5, price: str = "open") -> pd.DataFrame:
    """
    Return realisable from a decision made on bar `t`.

    price="open" (default): open_{t+1+h} / open_{t+1} - 1. This is execution-aligned --
    the backtester fills bar-`t` targets at the bar-`t+1` open.
    price="close": close_{t+h} / close_t - 1, the academic convention, kept so results
    can be compared with published cross-sectional studies.
    """
    if price == "open":
        px = panel.open
        return px.shift(-(1 + horizon)) / px.shift(-1) - 1.0
    if price == "close":
        px = panel.close
        return px.shift(-horizon) / px - 1.0
    raise ValueError("price must be 'open' or 'close'")
