"""
src/models/optimizer.py - Volatility-adjusted position sizing and portfolio construction.

The ranking in `utils/signals.py` decides *which* assets to hold and in which
direction. Everything here decides *how much*, which is where Track A's
"volatility-adjusted weighting" requirement is discharged.

Four intra-leg sizing rules, in increasing order of how much they trust the
covariance matrix: equal weight (DeMiguel et al. 2009), inverse volatility (naive
risk parity), equal risk contribution (Maillard et al. 2010), and minimum variance
(Markowitz 1952). The default is inverse volatility: it needs only N variance
estimates rather than N(N+1)/2 covariances, which is the difference between an
estimate that survives out of sample and one that does not.

Constraint note: the backtester's RuleGuard hard-clamps sum(|w|) <= 1.0 and counts a
violation when it has to. Everything here stays strictly inside that bound by
construction, so the guard never intervenes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "TRADING_DAYS",
    "SizingConfig",
    "select_names",
    "equal_weights",
    "inverse_vol_weights",
    "erc_weights",
    "min_variance_weights",
    "size_leg",
    "apply_caps",
    "ex_ante_vol",
    "full_cross_section_weights",
    "build_weights",
    "apply_no_trade_band",
]


TRADING_DAYS = 252


_EPS = 1e-12


@dataclass
class SizingConfig:
    """Portfolio construction settings."""

    selection: str = "topn"        # topn | full  (see build_weights)
    n_long: int = 6
    n_short: int = 6
    method: str = "inverse_vol"    # inverse_vol | erc | min_variance | equal
    dollar_neutral: bool = True
    max_gross: float = 0.95        # headroom below the RuleGuard's hard 1.0.
                                   # The organiser's limit is 1.0, not 0.95 -- the
                                   # 5% buffer absorbs drift between rebalances, so
                                   # actual holdings never approach the ceiling and
                                   # the guard never has to clamp us. Must match the
                                   # value in engine.py SIZING; a bare SizingConfig()
                                   # is reachable via build_weights(cfg=None).
    max_position: float = 0.12     # per-name cap on |w|
    target_vol: Optional[float] = 0.10   # annualised ex-ante target; None disables
    cov_window: int = 126
    no_trade_band: float = 0.005   # skip per-name changes smaller than this
    exit_buffer: int = 0           # hysteresis: hold a name until it falls past n+buffer


def select_names(
    scores: pd.Series,
    n_long: int,
    n_short: int,
    current: Optional[pd.Series] = None,
    exit_buffer: int = 0,
) -> tuple[List[str], List[str]]:
    """
    Pick the long and short baskets from the ranking.

    With `exit_buffer = 0` this is a plain top-N / bottom-N selection: a name leaves
    the book the instant it drops out of the top N.

    With `exit_buffer > 0` it applies **hysteresis**. A name already held is retained
    while it stays inside the top `n_long + exit_buffer`, and only sold once it falls
    past that wider boundary. New names still have to break into the top `n_long` to
    get in. The entry and exit thresholds are therefore different, which is the point:
    a name oscillating around rank 6 would otherwise be bought and sold repeatedly on
    rank noise that carries no information.

    This matters here because the chosen signal is fast -- 21-day reversal has a rank
    autocorrelation around 0.70 against momentum's 0.96, so roughly a third of the
    ranking reshuffles between rebalances. Hysteresis attacks that churn without
    touching the signal itself.

    `current` is the live weight vector; its sign decides which leg a held name
    belongs to. Pass None (or leave `exit_buffer` at 0) for stateless selection.
    """
    s = scores.dropna()
    if len(s) < n_long + n_short:
        k = len(s) // 2
        n_long = n_short = max(0, min(n_long, n_short, k))
    if n_long + n_short == 0:
        return [], []

    order = s.sort_values()                      # ascending: last entry is the best
    best_first = list(order.index[::-1])         # rank 0 = most attractive
    worst_first = list(order.index)              # rank 0 = least attractive

    if current is None or exit_buffer <= 0 or current.empty:
        shorts = worst_first[:n_short] if n_short else []
        longs = best_first[:n_long] if n_long else []
        return longs, _dedupe(longs, shorts, best_first, worst_first, n_long, n_short)

    held_long = {k for k, v in current.items() if v > 0}
    held_short = {k for k, v in current.items() if v < 0}

    longs = _apply_hysteresis(best_first, held_long, n_long, exit_buffer)
    shorts = _apply_hysteresis(worst_first, held_short, n_short, exit_buffer)
    return longs, _dedupe(longs, shorts, best_first, worst_first, n_long, n_short)


def _apply_hysteresis(
    ranked: Sequence[str], held: set, n: int, buffer: int
) -> List[str]:
    """
    One leg of the hysteresis rule. `ranked` is ordered most- to least-attractive
    for this leg.

    Retain held names still inside `n + buffer`, in rank order, then top up from the
    strict top `n` with names that are not already retained. Retained names are
    truncated to `n` best first, so the leg never exceeds its size.
    """
    if n <= 0:
        return []
    widened = ranked[: n + buffer]
    keep = [x for x in widened if x in held][:n]
    for name in ranked[:n]:
        if len(keep) >= n:
            break
        if name not in keep:
            keep.append(name)
    # Still short of n (tiny universe): fill from anywhere still unused.
    for name in ranked:
        if len(keep) >= n:
            break
        if name not in keep:
            keep.append(name)
    return keep[:n]


def _dedupe(
    longs: List[str],
    shorts: List[str],
    best_first: Sequence[str],
    worst_first: Sequence[str],
    n_long: int,
    n_short: int,
) -> List[str]:
    """
    Guard against a name landing in both legs.

    Possible once the hysteresis buffer widens both boundaries on a small universe.
    A name in both would net to roughly zero weight while still paying to trade, so
    the conflict is resolved in favour of the long leg and the short leg is refilled
    from the next eligible name.
    """
    clash = set(longs) & set(shorts)
    if not clash:
        return shorts
    cleaned = [x for x in shorts if x not in clash]
    for name in worst_first:
        if len(cleaned) >= n_short:
            break
        if name not in cleaned and name not in longs:
            cleaned.append(name)
    return cleaned[:n_short]


def equal_weights(names: Sequence[str], **_) -> pd.Series:
    """1/N. DeMiguel, Garlappi & Uppal (2009) -- the benchmark worth beating."""
    if not len(names):
        return pd.Series(dtype=float)
    return pd.Series(1.0 / len(names), index=list(names))


def inverse_vol_weights(names: Sequence[str], vol: pd.Series, **_) -> pd.Series:
    """
    w_i proportional to 1 / sigma_i -- "naive risk parity".

    Equalises each position's standalone volatility contribution. It ignores
    correlation, which is the honest trade: with 24 names it needs only N variance
    estimates rather than N(N+1)/2 covariance estimates, so it is dramatically more
    stable out of sample than anything requiring a matrix inverse.
    """
    if not len(names):
        return pd.Series(dtype=float)
    v = vol.reindex(names).astype(float)
    v = v.replace(0.0, np.nan).fillna(v.median() if np.isfinite(v.median()) else 1.0)
    inv = 1.0 / v.clip(lower=_EPS)
    return inv / inv.sum()


def erc_weights(
    names: Sequence[str], cov: pd.DataFrame, max_iter: int = 500, tol: float = 1e-10, **_
) -> pd.Series:
    """
    Equal Risk Contribution (Maillard, Roncalli & Teiletche 2010).

    Every holding contributes the same share of portfolio variance, which is what
    inverse-vol only approximates when correlations are heterogeneous. Solved by the
    fixed point w_i proportional to 1 / (Sigma w)_i, which holds exactly at the ERC
    solution and converges monotonically for a positive-definite Sigma.
    """
    names = [n for n in names if n in cov.index]
    if not names:
        return pd.Series(dtype=float)
    S = cov.loc[names, names].to_numpy(float)
    n = len(names)

    w = 1.0 / np.sqrt(np.clip(np.diag(S), _EPS, None))
    w /= w.sum()
    for _ in range(max_iter):
        mrc = S @ w
        w_new = 1.0 / np.clip(mrc, _EPS, None)
        w_new /= w_new.sum()
        if np.max(np.abs(w_new - w)) < tol:
            w = w_new
            break
        w = w_new
    return pd.Series(w, index=names)


def min_variance_weights(names: Sequence[str], cov: pd.DataFrame, **_) -> pd.Series:
    """
    Long-only minimum variance inside the leg: w proportional to Sigma^-1 * 1,
    clipped at zero and renormalised.

    Included for comparison. Markowitz (1952) sizing is the most sensitive of the
    four to covariance error -- the inverse amplifies exactly the small eigenvalues
    that shrinkage is least able to fix -- so it is the variant most likely to look
    best in sample and worst out of sample.
    """
    names = [n for n in names if n in cov.index]
    if not names:
        return pd.Series(dtype=float)
    S = cov.loc[names, names].to_numpy(float)
    try:
        inv_one = np.linalg.solve(S, np.ones(len(names)))
    except np.linalg.LinAlgError:
        return equal_weights(names)
    w = np.clip(inv_one, 0.0, None)
    if w.sum() <= _EPS:
        return equal_weights(names)
    return pd.Series(w / w.sum(), index=names)


def size_leg(
    names: Sequence[str], method: str, vol: pd.Series, cov: pd.DataFrame
) -> pd.Series:
    """Dispatch to the configured intra-leg sizing rule. Weights sum to 1.0."""
    fn = {
        "equal": equal_weights,
        "inverse_vol": inverse_vol_weights,
        "erc": erc_weights,
        "min_variance": min_variance_weights,
    }.get(method)
    if fn is None:
        raise ValueError(f"Unknown sizing method {method!r}")
    w = fn(names=names, vol=vol, cov=cov)
    if w.empty or not np.isfinite(w.to_numpy()).all() or w.sum() <= _EPS:
        return equal_weights(names)
    return w / w.sum()


def apply_caps(w: pd.Series, max_position: float, max_gross: float, n_iter: int = 20) -> pd.Series:
    """
    Impose the per-name cap and the gross cap together.

    Clipping breaks the normalisation and renormalising breaks the clip, so the two
    are alternated to a fixed point. If the caps are jointly infeasible (N *
    max_position < max_gross) the gross simply ends up below target, which is the
    safe direction to err.
    """
    if w.empty:
        return w
    for _ in range(n_iter):
        w = w.clip(lower=-max_position, upper=max_position)
        gross = float(w.abs().sum())
        if gross <= _EPS:
            return w * 0.0
        if gross > max_gross:
            w = w * (max_gross / gross)
        else:
            break
    return w.clip(lower=-max_position, upper=max_position)


def ex_ante_vol(w: pd.Series, cov: pd.DataFrame, annualise: bool = True) -> float:
    """Forecast portfolio volatility from the shrunk covariance."""
    names = [n for n in w.index if n in cov.index]
    if not names:
        return float("nan")
    wv = w.reindex(names).to_numpy(float)
    S = cov.loc[names, names].to_numpy(float)
    var = float(wv @ S @ wv)
    vol = np.sqrt(max(var, 0.0))
    return vol * np.sqrt(TRADING_DAYS) if annualise else vol


def full_cross_section_weights(
    scores: pd.Series,
    vol: pd.Series,
    min_names: int = 8,
) -> pd.Series:
    """
    Weight *every* ranked name, w_i proportional to score_i / sigma_i.

    Rationale (stated before measuring, so it can be falsified):

    Grinold & Kahn's Fundamental Law gives IR ~ IC * sqrt(breadth). A top-6/bottom-6
    cut on a 24-name universe ranks all 24 and then discards the middle 12 -- it takes
    12 bets where the data supports 24. Weighting the whole cross-section doubles
    breadth, so the law predicts an improvement of about sqrt(2) = 1.41x.

    It is also the right functional form rather than merely a wider one: under the
    linear signal model the IC already assumes, expected return is proportional to the
    score, so the optimal risk-adjusted weight is proportional to score / variance-risk
    -- which is exactly `score_i / sigma_i`. The top-N cut is a step-function
    approximation to that line, and it throws away the information that one name is
    ranked 7th and another 18th.

    There is no tunable parameter here to search over, which is the point: the change
    is justified in advance by theory, not selected after the fact by its backtest.

    Dollar-neutrality: the rank score already sums to zero, but dividing by a
    heterogeneous sigma breaks that, so the result is re-demeaned.
    """
    s = scores.dropna()
    if len(s) < min_names:
        return pd.Series(dtype=float)

    v = vol.reindex(s.index).astype(float)
    med = v.median()
    v = v.replace(0.0, np.nan).fillna(med if np.isfinite(med) else 1.0).clip(lower=_EPS)

    w = s / v
    w = w - w.mean()                       # restore dollar-neutrality
    gross = float(w.abs().sum())
    if gross <= _EPS:
        return pd.Series(dtype=float)
    return w / gross                       # unit gross; scaled by the caller


def build_weights(
    scores: pd.Series,
    vol: pd.Series,
    cov: pd.DataFrame,
    cfg: Optional[SizingConfig] = None,
    current: Optional[pd.Series] = None,
) -> pd.Series:
    """
    Full sizing pipeline for one rebalance date.

    scores : cross-sectional signal for this date (higher = more attractive)
    vol    : daily volatility estimate per name for this date
    cov    : shrunk covariance over the trailing window

    `cfg.selection`:
      "topn" -- long the best `n_long`, short the worst `n_short`, size within each
                leg by `cfg.method`. Optional hysteresis via `cfg.exit_buffer`.
      "full" -- weight every name by score/sigma across the whole cross-section.
                Doubles breadth; see `full_cross_section_weights`.

    Order of operations, and why:
      1. select names from the ranking (with hysteresis if configured)
      2. size by the configured risk rule
      3. set leg signs and impose dollar-neutrality
      4. scale the book to the ex-ante volatility target
      5. apply per-name and gross caps LAST, so no earlier step can reintroduce a
         breach of the backtester's leverage constraint
    """
    cfg = cfg or SizingConfig()

    if cfg.selection == "full":
        w = full_cross_section_weights(scores, vol)
        if w.empty:
            return pd.Series(dtype=float)
        w = w * cfg.max_gross
    elif cfg.selection == "topn":
        longs, shorts = select_names(
            scores, cfg.n_long, cfg.n_short, current=current, exit_buffer=cfg.exit_buffer
        )
        if not longs and not shorts:
            return pd.Series(dtype=float)

        w_long = size_leg(longs, cfg.method, vol, cov) if longs else pd.Series(dtype=float)
        w_short = size_leg(shorts, cfg.method, vol, cov) if shorts else pd.Series(dtype=float)

        if cfg.dollar_neutral and len(w_long) and len(w_short):
            long_share = short_share = 0.5
        else:
            total = len(w_long) + len(w_short)
            long_share = len(w_long) / total if total else 0.0
            short_share = len(w_short) / total if total else 0.0

        w = pd.concat([w_long * long_share, -w_short * short_share])
        w = w.groupby(level=0).sum()

        gross0 = float(w.abs().sum())
        if gross0 <= _EPS:
            return pd.Series(dtype=float)
        w = w * (cfg.max_gross / gross0)
    else:
        raise ValueError(f"selection must be 'topn' or 'full', got {cfg.selection!r}")

    if cfg.target_vol is not None and not cov.empty:
        ev = ex_ante_vol(w, cov)
        if np.isfinite(ev) and ev > _EPS:
            # Scale down to the target, but never lever up beyond max_gross.
            w = w * min(cfg.target_vol / ev, 1.0)

    return apply_caps(w, cfg.max_position, cfg.max_gross)


def apply_no_trade_band(
    target: pd.Series, current: pd.Series, band: float = 0.005
) -> pd.Series:
    """
    Suppress per-name changes smaller than `band`.

    Turnover is the tax on a fast signal: the reversal signal this project settles on
    has roughly 3x momentum's turnover, so a small deadband is what keeps commission
    from eating the edge. Names whose target moved less than the band keep their
    existing weight; everything else trades in full.
    """
    idx = target.index.union(current.index)
    t = target.reindex(idx).fillna(0.0)
    c = current.reindex(idx).fillna(0.0)
    keep = (t - c).abs() < band
    out = t.copy()
    out[keep] = c[keep]
    return out[out.abs() > _EPS]
