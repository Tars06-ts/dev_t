"""
src/research/signal_eval.py - Statistical evaluation of the ranking signals.

This is the "statistically justify your ranking signal" deliverable. Reporting a
backtest Sharpe is explicitly not sufficient, so the battery here tests the ranking
hypothesis directly:

  * **Information coefficient** -- per-date cross-sectional Spearman correlation
    between score and forward return, with a Newey-West t-statistic.
  * **Fama-MacBeth (1973)** -- per-date cross-sectional regression of forward returns
    on the standardised score; the hypothesis "this signal ranks assets" is exactly
    H0: E[lambda_t] = 0.
  * **Top-minus-bottom spread portfolio** -- the economic read, with block-bootstrap
    Sharpe intervals.
  * **Concentration profile** -- correlation and effective breadth of the top-ranked
    basket, since a ranking that repeatedly picks six names that move together is one
    bet with six tickers.

Overlapping forward horizons make naive t-statistics over-confident by roughly
sqrt(h), so every statistic here is autocorrelation-robust.

Forward returns are measured **open-to-open from t+1**, because the backtester fills
bar-`t` targets at the bar-`t+1` open. Scoring against close-to-close returns would
credit the signal with a move it could never have traded.

    python -m src.research.signal_eval
    python -m src.research.signal_eval --horizons 1,5,21,63 --standardise zscore
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from src.utils.data import Panel
from src.utils.math_helpers import (
    SEED,
    TRADING_DAYS,
    block_bootstrap_sharpe,
    cs_zscore,
    newey_west_tstat,
    two_sided_p,
)
from src.utils.signals import build_signals, forward_returns

__all__ = [
    "information_coefficient",
    "ic_summary",
    "fama_macbeth",
    "spread_portfolio",
    "top_basket_profile",
    "signal_persistence",
    "SignalEvaluation",
    "evaluate_signal",
    "evaluate_signal_static",
    "compare_signals",
]


def information_coefficient(
    score: pd.DataFrame, fwd: pd.DataFrame, method: str = "spearman", min_names: int = 5
) -> pd.Series:
    """Per-date cross-sectional correlation between the score and the forward return."""
    s, f = score.align(fwd, join="inner")
    out = {}
    for d in s.index:
        a, b = s.loc[d], f.loc[d]
        m = a.notna() & b.notna()
        if int(m.sum()) >= min_names:
            out[d] = a[m].corr(b[m], method=method)
    return pd.Series(out, dtype=float).rename("ic")


def ic_summary(ic: pd.Series, horizon: int) -> Dict[str, float]:
    """Mean IC, IC information ratio, and a HAC t-stat for `mean IC == 0`."""
    nw = newey_west_tstat(ic, lags=max(1, 2 * horizon))
    v = ic.dropna()
    sd = float(v.std(ddof=1)) if len(v) > 1 else np.nan
    return {
        "ic_mean": nw["mean"],
        "ic_std": sd,
        "ic_ir": nw["mean"] / sd if sd and np.isfinite(sd) and sd > 0 else np.nan,
        "ic_tstat_nw": nw["tstat"],
        "ic_pvalue": two_sided_p(nw["tstat"], nw["n"] - 1),
        "ic_hit_rate": float((v > 0).mean()) if len(v) else np.nan,
        "ic_n_dates": int(nw["n"]),
    }


def fama_macbeth(
    score: pd.DataFrame, fwd: pd.DataFrame, horizon: int, min_names: int = 5
) -> Dict[str, float]:
    """
    Fama-MacBeth (1973) test of the ranking hypothesis.

    Each date, regress the cross-section of forward returns on the cross-sectionally
    standardised score. The slope lambda_t is the return to a unit-exposure bet on the
    signal that day. The hypothesis "this signal ranks assets" is exactly
    H0: E[lambda_t] = 0, tested with a Newey-West HAC t-stat on the slope series.

    Also reports the average cross-sectional R^2 -- how much of the day-to-day
    dispersion in returns the ranking actually explains.
    """
    z = cs_zscore(score, min_names=min_names)
    z, f = z.align(fwd, join="inner")

    slopes, r2s = {}, {}
    for d in z.index:
        x, y = z.loc[d], f.loc[d]
        m = x.notna() & y.notna()
        if int(m.sum()) < min_names:
            continue
        xv, yv = x[m].to_numpy(float), y[m].to_numpy(float)
        xc = xv - xv.mean()
        denom = float(xc @ xc)
        if denom <= 0:
            continue
        beta = float(xc @ (yv - yv.mean())) / denom
        alpha = float(yv.mean() - beta * xv.mean())
        resid = yv - (alpha + beta * xv)
        tss = float(((yv - yv.mean()) ** 2).sum())
        slopes[d] = beta
        r2s[d] = 1.0 - float(resid @ resid) / tss if tss > 0 else np.nan

    lam = pd.Series(slopes, dtype=float)
    nw = newey_west_tstat(lam, lags=max(1, 2 * horizon))
    return {
        "fm_lambda_mean": nw["mean"],
        "fm_lambda_se_nw": nw["se"],
        "fm_tstat_nw": nw["tstat"],
        "fm_pvalue": two_sided_p(nw["tstat"], nw["n"] - 1),
        "fm_avg_r2": float(pd.Series(r2s).mean()) if r2s else np.nan,
        "fm_n_dates": int(nw["n"]),
        "_lambda_series": lam,
    }


def spread_portfolio(
    score: pd.DataFrame,
    panel: Panel,
    top_n: int = 6,
    rebalance: int = 5,
    n_boot: int = 2000,
) -> Dict[str, float]:
    """
    Economic read on the ranking: an equal-weight long-top-N / short-bottom-N book,
    held between rebalances, marked on execution-aligned open-to-open returns.

    This is deliberately *not* the final strategy -- there is no volatility sizing and
    no cost model. It isolates one question: does the ordering the signal produces
    separate winners from losers at all?
    """
    daily_fwd = forward_returns(panel, horizon=1, price="open")

    held: Optional[pd.Series] = None
    rets, dates = [], []
    for i, d in enumerate(score.index):
        if i % rebalance == 0:
            row = score.loc[d].dropna()
            if len(row) >= 2 * top_n:
                order = row.sort_values()
                w = pd.Series(0.0, index=score.columns)
                w[order.index[-top_n:]] = 0.5 / top_n
                w[order.index[:top_n]] = -0.5 / top_n
                held = w
        if held is None:
            continue
        f = daily_fwd.loc[d]
        contrib = (held * f.reindex(held.index)).dropna()
        if len(contrib):
            rets.append(float(contrib.sum()))
            dates.append(d)

    r = pd.Series(rets, index=pd.DatetimeIndex(dates), dtype=float)
    if len(r) < 60:
        return {"spread_ann_return_pct": np.nan, "spread_ann_vol_pct": np.nan,
                "spread_sharpe": np.nan, "spread_sharpe_ci_low": np.nan,
                "spread_sharpe_ci_high": np.nan, "spread_tstat_nw": np.nan,
                "spread_max_dd_pct": np.nan, "spread_n_days": len(r)}

    mu, sd = r.mean(), r.std(ddof=1)
    sharpe = float(mu / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else np.nan
    nw = newey_west_tstat(r, lags=max(1, rebalance))
    ci = block_bootstrap_sharpe(r.to_numpy(float), n_boot=n_boot)
    eq = (1.0 + r).cumprod()
    dd = float((eq / eq.cummax() - 1.0).min() * 100.0)
    return {
        "spread_ann_return_pct": float(mu * TRADING_DAYS * 100.0),
        "spread_ann_vol_pct": float(sd * np.sqrt(TRADING_DAYS) * 100.0),
        "spread_sharpe": sharpe,
        "spread_sharpe_ci_low": ci["sharpe_ci_low"],
        "spread_sharpe_ci_high": ci["sharpe_ci_high"],
        "spread_tstat_nw": nw["tstat"],
        "spread_max_dd_pct": dd,
        "spread_n_days": int(len(r)),
        "_spread_returns": r,
    }


def top_basket_profile(
    score: pd.DataFrame, panel: Panel, top_n: int = 6, corr_window: int = 126
) -> Dict[str, float]:
    """
    Correlation / concentration analysis of the top-ranked assets (deliverable 2).

    A ranking that keeps selecting six names which all move together is not a
    diversified book -- it is one bet with six tickers, and an equal-weight sizing
    rule would badly understate its risk. Reported:

      * mean pairwise trailing correlation inside the long basket, versus the same
        statistic for the universe -- the excess is the concentration the signal adds
      * effective number of bets, 1 / sum(w_eff^2) on the equal-correlation
        approximation, i.e. N / (1 + (N-1) * rho_bar)
      * basket persistence (overlap with the previous rebalance) and name turnover
    """
    rets = panel.returns_est
    bask_rho, univ_rho, overlap = [], [], []
    prev: Optional[set] = None

    probe = score.index[corr_window::5]
    for d in probe:
        row = score.loc[d].dropna()
        if len(row) < 2 * top_n:
            continue
        names = list(row.sort_values().index[-top_n:])
        win = rets.loc[:d].tail(corr_window)

        c = win[names].corr()
        iu = np.triu_indices(len(names), k=1)
        if len(iu[0]):
            bask_rho.append(float(np.nanmean(c.to_numpy()[iu])))

        cu = win.corr()
        iu2 = np.triu_indices(cu.shape[0], k=1)
        if len(iu2[0]):
            univ_rho.append(float(np.nanmean(cu.to_numpy()[iu2])))

        if prev is not None:
            overlap.append(len(prev & set(names)) / float(top_n))
        prev = set(names)

    rho_b = float(np.nanmean(bask_rho)) if bask_rho else np.nan
    rho_u = float(np.nanmean(univ_rho)) if univ_rho else np.nan
    eff_n = top_n / (1.0 + (top_n - 1) * rho_b) if np.isfinite(rho_b) and rho_b > -1 else np.nan
    return {
        "top_mean_pairwise_corr": rho_b,
        "universe_mean_pairwise_corr": rho_u,
        "excess_corr_vs_universe": rho_b - rho_u if np.isfinite(rho_b) and np.isfinite(rho_u) else np.nan,
        "top_effective_n_bets": eff_n,
        "top_basket_overlap": float(np.nanmean(overlap)) if overlap else np.nan,
    }


def signal_persistence(score: pd.DataFrame, rebalance: int = 5) -> Dict[str, float]:
    """
    Autocorrelation of the cross-sectional ranking and the turnover it implies.

    A signal whose ranking decays in days cannot survive transaction costs no matter
    how strong its IC; this is the cost-side counterweight to the IC table.
    """
    lagged = score.shift(rebalance)
    rho = [
        score.loc[d].corr(lagged.loc[d], method="spearman")
        for d in score.index[rebalance:]
        if score.loc[d].notna().sum() >= 5 and lagged.loc[d].notna().sum() >= 5
    ]
    w = score.div(score.abs().sum(axis=1).replace(0.0, np.nan), axis=0)
    turn = w.diff(rebalance).abs().sum(axis=1).replace(0.0, np.nan).dropna()
    return {
        "rank_autocorr": float(np.nanmean(rho)) if rho else np.nan,
        "weight_turnover_per_rebalance": float(turn.mean()) if len(turn) else np.nan,
    }


@dataclass
class SignalEvaluation:
    name: str
    horizon: int
    stats: Dict[str, float] = field(default_factory=dict)
    ic_series: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    spread_returns: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    lambda_series: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))


def evaluate_signal(
    name: str,
    score: pd.DataFrame,
    panel: Panel,
    horizon: int = 5,
) -> SignalEvaluation:
    """
    Horizon-dependent battery for one signal: IC and Fama-MacBeth at `horizon`.

    Statistics that do not depend on the evaluation horizon -- the spread portfolio,
    the concentration profile, persistence -- live in `evaluate_signal_static` so they
    are computed once per signal rather than once per horizon.
    """
    fwd = forward_returns(panel, horizon=horizon, price="open")

    ic = information_coefficient(score, fwd)
    stats: Dict[str, float] = {"horizon_days": horizon}
    stats.update(ic_summary(ic, horizon))

    fm = fama_macbeth(score, fwd, horizon=horizon)
    lam = fm.pop("_lambda_series")
    stats.update(fm)

    return SignalEvaluation(name=name, horizon=horizon, stats=stats, ic_series=ic,
                            lambda_series=lam)


def evaluate_signal_static(
    name: str,
    score: pd.DataFrame,
    panel: Panel,
    top_n: int = 6,
    rebalance: int = 5,
    n_boot: int = 2000,
) -> SignalEvaluation:
    """Horizon-invariant battery: spread portfolio, concentration, persistence."""
    sp = spread_portfolio(score, panel, top_n=top_n, rebalance=rebalance, n_boot=n_boot)
    spr = sp.pop("_spread_returns", pd.Series(dtype=float))
    stats: Dict[str, float] = dict(sp)
    stats.update(top_basket_profile(score, panel, top_n=top_n))
    stats.update(signal_persistence(score, rebalance=rebalance))
    return SignalEvaluation(name=name, horizon=rebalance, stats=stats, spread_returns=spr)


def compare_signals(
    panel: Panel,
    names: Optional[Sequence[str]] = None,
    horizons: Sequence[int] = (1, 5, 21),
    standardise: str = "rank",
    top_n: int = 6,
    rebalance: int = 5,
    n_boot: int = 2000,
) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, SignalEvaluation]]:
    """
    Evaluate every signal and return

      ic_table      -- (signal x horizon): IC and Fama-MacBeth statistics
      static_table  -- (signal): spread portfolio, concentration, persistence
      evals         -- the underlying evaluation objects, keyed ("name", horizon)
                       for horizon-dependent runs and ("name", "static") otherwise,
                       so the report can plot IC and equity series directly.
    """
    scores = build_signals(panel, names, standardise=standardise)
    ic_rows, static_rows, evals = [], [], {}
    for name, score in scores.items():
        for h in horizons:
            ev = evaluate_signal(name, score, panel, horizon=h)
            evals[(name, h)] = ev
            ic_rows.append({"signal": name, **ev.stats})
        sev = evaluate_signal_static(name, score, panel, top_n=top_n,
                                     rebalance=rebalance, n_boot=n_boot)
        evals[(name, "static")] = sev
        static_rows.append({"signal": name, **sev.stats})

    ic_table = pd.DataFrame(ic_rows).set_index(["signal", "horizon_days"]).sort_index()
    static_table = pd.DataFrame(static_rows).set_index("signal").sort_index()
    return ic_table, static_table, evals


def _print_comparison(ic_table: pd.DataFrame, static_table: pd.DataFrame) -> None:
    ic_cols = ["ic_mean", "ic_std", "ic_ir", "ic_tstat_nw", "ic_pvalue", "ic_hit_rate", "ic_n_dates"]
    fm_cols = ["fm_lambda_mean", "fm_tstat_nw", "fm_pvalue", "fm_avg_r2", "fm_n_dates"]
    sp_cols = ["spread_ann_return_pct", "spread_ann_vol_pct", "spread_sharpe",
               "spread_sharpe_ci_low", "spread_sharpe_ci_high", "spread_tstat_nw",
               "spread_max_dd_pct", "spread_n_days"]
    cc_cols = ["top_mean_pairwise_corr", "universe_mean_pairwise_corr",
               "excess_corr_vs_universe", "top_effective_n_bets", "top_basket_overlap",
               "rank_autocorr", "weight_turnover_per_rebalance"]

    blocks = [
        ("INFORMATION COEFFICIENT (Spearman, Newey-West t)", ic_table, ic_cols),
        ("FAMA-MACBETH CROSS-SECTIONAL REGRESSION", ic_table, fm_cols),
        ("TOP-MINUS-BOTTOM SPREAD PORTFOLIO (block-bootstrap 95% CI)", static_table, sp_cols),
        ("CORRELATION / CONCENTRATION & PERSISTENCE", static_table, cc_cols),
    ]
    for title, tbl, cols in blocks:
        print(f"\n-- {title} " + "-" * max(2, 86 - len(title)))
        print(tbl[[c for c in cols if c in tbl.columns]].round(4).to_string())


if __name__ == "__main__":
    import argparse

    from src.utils.data import load_panel
    from src.utils.signals import assert_causal

    ap = argparse.ArgumentParser(
        description="Build and statistically compare Track A ranking signals."
    )
    ap.add_argument("--data", default=None, help="Path to Dataset_PS-A.csv")
    ap.add_argument("--standardise", default="rank", choices=["rank", "zscore", "raw"])
    ap.add_argument("--top-n", type=int, default=6)
    ap.add_argument("--rebalance", type=int, default=5)
    ap.add_argument("--horizons", default="1,5,21")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    np.random.seed(SEED)
    panel = load_panel(args.data)
    print(panel.report.summary())

    print("\n[causality] rebuilding each signal on truncated panels ...")
    chk = assert_causal(panel, standardise=args.standardise)
    print(f"[causality] PASS - {len(chk)} point-in-time checks, "
          f"max abs deviation {chk['max_abs_diff'].max():.2e}")

    horizons = tuple(int(h) for h in args.horizons.split(","))
    ic_table, static_table, _ = compare_signals(
        panel, horizons=horizons, standardise=args.standardise,
        top_n=args.top_n, rebalance=args.rebalance, n_boot=args.n_boot)
    print("\n" + "=" * 90)
    print(f"{'CROSS-SECTIONAL SIGNAL COMPARISON':^90}")
    print("=" * 90)
    _print_comparison(ic_table, static_table)
    print()
