"""
src/research/walkforward.py - Walk-forward validation.

The required deliverable the rest of the repo was missing. Everything else here is
single-split in-sample: parameters were chosen by looking at the whole history, so an
in-sample Sharpe cannot tell you whether the *selection procedure* generalises.

What is actually being validated
--------------------------------
This strategy has no fitted coefficients -- there is no regression to re-estimate each
fold. What *is* fitted is the researcher's parameter choice: signal, cadence, sizing
rule, volatility model. That choice used information from the whole sample, and it is
the thing most likely not to survive.

So each fold does:

    1. look only at the TRAIN window, pick the best configuration from a candidate set
    2. apply that configuration, unchanged, to the following TEST window
    3. record how it did

Stitching the test windows together gives an out-of-sample equity curve for the whole
*procedure*, not for a strategy that already knew the answer.

A note on honesty: the candidate set below was itself shaped by earlier in-sample
work, so this is not a pristine OOS experiment -- nothing run on data you have already
looked at can be. What it does measure is whether picking a configuration on past data
beats picking one at random, and whether the chosen configuration is stable across
folds. Fold-to-fold instability in the selected parameters is the real warning sign,
and is reported explicitly.

Implementation note: the strategy is run ONCE per configuration over the full history
(which is what live trading looks like -- state carries forward), and the resulting
daily equity curve is then sliced into windows. Re-running from scratch inside each
window would discard the warmup and measure something different.

    python -m src.research.walkforward
    python -m src.research.walkforward --train 504 --test 126
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

from backtester import Backtester, BacktestConfig   # noqa: E402
from src.engine import ParticipantStrategy          # noqa: E402

DATA = "data/Dataset_PS-A.csv"
TRADING_DAYS = 252

# Candidate configurations the walk-forward chooses between each fold.
# Kept deliberately small and sensible: a large grid searched on short training
# windows would select noise, which is the very failure this is meant to detect.
CANDIDATES: Dict[str, dict] = {
    "baseline_topn_21d":   {"smooth": 1},
    "ENSEMBLE_adopted":    {},
    "cadence_42d":         {"rebalance_cadence": 42},
    "sizing_erc":          {"sizing_method": "erc"},
    "hysteresis_4":        {"exit_buffer": 4},
    "hysteresis_9":        {"exit_buffer": 9},
    "smooth_10":           {"smooth": 10},
    "smooth_16":           {"smooth": 16},
    "full_cross_section":  {"selection": "full"},
    "momentum":            {"signal": "MOM_12_1"},
}


def _run(slippage: float, **overrides) -> pd.Series:
    """Run one configuration over the full history; return its daily equity curve."""
    cfg = BacktestConfig(
        track="A", data_path=DATA, initial_cash=100_000.0,
        slippage_rate=slippage, commission_rate=0.0001,
        save_results=False, generate_plot=False,
    )
    strat = ParticipantStrategy(**overrides)
    with contextlib.redirect_stdout(io.StringIO()):
        Backtester(cfg).run(strat)
    return strat.equity_frame()


def window_metrics(equity: pd.Series) -> Dict[str, float]:
    """Sharpe / return / drawdown computed inside one window of an equity curve."""
    r = equity.pct_change().dropna()
    if len(r) < 20 or r.std(ddof=1) == 0:
        return {"sharpe": np.nan, "ann_return_pct": np.nan, "max_dd_pct": np.nan, "n": len(r)}
    sharpe = float(r.mean() / r.std(ddof=1) * np.sqrt(TRADING_DAYS))
    eq = (1 + r).cumprod()
    return {
        "sharpe": sharpe,
        "ann_return_pct": float(r.mean() * TRADING_DAYS * 100),
        "max_dd_pct": float((eq / eq.cummax() - 1).min() * 100),
        "n": len(r),
    }


def walk_forward(
    train: int = 504,
    test: int = 126,
    slippage: float = 0.0005,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Rolling train/test evaluation.

    train / test are in trading sessions (504 ~ 2 years, 126 ~ 6 months).
    Returns (per-fold table, stitched OOS return series, baseline OOS return series).
    """
    print(f"Running {len(CANDIDATES)} candidate configurations over the full history "
          f"(slippage {slippage*1e4:.0f}bp) ...")
    curves = {}
    for name, ov in CANDIDATES.items():
        curves[name] = _run(slippage, **ov)
        print(f"  {name:<26} done")

    dates = curves["baseline_topn_21d"].index
    # Skip the strategy warmup: equity is flat there and would distort window metrics.
    first = max(160, train)

    rows: List[dict] = []
    oos_parts: List[pd.Series] = []
    base_parts: List[pd.Series] = []

    start = first
    while start + test <= len(dates):
        tr_slice = slice(start - train, start)
        te_slice = slice(start, start + test)

        # 1. choose using TRAIN only
        train_scores = {n: window_metrics(c.iloc[tr_slice])["sharpe"] for n, c in curves.items()}
        train_scores = {k: v for k, v in train_scores.items() if np.isfinite(v)}
        if not train_scores:
            start += test
            continue
        chosen = max(train_scores, key=train_scores.get)

        # 2. apply it, unchanged, to TEST
        te = window_metrics(curves[chosen].iloc[te_slice])
        base = window_metrics(curves["baseline_topn_21d"].iloc[te_slice])

        rows.append({
            "fold": len(rows) + 1,
            "train_start": dates[start - train].date(),
            "test_start": dates[start].date(),
            "test_end": dates[start + test - 1].date(),
            "chosen": chosen,
            "train_sharpe": round(train_scores[chosen], 3),
            "test_sharpe": round(te["sharpe"], 3),
            "baseline_test_sharpe": round(base["sharpe"], 3),
            "test_return%": round(te["ann_return_pct"], 2),
            "test_maxDD%": round(te["max_dd_pct"], 2),
        })
        oos_parts.append(curves[chosen].iloc[te_slice].pct_change().dropna())
        base_parts.append(curves["baseline_topn_21d"].iloc[te_slice].pct_change().dropna())
        start += test

    folds = pd.DataFrame(rows)
    oos = pd.concat(oos_parts) if oos_parts else pd.Series(dtype=float)
    baseline = pd.concat(base_parts) if base_parts else pd.Series(dtype=float)
    return folds, oos, baseline


def _summary(r: pd.Series, label: str) -> Dict[str, float]:
    if len(r) < 20:
        return {}
    sharpe = float(r.mean() / r.std(ddof=1) * np.sqrt(TRADING_DAYS))
    eq = (1 + r).cumprod()
    return {
        "label": label,
        "sharpe": round(sharpe, 3),
        "ann_return%": round(float(r.mean() * TRADING_DAYS * 100), 2),
        "ann_vol%": round(float(r.std(ddof=1) * np.sqrt(TRADING_DAYS) * 100), 2),
        "max_dd%": round(float((eq / eq.cummax() - 1).min() * 100), 2),
        "total_return%": round(float(eq.iloc[-1] - 1) * 100, 2),
        "days": len(r),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Walk-forward validation for Track A.")
    ap.add_argument("--train", type=int, default=504, help="training window, sessions")
    ap.add_argument("--test", type=int, default=126, help="test window, sessions")
    ap.add_argument("--slippage-bp", type=float, default=5.0,
                    help="execution slippage in basis points (default 5)")
    args = ap.parse_args()

    folds, oos, baseline = walk_forward(args.train, args.test, args.slippage_bp / 1e4)

    print("\n" + "=" * 104)
    print(f"{'WALK-FORWARD FOLDS':^104}")
    print("=" * 104)
    print(folds.to_string(index=False))

    print("\n" + "=" * 104)
    print(f"{'STITCHED OUT-OF-SAMPLE RESULT':^104}")
    print("=" * 104)
    summ = pd.DataFrame([s for s in [
        _summary(oos, "walk-forward selection"),
        _summary(baseline, "fixed baseline (no selection)"),
    ] if s])
    print(summ.to_string(index=False))

    if len(folds):
        picks = folds["chosen"].value_counts()
        print(f"\nConfiguration chosen per fold ({len(folds)} folds):")
        for k, v in picks.items():
            print(f"  {k:<28} {v} fold(s)")
        print(f"\nStability: {len(picks)} distinct configurations selected across "
              f"{len(folds)} folds.")
        print("  Many different winners => the selection is fitting noise, and the")
        print("  in-sample parameter choice should not be trusted.")
        beat = (folds["test_sharpe"] > folds["baseline_test_sharpe"]).mean()
        print(f"\nSelection beat the fixed baseline in {beat:.0%} of folds.")
        print("  At or below ~50%, training-window selection is adding nothing and the")
        print("  simpler fixed configuration is the honest choice.")

    import os
    os.makedirs("results", exist_ok=True)
    folds.to_csv("results/walkforward.csv", index=False)
    print("\nSaved -> results/walkforward.csv")


if __name__ == "__main__":
    main()
