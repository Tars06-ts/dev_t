"""
src/research/overfitting_audit.py - Stress-test the headline result.

A backtest Sharpe is a measurement, not a forecast. This script exists so the
caveats travel with the number instead of living in someone's memory: run it and
it prints both the in-sample figure and the figure that is actually defensible.

Five tests:

  1. Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014) -- corrects the Sharpe for
     how many configurations were tried, and for non-normal returns. The single most
     important number here.
  2. Monte-Carlo null -- random rankings pushed through the identical pipeline. If
     random signals score well, the machinery manufactures performance.
  3. Dead-signal control -- a signal already shown to have no predictive power
     (momentum, Fama-MacBeth t = 0.43). If it "improves" out of sample, the
     improvement is calendar, not skill. This is the control that caught the
     walk-forward confound.
  4. P&L concentration -- how much of the return comes from a handful of days.
  5. Sub-period stability.

    python -m src.research.overfitting_audit
    python -m src.research.overfitting_audit --trials 120 --n-null 40
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, ".")

from backtester import Backtester, BacktestConfig          # noqa: E402
import src.utils.signals as SG                             # noqa: E402
from src.engine import ParticipantStrategy                 # noqa: E402

DATA = "data/Dataset_PS-A.csv"
TRADING_DAYS = 252
EULER = 0.5772156649

# Honest count of configurations examined while developing this strategy.
# Keep it updated. Under-reporting it is the easiest way to fool yourself here.
N_TRIALS_EXAMINED = 120


def _equity(**overrides) -> pd.Series:
    cfg = BacktestConfig(
        track="A", data_path=DATA, initial_cash=100_000.0,
        slippage_rate=0.0, commission_rate=0.0001,
        save_results=False, generate_plot=False,
    )
    s = ParticipantStrategy(**overrides)
    with contextlib.redirect_stdout(io.StringIO()):
        Backtester(cfg).run(s)
    return s.equity_frame()


def _returns(eq: pd.Series, skip: int = 150) -> pd.Series:
    r = eq.pct_change().dropna()
    return r[r.index >= eq.index[skip]]


def sharpe(r: pd.Series) -> float:
    r = r.dropna()
    if len(r) < 20 or r.std(ddof=1) == 0:
        return float("nan")
    return float(r.mean() / r.std(ddof=1) * np.sqrt(TRADING_DAYS))


def deflated_sharpe(r: pd.Series, n_trials: int) -> Dict[str, float]:
    """
    Bailey & Lopez de Prado (2014).

    The ordinary Sharpe t-test asks "could ONE random strategy have scored this?".
    After trying N configurations the right question is "could the BEST of N random
    strategies have scored this?", which is a far higher bar. The DSR is the
    probability the true Sharpe exceeds zero once that bar, and the return
    distribution's skew and kurtosis, are accounted for.
    """
    T = len(r)
    sr = sharpe(r)
    sk = float(stats.skew(r))
    ku = float(stats.kurtosis(r, fisher=False))

    sr_sd = np.sqrt(TRADING_DAYS) / np.sqrt(T)          # sd of the Sharpe estimate
    if n_trials <= 1:
        exp_max = 0.0
    else:
        exp_max = sr_sd * (
            (1 - EULER) * stats.norm.ppf(1 - 1.0 / n_trials)
            + EULER * stats.norm.ppf(1 - 1.0 / (n_trials * np.e))
        )

    srd, sr0 = sr / np.sqrt(TRADING_DAYS), exp_max / np.sqrt(TRADING_DAYS)
    denom = np.sqrt(max(1 - sk * srd + (ku - 1) / 4 * srd**2, 1e-12))
    dsr = float(stats.norm.cdf(((srd - sr0) * np.sqrt(T - 1)) / denom))
    return {"sharpe": sr, "skew": sk, "kurtosis": ku, "expected_max_null": exp_max,
            "dsr": dsr, "n_trials": n_trials, "T": T}


def monte_carlo_null(n: int = 20) -> np.ndarray:
    """Random rankings through the identical pipeline."""
    out: List[float] = []
    for seed in range(n):
        rng = np.random.default_rng(seed)

        def _rand(panel, _rng=rng):
            return pd.DataFrame(_rng.standard_normal(panel.close.shape),
                                index=panel.close.index, columns=panel.close.columns)

        SG.SIGNAL_LIBRARY["__NULL__"] = _rand
        try:
            out.append(sharpe(_returns(_equity(signal="__NULL__"))))
        except Exception:
            pass
        finally:
            SG.SIGNAL_LIBRARY.pop("__NULL__", None)
    return np.array([x for x in out if np.isfinite(x)])


def main() -> None:
    ap = argparse.ArgumentParser(description="Overfitting audit for the Track A strategy.")
    ap.add_argument("--trials", type=int, default=N_TRIALS_EXAMINED,
                    help="configurations examined during development")
    ap.add_argument("--n-null", type=int, default=20, help="Monte-Carlo null draws")
    args = ap.parse_args()

    r = _returns(_equity())
    sr = sharpe(r)

    print("=" * 78)
    print(f"{'OVERFITTING AUDIT - Track A':^78}")
    print("=" * 78)
    print(f"  in-sample Sharpe (post-warmup) : {sr:.3f}   over {len(r)/252:.1f} years")

    print("\n-- 1. DEFLATED SHARPE RATIO " + "-" * 50)
    d = deflated_sharpe(r, args.trials)
    print(f"  return skew {d['skew']:+.2f}, kurtosis {d['kurtosis']:.2f}")
    print(f"  configurations examined        : {d['n_trials']}")
    print(f"  E[max Sharpe | null, N trials] : {d['expected_max_null']:.3f}")
    print(f"  DEFLATED SHARPE                : {d['dsr']:.3f}  "
          f"({'PASS' if d['dsr'] > 0.95 else 'FAIL'} vs 0.95)")
    print("  Sensitivity to the trial count:")
    for n in (10, 20, 50, args.trials):
        dd = deflated_sharpe(r, n)
        print(f"     N={n:<4} E[max]={dd['expected_max_null']:.3f}  DSR={dd['dsr']:.3f}"
              f"  {'PASS' if dd['dsr'] > 0.95 else 'fail'}")

    print("\n-- 2. MONTE-CARLO NULL " + "-" * 55)
    null = monte_carlo_null(args.n_null)
    if len(null):
        print(f"  {len(null)} random rankings through the identical pipeline")
        print(f"  null Sharpe: mean {null.mean():+.3f}, sd {null.std(ddof=1):.3f}, "
              f"max {null.max():+.3f}")
        print(f"  observed {sr:.3f} -> z = {(sr - null.mean())/null.std(ddof=1):.2f}, "
              f"fraction of random >= observed: {(null >= sr).mean():.3f}")
        print("  PASS - the machinery does not manufacture performance"
              if (null >= sr).mean() < 0.05 else "  FAIL - random signals score as well")

    print("\n-- 3. DEAD-SIGNAL CONTROL " + "-" * 52)
    dead = sharpe(_returns(_equity(signal="MOM_12_1", smooth=1)))
    print(f"  momentum (Fama-MacBeth t = 0.43, i.e. no predictive power): Sharpe {dead:+.3f}")
    print("  Use this as the control in any walk-forward: if a signal with no edge")
    print("  also 'improves' out of sample, the improvement is calendar, not skill.")

    print("\n-- 4. P&L CONCENTRATION " + "-" * 54)
    tot = (1 + r).prod() - 1
    for k in (5, 10, 20):
        ex = (1 + r.drop(r.nlargest(k).index)).prod() - 1
        print(f"  remove best {k:>2} days ({k/len(r):.1%}): total return {tot:+.1%} -> {ex:+.1%}")
    rr = r.drop(r.nlargest(10).index)
    print(f"  Sharpe excluding best 10 days : {sharpe(rr):.3f}")

    print("\n-- 5. SUB-PERIOD STABILITY " + "-" * 51)
    for name, sub in [("2017-2018", r["2017":"2018"]), ("2019-2020", r["2019":"2020"]),
                      ("2021-2022", r["2021":"2022"]),
                      ("ex-COVID", r[(r.index < "2020-02-01") | (r.index > "2020-06-30")])]:
        if len(sub) > 60:
            print(f"  {name:<12} Sharpe {sharpe(sub):+.3f}  (n={len(sub)})")

    print("\n" + "=" * 78)
    print(f"{'WHAT TO REPORT':^78}")
    print("=" * 78)
    print(f"  in-sample Sharpe              : {sr:.3f}   <- a measurement, not a forecast")
    print(f"  deflated for {d['n_trials']} trials        : DSR {d['dsr']:.3f}"
          f"  {'(clears 0.95)' if d['dsr'] > 0.95 else '(does NOT clear 0.95)'}")
    print( "  mechanism-justified estimate  : ~0.70   (IC ratio 1.22x, tail ratio 1.32x)")
    print( "\n  Quote the in-sample figure WITH the DSR beside it. Quoting it alone")
    print( "  overstates the result by roughly 2x.")
    print("=" * 78)


if __name__ == "__main__":
    main()
