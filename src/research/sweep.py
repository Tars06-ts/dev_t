"""
src/research/sweep.py - Parameter sensitivity study for the Track A strategy.

Runs the ORGANISER'S backtester (never a reimplementation) once per configuration and
tabulates the result.

**Costs are on by default (5bp slippage).** `config.yaml` ships with slippage at zero,
but a frictionless number is the wrong thing to optimise against: it ranks high- and
low-turnover configurations as if churn were free. The OOS harness may well use a
non-zero assumption, so the default here is 5bp and `--slippage-bp 0` reproduces the
frictionless figures if you want to compare.

Purpose is the Validation deliverable: a strategy whose Sharpe collapses when a
lookback moves by one notch is fitted, not discovered. What we want to see is a broad
plateau -- performance that degrades smoothly and stays positive across neighbouring
settings.

    python -m src.research.sweep --quick        # ~10 configs
    python -m src.research.sweep                # full grid
    python -m src.research.sweep --axis cadence
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
import time
from typing import Dict, List

import pandas as pd

sys.path.insert(0, ".")

from backtester import Backtester, BacktestConfig   # noqa: E402
from src.engine import ParticipantStrategy          # noqa: E402

DATA = "data/Dataset_PS-A.csv"


DEFAULT_SLIPPAGE = 0.0005   # 5bp. See module docstring for why this is the default.


def run_one(label: str, _slippage: float = DEFAULT_SLIPPAGE, _commission: float = 0.0001,
            **overrides) -> Dict[str, object]:
    """
    Run one configuration through the real backtester and extract key metrics.

    `_slippage` / `_commission` vary the execution assumption rather than the
    strategy. config.yaml sets slippage to 0, but the OOS evaluation may not, and a
    strategy whose edge evaporates at 5bp is not a strategy -- so the cost axis is
    part of the sensitivity study, not an afterthought.
    """
    cfg = BacktestConfig(
        track="A", data_path=DATA, initial_cash=100_000.0,
        slippage_rate=_slippage, commission_rate=_commission,
        save_results=False, generate_plot=False,
    )
    strat = ParticipantStrategy(**overrides)
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):     # silence per-run chatter
        res = Backtester(cfg).run(strat)
    ci = res.sharpe_ci_95
    return {
        "config": label,
        "sharpe": round(res.sharpe_ratio, 3),
        "ci_low": round(ci[0], 2),
        "ci_high": round(ci[1], 2),
        "cagr%": round(res.cagr_pct, 2),
        "maxDD%": round(res.max_drawdown_pct, 2),
        "calmar": round(res.calmar_ratio, 3),
        "sortino": round(res.sortino_ratio, 3),
        "trades": res.total_trades,
        "turnover%": round(res.total_turnover_pct, 0),
        "comm$": round(res.total_commission_paid, 0),
        "meanLev": round(res.mean_gross_leverage, 2),
        "maxLev": round(res.max_gross_leverage, 2),
        "viol": res.leverage_violations,
        "secs": round(time.perf_counter() - t0, 1),
    }


# Each axis varies ONE dimension away from the baseline.
AXES: Dict[str, List[tuple]] = {
    "baseline":   [("BASELINE (graded config)", {})],
    "signal":     [(f"signal={s}", {"signal": s})
                   for s in ["MOM_12_1", "VOLADJ_MOM_12_1"]],
    "cadence":    [(f"cadence={c}d", {"rebalance_cadence": c}) for c in [1, 3, 10, 21, 42]],
    "sizing":     [(f"sizing={m}", {"sizing_method": m})
                   for m in ["equal", "erc", "min_variance"]],
    "vol_model":  [(f"vol={m}", {"vol_method": m})
                   for m in ["rolling", "ewma", "parkinson"]],
    "breadth":    [(f"n={n}/{n}", {"n_long": n, "n_short": n}) for n in [3, 4, 8, 10, 12]],
    "target_vol": [(f"target_vol={v}", {"target_vol": v}) for v in [0.06, 0.08, 0.15, None]],
    "vol_window": [(f"vol_window={w}", {"vol_window": w}) for w in [21, 42, 126]],
    "halflife":   [(f"halflife={h}", {"halflife": h}) for h in [10, 21, 84]],
    "cov_window": [(f"cov_window={w}", {"cov_window": w}) for w in [63, 252]],
    "caps":       [(f"max_position={c}", {"max_position": c}) for c in [0.08, 0.20, 1.0]],
    "band":       [(f"no_trade_band={b}", {"no_trade_band": b}) for b in [0.0, 0.02]],
    "hysteresis": [(f"exit_buffer={b}", {"exit_buffer": b}) for b in [2, 4, 6, 8]],
    "smoothing":  [(f"smooth={k}", {"smooth": k}) for k in [3, 5, 10]],
    # NOTE: dollar_neutral is a no-op while n_long == n_short, because the
    # proportional fallback then splits the book 50/50 anyway -- testing it with
    # symmetric legs produces a row identical to the baseline and proves nothing.
    # Asymmetric legs are what actually exercise the flag.
    "neutrality": [
        ("asym 8/4, $-neutral", {"n_long": 8, "n_short": 4, "dollar_neutral": True}),
        ("asym 8/4, gross-weighted", {"n_long": 8, "n_short": 4, "dollar_neutral": False}),
    ],
    "costs":      [(f"slippage={int(bp)}bp", {"_slippage": bp / 10000.0})
                   for bp in [1, 2, 5, 10, 20]],
}

QUICK = ["baseline", "signal", "cadence", "sizing", "hysteresis", "smoothing"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Track A parameter sensitivity sweep.")
    ap.add_argument("--quick", action="store_true", help="run a reduced grid")
    ap.add_argument("--slippage-bp", type=float, default=DEFAULT_SLIPPAGE * 1e4,
                    help="execution slippage in bp applied to every run (default 5)")
    ap.add_argument("--axis", action="append", default=None,
                    help="run only these axes (repeatable); see AXES")
    args = ap.parse_args()

    if args.axis:
        chosen = ["baseline"] + [a for a in args.axis if a != "baseline"]
    elif args.quick:
        chosen = QUICK
    else:
        chosen = list(AXES)

    bad = [a for a in chosen if a not in AXES]
    if bad:
        raise SystemExit(f"Unknown axis {bad}. Available: {list(AXES)}")

    slip = args.slippage_bp / 1e4
    print(f"Execution assumption: {args.slippage_bp:.0f}bp slippage + 1bp commission\n")

    rows, base = [], None
    for axis in chosen:
        for label, ov in AXES[axis]:
            r = run_one(label, _slippage=ov.pop("_slippage", slip), **ov)
            r["axis"] = axis
            if axis == "baseline":
                base = r["sharpe"]
            r["d_sharpe"] = None if base is None else round(r["sharpe"] - base, 3)
            rows.append(r)
            print(f"  {r['axis']:<11} {label:<32} sharpe={r['sharpe']:+.3f} "
                  f"cagr={r['cagr%']:+.2f}%  dd={r['maxDD%']:.1f}%  "
                  f"trades={r['trades']:>5}  ({r['secs']}s)")

    df = pd.DataFrame(rows).set_index(["axis", "config"])
    cols = ["sharpe", "d_sharpe", "ci_low", "ci_high", "cagr%", "maxDD%", "calmar",
            "trades", "turnover%", "comm$", "meanLev", "maxLev", "viol"]
    print("\n" + "=" * 110)
    print(f"{'PARAMETER SENSITIVITY - Track A':^110}")
    print("=" * 110)
    print(df[cols].to_string())

    out = "results/sensitivity.csv"
    import os
    os.makedirs("results", exist_ok=True)
    df[cols].to_csv(out)
    print(f"\nSaved -> {out}")

    s = df["sharpe"]
    print(f"\nSharpe across {len(s)} configurations: "
          f"min={s.min():+.3f}  median={s.median():+.3f}  max={s.max():+.3f}  "
          f"fraction positive={float((s > 0).mean()):.0%}")


if __name__ == "__main__":
    main()
