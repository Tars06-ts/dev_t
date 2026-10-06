"""
src/research/validate.py - Self-checks for the Track A pipeline.

Run this before trusting any backtest number:

    python -m src.research.validate

It is deliberately adversarial. Sections 1-4 check that the data layer is internally
consistent; section 5 plants two signals that genuinely cheat (one reads tomorrow's
return, one uses a centred window) and FAILS if the look-ahead guard does not catch
them -- a guard that has never been shown to fire is not evidence of anything.
Sections 6-9 cover the signal statistics, and 10-12 the portfolio and strategy layers.

Exit code 0 means every check passed.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

from src.utils import signals as SG                            # noqa: E402
from src.utils import math_helpers as MH                        # noqa: E402
from src.utils.data import load_panel, panel_from_wide         # noqa: E402
from src.models import covariance as CV                        # noqa: E402
from src.models import optimizer as OPT                         # noqa: E402
from src.models import volatility as VOL                        # noqa: E402

DATA = "data/Dataset_PS-A.csv"
_n = 0


def ok(msg: str) -> None:
    global _n
    _n += 1
    print(f"  PASS  {msg}")


def section(title: str) -> None:
    print(f"\n[{title}]")


def main() -> int:
    panel = load_panel(DATA)

    section("1. panel integrity")
    frames = ["open", "high", "low", "close", "volume", "returns", "log_returns",
              "returns_est", "observed", "filled", "tradable"]
    for nm in frames:
        f = getattr(panel, nm)
        assert f.shape == panel.close.shape, nm
        assert f.index.equals(panel.close.index) and list(f.columns) == panel.tickers, nm
    ok(f"all {len(frames)} frames share index/columns/shape")
    assert panel.close.index.is_monotonic_increasing and panel.close.index.is_unique
    ok("calendar strictly increasing and unique")
    hi, lo = panel.high.dropna(how="all"), panel.low.dropna(how="all")
    assert (hi >= lo).all().all()
    ok("high >= low everywhere after repair")
    assert ((panel.close > 0) | panel.close.isna()).all().all()
    ok("no non-positive closes survive")

    section("2. stale-session mask: both endpoints, no look-ahead")
    f = panel.filled
    expect = (f | f.shift(1, fill_value=False)) & panel.log_returns.notna()
    assert panel.returns_est[expect].isna().all().all()
    extra = panel.returns_est.isna() & panel.log_returns.notna() & ~expect
    assert int(extra.to_numpy().sum()) == 0
    ok(f"{int(expect.to_numpy().sum())} stale-adjacent returns masked, none over-masked")

    section("3. up_to() is a true point-in-time slice")
    d = panel.dates[1000]
    sub = panel.up_to(d)
    assert sub.dates.max() == d and len(sub.dates) == 1001
    assert sub.close.equals(panel.close.loc[:d])
    ok("up_to truncates every frame identically")

    section("4. research and live panel construction agree")
    q = panel_from_wide(panel.open, panel.high, panel.low, panel.close, panel.volume)
    assert np.allclose(q.close.values, panel.close.values)
    assert np.allclose(q.returns_est.fillna(-9).values, panel.returns_est.fillna(-9).values)
    assert (q.tradable.values == panel.tradable.values).all()
    ok("panel_from_wide reproduces load_panel exactly (live == research)")

    section("5. look-ahead guard actually fires")
    chk = SG.assert_causal(panel)
    ok(f"clean signals pass ({len(chk)} point-in-time checks, "
       f"max dev {chk['max_abs_diff'].max():.1e})")
    planted = {
        "LEAKY": lambda p: p.log_returns.shift(-1).rolling(20, min_periods=10).sum(),
        "CENTRED": lambda p: p.log_returns.rolling(41, center=True, min_periods=20).mean(),
    }
    for name, fn in planted.items():
        SG.SIGNAL_LIBRARY[name] = fn
        try:
            SG.assert_causal(panel, names=[name])
        except AssertionError:
            ok(f"planted leak {name!r} detected")
        else:
            print(f"  FAIL  planted leak {name!r} NOT detected")
            return 1
        finally:
            del SG.SIGNAL_LIBRARY[name]

    section("6. forward returns are execution-aligned")
    fwd = SG.forward_returns(panel, horizon=5, price="open")
    t, tk = panel.dates[500], panel.tickers[0]
    man = panel.open.loc[panel.dates[506], tk] / panel.open.loc[panel.dates[501], tk] - 1
    assert abs(fwd.loc[t, tk] - man) < 1e-12
    ok("matches hand-computed open_{t+1+h}/open_{t+1} - 1")
    assert fwd.iloc[-6:].isna().all().all()
    ok("last h+1 rows NaN (nothing left to trade into)")

    section("7. cross-sectional standardisation")
    for mode in ["rank", "zscore", "raw"]:
        sc = SG.build_signals(panel, ["MOM_12_1"], standardise=mode)["MOM_12_1"]
        assert sc.shape == panel.close.shape
    ok("rank / zscore / raw all build")
    r = SG.build_signals(panel, ["MOM_12_1"], standardise="rank")["MOM_12_1"].iloc[800].dropna()
    assert abs(r.mean()) < 1e-9
    ok("rank scores sum to zero (no hidden long tilt)")
    z = SG.build_signals(panel, ["MOM_12_1"], standardise="zscore")["MOM_12_1"].iloc[800].dropna()
    assert abs(z.mean()) < 1e-9 and abs(z.std(ddof=1) - 1) < 1e-9
    ok("zscore scores are mean 0 / sd 1")

    section("8. Newey-West estimator")
    rng = np.random.default_rng(0)
    iid = pd.Series(rng.normal(0, 1, 2000))
    nw = MH.newey_west_tstat(iid, lags=0)
    assert abs(nw["se"] - iid.std(ddof=0) / np.sqrt(2000)) < 1e-9
    ok("lags=0 reproduces the plain standard error exactly")
    ar = pd.Series(np.convolve(rng.normal(0, 1, 2200), np.ones(20) / 20, "valid"))
    se0 = MH.newey_west_tstat(ar, lags=0)["se"]
    se1 = MH.newey_west_tstat(ar, lags=40)["se"]
    assert se1 > 2 * se0
    ok(f"HAC widens se on autocorrelated data ({se0:.5f} -> {se1:.5f}, {se1/se0:.1f}x)")

    section("9. bootstrap determinism")
    a = MH.block_bootstrap_sharpe(rng.normal(0, 0.01, 1200), n_boot=500)
    b = MH.block_bootstrap_sharpe(np.random.default_rng(1).normal(0, 0.01, 1200), n_boot=500)
    c = MH.block_bootstrap_sharpe(np.random.default_rng(1).normal(0, 0.01, 1200), n_boot=500)
    assert b == c
    ok("fixed seed gives identical CIs across runs")

    section("10. volatility estimators")
    for m in ["rolling", "ewma", "parkinson", "blended"]:
        v = VOL.estimate_vol(panel, VOL.VolConfig(method=m))
        assert v.shape == panel.close.shape
        tail = v.iloc[300:]
        assert tail.notna().any().all(), m
        assert (tail.stack() >= 0).all(), m
    ok("rolling / ewma / parkinson / blended all produce non-negative estimates")
    v = VOL.estimate_vol(panel, VOL.VolConfig(method="blended", floor=0.004, cap=0.09))
    s = v.stack()
    assert s.min() >= 0.004 - 1e-12 and s.max() <= 0.09 + 1e-12
    ok("floor and cap are respected")
    # causality: estimating on a truncated panel must reproduce the same last row
    cut = panel.up_to(panel.dates[900])
    full = VOL.estimate_vol(panel, VOL.VolConfig(method="rolling"))
    part = VOL.estimate_vol(cut, VOL.VolConfig(method="rolling"))
    both = full.loc[panel.dates[900]].notna() & part.iloc[-1].notna()
    assert np.allclose(full.loc[panel.dates[900]][both], part.iloc[-1][both])
    ok("rolling vol on a truncated panel matches the full-sample row (causal)")

    section("11. covariance and sizing")
    cov = CV.ledoit_wolf_cov(panel.returns_est.tail(126))
    assert cov.shape[0] == cov.shape[1] == 24
    assert np.allclose(cov.to_numpy(), cov.to_numpy().T)
    ok("shrunk covariance is square and symmetric")
    assert np.linalg.eigvalsh(cov.to_numpy()).min() > 0
    ok("shrunk covariance is positive definite (safe to invert)")
    names = list(cov.index[:8])
    w_erc = OPT.erc_weights(names, cov)
    rc = w_erc.to_numpy() * (cov.loc[names, names].to_numpy() @ w_erc.to_numpy())
    assert rc.std() / rc.mean() < 0.01
    ok(f"ERC risk contributions are equal (spread {rc.std()/rc.mean():.2e} of mean)")
    vol_row = VOL.estimate_vol(panel).iloc[-1]
    inv = OPT.inverse_vol_weights(names, vol_row)
    scaled = inv * vol_row.reindex(names)
    assert scaled.std() / scaled.mean() < 1e-9
    ok("inverse-vol gives every name equal standalone risk")

    section("12. weight construction respects every constraint")
    scores = SG.build_signals(panel, ["ST_REV_21"])["ST_REV_21"].iloc[-1]
    for meth in ["equal", "inverse_vol", "erc", "min_variance"]:
        cfg = OPT.SizingConfig(method=meth, max_gross=0.95, max_position=0.12)
        w = OPT.build_weights(scores, vol_row, cov, cfg)
        assert float(w.abs().sum()) <= 0.95 + 1e-9, meth
        assert float(w.abs().max()) <= 0.12 + 1e-9, meth
        assert abs(float(w.sum())) < 0.05, meth
        assert np.isfinite(w.to_numpy()).all(), meth
    ok("all 4 sizing methods: gross<=0.95, |w_i|<=0.12, ~dollar-neutral, finite")
    w = OPT.build_weights(scores, vol_row, cov, OPT.SizingConfig(max_position=1.0, target_vol=None))
    assert float(w.abs().sum()) <= 1.0 + 1e-9
    ok("gross cap holds even with per-name cap and vol target disabled")
    empty = OPT.build_weights(pd.Series(dtype=float), vol_row, cov, OPT.SizingConfig())
    assert empty.empty
    ok("empty score cross-section returns no position rather than raising")

    print(f"\n{'=' * 64}\n  ALL {_n} CHECKS PASSED\n{'=' * 64}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
