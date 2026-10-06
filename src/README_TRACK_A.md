# Track A — how to run and verify

Everything below runs from the repo root (`/home/tar/dev_t`) with the project venv.
Prefix commands with `.venv/bin/python`, or activate first: `source .venv/bin/activate`.

The dataset must be at `data/Dataset_PS-A.csv` (that is what `config.yaml` points to).
A copy already lives there; the original upload is untouched in `Track-A/Track-A/`.

---

## 1. Start here — does the whole pipeline hold together?

```bash
.venv/bin/python -m src.research.validate
```

28 self-checks across the data, signal, volatility and sizing layers. **Section 5 is
the one to watch**: it plants two signals that genuinely cheat (one reads tomorrow's
return, one uses a centred window) and fails if the look-ahead guard does *not* catch
them. A guard that has never been shown to fire proves nothing.

Expect `ALL 28 CHECKS PASSED`. Exit code 0.

## 2. Look at the data

```python
from src.utils.data import load_panel
panel = load_panel("data/Dataset_PS-A.csv")
print(panel.report.summary())
print(panel.describe())
```

`panel.report` carries the quality audit (what the cleaner repaired, per-ticker
coverage) and `panel.describe()` the annualised return characteristics. Work through
this in `src/research/data-analysis.ipynb` — Part 1 of `TODO_RESEARCH.md` is the
question list.

## 3. Compare the ranking signals

```bash
.venv/bin/python -m src.research.signal_eval --data data/Dataset_PS-A.csv
```

Takes ~20s. Prints the causality check, then four tables: information coefficients
with Newey-West t-stats, Fama-MacBeth regressions, top-minus-bottom spread portfolios
with block-bootstrap Sharpe CIs, and the concentration/persistence profile.

Useful flags:

```bash
--horizons 1,5,21,63        # forward horizons to score against
--standardise zscore        # rank (default) | zscore | raw
--top-n 4 --rebalance 21    # spread portfolio construction
```

**What you should see:** `ST_REV_21` is the only signal with statistical support
(IC t = +3.6, FM t = +3.4). Both momentum variants are indistinguishable from zero.
They also correlate 0.96 with each other, which is why vol-adjusting the *signal* is
nearly a no-op and the vol adjustment has to live in the sizing layer instead.

## 4. Run the actual backtest

```bash
.venv/bin/python run_backtest.py --track A
```

This is the organiser's runner calling `src/engine.py`. Drop `--no-plot` to skip the
tearsheet. Outputs land in `results/` (`summary.json`, `trades.csv`, `tearsheet.png`).

Current in-sample result: Sharpe 0.55, CAGR 4.2%, max DD 12.4%, 1588 trades,
mean gross leverage 0.83, **0 leverage violations**, no bankruptcy.

## 5. The notebooks

```bash
.venv/bin/pip install nbformat nbconvert ipykernel jupyterlab   # not needed to run the strategy
.venv/bin/jupyter lab src/research/
```

**These are skeletons — headings and instructions only, no code.** Fill them in
yourself; that is the research deliverable.

- **`TODO_RESEARCH.md`** — the question list. 36 questions in six parts, plus Part 6
  which lists every decision already hardcoded in `src/` with a hint for re-deriving
  it. Start here.
- **`data-analysis.ipynb`** — EDA. Maps to Part 1 of the questionnaire.
- **`model-training.ipynb`** — signal selection, robustness, sizing, sensitivity.
  Maps to Parts 2–5.

## 6. Sensitivity analysis

```bash
.venv/bin/python -m src.research.sweep --quick          # ~11 configs, ~1 min
.venv/bin/python -m src.research.sweep                  # full grid, ~8 min
.venv/bin/python -m src.research.sweep --axis cadence --axis sizing
```

Runs the real backtester once per configuration, one parameter varied at a time off
the graded baseline. Writes `results/sensitivity.csv`. Available axes: `signal`,
`cadence`, `sizing`, `vol_model`, `breadth`, `target_vol`, `vol_window`, `halflife`,
`cov_window`, `caps`, `band`, `neutrality`, `costs`.

`costs` varies the execution assumption rather than a strategy parameter. Worth
running: `config.yaml` sets slippage to 0, but `BacktestConfig` defaults to 5bp, and
at 5bp the Sharpe falls from 0.55 to 0.43. The OOS evaluation may not use 0, so the
cost curve is part of the result, not a footnote.

## 7. Overfitting audit — run this before quoting any number

```bash
.venv/bin/python -m src.research.overfitting_audit
```

Recomputes the caveats alongside the headline so they cannot drift apart: Deflated
Sharpe at several trial counts, a Monte-Carlo null, a dead-signal control, P&L
concentration, sub-period stability, and a closing "what to report" block.

**Current state: in-sample Sharpe 1.32, but DSR = 0.74, which does not clear 0.95.**
The defensible figure is ~0.70. Quote the in-sample number *with* the DSR beside it.

`N_TRIALS_EXAMINED = 120` at the top of that file is the honest count of
configurations tried. Keep it updated — the DSR is very sensitive to it.

## 8. Organiser's own pre-submission test

```bash
.venv/bin/python test_submission.py
```

Checks file structure, class inheritance, Track A and Track B bar handling, and the
leverage/capital constraints. All 5 pass.

---

## Poking at it yourself

Every parameter can be overridden without editing a file:

```python
import sys; sys.path.insert(0, ".")
from backtester import Backtester, BacktestConfig
from src.engine import ParticipantStrategy

cfg = BacktestConfig(track="A", data_path="data/Dataset_PS-A.csv",
                     initial_cash=100_000.0, commission_rate=0.0001,
                     save_results=False, generate_plot=False)

strat = ParticipantStrategy(signal="MOM_12_1", rebalance_cadence=42,
                            sizing_method="erc", target_vol=0.15)
res = Backtester(cfg).run(strat)
print(res.sharpe_ratio, res.max_drawdown_pct)

print(strat.diagnostics_frame().head())   # per-rebalance gross/net/turnover/ex-ante vol
```

Warmup and buffer depth are derived from whichever lookbacks you configure
(`ST_REV_21` needs 151 sessions, the momentum variants 298), so switching signals
cannot silently starve one of them.

`ParticipantStrategy()` with no arguments is always the graded configuration — the
backtester constructs it exactly that way. Overrides accept any key of `PARAMS`,
`VolConfig` or `SizingConfig` (see `src/engine.py`). `method` exists in both configs, so
it must be qualified as `vol_method` or `sizing_method`; bad keys raise immediately
rather than being silently ignored.

---

## What is where

Laid out per README.md §4 (`models/` for statistical models, `utils/` for feature
extraction and helpers, `research/` for analysis, `engine.py` kept concise).

| file | role |
|---|---|
| `src/engine.py` | `ParticipantStrategy` — the graded entry point |
| `src/utils/data.py` | load, clean, calendar-align, returns + stale masks, `Panel` |
| `src/utils/signals.py` | ranking signal builders + look-ahead guard |
| `src/utils/math_helpers.py` | cross-sectional transforms, Newey-West, block bootstrap |
| `src/models/volatility.py` | rolling / EWMA / Parkinson / blended vol estimators |
| `src/models/covariance.py` | Ledoit-Wolf shrinkage covariance |
| `src/models/optimizer.py` | sizing rules, caps, vol targeting, no-trade band |
| `src/research/TODO_RESEARCH.md` | **start here** — 36 questions + hints for re-deriving hardcoded choices |
| `src/research/data-analysis.ipynb` | EDA notebook — **skeleton, you write it** |
| `src/research/model-training.ipynb` | signal/model notebook — **skeleton, you write it** |
| `src/research/signal_eval.py` | tool: IC / Fama-MacBeth / spread / bootstrap |
| `src/research/sweep.py` | tool: parameter sensitivity via the real backtester |
| `src/research/validate.py` | 28 self-checks incl. planted look-ahead leaks |
| `src/research/overfitting_audit.py` | DSR, Monte-Carlo null, dead-signal control |
| `src/research/walkforward.py` | rolling train/test validation |

## Two design choices worth knowing before you read the code

**Holding between rebalances submits an empty weight dict.** `_run_track_a` executes
pending orders only `if pending_target_weights`, and `{}` is falsy, so nothing fills
and the book carries untouched. The alternatives are both wrong: re-submitting the
previous targets makes the ledger recompute share counts from each new open and
silently drift-corrects *daily*, and restating actual drifted weights still trades the
overnight gap on every name every day. Measured, those cost ~15,600 fills over the
sample versus ~1,400 for the same intended positions.

**The strategy only ever holds bars the backtester has handed it.** No file read, no
global fit, no precomputed frame. Look-ahead is impossible by construction rather than
by convention, and `src/utils/data.panel_from_wide` is shared with the research path
so live and research numbers cannot drift apart (check 4 in `validate.py` asserts this).
