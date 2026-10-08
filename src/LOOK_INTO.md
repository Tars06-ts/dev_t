# What each file does, and why each parameter is what it is

Read this before `research/TODO_RESEARCH.md`. This is the map; that is the homework.
Everything here is one or two lines — the reasoning in depth lives in the questionnaire.

---

## The pipeline, in order

Bars arrive one day at a time from the organiser's backtester:

```
utils/data.py        clean the prices, build returns
        |
utils/signals.py     rank the 24 stocks, best to worst
        |
models/volatility.py how risky is each stock right now
models/covariance.py how do they move together
        |
models/optimizer.py  turn the ranking into "hold this much of each"
        |
engine.py            hand the weights back to the backtester
```

---

## `src/engine.py` — the strategy the grader runs

Holds only the bars it has already been given, rebuilds the panel each rebalance, and
submits target weights. **Look-ahead is impossible by construction** — there is no file
read and no global fit, so it physically cannot see the future.

| parameter | value | why |
|---|---|---|
| `signal` | `ST_REV_21` | 21-day reversal. Momentum was the hypothesis and tested insignificant (Fama-MacBeth t = 0.43); reversal was added as a *control* and is the only one that works (t = +3.4). |
| `smooth` | `(1, 5, 10, 21)` | An **ensemble**, not a tuned value. Picking the best window from a sweep would inflate it by selection bias. Averaging across the range removes the argmax. See 8c below. |
| `rebalance_cadence` | `21` | Matched to the horizon where the signal actually predicts (IC t = 3.6 at h=21, only 1.8 at h=5). Chosen from the *signal study*, not from backtest Sharpe. |
| `standardise` | `rank` | This universe has excess kurtosis up to 29.9. Ranks are immune to outliers; z-scores are not. |
| `max_history` | derived | Buffer depth computed from whichever lookbacks are configured. A fixed value once sat below momentum's 273-day lookback, which made momentum look *bad* when it was actually *starved*. |

**The one trick worth knowing:** between rebalances the strategy submits an empty dict
`{}`. The backtester only executes orders `if pending_target_weights`, and `{}` is
falsy — so nothing trades and the book carries. Re-submitting the old weights instead
would silently re-trade every day (15,600 fills vs 1,554).

---

## `src/utils/data.py` — load and clean

Reads the CSV, removes duplicates, repairs OHLC violations, aligns every ticker to one
calendar, builds returns. Counts every repair rather than silently dropping rows.

**On this dataset it does nothing** — the panel is perfectly clean. It is out-of-sample
insurance, not in-sample work. Say that plainly in the report.

| parameter | value | why |
|---|---|---|
| `ffill_limit` | 5 | Carry a stale price across a short gap so one hole doesn't destroy a 252-day window; past 5 sessions the name is treated as untradable instead. |
| `SUSPECT_LOG_RETURN` | 0.40 | A 40% daily move in a liquid large-cap is worth *flagging*, never deleting — real gaps happen. |

`returns_est` masks forward-filled sessions, because a stale zero would understate a
stock's volatility and the sizing layer would then buy too much of it.

---

## `src/utils/signals.py` — the ranking

Three signals. Each returns a score per stock per day; row `t` uses only rows `<= t`.

| signal | idea |
|---|---|
| `MOM_12_1` | winners keep winning, measured over 12 months skipping the last one |
| `VOLADJ_MOM_12_1` | same, divided by volatility |
| `ST_REV_21` | **the one in use** — last month's losers bounce back |

`momentum_12_1` skips the most recent 21 days because short-horizon reversal has the
opposite sign and would otherwise cancel the medium-term signal.

**`assert_causal` is the look-ahead guard** — it rebuilds each signal on truncated data
and demands an exact match. `research/validate.py` plants two cheating signals and
fails if the guard misses them, so the guard is proven to work rather than assumed to.

---

## `src/utils/math_helpers.py` — shared statistics

| function | why it exists |
|---|---|
| `cs_rank` | ranks mapped to sum exactly zero. The obvious `rank(pct=True) - 0.5` averages to `(n+1)/2n`, which would bake a permanent **+2% long tilt** into every weight vector. |
| `newey_west_tstat` | overlapping forward returns make naive t-stats too confident by roughly √h. Every t-statistic in this project goes through this. |
| `block_bootstrap_sharpe` | confidence intervals without assuming normal returns, using month-long blocks so volatility clustering survives. |

---

## `src/models/volatility.py` — how risky is each stock

| estimator | trade-off |
|---|---|
| `rolling` | simple, but a shock leaves the estimate abruptly N days later — plateau then cliff |
| `ewma` | geometric decay, the empirically right shape |
| `parkinson` | uses the high-low range, ~5× more information per day, but **biased low** because it cannot see overnight gaps |
| `blended` | **in use** — EWMA + Parkinson averaged in variance space |

| parameter | value | why |
|---|---|---|
| `method` | `blended` | The two have complementary errors: Parkinson is efficient but misses gaps, EWMA is unbiased but noisy. |
| `halflife` | 42 | EWMA is the IGARCH(1,1) special case, so it gets GARCH's clustering response with **no per-asset fitting** — which matters because fitted parameters are what break out of sample. |
| `floor` / `cap` | 0.002 / 0.25 | A near-zero vol estimate would produce an enormous position. |

---

## `src/models/covariance.py` — how stocks move together

Ledoit-Wolf shrinkage toward a constant-correlation target.

**Why it is mandatory, not a refinement:** 24 assets on a 126-day window is T/N ≈ 5.
At that ratio the sample covariance's smallest eigenvalues are near-pure noise, and any
optimiser that inverts it loads onto exactly those directions.

---

## `src/models/optimizer.py` — how much to hold

Four sizing rules, ordered by how much they trust the covariance matrix:

| rule | trusts covariance |
|---|---|
| `equal` | not at all |
| `inverse_vol` | **in use** — variances only |
| `erc` | full matrix |
| `min_variance` | full matrix, inverted — most fragile |

| parameter | value | why |
|---|---|---|
| `method` | `inverse_vol` | Needs N variance estimates instead of N(N+1)/2 covariances. Fewer things to estimate wrongly. Also won the backtest comparison. |
| `n_long` / `n_short` | 6 / 6 | Symmetric and dollar-neutral. An 8/4 tilt scored a better Sharpe and was **rejected** — every stock in this sample rose, so a long tilt harvests bull-market beta, not alpha. |
| `max_gross` | 0.95 | Hard limit is 1.0 and the book drifts between rebalances; this is the headroom. |
| `max_position` | 0.12 | Stops one name dominating a 12-name book. |
| `target_vol` | 0.10 | Ex-ante annualised, and **one-sided** — `min(target/forecast, 1.0)` can only cut risk, never add it, because levering up would breach the 1.0 limit. It binds in just 9 of 91 rebalances, so it is a brake for high-volatility periods, not a scaling knob. At 0.15 it never binds and is identical to switching it off. Forecast averages 7.7% against 7.7% realised — well calibrated. |
| `no_trade_band` | 0.005 | Don't trade to close a 0.5% weight gap; the commission exceeds the benefit. |
| `exit_buffer` | 0 (off) | Hysteresis. **Available but not adopted** — see 8b below. |
| `selection` | `topn` | Full cross-section weighting available but **failed its test** — see 8a. |

---

## `src/research/` — your work

| file | what |
|---|---|
| `TODO_RESEARCH.md` | the questionnaire — 91 questions in 9 parts, start here |
| `data-analysis.ipynb` | EDA skeleton, you write it |
| `model-training.ipynb` | signal/model skeleton, you write it |
| `signal_eval.py` | tool: IC, Fama-MacBeth, spread portfolios, bootstrap |
| `sweep.py` | tool: re-runs the backtest across settings (defaults to 5bp slippage) |
| `walkforward.py` | tool: rolling train/test validation |
| `overfitting_audit.py` | tool: DSR, Monte-Carlo null, dead-signal control — **run before quoting any number** |
| `validate.py` | 28 correctness checks — **run after every change** |

---

## Three experiments, and why two were rejected

This is the part worth understanding — the rejections show the method.

**8a. Full cross-section weighting — FAILED.** Weight all 24 names instead of 12.
Prediction written first: Grinold's law says doubling breadth gives √2 = 1.41×, so
0.55 → 0.78. **Measured 1.04×.** The law counts *independent* bets; at 0.26 average
correlation the extra names add little new information. Kept as a documented negative
result.

**8b. Rank hysteresis — works in backtest, REJECTED.** Hold a name until it falls past
rank `6 + buffer`. Took Sharpe 0.435 → 1.05. Rejected because **the arithmetic doesn't
work**: it earns by cutting trading costs, turnover fell 38%, which at 5bp is worth
0.42%/yr ≈ **0.05 Sharpe**. The measured gain was **+0.62 — ten times what the
mechanism can pay for.** The curve is also jagged (buffer 5 → 0.68, 6 → **0.41**,
7 → 0.84), and a 0.4 swing from a one-step change is noise.

**8c. Signal smoothing — ADOPTED, as an ensemble.** Averaging the score over several
days raises the Information Coefficient itself (0.071 → 0.089), which is better
*prediction*, not better trading.

But the honest test — a *paired* test of whether `smooth=10` beats `smooth=1` day by
day — gives **t = 1.85, not significant**. And the value was picked from 10 trials.

So instead of one tuned window, the strategy uses an **ensemble `(1, 5, 10, 21)`**:
every member ranked, ranks averaged. No argmax means no selection bias — the number
you get is an honest estimate rather than a maximum. Its IC lands between the members
(0.087) exactly as theory predicts.

---

## Four framings to avoid (corrected in `research/EXTRA_SHI.md`)

Easy mistakes to make when writing this up, each of which an interviewer can unpick:

1. **Do not credit Grinold's law with the ~0.70 estimate.** That comes from IC-ratio
   scaling (0.55 × 1.22 = 0.67) and tail-spread scaling (0.55 × 1.32 = 0.73). Grinold
   applied directly gives `0.07 × √288 ≈ 1.19`, a ceiling. Worse, its breadth
   prediction was tested here and *failed* (1.41× predicted, 1.04× delivered) — keep
   that as a separate and genuinely strong point.
2. **Realised vol is not 7.7% because a cap "overrode" the target.** The target is
   one-sided and mostly does not bind. See the `target_vol` row above.
3. **Selecting a signal on a rolling training window is not look-ahead bias.** It is
   legitimate, and `walkforward.py` does it. The reason `ST_REV_21` is hardcoded is
   empirical: per-fold selection *underperformed* a fixed choice (1.32 vs 1.73, with
   4 different winners across 7 folds).
4. **The empty-dict hold is not an "exploit"**, and commission was not eating the
   alpha — it is only 0.225%/yr. The larger effect of restating weights daily was
   that it silently converted a 21-day schedule into daily drift-correction.

## The number to quote, and the number not to

Current graded run: **Sharpe 1.32, CI [0.52, 2.21], CAGR 11.35%, max DD 7.45%,
0 leverage violations.**

**Do not report 1.32 unqualified.** IR is proportional to IC; the ensemble's IC is
1.22× the plain signal's, so the defensible expectation is `0.55 × 1.22 ≈ 0.67`.
The backtest prints roughly double that and **the excess is unexplained**.

The honest framing is both numbers: the mechanism justifies ~0.67, the backtest shows
1.32, and the gap is sample luck until something out-of-sample says otherwise.

Across this project 100+ configurations were examined on 1484 days. Harvey, Liu & Zhu
(2016) argue for a t-stat hurdle of ~3.0 rather than 2.0 for exactly this reason.

---

# Overfitting audit — the honest verdict

Six tests were run specifically to try to break the result. Three pass, three fail,
and one apparent confirmation turned out to be a confound.

## Passes

**Monte-Carlo null.** 40 random rankings pushed through the *exact* final pipeline:
mean −0.047, sd 0.394, max 0.695. **Zero of 40 reached 1.32** (z = 3.47). The
machinery — vol sizing, neutrality, caps, ensembling — does not manufacture
performance. If it did, random signals would score well.

**Sub-period stability.** 2017-18: 1.28 · 2019-20: 0.95 · 2021-22: 2.19 ·
ex-COVID: 1.60. Positive everywhere, strongest outside the crash.

**The core signal predates the search.** Reversal's IC t = 3.64 and Fama-MacBeth
t = 3.41 were measured before any smoothing work. That finding is uncontaminated.

## Fails

**Deflated Sharpe Ratio = 0.736** (Bailey & Lopez de Prado). Given ~120
configurations examined, the expected *maximum* Sharpe under the null is **1.127**;
observed is 1.397. It only clears 0.95 if fewer than ~20 effectively independent
configurations were tried. Far more were. Two independent methods agree on the
threshold: the Monte-Carlo null's N(0, 0.394) gives an expected max over 120 draws of
1.13, matching the analytic 1.127.

**An unexplained gap.** IC rose 1.22x; the tail spread the portfolio actually trades
rose 1.32x. Both predict Sharpe near 0.73. Observed 1.32. The best available
mechanism explains roughly half the performance, and unexplained performance is the
most reliable overfitting signature there is.

**P&L concentration.** Removing the best 10 days (0.7% of the sample) cuts total
return 88.4% -> 51.5% and Sharpe 1.40 -> 1.00.

## The confound — why the walk-forward numbers are NOT the vindication they look like

Walk-forward appeared to endorse everything: selection 1.39 vs baseline 0.53. But the
train->test decay is **positive for every single configuration**, including
`momentum` (+0.18), which is a known-dead signal (Fama-MacBeth t = 0.43).

A dead signal cannot "generalise". The test windows are simply *later in the sample*
than the training windows, and the whole strategy family does better in 2020-22 than
in 2016-19. The apparent out-of-sample success is substantially a calendar effect, not
evidence of generalisation.

**Anyone reading that table without checking the dead-signal control would conclude
the strategy validates out of sample. It does not.**

## What the walk-forward does still show

In the three folds where the ensemble was actually selected on training data, it
degraded in absolute terms but **beat the baseline in all three**:

| fold | train | test | baseline test |
|---|---|---|---|
| 2 | +1.464 | +0.832 | +0.357 |
| 3 | +1.926 | −0.344 | −1.033 |
| 4 | +1.174 | +0.039 | −0.952 |

Held fixed across the whole 875-day OOS region: ensemble **1.683** vs baseline
**0.526**. Degradation from a training peak is expected for any selected
configuration (regression to the mean); consistently beating the alternative is the
part that matters, and it does.

## Verdict

**The signal is real. The headline Sharpe is not.**

- the ranking signal survives every test that does not depend on the search
- the ensemble beats the plain baseline consistently out of sample
- but 1.32 is roughly what the best of ~120 tries produces under the null, and
  half of it has no mechanism behind it

**Claim 0.7, not 1.32.** Report the in-sample 1.32 with the DSR of 0.736 beside it.
On a methodology-weighted rubric, demonstrating that you tested your own result and
found it wanting is worth more than the number.

The mistake to avoid repeating: searching, keeping the winner, then constructing a
mechanism story for it. Removing the final selection step (the ensemble) does not
undo the search that found the family in the first place.

---

# DECISION: the ensemble stays. Here is why, and what changed instead.

**The question was: is Sharpe 1.32 a "wrong value"?**

No. It is a correct measurement of what the strategy did on this data. What would be
wrong is *presenting it as an expected future Sharpe*. Those are different failures,
and they need different fixes. The number does not need changing; the way it is
reported does.

## Why the ensemble was kept rather than reverted

Three reasons, in order of weight:

1. **It beats the plain baseline out of sample, cleanly.** In the three walk-forward
   folds where the ensemble was selected *using training data only* -- no hindsight --
   it beat the baseline in all three (+0.83 vs +0.36, −0.34 vs −1.03, +0.04 vs −0.95).
   Held fixed across the whole 875-day OOS region: **1.683 vs 0.526**. Degrading from
   a training peak is expected for any selected configuration; consistently beating
   the alternative is the part that counts.

2. **Its construction has no tuned parameter.** The windows `(1, 5, 10, 21)` are
   1 day, 1 week, 2 weeks, 1 month -- calendar-natural units, not optimised values.
   Nothing was selected by its backtest score. Had the range been `(1, 7, 13, 19)` the
   argument would be identical, which is the test of whether a choice is a priori.

3. **Reverting would cost real out-of-sample performance for no methodological gain**,
   because the Deflated Sharpe can be disclosed either way. Honesty is a reporting
   property, not a reason to hold a worse strategy.

The case against -- DSR 0.736, ~half the performance unexplained -- is an argument
about **what to claim**, not about which strategy to run.

## What changed in the code instead

`src/research/overfitting_audit.py` -- one command that recomputes the caveats
alongside the headline, so they cannot drift apart:

```bash
python -m src.research.overfitting_audit
```

It prints the in-sample Sharpe, the Deflated Sharpe at several trial counts, a
Monte-Carlo null, the dead-signal control, P&L concentration, sub-period stability,
and finishes with an explicit "what to report" block.

`N_TRIALS_EXAMINED = 120` at the top of that file is the honest count of
configurations tried during development. **Keep it updated as you experiment.**
Under-reporting it is the easiest way to fool yourself -- the DSR is highly sensitive
to it, and lowering it silently turns a FAIL into a PASS.

## The sentence to use in the report

> In-sample Sharpe 1.32 (95% CI [0.52, 2.21]). Deflating for the ~120 configurations
> examined gives a Deflated Sharpe Ratio of 0.74, below the 0.95 threshold, so the
> in-sample figure should not be read as an expected out-of-sample Sharpe. The
> mechanism-justified estimate -- from a 1.22x improvement in information coefficient
> and 1.32x in the traded tail spread -- is approximately 0.70. Walk-forward confirms
> the configuration outperforms the baseline out of sample (1.68 vs 0.53), but the
> absolute level is not supported.

That is a stronger submission than quoting 1.32, because it demonstrates the
validation work rather than only its output.
