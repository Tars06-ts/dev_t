# Track A — research questionnaire

**90 questions in 9 parts.** Parts 1–6 are numbered 1–36; Parts 7–9 use Q37–Q90.

Work through this yourself. Every question below is one you should be able to answer
from the data, and together they justify every choice baked into `src/`.

**No answers here on purpose.** Where a decision is already hardcoded in the strategy,
it is listed in Part 6 with a hint about how it was derived — reproduce it, and if you
disagree, change the code.

| part | topic |
|---|---|
| 1–5 | the core research: data, signal, robustness, portfolio, walk-forward |
| 6 | every hardcoded parameter, with a hint for re-deriving it |
| 7 | an external critique — which claims held up and which did not |
| 8 | three improvements built and tested; one adopted, two rejected |
| 9 | **the overfitting audit** — why the headline Sharpe is not the number to quote |

Tools you have:
- `signal_eval.py` — IC, Fama-MacBeth, spread portfolios, bootstrap
- `sweep.py` — re-runs the backtest across parameter settings (5bp slippage by default)
- `walkforward.py` — rolling train/test validation
- `overfitting_audit.py` — Deflated Sharpe, Monte-Carlo null, dead-signal control
- `validate.py` — 28 correctness checks; run it after any change you make

Read `../LOOK_INTO.md` first if you have not — it is the map, this is the homework.

---

## Part 1 — Know your data
*Goes in `data-analysis.ipynb`. Do this first; it constrains everything after.*

1. How many sessions, how many tickers, what date range? Is every ticker present on
   every date, or are there holes?
2. Does the cleaning layer in `utils/data.py` actually change anything on this file?
   If it changes nothing, what is it there for?
3. What is the skew and excess kurtosis of daily returns per ticker? How far from
   Normal is the worst one?
4. **Why does that matter?** If returns were Normal, which of the methods in `src/`
   could you replace with something simpler? Which would break?
5. How many tickers have a positive annualised return over the sample? What does that
   imply about any strategy that ends up net long?
6. Plot rolling 63-day volatility for all 24 names. What is the ratio between the most
   and least volatile name at a typical point in time?
7. **Why does that matter?** If that ratio were 1.0, what would inverse-volatility
   sizing reduce to?
8. Plot the autocorrelation of mean |daily return| out to 60 lags. Is volatility
   predictable? Are returns? Why does that asymmetry justify risk-based sizing?
9. What is the mean pairwise correlation across the universe? If you hold 6 names,
   how many independent bets is that really?
10. You estimate covariance on a 126-day window with 24 assets. What is T/N? Look up
    why that ratio is a problem for matrix inversion.

## Part 2 — Does the signal work?
*Goes in `model-training.ipynb`.*

11. Three signals are defined in `utils/signals.py`. Read them. What is the economic
    story for each — why *should* each one predict returns?
12. Why does `momentum_12_1` skip the most recent 21 days? What would contaminate it
    if you did not?
13. Compute the Information Coefficient for each signal at horizons 1, 5 and 21 days.
    Which signals are significant?
14. **Was that the result you expected?** Which signal was included as a control
    rather than a candidate, and what happened to it?
15. Run the Fama-MacBeth regression. Does it agree with the IC? Why run both?
16. Forward returns here are open-to-open from t+1, not close-to-close. Find the line
    in the backtester that makes that the correct choice. What would you be measuring
    if you used close-to-close?
17. The t-statistics are Newey-West corrected. Compute one naive t-stat and one
    corrected, on the same 21-day-horizon IC series. How big is the difference? Why
    does overlapping data cause it?

## Part 3 — Can you break the result?
*The part that separates a finding from a fluke.*

18. Split the sample into sub-periods. Is the signal positive in all of them, or is
    one period carrying it?
19. The sample contains Feb–Mar 2020. That crash-and-rebound is itself a giant
    reversal event. Recompute excluding roughly Feb–Jun 2020. Does the result survive?
20. Sweep the reversal lookback over 5 / 10 / 21 / 42 / 63 / 126 days. Plot IC t-stat
    against lookback. **Is it a plateau or a single spike?** Why does the shape matter
    more than the peak value?
21. The strategy uses 21 days. Is 21 the argmax of your sweep? If not, why keep it?
22. Compute the cross-sectional rank correlation between the three signals. Two of
    them are nearly the same thing — which two, and what does that tell you about
    where volatility adjustment belongs?
23. Correlate each signal against trailing volatility. Could any of them just be a
    disguised low-volatility bet?

## Part 4 — Turning a signal into a portfolio

24. Build an equal-weight top-6 / bottom-6 spread portfolio. Is its Sharpe
    significant? Compare to the IC result — if the IC is significant but the portfolio
    is not, where did the edge go?
25. Re-run that spread portfolio holding for 5, 21 and 63 days. What happens, and how
    does it relate to the horizon where the IC was strongest?
26. Four volatility estimators exist in `models/volatility.py`. Plot all four for one
    ticker. Which responds fastest to a shock? Which has a "plateau then cliff" shape,
    and why?
27. Why is EWMA preferred over fitting a GARCH model per asset? (Look up the
    relationship between EWMA and IGARCH(1,1).)
28. Parkinson's estimator uses the high-low range. Why is it more statistically
    efficient than close-to-close? What does it miss?
29. Compare the condition number of the raw sample covariance against the shrunk one.
    What does a large condition number do to a minimum-variance optimiser?
30. Four sizing rules exist in `models/optimizer.py`. Rank them by how much they rely
    on the covariance matrix being correct. Which should be most fragile out of sample?
31. Run `sweep.py --axis sizing`. Does the backtest ranking match your prediction?
32. Run the full `sweep.py`. **What fraction of configurations have positive Sharpe?**
    Argue why that distribution is a better robustness statement than any single cell.
33. Several settings beat the baseline. Should you adopt the best one? What is the
    width of the Sharpe confidence interval, and how does that answer the question?
34. Run `sweep.py --axis costs`. How much Sharpe do you lose per basis point of
    slippage? `config.yaml` sets slippage to 0 — is that what the graders will use?

## Part 5 — Still missing (required deliverable)

35. **Walk-forward validation is not implemented.** The problem statement requires
    rolling train/test windows. Everything currently in the repo is single-split
    in-sample. Design it: how long is each training window, how far does it step, and
    what exactly are you re-fitting in each one given the strategy has no fitted
    parameters?
36. What would you conclude if walk-forward Sharpe were materially below in-sample
    Sharpe? What if it were higher?

---

## Part 6 — Decisions already hardcoded: reproduce these

Each of these is live in the code. The reasoning came from analysis that is no longer
in this folder, so re-derive it. Hints only.

**(a) `signal="ST_REV_21"` in `engine.py`, not momentum.**
Hint: compute mean IC and its Newey-West t-stat at h=21 for all three signals. One
clears |t| > 3; two sit near zero. The one that works was added as a falsification
control, not as a candidate — which is why Q14 is phrased the way it is.

**(b) `rebalance_cadence: 21` in `config.yaml`, changed from the default 5.**
Hint: this was derived from the *signal* study, not from backtest Sharpe. Look at how
IC t-stat varies with forward horizon for `ST_REV_21` — h=1, h=5, h=21. The holding
period was matched to the horizon where the edge is measurable. Note that a longer
cadence scores even better in the backtest; work out why it was not adopted.

**(c) `sizing_method="inverse_vol"`, not ERC or minimum-variance.**
Hint: count how many parameters each rule has to estimate from data. For 24 assets,
inverse-vol needs N numbers; the others need N(N+1)/2. Then check whether the backtest
agrees with the argument from estimation error.

**(d) `vol_method="blended"` (EWMA + Parkinson in variance space).**
Hint: one estimator is unbiased but noisy, the other is ~5x more efficient per
observation but systematically biased low. Work out *why* it is biased low — what
happens overnight that a high-low range cannot see?

**(e) `cs_rank` maps ranks to `(rank - (n+1)/2) / n`, not `rank(pct=True) - 0.5`.**
Hint: compute the mean of each version across a 24-name cross-section. One of them is
not zero. Work out the size of the resulting permanent tilt, and why that matters for
a strategy that is supposed to be dollar-neutral.

**(f) `max_gross=0.95`, not 1.0.**
Hint: the hard limit is 1.0 and the RuleGuard logs a violation if you exceed it. The
book drifts between rebalances. What is the headroom for?

**(g) Symmetric 6 long / 6 short, dollar-neutral — and an asymmetric 8/4 book was
rejected despite scoring the best Sharpe of anything tested.**
Hint: run `sweep.py --axis neutrality` and look at the Sharpe *and* the max drawdown.
Then go back to your answer to Q5. **This one is a judgement call, not a fact — if you
disagree with the rejection, change it and be ready to defend the change.**

**(h) Between rebalances the strategy submits an empty weight dict `{}`.**
Hint: find the line in `backtester.py::_run_track_a` that decides whether to execute
pending orders. What does an empty dict do there? Then work out what happens instead
if you re-submit the previous weights every bar — specifically, what the ledger does
with `desired_shares = equity * w / open_{t+1}` when the open has moved.

---

# Part 7 — Responding to an external critique

An outside analysis of the backtest output raised five points. Most were wrong, but
two real problems were buried in it. Work through this yourself — learning to tell a
valid criticism from a confident-sounding wrong one matters more than the fixes.

## 7a. Check the claims before acting on them

For each claim below, verify it from the data before you believe it. The commands are
hints, not answers.

**Claim: "1.07 trades/day — you are trading daily and bleeding fees."**
- Q37: Count the number of *distinct days* on which any fill occurred, vs total
  sessions. (`Backtester(...).ledger.trades`, count unique `.step`.) Is the book
  trading daily?
- Q38: Divide total fills by number of rebalances. For a 12-long/12-short... actually
  a 6-long/6-short book, how many fills *should* one rebalance produce? Does the
  number match?
- Q39: What is total commission as a % per year, against CAGR? Is "bleeding" fair?

**Claim: "rebalance monthly (every 21 days)" and "add a turnover threshold".**
- Q40: Read `config.yaml` and `SizingConfig` in `models/optimizer.py`. Are these two
  suggestions already implemented? (This is a lesson in reading the code before
  critiquing output.)

**Claim: "$0.00 slippage is an overfit illusion; realistic slippage would make this a
net loss."**
- Q41: Who set slippage to zero — you, or the organiser's `config.yaml`?
- Q42: Run `sweep.py --axis costs`. At what slippage does CAGR actually cross zero?
- Q43: Compute average trade size, and compare it to median daily dollar volume per
  ticker. What fraction of ADV is a single trade? Is market impact plausible at that
  size? **This is the number that decides whether the criticism applies.**

**Claim: "you took on equity market risk for a 4.23% return."**
- Q44: Look at `net` in `strat.diagnostics_frame()`. What is the average net exposure?
  Does a dollar-neutral long/short book carry equity market risk?

**Claim: "Sharpe CI includes zero, so you fail the rigorous statistical test of the
ranking hypothesis."**
- Q45: Two different questions are being conflated. Which test does the problem
  statement actually ask for on the *ranking hypothesis* — and does it pass?
- Q46: Use SE(Sharpe) ≈ √((1 + SR²/2) / T_years). For SR = 0.55, how many years of
  data would you need for the lower CI bound to clear zero? Compare to the 5.9 years
  you have. **Is the wide CI a flaw in the method, or a limit of the sample?**

**Claim: "ensure you ignore 1-month short-term reversal noise if using momentum."**
- Q47: Which signal is the strategy actually using? What would happen if you followed
  this advice?

## 7b. The two criticisms that ARE valid

**Turnover is genuinely high.** Not because the book trades daily — it does not — but
because roughly 117% of it turns over at each rebalance.

- Q48: Why does a reversal signal churn more than a momentum signal? Compare the rank
  autocorrelation of the two (`signal_persistence` in `signal_eval.py`).
- Q49: Is high turnover a problem *here*, given this account size and these ADVs?
  Answer it with the cost curve from Q42, not with intuition.

**CAGR and Sharpe are modest.** 4.23% and 0.55. Also, the book realises ~7.7%
volatility against a 10% target — it is under-using its risk budget.

- Q50: Why does the realised vol fall short of the target? (Look at `gross` and
  `ex_ante_vol` in the diagnostics frame, and at where `apply_caps` binds.)
- Q51: The sweep shows `target_vol=0.15` scores *worse* (0.388) than `target_vol=0.10`
  (0.55). That is backwards — raising the risk target should scale return and leave
  Sharpe roughly unchanged. **Work out why.** This is an unexplained result in the
  current code and the most interesting open question in the project.

## 7c. Improvements worth trying

In rough order of expected payoff. Each is an experiment, not a known win — measure
before and after, and keep the one the evidence supports.

**1. Rank hysteresis (buffer zones).** Today a name leaves the book the moment it
drops out of the top 6. Instead: enter at top 6, but only exit when it falls below
rank 10. Standard fix for ranking churn; attacks turnover without touching the signal.
- Q52: Implement it in `select_names`. How much does turnover fall? What happens to
  Sharpe net of 5bp slippage?

**2. Signal smoothing.** Average the score over the last 3–5 days before ranking, so
one noisy session cannot flip a position.
- Q53: Does smoothing help the IC, hurt it, or just cut turnover? Smoothing a signal
  always reduces churn — the question is whether it also destroys the edge.

**3. Make net-of-cost the headline number.** Evaluate at 5bp by default rather than 0.
- Q54: Which configurations in the sweep survive a realistic cost assumption? Does the
  *ranking* of configurations change, or just the level?

**4. Walk-forward validation.** Still the biggest gap (Q35–36).

## 7d. How good can this realistically get?

Before chasing a high Sharpe, work out the ceiling.

- Q55: Look up **Grinold's Fundamental Law of Active Management**: IR ≈ IC × √breadth.
- Q56: With an IC of about 0.07, 24 assets, and rebalancing every 21 days, what is
  breadth per year — and what IR does the law imply as an upper bound?
- Q57: That bound assumes perfect implementation using *all* assets. You use 12 of 24
  and have gross/position caps. What does that do to the realistic target?
- Q58: Given your answer, is the current 0.55 a failure, or close to the achievable
  range for this dataset? **If someone offers you a Sharpe of 2 on this data, what
  should you suspect?**

---

# Part 8 — What was built after Part 7, and why

Three improvements were implemented and tested. **One was adopted, two were not.**
The reasoning matters more than the result: in each case the test was "does the
measured gain match what the mechanism can justify?", not "is the Sharpe higher?".

## 8a. Full cross-section weighting — FAILED, not adopted

`models/optimizer.py::full_cross_section_weights`, enabled with `selection="full"`.

Instead of long top-6 / short bottom-6, weight every one of the 24 names by
`score_i / sigma_i`.

**Prediction written down before running:** Grinold's law gives IR ~ IC * sqrt(breadth).
Going from 12 bets to 24 doubles breadth, so Sharpe should improve by sqrt(2) = 1.41x,
i.e. 0.55 -> 0.78.

**Result: 0.574. A ratio of 1.04x, not 1.41x.** The prediction failed.

- Q59: Why? Grinold's breadth counts *independent* bets. What is the mean pairwise
  correlation of this universe (from `data-analysis.ipynb`), and what does that do to
  the effective number of independent bets when you add the middle-ranked names?
- Q60: The middle-ranked names also carry the weakest signal. Under a linear model
  they get small weights. How much new information can they actually add?
- Q61: This is a useful negative result for the report. Write up why a theoretically
  motivated change can still fail, and what that tells you about applying textbook
  formulas to correlated assets.

## 8b. Rank hysteresis — works, but NOT adopted

`models/optimizer.py::select_names(..., exit_buffer=N)`. A held name is kept while it
stays inside the top `n + N`, instead of being dropped the moment it leaves the top 6.

The backtest loved it: `exit_buffer=9` took Sharpe from 0.435 to 1.05 at 5bp.
**It was still rejected.** Two reasons:

1. **The curve is jagged.** buffer 5 -> 0.68, buffer 6 -> **0.41**, buffer 7 -> 0.84.
   A 0.4 Sharpe swing from a one-step parameter change is noise, not an effect.
2. **The arithmetic does not work.** Hysteresis works by cutting trading costs.
   Turnover fell 38% (12,934% -> 8,034%), which at 5bp saves **0.42% per year**. On
   ~8% volatility that is worth about **0.05 Sharpe**. The measured gain was **+0.62**
   — roughly ten times what the mechanism can pay for.

- Q62: Reproduce that cost calculation yourself. If a change earns ten times more than
  its stated mechanism can deliver, what is the most likely explanation?
- Q63: Hysteresis is left available as `exit_buffer=N`. Can you find a *different*
  justification for it that the evidence supports? (Hint: does it change the
  signal's information content at all, or only when you trade?)

## 8c. Signal smoothing — adopted, but as an ENSEMBLE (not `smooth=10`)

`utils/signals.py::build_signals(..., smooth=k)` averages the raw score over the
trailing k sessions before ranking.

**First attempt: adopt `smooth=10`.** The justification was that smoothing raises the
Information Coefficient itself — better *prediction*, not just better trading:

| smooth | mean IC (h=21) | NW t |
|---|---|---|
| 1  | 0.0713 | 3.64 |
| 10 | 0.0888 | **3.98** |
| 16 | 0.0912 | 3.81 |

The mechanism is ordinary signal processing: if the daily score is
`true signal + noise` and the true part moves slowly, averaging suppresses the noise.

**Then the proper test was run, and it failed.** Comparing two IC numbers is not a
test. The right one is *paired*: does IC(smooth=k) beat IC(smooth=1) day by day?
The two series correlate 0.81, so the paired test is the powerful one.

| comparison | mean IC gain | NW t | |
|---|---|---|---|
| smooth=5 vs 1  | +0.0075 | +1.64 | not significant |
| smooth=10 vs 1 | +0.0175 | +1.85 | **not significant** |
| smooth=16 vs 1 | +0.0199 | +1.35 | not significant |

So the IC *level* is significantly positive (t = 3.98 vs zero) but the *improvement
over no smoothing* is not. The value was also picked from 10 trials, and the maximum
of 10 correlated searches is inflated.

**What was adopted instead: an ensemble, `smooth = (1, 5, 10, 21)`.** Every member is
ranked separately and the ranks averaged. The point is that there is **no argmax** —
nothing is selected on its backtest score, so there is no selection bias to correct.
The windows are 1 day, 1 week, 2 weeks, 1 month: calendar-natural units, not tuned
values.

Its IC lands at 0.087 (t = 3.84), *between* its members, exactly as theory predicts
for an average.

- Q64: Re-derive why IR is proportional to IC, and what Sharpe a 1.22x IC improvement
  justifies.
- Q65: Smoothing did *not* reduce turnover (12,577% -> 12,342%). The original
  rationale was "it reduces churn". That was wrong. Why does averaging the score not
  reduce trading here?
- Q66: What is a smoothed 21-day reversal signal, mechanically? Work out the effective
  weighting over past returns and relate it to simply using a longer lookback. Does
  the lookback sweep from Q20 agree?
- Q67: Run the paired test yourself for a smoothing value of your choice. Why is the
  paired version more powerful than comparing two independent t-statistics?
- Q68: **Is an ensemble genuinely immune to the search that found it?** Removing the
  final selection step does not undo the fact that the smoothing *family* was
  discovered by searching. Argue both sides.

## 8d. Walk-forward validation — built

`src/research/walkforward.py`. Rolling 504-session train / 126-session test. Each fold
picks the best configuration using *only* its training window, then applies it
unchanged to the test window.

Results (5bp slippage, 7 folds, 875 OOS days):

| | Sharpe |
|---|---|
| walk-forward selection | 1.32 |
| fixed baseline, no selection | 0.53 |
| **`smooth_16` held fixed** | **1.73** |

- Q69: The adaptive selection (1.32) does **worse** than simply holding one good
  configuration for the whole period (1.73). What does that tell you about selecting
  parameters on a 2-year training window?
- Q70: 4 different configurations won across 7 folds. Is that stability or noise?
- Q71: Drop the best fold and the selection result falls 1.32 -> 0.96. How much should
  one 6-month window be allowed to carry a conclusion?
- Q72: **The deepest problem with 8d.** The candidate list in `walkforward.py` was
  written *after* looking at in-sample sweep results — `smooth_16` and `hysteresis_9`
  are in there precisely because the in-sample sweep flagged them. In what sense are
  these numbers "out of sample"? What would a genuinely clean test require?

## 8e. Current state after Part 8

Adopted: the ensemble `smooth = (1, 5, 10, 21)`. Hysteresis and full cross-section
remain available as opt-in flags but are **not** in the graded configuration.

Graded run: Sharpe **1.32**, CI [0.52, 2.21], CAGR 11.35%, max DD 7.45%, 1554 trades,
0 leverage violations.

**That number does not survive Part 9. Read on before quoting it.**

---

# Part 9 — The overfitting audit

Sharpe went from 0.55 to 1.32 over Parts 7–8. That is the point at which a careful
person stops celebrating and starts trying to break their own result.

Six tests were run. **Three pass, three fail, and one apparent confirmation turned out
to be a confound.** Reproduce them:

```bash
python -m src.research.overfitting_audit
```

## 9a. The test that matters most — Deflated Sharpe

- Q73: Read about the **Deflated Sharpe Ratio** (Bailey & López de Prado 2014). The
  ordinary Sharpe t-test asks "could ONE random strategy have scored this?". After
  trying N configurations, what is the correct question instead?
- Q74: Run the audit. Given ~120 configurations examined, what is the *expected
  maximum* Sharpe under the null? Compare it to the observed Sharpe. How much margin
  is there really?
- Q75: The audit prints DSR against several trial counts. **Below roughly how many
  independent trials does the result clear 0.95?** Is that plausible given what was
  actually tried?
- Q76: `N_TRIALS_EXAMINED = 120` is hardcoded at the top of `overfitting_audit.py`.
  Lowering it turns a FAIL into a PASS. Why is honestly maintaining that number the
  single most important discipline in this whole file?
- Q77: Many of the 120 configurations were near-duplicates (e.g. smooth=10 vs 11).
  Does that mean the *effective* number of independent trials is lower? How would you
  estimate it, and does it change the verdict?

## 9b. Tests the strategy passes

- Q78: **Monte-Carlo null.** 40 random rankings were pushed through the identical
  pipeline: mean −0.05, sd 0.39, max 0.70, and none reached 1.32. What exactly does
  this rule out — and, importantly, what does it *not* rule out?
- Q79: Why does the Monte-Carlo null's sd (0.39) match the analytic `1/sqrt(years)`?
  Verify that the expected max of 120 draws from that null matches the DSR's figure
  of 1.127. Two independent methods agreeing is worth noting in the report.
- Q80: Sub-period Sharpes are 1.28 / 0.95 / 2.19, and 1.60 excluding COVID. Positive
  everywhere. Is consistent sub-period performance sufficient evidence against
  overfitting? Why not?

## 9c. The confound — the most instructive finding here

Walk-forward appeared to vindicate everything: selection 1.39 vs baseline 0.53. It
does not.

- Q81: Compute train→test decay for every configuration. It is **positive for all of
  them**, including `momentum` (+0.18) — a signal already shown to have no predictive
  power (Fama-MacBeth t = 0.43). **A dead signal cannot generalise.** What is actually
  causing the apparent improvement?
- Q82: Given that, how much of the walk-forward "out-of-sample success" is skill and
  how much is calendar? What does this tell you about always including a known-dead
  control in any validation you run?
- Q83: Despite the confound, in the three folds where the ensemble was selected *on
  training data only* it beat the baseline all three times (+0.83 vs +0.36, −0.34 vs
  −1.03, +0.04 vs −0.95). Why is "beats the alternative" a more robust claim than
  "achieves Sharpe X"?

## 9d. The unexplained half

- Q84: IC improved 1.22x. The tail spread the portfolio actually trades improved
  1.32x. Both predict a Sharpe near 0.73. The backtest shows 1.32. **Roughly half the
  performance has no mechanism behind it.** Why is unexplained performance the most
  reliable overfitting signature there is — more reliable than any single statistic?
- Q85: P&L concentration: removing the best 10 days (0.7% of the sample) cuts Sharpe
  1.40 → 1.00. Is that fragility, or is it normal for a market-neutral book? Find a
  comparison point before deciding.

## 9e. The decision, and whether you agree with it

The ensemble was **kept**, not reverted. The reasoning is in `../LOOK_INTO.md`:
it beats the baseline out of sample, its construction has no tuned parameter, and
reverting would cost real performance for no methodological gain — because the
Deflated Sharpe can be disclosed either way.

- Q86: Do you agree? The alternative is reverting to `smooth=1` (Sharpe 0.55, DSR
  passes comfortably, nothing unexplained). Argue the other side.
- Q87: The grading rubric is roughly 50% methodology, 15% OOS performance. How should
  that weighting affect the choice between "higher number" and "fully defensible
  number"?
- Q88: Write the results paragraph for your report in your own words. It must contain
  the in-sample figure, the DSR, and the mechanism-justified estimate. There is a
  draft in `LOOK_INTO.md` — do not copy it, argue it yourself, because this is what
  the interview will probe.

## 9f. The mistake to avoid repeating

The sequence in Parts 7–8 was: search → keep the winner → construct a mechanism story
for it. That is the standard way to overfit, and it happened here even while actively
trying to avoid it.

- Q89: The ensemble removed the *final* selection step. Why does that not undo the
  search that found the smoothing family in the first place?
- Q90: Design the protocol you would follow if you started again. When would you
  decide what to test? When would you look at a backtest? What would you have to
  write down in advance?
