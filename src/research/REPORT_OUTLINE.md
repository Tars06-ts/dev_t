# Track A — Condensed Research & Report Outline

**Objective:** This is the filtered, high-impact questionnaire mapped directly to the 10-page report structure. It strips out the self-guided exploration and leaves only the core architectural questions required to provide mathematical proof for the judges.

## Part 1: Data Preprocessing & The Environment
*Focus: Prove you understand the difference between risk modeling and trade execution, and mathematically justify your data transformations.*

1. **The Execution Gap:** Why does `data.py` explicitly calculate historical close-to-close returns for the signal/risk models, while the backtester executes trades open-to-open from $t+1$? What would happen to your risk model if it only looked at open-to-open data?
2. **The Kurtosis Problem:** What is the excess kurtosis of daily returns for this specific 24-ticker universe? How does this extreme outlier behavior mathematically justify your choice to strictly use cross-sectional ranking (`cs_rank`) instead of z-scores (`cs_zscore`)?
3. **The Matrix Noise (T/N Ratio):** You estimate covariance on a 126-day window with 24 assets. What is the T/N ratio? Why does this specific ratio mathematically break standard optimizers, making Ledoit-Wolf shrinkage a mandatory precondition rather than a minor refinement?

## Part 2: Signal Selection & Guarding Against Look-Ahead Bias
*Focus: Defend the `ST_REV_21` hardcoding and the ensemble smoothing choice using rigorous statistics.*

4. **The Holy Trinity of Quant Proof:** Report the Information Coefficient (IC) and the Fama-MacBeth regression t-statistics for the 21-day short-term reversal signal. How do these numbers compare to the 12-1 momentum variants?
5. **Autocorrelation & Newey-West:** Why must the t-statistics for a 21-day holding period overlapping day-by-day be Newey-West corrected? What happens to the significance if you use a naive iid standard error?
6. **The Plateau vs. The Spike:** When sweeping the reversal lookback across 5, 10, 21, 42, and 63 days, does the IC t-stat form a smooth plateau or a single sharp spike? Why is a plateau the ultimate proof of a real economic effect rather than an overfitted parameter?
7. **The Dynamic Selection Trap:** Why is `signal="ST_REV_21"` hardcoded in `engine.py` rather than letting the code pick the best-performing variant on the fly? Distinguish two cases: selecting on the **whole sample** (genuine look-ahead bias) versus selecting on a **rolling training window** (legitimate — it is what `walkforward.py` does). Given the second is legal, what is the *empirical* argument against it here? Cite the walk-forward result: per-fold selection scored 1.32–1.39 against 1.73 for simply holding one fixed configuration, with 4 different winners across 7 folds.
8. **Ensemble over Argmax:** Why was the final signal smoothed using an ensemble average of calendar-natural units `smooth=(1, 5, 10, 21)` instead of just picking the single best-performing window (e.g., `smooth=10`)? How does this eliminate selection bias?

## Part 3: Volatility, Sizing & Transaction Cost Mechanics
*Focus: Justify the plumbing of `optimizer.py` and `volatility.py`.*

9. **Blended Volatility:** Chart the rolling volatility estimator against the Parkinson (high-low range) estimator. Why does Parkinson systematically underestimate risk (what does it miss?), and how does blending it with EWMA close-to-close variance surgically fix the flaw while keeping the statistical efficiency?
10. **Parameter Counting (Inverse Vol vs. ERC):** Why is inverse volatility (naive risk parity) mathematically safer out of sample for a 24-ticker universe than Equal Risk Contribution (ERC) or Minimum Variance? Base your answer on the exact number of parameters each model must estimate from the noisy data.
11. **The No-Order Path:** Explain the mechanics of submitting `context.set_target_weights({})` between rebalances — specifically, which line in `backtester.py::_run_track_a` makes an empty dict a true no-trade hold. How many fills does this save versus restating the targets daily (~15,600 → ~1,554)? Then check the magnitude honestly: commission is only **0.225%/yr** against a 4.23% CAGR, so what was the *larger* effect — the fees saved, or the fact that restating weights silently converts a 21-day schedule into daily drift-correction?
12. **The One-Sided Risk Budget:** Realised portfolio volatility is ~7.7% against a 10.0% target. The cause is *not* a gross cap overriding the target — find the line `w = w * min(cfg.target_vol / ev, 1.0)` in `build_weights` and explain why the `min(..., 1.0)` makes the target **one-sided**: it can only reduce risk, never add it. Forecast vol averages 7.7% and exceeds the target in just **9 of 91 rebalances**, so the book mostly sits at the gross cap. Why is refusing to lever up to reach a vol target the correct behaviour under a hard 1.0 leverage constraint?

## Part 4: The Overfitting Audit & Walk-Forward Validation
*Focus: The humility check. Prove you tried to break your own 1.32 Sharpe ratio.*

13. **The Deflated Sharpe Ratio (DSR):** Out of the ~120 configurations explored during the sweep, what is the *expected maximum* Sharpe ratio under the null hypothesis? Does the final backtest Sharpe clear the 95% confidence threshold for the DSR? 
14. **The Monte-Carlo Null:** When 40 completely random, dead rankings were pushed through this exact sizing and risk pipeline, what was the highest Sharpe they achieved? Does this independently confirm the DSR threshold?
15. **The Unexplained Half:** The mechanism-justified estimate is ~0.70 — derived from a **1.22× improvement in IC** (0.0713 → 0.0871) and **1.32× in the traded tail spread**, giving `0.55 × 1.22 = 0.67` and `0.55 × 1.32 = 0.73`. **Do not attribute this to Grinold's law** — applied directly it gives `0.07 × √288 ≈ 1.19`, a ceiling, not 0.73. Since the backtest produced 1.32, why is stating plainly that roughly half the performance is unexplained the strongest proof of quantitative maturity in the report?

16. **Grinold Tested and Failed (separate point):** The Fundamental Law predicts that doubling breadth from a 12-stock to a 24-stock book improves IR by √2 = 1.41× (0.55 → 0.78). Implemented as `selection="full"`, it delivered **1.04× (0.574)**. Why? Grinold counts *independent* bets, and mean pairwise correlation here is 0.26. Use this as a standalone result on the limits of applying textbook formulas to correlated assets.

17. **The Dead-Signal Control:** Walk-forward appeared to validate every configuration — until `momentum`, a signal already shown to have no predictive power (Fama-MacBeth t = 0.43), *also* improved from train to test (+0.18). A dead signal cannot generalise. What was actually causing the apparent improvement, and why should every validation you ever run include a known-dead control?

---

## Verified figures (use these exact numbers)

| quantity | value |
|---|---|
| excess kurtosis range | 2.5 – 29.9 (worst: skew −1.60, kurtosis 29.9) |
| covariance T/N | 126 / 24 = 5.25 |
| reversal IC (h=21) / Newey-West t | 0.0713 / **+3.64** |
| reversal Fama-MacBeth t | **+3.41** |
| momentum IC t / FM t | +0.87 / **+0.43** (no predictive power) |
| lookback sweep | plateau at 21–42d, gone by 126d |
| ensemble IC (h=21) / t | 0.0871 / +3.84 |
| parameters estimated: inverse-vol vs ERC | N vs N(N+1)/2 |
| ex-ante vol forecast vs realised | 7.7% vs 7.7% (well calibrated) |
| in-sample Sharpe / 95% CI | **1.32** / [0.52, 2.21] |
| E[max Sharpe \| null, ~120 trials] | **1.127** |
| **Deflated Sharpe Ratio** | **0.736 — does NOT clear 0.95** |
| Monte-Carlo null (40 random rankings) | mean −0.047, sd 0.394, **max 0.695**, none ≥ 1.32 |
| mechanism-justified estimate | **~0.70** |

**Changelog:** Q7, Q11, Q12 and Q15 were corrected against measured results — Q15 had
misattributed the 0.73 to Grinold's law. Q16 (the failed breadth prediction) and Q17
(the dead-signal control) were added; both are strong report material that the
original outline omitted.
