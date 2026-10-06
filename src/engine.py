"""
src/engine.py - Mandatory Participant Strategy Entry Point
Inter IIT Tech Meet 15.0 Quant Evaluation

Track A: volatility-adjusted cross-sectional long/short strategy.

Pipeline, executed every `rebalance_cadence` sessions:

    bars streamed in  ->  utils.data.panel_from_wide    (returns, masks, tradability)
                      ->  utils.signals.build_signals   (cross-sectional ranking)
                      ->  models.volatility.estimate_vol + models.covariance
                      ->  models.optimizer.build_weights (sizing + constraints)
                      ->  context.set_target_weights

Why it is built this way
------------------------
The strategy holds *only* the bars the backtester has already handed it. There is no
file read, no global fit and no precomputed frame: look-ahead is impossible by
construction rather than by convention. The panel is rebuilt from those buffers using
the same `utils.data.panel_from_wide` the research notebooks use, so live and research
numbers cannot drift apart.

Signal choice is evidence-led and documented in `utils.signals.short_term_reversal`: on the
in-sample panel, 21-day reversal is the only one of the three candidates with
statistical support (IC t = +3.6, Fama-MacBeth t = +3.4, stable across sub-periods and
stronger excluding COVID), while 12-1 momentum is indistinguishable from zero.

Track B is not implemented -- this submission targets Track A. The handler stays flat
so the module remains importable and the organiser's sanity tests pass.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from strategy_base import BaseStrategy, Context, Bar

from src.utils.data import panel_from_wide
from src.utils.signals import build_signals
from src.models.covariance import ledoit_wolf_cov
from src.models.optimizer import (
    SizingConfig,
    apply_caps,
    apply_no_trade_band,
    build_weights,
    ex_ante_vol,
)
from src.models.volatility import VolConfig, estimate_vol

# --------------------------------------------------------------------------------------
# Strategy parameters. Kept in one block so the sensitivity analysis in the report can
# sweep them without touching logic.
# --------------------------------------------------------------------------------------
# Minimum sessions of history each signal needs before it can produce a score.
# Must stay in sync with the lookbacks in utils/signals.py.
SIGNAL_MIN_HISTORY = {
    "MOM_12_1": 252 + 21,
    "VOLADJ_MOM_12_1": 252 + 21,
    "ST_REV_21": 21,
}

PARAMS = dict(
    signal="ST_REV_21",        # see utils.signals.SIGNAL_LIBRARY
    standardise="rank",        # rank is outlier-proof; this universe has kurtosis up to 30
    smooth=(1, 5, 10, 21),     # ENSEMBLE, not a single tuned window.
                               # Picking the best smoothing window from a sweep would
                               # import selection bias: the winner's score is inflated
                               # by however many settings were tried, and the paired
                               # test shows smooth=10's IC gain over smooth=1 is NOT
                               # significant (t=1.85). Averaging ranks across the whole
                               # plausible range removes the argmax entirely, so the
                               # result is an honest estimate rather than a maximum.
                               # IC lands between the members (0.087, t=3.84) exactly
                               # as theory predicts. IR is proportional to IC, so the
                               # defensible expectation is ~0.55 * 1.22 = 0.67; the
                               # backtest prints higher and that excess is unexplained.
    rebalance_cadence=21,      # sessions between rebalances (overridden by config.yaml).
                               # 21 is chosen from the IC horizon study in signals.py,
                               # NOT from backtest Sharpe: the reversal signal's
                               # predictive power peaks at h=21 (t=+3.6) and is far
                               # weaker at h=5 (t=+1.8), so the holding period is
                               # matched to the horizon where the edge was measured.
    max_history=None,          # trailing buffer depth; None = derive from the lookbacks
                               # actually in use. A fixed value is a trap: set it below
                               # a signal's lookback and that signal can never compute,
                               # which reads as "the signal is bad" rather than
                               # "the buffer is too short".
    drift_tolerance=1.03,      # early rebalance once gross exceeds max_gross * this
    seed=42,
)

VOL = VolConfig(
    method="blended",          # EWMA close-to-close blended with Parkinson range
    window=63,
    halflife=42,
    blend_weight=0.5,
    min_periods=20,
)

SIZING = SizingConfig(
    selection="topn",          # "full" weights all 24 names (2x breadth, see optimizer)
    n_long=6,
    n_short=6,
    method="inverse_vol",      # naive risk parity: stable, needs only N variances
    dollar_neutral=True,
    max_gross=0.95,            # strictly inside the RuleGuard's 1.0, with drift headroom
    max_position=0.12,
    target_vol=0.10,           # annualised ex-ante
    cov_window=126,
    no_trade_band=0.005,
    exit_buffer=0,             # hysteresis: 0 = plain top-N; >0 widens the exit boundary
)


class ParticipantStrategy(BaseStrategy):
    """Volatility-adjusted cross-sectional long/short strategy (Track A)."""

    def __init__(self, **overrides) -> None:
        """
        `ParticipantStrategy()` with no arguments is the graded configuration -- the
        backtester constructs it exactly that way. The keyword overrides exist so the
        sensitivity study in `src/research/sweep.py` can vary one parameter at a time
        without editing this file; they accept any key of PARAMS, VolConfig or
        SizingConfig.
        """
        self._overrides = dict(overrides)

    # ----------------------------------------------------------------------------------
    # Lifecycle
    # ----------------------------------------------------------------------------------
    def initialize(self, context: Context) -> None:
        import dataclasses

        vol_fields = {f.name for f in dataclasses.fields(VolConfig)}
        size_fields = {f.name for f in dataclasses.fields(SizingConfig)}
        # `method` exists in BOTH configs, so a bare override would silently set the
        # volatility estimator and the sizing rule at once. Names that collide must be
        # qualified as vol_<name> / sizing_<name>; everything unique stays bare.
        ambiguous = vol_fields & size_fields

        ov = dict(getattr(self, "_overrides", {}))
        params, vol_kw, size_kw, unknown = dict(PARAMS), {}, {}, []
        for k, v in ov.items():
            if k.startswith("vol_") and k[4:] in vol_fields:
                vol_kw[k[4:]] = v
            elif k.startswith("sizing_") and k[7:] in size_fields:
                size_kw[k[7:]] = v
            elif k in ambiguous:
                unknown.append(f"{k} (ambiguous: use vol_{k} or sizing_{k})")
            elif k in params:
                params[k] = v
            elif k in vol_fields:
                vol_kw[k] = v
            elif k in size_fields:
                size_kw[k] = v
            else:
                unknown.append(k)
        if unknown:
            raise KeyError(f"Unknown strategy override(s): {sorted(unknown)}")

        vol_cfg = dataclasses.replace(VOL, **vol_kw)
        size_cfg = dataclasses.replace(SIZING, **size_kw)

        np.random.seed(params["seed"])

        self.signal_name: str = params["signal"]
        self.standardise: str = params["standardise"]
        self.smooth = params["smooth"]   # int, or a sequence for ensembling
        self.cadence: int = int(
            ov["rebalance_cadence"] if "rebalance_cadence" in ov
            else self._config_cadence(params["rebalance_cadence"])
        )
        # Derive the history requirement from what is actually configured, so a
        # longer-lookback signal or a wider covariance window cannot silently starve.
        need = max(
            SIGNAL_MIN_HISTORY.get(self.signal_name, 252),
            size_cfg.cov_window,
            vol_cfg.window,
            vol_cfg.halflife * 3,
        )
        self.warmup: int = need + 25
        self.max_history: int = int(
            params["max_history"] or (self.warmup + 2 * self.cadence + 10)
        )
        if self.max_history < self.warmup:
            raise ValueError(
                f"max_history={self.max_history} is below the {self.warmup}-session "
                f"warmup needed by signal {self.signal_name!r}; it could never compute."
            )
        self.drift_tolerance: float = float(params["drift_tolerance"])
        self.vol_cfg: VolConfig = vol_cfg
        self.size_cfg: SizingConfig = size_cfg

        # Rolling buffers of streamed bars: one dict {ticker: value} per session.
        self._dates: List = []
        self._open: List[Dict[str, float]] = []
        self._high: List[Dict[str, float]] = []
        self._low: List[Dict[str, float]] = []
        self._close: List[Dict[str, float]] = []
        self._volume: List[Dict[str, float]] = []


        # Monotonic count of bars seen. MUST be separate from len(self._dates):
        # that list is capped at max_history, so using it for the rebalance schedule
        # freezes the modulo at a constant and the strategy silently stops trading
        # once the buffer fills.
        self._n_bars: int = 0
        self._last_weights: pd.Series = pd.Series(dtype=float)
        self.diagnostics: List[Dict[str, float]] = []
        # Per-bar equity, recorded through the public Context API. The backtester does
        # not expose its equity curve on the result object, and walk-forward needs to
        # score performance inside sub-windows of a single continuous run.
        self.equity_log: List[tuple] = []
        self._n_rebalances: int = 0

        if context.symbol is None:
            print(
                f"[ParticipantStrategy] Track A | signal={self.signal_name} "
                f"| sizing={self.size_cfg.method} | vol={self.vol_cfg.method} "
                f"| cadence={self.cadence}d | warmup={self.warmup}d "
                f"| universe={len(context.universe)}"
            )
        else:
            print(f"[ParticipantStrategy] Track B detected ({context.symbol}) - staying flat.")

    @staticmethod
    def _config_cadence(default: int) -> int:
        """Honour `track_a.rebalance_cadence` from config.yaml when it is readable."""
        try:
            from src.utils.data import repo_root
            import yaml

            cfg_path = repo_root() / "config.yaml"
            if cfg_path.exists():
                cfg = yaml.safe_load(cfg_path.read_text()) or {}
                return int((cfg.get("track_a") or {}).get("rebalance_cadence", default))
        except Exception:
            pass
        return default

    def on_bar(self, context: Context, bars: Dict[str, Bar]) -> None:
        if context.symbol is not None or len(bars) <= 1:
            self._handle_track_b(context, bars)
        else:
            self._handle_track_a(context, bars)

    # ----------------------------------------------------------------------------------
    # Track B - out of scope for this submission; hold no position.
    # ----------------------------------------------------------------------------------
    def _handle_track_b(self, context: Context, bars: Dict[str, Bar]) -> None:
        return

    # ----------------------------------------------------------------------------------
    # Track A
    # ----------------------------------------------------------------------------------
    def _handle_track_a(self, context: Context, bars: Dict[str, Bar]) -> None:
        self._append(context.timestamp, bars)
        self.equity_log.append((pd.Timestamp(context.timestamp), float(context.portfolio_value)))

        # Not enough history to estimate anything honestly -> stay in cash.
        if self._n_bars < self.warmup:
            return

        current = self._current_weights(context, bars)

        # Hold between rebalances by submitting an EMPTY weight dict.
        #
        # This exploits a specific property of the engine: `_run_track_a` executes
        # pending orders only `if pending_target_weights`, and an empty dict is falsy,
        # so nothing is filled and the existing book simply carries. The two obvious
        # alternatives are both wrong -- submitting the previous *targets* makes the
        # ledger recompute desired_shares from the new open each day and silently
        # drift-corrects (that is daily rebalancing, not a 5-day schedule), and
        # restating actual drifted weights still trades the overnight gap on every
        # name every day. Measured: those cost ~15,600 round-trip fills over the
        # sample against ~1,400 here, for the same intended positions.
        on_schedule = (self._n_bars - self.warmup) % self.cadence == 0

        # Risk trigger: if the book drifts materially past the gross cap between
        # scheduled rebalances, rebalance early rather than approach the RuleGuard's
        # hard 1.0. The tolerance matters -- triggering at exactly max_gross fires on
        # almost every bar, because the book oscillates around that level, which
        # collapses the 5-day schedule back into daily trading.
        limit = self.size_cfg.max_gross * self.drift_tolerance
        drifted = len(current) > 0 and float(current.abs().sum()) > limit

        if not (on_schedule or drifted):
            context.set_target_weights({})
            return

        weights = self._compute_target(context, current)
        if weights is None:
            context.set_target_weights({})
            return

        context.set_target_weights(weights.to_dict())
        self._last_weights = weights

    # ----------------------------------------------------------------------------------
    # Internals
    # ----------------------------------------------------------------------------------
    def _append(self, timestamp, bars: Dict[str, Bar]) -> None:
        """Push one session of bars onto the rolling buffers."""
        self._n_bars += 1
        self._dates.append(pd.Timestamp(timestamp))
        self._open.append({t: b.open for t, b in bars.items()})
        self._high.append({t: b.high for t, b in bars.items()})
        self._low.append({t: b.low for t, b in bars.items()})
        self._close.append({t: b.close for t, b in bars.items()})
        self._volume.append({t: b.volume for t, b in bars.items()})

        if len(self._dates) > self.max_history:
            cut = len(self._dates) - self.max_history
            del self._dates[:cut]
            for buf in (self._open, self._high, self._low, self._close, self._volume):
                del buf[:cut]

    def _build_panel(self):
        idx = pd.DatetimeIndex(self._dates)
        frames = [
            pd.DataFrame(buf, index=idx)
            for buf in (self._open, self._high, self._low, self._close, self._volume)
        ]
        cols = sorted(set().union(*[f.columns for f in frames]))
        frames = [f.reindex(columns=cols) for f in frames]
        return panel_from_wide(*frames)

    def _current_weights(self, context: Context, bars: Dict[str, Bar]) -> pd.Series:
        """Actual portfolio weights right now, marked at today's close."""
        equity = float(context.portfolio_value)
        if equity <= 0.0 or not context.positions:
            return pd.Series(dtype=float)
        w = {
            t: (shares * bars[t].close) / equity
            for t, shares in context.positions.items()
            if t in bars and shares
        }
        # Deliberately NOT capped: this is a measurement of what we actually hold, and
        # the drift trigger in _handle_track_a needs to see a real breach to fire.
        # Capping belongs on the submitted target, not on the observation.
        return pd.Series(w, dtype=float)

    def _compute_target(self, context: Context, current: pd.Series) -> Optional[pd.Series]:
        """Full rebalance. Returns None when the inputs are too thin to act on."""
        try:
            panel = self._build_panel()

            score = build_signals(panel, [self.signal_name],
                                  standardise=self.standardise, smooth=self.smooth)
            scores = score[self.signal_name].iloc[-1].dropna()
            if len(scores) < self.size_cfg.n_long + self.size_cfg.n_short:
                return None

            vol = estimate_vol(panel, self.vol_cfg).iloc[-1]
            cov = ledoit_wolf_cov(panel.returns_est.tail(self.size_cfg.cov_window))

            # `current` lets the optimiser apply hysteresis: a held name is kept while
            # it stays inside the widened boundary instead of being churned on noise.
            target = build_weights(scores, vol, cov, self.size_cfg, current=current)
            if target.empty:
                return None

            target = apply_no_trade_band(target, current, self.size_cfg.no_trade_band)
            target = apply_caps(target, self.size_cfg.max_position, self.size_cfg.max_gross)
            if target.empty or not np.isfinite(target.to_numpy()).all():
                return None

            self._n_rebalances += 1
            self.diagnostics.append(
                {
                    "date": self._dates[-1],
                    "gross": float(target.abs().sum()),
                    "net": float(target.sum()),
                    "n_positions": int((target.abs() > 1e-9).sum()),
                    "ex_ante_vol": ex_ante_vol(target, cov),
                    "turnover": float((target.reindex(
                        target.index.union(current.index)).fillna(0.0)
                        - current.reindex(
                        target.index.union(current.index)).fillna(0.0)).abs().sum()),
                }
            )
            return target

        except Exception as exc:  # never take the whole backtest down on one bad bar
            print(f"[ParticipantStrategy] rebalance skipped at {self._dates[-1]}: {exc}")
            return None

    def equity_frame(self) -> pd.Series:
        """Daily marked-to-market equity, indexed by date."""
        if not self.equity_log:
            return pd.Series(dtype=float)
        idx, val = zip(*self.equity_log)
        return pd.Series(val, index=pd.DatetimeIndex(idx), name="equity")

    def diagnostics_frame(self) -> pd.DataFrame:
        """Per-rebalance gross/net/turnover/ex-ante vol, for the report's exhibits."""
        return pd.DataFrame(self.diagnostics).set_index("date") if self.diagnostics else pd.DataFrame()
