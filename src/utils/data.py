"""
src/utils/data.py - Track A data loading, cleaning and return construction.

Responsibilities
----------------
1. Load the long-format daily OHLCV CSV (`date, ticker, open, high, low, close, volume`).
2. Clean it: de-duplicate, repair/flag OHLC inconsistencies, align every ticker onto a
   common trading calendar, and record exactly what was changed.
3. Build the wide (date x ticker) matrices and the daily return series the signal and
   sizing layers consume.

Design rules
------------
* Nothing here peeks ahead. Every object returned is a panel indexed by date; any
  downstream consumer is responsible for slicing to `<= t`, and `signals.py` does.
* Cleaning never silently deletes data. Repairs are applied and counted, and the
  resulting `QualityReport` is what the written report quotes.
* No absolute paths: the default dataset location is read from `config.yaml`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

__all__ = [
    "Panel",
    "QualityReport",
    "REQUIRED_COLUMNS",
    "repo_root",
    "default_data_path",
    "load_raw",
    "clean_long",
    "to_panel",
    "load_panel",
    "panel_from_wide",
]

REQUIRED_COLUMNS: List[str] = ["date", "ticker", "open", "high", "low", "close", "volume"]
PRICE_COLUMNS: List[str] = ["open", "high", "low", "close"]

# A daily log return larger than this in absolute value is flagged as a suspect print.
# 40% in one session is far outside anything a liquid large-cap does organically, so it
# is worth surfacing, but it is *flagged*, never dropped -- real gaps do happen.
SUSPECT_LOG_RETURN = 0.40

# Maximum number of consecutive sessions a stale price is carried forward before the
# ticker is treated as untradable rather than merely quiet.
DEFAULT_FFILL_LIMIT = 5


# --------------------------------------------------------------------------------------
# Containers
# --------------------------------------------------------------------------------------

@dataclass
class QualityReport:
    """Everything the cleaner changed or noticed, for the EDA section of the report."""

    n_rows_raw: int = 0
    n_rows_clean: int = 0        # rows surviving clean_long
    n_panel_cells: int = 0       # non-NaN closes in the wide panel (incl. forward fills)
    n_dates: int = 0
    n_tickers: int = 0
    date_min: Optional[pd.Timestamp] = None
    date_max: Optional[pd.Timestamp] = None

    n_duplicate_rows: int = 0
    n_nonpositive_prices: int = 0
    n_ohlc_repaired: int = 0
    n_missing_cells: int = 0          # ticker-days absent from the raw file
    n_filled_cells: int = 0           # ticker-days reconstructed by forward fill
    n_zero_volume: int = 0
    n_suspect_returns: int = 0

    per_ticker: pd.DataFrame = field(default_factory=pd.DataFrame)
    suspect_returns: pd.DataFrame = field(default_factory=pd.DataFrame)

    def summary(self) -> str:
        lines = [
            "=" * 72,
            f"{'TRACK A DATA QUALITY REPORT':^72}",
            "=" * 72,
            f"  Raw rows                     : {self.n_rows_raw:,}",
            f"  Clean rows kept              : {self.n_rows_clean:,}",
            f"  Panel cells with a close     : {self.n_panel_cells:,}",
            f"  Trading days                 : {self.n_dates:,}",
            f"  Universe size                : {self.n_tickers:,}",
            f"  Date range                   : {self.date_min} -> {self.date_max}",
            "-" * 72,
            f"  Duplicate (date,ticker) rows : {self.n_duplicate_rows:,}",
            f"  Non-positive prices -> NaN   : {self.n_nonpositive_prices:,}",
            f"  OHLC bounds repaired         : {self.n_ohlc_repaired:,}",
            f"  Missing ticker-days          : {self.n_missing_cells:,}",
            f"  Forward-filled ticker-days   : {self.n_filled_cells:,}",
            f"  Zero-volume sessions         : {self.n_zero_volume:,}",
            f"  Suspect returns (|r| > {SUSPECT_LOG_RETURN:.0%})  : {self.n_suspect_returns:,}",
            "=" * 72,
        ]
        return "\n".join(lines)


@dataclass
class Panel:
    """
    Wide (date x ticker) view of the cleaned universe.

    All frames share the same index (the trading calendar) and the same columns
    (the sorted universe), so they can be combined elementwise without alignment
    surprises.
    """

    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame

    returns: pd.DataFrame          # simple close-to-close returns
    log_returns: pd.DataFrame      # log close-to-close returns
    returns_est: pd.DataFrame      # log returns with stale sessions masked to NaN

    observed: pd.DataFrame         # True where the raw file actually had the ticker-day
    filled: pd.DataFrame           # True where the close was carried forward
    tradable: pd.DataFrame         # True where the ticker can be held that session

    report: QualityReport

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.close.index

    @property
    def tickers(self) -> List[str]:
        return list(self.close.columns)

    @property
    def dollar_volume(self) -> pd.DataFrame:
        return self.close * self.volume

    def up_to(self, date) -> "Panel":
        """
        Point-in-time slice: every frame truncated to rows with index <= `date`.

        This is the only sanctioned way for research code to simulate what was
        knowable at `date`, and it is what the causality test in `signals.py`
        exercises.
        """
        ts = pd.Timestamp(date)
        mask = self.close.index <= ts
        return Panel(
            open=self.open.loc[mask],
            high=self.high.loc[mask],
            low=self.low.loc[mask],
            close=self.close.loc[mask],
            volume=self.volume.loc[mask],
            returns=self.returns.loc[mask],
            log_returns=self.log_returns.loc[mask],
            returns_est=self.returns_est.loc[mask],
            observed=self.observed.loc[mask],
            filled=self.filled.loc[mask],
            tradable=self.tradable.loc[mask],
            report=self.report,
        )

    def describe(self) -> pd.DataFrame:
        """Per-ticker summary of the return series (annualised, 252 sessions)."""
        r = self.returns_est
        out = pd.DataFrame(
            {
                "n_obs": r.notna().sum(),
                "ann_return_pct": (r.mean() * 252.0) * 100.0,
                "ann_vol_pct": (r.std(ddof=1) * np.sqrt(252.0)) * 100.0,
                "skew": r.skew(),
                "excess_kurtosis": r.kurt(),
                "min_pct": r.min() * 100.0,
                "max_pct": r.max() * 100.0,
            }
        )
        out["ann_sharpe"] = out["ann_return_pct"] / out["ann_vol_pct"].replace(0.0, np.nan)
        return out.sort_values("ann_sharpe", ascending=False)


# --------------------------------------------------------------------------------------
# Path helpers
# --------------------------------------------------------------------------------------

def repo_root() -> Path:
    """Project root, resolved relative to this file -- never an absolute hardcode."""
    # src/utils/data.py -> src/utils -> src -> repo root
    return Path(__file__).resolve().parents[2]


def default_data_path(config_name: str = "config.yaml") -> Path:
    """Track A dataset path from config.yaml, falling back to the documented default."""
    rel = "data/Dataset_PS-A.csv"
    cfg_path = repo_root() / config_name
    if cfg_path.exists():
        try:
            import yaml

            cfg = yaml.safe_load(cfg_path.read_text()) or {}
            rel = (cfg.get("track_a") or {}).get("data_path", rel)
        except Exception:
            pass
    p = Path(rel)
    return p if p.is_absolute() else repo_root() / p


# --------------------------------------------------------------------------------------
# Load
# --------------------------------------------------------------------------------------

def load_raw(path: Optional[str | Path] = None) -> pd.DataFrame:
    """
    Read the long-format CSV with minimal interference.

    Only type coercion happens here -- every judgement call lives in `clean_long`
    so that it is visible and countable.
    """
    csv_path = Path(path) if path is not None else default_data_path()
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Track A dataset not found at {csv_path}. "
            "Download Dataset_PS-A.csv from the organiser's drive and place it there."
        )

    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip().lower() for c in df.columns]

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path.name} is missing required column(s): {missing}")

    df = df[REQUIRED_COLUMNS].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    for c in PRICE_COLUMNS + ["volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


# --------------------------------------------------------------------------------------
# Clean
# --------------------------------------------------------------------------------------

def clean_long(df: pd.DataFrame, report: QualityReport) -> pd.DataFrame:
    """
    Clean the long frame in place-ish, recording every intervention on `report`.

    Steps, in order:
      1. drop rows with an unparseable date or ticker
      2. drop duplicate (date, ticker) keys, keeping the last occurrence
      3. map non-positive prices to NaN (a zero or negative price is not a price)
      4. repair OHLC bound violations by widening high/low to contain open/close
    """
    report.n_rows_raw = len(df)

    df = df.dropna(subset=["date", "ticker"])
    df = df[df["ticker"].str.len() > 0]

    dup_mask = df.duplicated(subset=["date", "ticker"], keep="last")
    report.n_duplicate_rows = int(dup_mask.sum())
    df = df.loc[~dup_mask].copy()

    # 3. Non-positive prices cannot be used in a log return or a ratio.
    nonpos = pd.DataFrame(False, index=df.index, columns=PRICE_COLUMNS)
    for c in PRICE_COLUMNS:
        nonpos[c] = df[c] <= 0.0
        df.loc[nonpos[c], c] = np.nan
    report.n_nonpositive_prices = int(nonpos.to_numpy().sum())

    # Volume may legitimately be zero (a halt), but never negative.
    df.loc[df["volume"] < 0.0, "volume"] = np.nan

    # 4. The bar must contain its own open and close. Where it does not, widen the
    #    range rather than discard the session -- the close is the quantity we trade on
    #    and it is almost always the trustworthy field.
    body_hi = df[["open", "close"]].max(axis=1)
    body_lo = df[["open", "close"]].min(axis=1)
    bad_hi = df["high"] < body_hi
    bad_lo = df["low"] > body_lo
    report.n_ohlc_repaired = int((bad_hi | bad_lo).sum())
    df.loc[bad_hi, "high"] = body_hi[bad_hi]
    df.loc[bad_lo, "low"] = body_lo[bad_lo]

    out = df.sort_values(["date", "ticker"]).reset_index(drop=True)
    report.n_rows_clean = len(out)
    return out


def _pivot(df: pd.DataFrame, value: str, index: pd.DatetimeIndex, cols: List[str]) -> pd.DataFrame:
    wide = df.pivot(index="date", columns="ticker", values=value)
    return wide.reindex(index=index, columns=cols)


def to_panel(
    df: pd.DataFrame,
    report: QualityReport,
    ffill_limit: int = DEFAULT_FFILL_LIMIT,
    min_history: int = 1,
) -> Panel:
    """
    Pivot the cleaned long frame to wide matrices and derive returns.

    Calendar handling
    -----------------
    The union of all observed dates is the trading calendar; every ticker is reindexed
    onto it. A ticker-day that is absent, or whose close is NaN, is carried forward for
    at most `ffill_limit` sessions and marked in `filled`. Those sessions produce a
    zero simple return (the price genuinely did not move in our data) but are masked out
    of `returns_est`, which is what volatility and signal estimation use -- a stale zero
    would bias realised vol downward.
    """
    dates = pd.DatetimeIndex(sorted(df["date"].unique()))
    tickers = sorted(df["ticker"].unique())

    raw = {c: _pivot(df, c, dates, tickers) for c in PRICE_COLUMNS + ["volume"]}

    observed = raw["close"].notna()
    report.n_missing_cells = int((~observed).to_numpy().sum())

    # Forward-fill prices across short gaps so that a one-off hole does not destroy a
    # 252-day momentum window. Volume is filled with 0.0: no print means no turnover.
    filled_px = {c: raw[c].ffill(limit=ffill_limit) for c in PRICE_COLUMNS}
    volume = raw["volume"].fillna(0.0)

    close = filled_px["close"]
    filled = close.notna() & (~observed)
    report.n_filled_cells = int(filled.to_numpy().sum())
    report.n_zero_volume = int(((volume <= 0.0) & observed).to_numpy().sum())

    # Returns. `returns` is the tradable P&L series; `log_returns` is the estimation
    # series; `returns_est` additionally blanks stale sessions.
    simple = close.pct_change()
    logret = np.log(close).diff()

    # A log return is unusable for estimation if either endpoint of the pair is a
    # carried-forward price: the stale day itself reads as a spurious 0.0, and the day
    # the real price resumes compresses a multi-session move into one observation.
    # Both endpoints are strictly at or before t, so this mask introduces no look-ahead.
    returns_est = logret.mask(filled | filled.shift(1, fill_value=False))

    # Suspect prints: flagged, not removed.
    suspect = returns_est.abs() > SUSPECT_LOG_RETURN
    report.n_suspect_returns = int(suspect.to_numpy().sum())
    if report.n_suspect_returns:
        idx = suspect.stack()
        idx = idx[idx].index
        report.suspect_returns = pd.DataFrame(
            {
                "date": [i[0] for i in idx],
                "ticker": [i[1] for i in idx],
                "log_return": [returns_est.at[i[0], i[1]] for i in idx],
            }
        ).sort_values("log_return", key=np.abs, ascending=False)

    # Tradability: a real, non-stale price with some history behind it.
    history = observed.cumsum()
    tradable = close.notna() & (~filled) & (history >= min_history) & (volume > 0.0)

    report.n_panel_cells = int(close.notna().to_numpy().sum())
    report.n_dates = len(dates)
    report.n_tickers = len(tickers)
    report.date_min = dates.min() if len(dates) else None
    report.date_max = dates.max() if len(dates) else None
    report.per_ticker = pd.DataFrame(
        {
            "first_date": observed.idxmax().where(observed.any()),
            "n_observed": observed.sum(),
            "n_missing": (~observed).sum(),
            "n_filled": filled.sum(),
            "n_zero_volume": ((volume <= 0.0) & observed).sum(),
            "n_suspect": suspect.sum(),
            "median_dollar_volume": (close * volume).median(),
        }
    )

    return Panel(
        open=filled_px["open"],
        high=filled_px["high"],
        low=filled_px["low"],
        close=close,
        volume=volume,
        returns=simple.fillna(0.0).where(close.notna()),
        log_returns=logret,
        returns_est=returns_est,
        observed=observed,
        filled=filled,
        tradable=tradable,
        report=report,
    )


def panel_from_wide(
    open_: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    volume: pd.DataFrame,
    ffill_limit: int = DEFAULT_FFILL_LIMIT,
    min_history: int = 1,
) -> Panel:
    """
    Build a Panel from wide frames that are already in memory.

    This exists so the live strategy in `engine.py` can assemble a Panel from the bars
    the backtester has streamed to it and get *byte-identical* returns, masks and
    volatility inputs to the research path. Any divergence between research and live
    construction is a silent source of backtest-vs-deployment mismatch, so there is
    deliberately only one implementation of that logic.
    """
    long = (
        pd.concat(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
            axis=1,
        )
        .stack(level=1, future_stack=True)
        .rename_axis(index=["date", "ticker"])
        .reset_index()
    )
    long = long.dropna(subset=["close"])
    report = QualityReport()
    clean = clean_long(long, report)
    return to_panel(clean, report, ffill_limit=ffill_limit, min_history=min_history)


def load_panel(
    path: Optional[str | Path] = None,
    ffill_limit: int = DEFAULT_FFILL_LIMIT,
    min_history: int = 1,
) -> Panel:
    """Load -> clean -> pivot. The single entry point the rest of the project uses."""
    report = QualityReport()
    raw = load_raw(path)
    clean = clean_long(raw, report)
    return to_panel(clean, report, ffill_limit=ffill_limit, min_history=min_history)
