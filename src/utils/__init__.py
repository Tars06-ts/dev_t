"""
Participant utilities: data loading, feature extraction, and shared math.
"""

from .data import Panel, QualityReport, load_panel, panel_from_wide
from .math_helpers import cs_rank, cs_winsorize, cs_zscore, newey_west_tstat
from .signals import SIGNAL_LIBRARY, assert_causal, build_signals, forward_returns

__all__ = [
    "Panel", "QualityReport", "load_panel", "panel_from_wide",
    "cs_rank", "cs_winsorize", "cs_zscore", "newey_west_tstat",
    "SIGNAL_LIBRARY", "assert_causal", "build_signals", "forward_returns",
]
