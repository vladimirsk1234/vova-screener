"""
Pine "Sequence Vova Indicator" inputs, chart visibility, and theme defaults.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields, asdict
from typing import Any


@dataclass
class IndicatorParams:
    # Strategy / MAs
    len_fast: int = 20
    len_slow: int = 40
    length_major: int = 200

    # Elder envelope
    lookback: int = 100
    multiplier: float = 2.0
    elder_bull_color: str = "#00c853"
    elder_bear_color: str = "#ff1744"
    elder_neut_color: str = "#4eadfc"
    env_upper_color: str = "rgba(128,128,128,0.5)"
    env_lower_color: str = "rgba(128,128,128,0.5)"

    # EMA / SMA colors
    short_ema_color: str = "#2196f3"
    center_ema_color: str = "#f44336"
    sma_major_color: str = "#ff9800"

    # Structure
    hhll_color: str = "#000000"
    hhll_label_size: int = 12  # px proxy for Pine size.normal

    # Critical level
    crit_stop_color_up: str = "#00c853"
    crit_stop_color_down: str = "#f44336"
    crit_custom_color: str = "#000000"
    crit_lbl_offset: int = 10

    # Fibonacci
    fib_color: str = "#000000"
    fib_width: int = 2

    # Bollinger
    bb_length: int = 20
    bb_mult: float = 2.0
    bb_basis_color: str = "#2196f3"
    bb_upper_color: str = "#9e9e9e"
    bb_lower_color: str = "#9e9e9e"
    bb_fill_color: str = "rgba(158,158,158,0.15)"

    # Dashboard / risk
    atr_len: int = 14
    atr_low_thresh: float = 3.0
    atr_high_thresh: float = 5.0
    adx_len: int = 14
    adx_thresh: int = 20
    min_rr: float = 1.5
    use_last_hl_sl: bool = True
    risk_dollars: float = 100.0

    # Watermark
    wm_text_color: str = "#e0e0e0"
    wm_font_size: int = 11

    # Chart theme (TradingView grey plot + high-contrast candles)
    bg_color: str = "#434651"
    paper_color: str = "#2a2e39"
    grid_color: str = "#363a45"
    candle_up: str = "#089981"
    candle_down: str = "#f23645"
    candle_border: str = "#000000"
    candle_wick: str = "#000000"

    # Visibility (defaults: structure + critical only)
    show_crit_level: bool = True
    show_hhll: bool = True
    show_extension_lines: bool = True
    show_fib: bool = False
    show_short_ema: bool = False
    show_center_ema: bool = False
    show_sma_major: bool = False
    show_elder_envelope: bool = False
    show_elder_impulse: bool = False
    show_bb: bool = False
    show_bb_background: bool = True
    show_breaks: bool = False
    show_tp_sl: bool = False
    show_watermark: bool = True  # always rendered on chart; kept for session compat

    def to_runner_kwargs(self) -> dict[str, Any]:
        return {
            "atr_len": self.atr_len,
            "min_rr": self.min_rr,
            "use_last_hl_sl": self.use_last_hl_sl,
            "risk_dollars": self.risk_dollars,
            "len_fast": self.len_fast,
            "len_slow": self.len_slow,
            "length_major": self.length_major,
            "lookback": self.lookback,
            "multiplier": self.multiplier,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "IndicatorParams":
        if not d:
            return cls()
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_chart_params() -> IndicatorParams:
    return IndicatorParams()
