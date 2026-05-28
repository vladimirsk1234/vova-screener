"""
Streamlit UI for Plotly chart indicator parameters (not frozen during scan).
"""
from __future__ import annotations

import streamlit as st

from indicator_params import IndicatorParams, default_chart_params


def _rgba_to_hex(color: str) -> str:
    """Best-effort rgba(...) -> #rrggbb for Streamlit color_picker."""
    c = (color or "").strip()
    if c.startswith("#"):
        return c
    if c.startswith("rgba(") and c.endswith(")"):
        parts = [p.strip() for p in c[5:-1].split(",")]
        if len(parts) >= 3:
            try:
                r, g, b = (int(float(parts[0])), int(float(parts[1])), int(float(parts[2])))
                return f"#{r:02x}{g:02x}{b:02x}"
            except ValueError:
                pass
    return "#808080"


def _init_chart_params() -> None:
    if "chart_params" not in st.session_state:
        st.session_state.chart_params = default_chart_params().as_dict()


def get_chart_params() -> IndicatorParams:
    _init_chart_params()
    return IndicatorParams.from_dict(st.session_state.chart_params)


def render_chart_settings() -> IndicatorParams:
    """Chart settings expander; returns current IndicatorParams."""
    _init_chart_params()
    p = IndicatorParams.from_dict(st.session_state.chart_params)

    with st.expander("Chart indicator settings", expanded=False):
        st.caption("Changes apply immediately to the selected chart (no re-scan).")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Visibility**")
            p.show_crit_level = st.checkbox("Critical level", value=p.show_crit_level)
            p.show_hhll = st.checkbox("HH / LL labels", value=p.show_hhll)
            p.show_extension_lines = st.checkbox("Extension lines", value=p.show_extension_lines)
            p.show_fib = st.checkbox("Fibonacci", value=p.show_fib)
            p.show_short_ema = st.checkbox("Short EMA", value=p.show_short_ema)
            p.show_center_ema = st.checkbox("Center EMA", value=p.show_center_ema)
            p.show_sma_major = st.checkbox(f"SMA {p.length_major}", value=p.show_sma_major)
        with c2:
            st.markdown("**More overlays**")
            p.show_elder_envelope = st.checkbox("Elder envelope", value=p.show_elder_envelope)
            p.show_elder_impulse = st.checkbox("Elder impulse tint", value=p.show_elder_impulse)
            p.show_bb = st.checkbox("Bollinger Bands", value=p.show_bb)
            p.show_bb_background = st.checkbox("BB fill", value=p.show_bb_background)
            p.show_breaks = st.checkbox("Break triangles", value=p.show_breaks)
            p.show_tp_sl = st.checkbox("TP / SL lines", value=p.show_tp_sl)
            p.show_watermark = st.checkbox("Watermark table", value=p.show_watermark)
        with c3:
            st.markdown("**Moving averages**")
            p.len_fast = st.number_input("Fast EMA", value=int(p.len_fast), min_value=1, step=1)
            p.len_slow = st.number_input("Center EMA", value=int(p.len_slow), min_value=1, step=1)
            p.length_major = st.number_input("Major SMA", value=int(p.length_major), min_value=1, step=1)
            p.lookback = st.number_input("Envelope lookback", value=int(p.lookback), min_value=0, step=1)
            p.multiplier = st.number_input("Envelope mult", value=float(p.multiplier), min_value=0.0, step=0.1)
            p.bb_length = st.number_input("BB length", value=int(p.bb_length), min_value=1, step=1)
            p.bb_mult = st.number_input("BB std mult", value=float(p.bb_mult), min_value=0.01, step=0.1)

        st.markdown("**Colors & theme**")
        r1, r2, r3, r4 = st.columns(4)
        with r1:
            p.bg_color = st.color_picker("Background", p.bg_color)
            p.paper_color = st.color_picker("Paper", p.paper_color)
            p.grid_color = st.color_picker("Grid", p.grid_color)
            p.candle_up = st.color_picker("Candle up", p.candle_up)
            p.candle_down = st.color_picker("Candle down", p.candle_down)
        with r2:
            p.hhll_color = st.color_picker("HH/LL labels", p.hhll_color)
            p.crit_stop_color_up = st.color_picker("Critical (up)", p.crit_stop_color_up)
            p.crit_stop_color_down = st.color_picker("Critical (down)", p.crit_stop_color_down)
            p.crit_custom_color = st.color_picker("Critical label text", p.crit_custom_color)
            p.fib_color = st.color_picker("Fibonacci", p.fib_color)
            p.fib_width = int(st.number_input("Fib line width", value=int(p.fib_width), min_value=1, max_value=5, step=1))
        with r3:
            p.short_ema_color = st.color_picker("Short EMA", p.short_ema_color)
            p.center_ema_color = st.color_picker("Center EMA", p.center_ema_color)
            p.sma_major_color = st.color_picker("Major SMA", p.sma_major_color)
            p.elder_bull_color = st.color_picker("Impulse bull", p.elder_bull_color)
            p.elder_bear_color = st.color_picker("Impulse bear", p.elder_bear_color)
            p.elder_neut_color = st.color_picker("Impulse neutral", p.elder_neut_color)
        with r4:
            p.env_upper_color = st.color_picker("Envelope upper", _rgba_to_hex(p.env_upper_color))
            p.env_lower_color = st.color_picker("Envelope lower", _rgba_to_hex(p.env_lower_color))
            p.bb_basis_color = st.color_picker("BB basis", p.bb_basis_color)
            p.bb_upper_color = st.color_picker("BB upper", p.bb_upper_color)
            p.bb_lower_color = st.color_picker("BB lower", p.bb_lower_color)
            p.bb_fill_color = st.color_picker("BB fill", _rgba_to_hex(p.bb_fill_color))
            p.wm_text_color = st.color_picker("Watermark text", p.wm_text_color)

        st.markdown("**Dashboard / risk (watermark & trade row)**")
        d1, d2, d3 = st.columns(3)
        with d1:
            p.min_rr = st.number_input("Min R/R", value=float(p.min_rr), min_value=0.5, step=0.1)
            p.atr_low_thresh = st.number_input("ATR low %", value=float(p.atr_low_thresh), min_value=0.1, step=0.1)
            p.atr_high_thresh = st.number_input("ATR high %", value=float(p.atr_high_thresh), min_value=0.1, step=0.1)
        with d2:
            p.adx_len = st.number_input("ADX length", value=int(p.adx_len), min_value=1, step=1)
            p.adx_thresh = st.number_input("ADX threshold", value=int(p.adx_thresh), min_value=1, step=1)
            p.atr_len = st.number_input("ATR length", value=int(p.atr_len), min_value=1, step=1)
        with d3:
            p.risk_dollars = float(
                st.session_state.get("run_params", {}).get("risk_per_trade", p.risk_dollars)
            )
            p.use_last_hl_sl = bool(
                st.session_state.get("run_params", {}).get("use_last_hl_sl", p.use_last_hl_sl)
            )

        if st.button("Reset chart settings to defaults", use_container_width=True):
            st.session_state.chart_params = default_chart_params().as_dict()
            st.rerun()

    st.session_state.chart_params = p.as_dict()
    return p
