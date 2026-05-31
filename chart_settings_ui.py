"""
Streamlit UI for Plotly chart indicator parameters (not frozen during scan).
"""
from __future__ import annotations

import streamlit as st

from indicator_params import IndicatorParams, default_chart_params


def _init_chart_params() -> None:
    if "chart_params" not in st.session_state:
        st.session_state.chart_params = default_chart_params().as_dict()


def get_chart_params() -> IndicatorParams:
    _init_chart_params()
    return IndicatorParams.from_dict(st.session_state.chart_params)


def _apply_hardcoded_params(p: IndicatorParams) -> None:
    """Fixed visibility, BB, Elder off, dashboard/watermark from scan defaults."""
    p.show_crit_level = True
    p.show_hhll = True
    p.show_extension_lines = True
    p.show_breaks = True
    p.show_watermark = True
    p.show_elder_envelope = False
    p.show_elder_impulse = False
    p.bb_length = 20
    p.bb_mult = 2.0
    p.show_bb_background = False
    p.atr_len = 14
    p.atr_low_thresh = 3.0
    p.atr_high_thresh = 5.0
    p.adx_len = 14
    p.adx_thresh = 20
    p.wm_text_color = "#e0e0e0"
    run = st.session_state.get("run_params", {})
    p.risk_dollars = float(run.get("risk_per_trade", 100))
    p.use_last_hl_sl = bool(run.get("use_last_hl_sl", True))
    p.min_rr = float(run.get("rr", 1.5))


def render_chart_settings() -> IndicatorParams:
    """Chart settings expander; returns current IndicatorParams."""
    _init_chart_params()
    p = IndicatorParams.from_dict(st.session_state.chart_params)

    with st.expander("Chart indicator settings", expanded=False):
        st.caption("Changes apply immediately to the selected chart (no re-scan).")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Visibility**")
            p.show_fib = st.checkbox("Fibonacci", value=p.show_fib)
            p.show_short_ema = st.checkbox("Short EMA", value=p.show_short_ema)
            p.show_center_ema = st.checkbox("Center EMA", value=p.show_center_ema)
            p.show_sma_major = st.checkbox(f"SMA {p.length_major}", value=p.show_sma_major)
        with c2:
            st.markdown("**More overlays**")
            p.show_bb = st.checkbox("Bollinger Bands", value=p.show_bb)
            p.show_tp_sl = st.checkbox("TP / SL lines", value=p.show_tp_sl)
        with c3:
            st.markdown("**Moving averages**")
            p.len_fast = st.number_input("Fast EMA", value=int(p.len_fast), min_value=1, step=1)
            p.len_slow = st.number_input("Center EMA", value=int(p.len_slow), min_value=1, step=1)
            p.length_major = st.number_input("Major SMA", value=int(p.length_major), min_value=1, step=1)

        st.markdown("**Colors & theme**")
        r1, r2, r3, r4 = st.columns(4)
        with r1:
            p.bg_color = st.color_picker("Background", p.bg_color)
            p.paper_color = st.color_picker("Paper", p.paper_color)
            p.grid_color = st.color_picker("Grid", p.grid_color)
            p.candle_up = st.color_picker("Candle body up", p.candle_up)
            p.candle_down = st.color_picker("Candle body down", p.candle_down)
            p.candle_border = st.color_picker("Candle border", p.candle_border)
            p.candle_wick = st.color_picker("Candle wick", p.candle_wick)
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
        with r4:
            p.bb_basis_color = st.color_picker("BB basis", p.bb_basis_color)
            p.bb_upper_color = st.color_picker("BB upper", p.bb_upper_color)
            p.bb_lower_color = st.color_picker("BB lower", p.bb_lower_color)

        if st.button("Reset chart settings to defaults", use_container_width=True):
            st.session_state.chart_params = default_chart_params().as_dict()
            st.rerun()

    _apply_hardcoded_params(p)
    st.session_state.chart_params = p.as_dict()
    return p
