"""
Watermark table lines: D/W/M sequence & structure status, trade row (Pine dashboard port).
"""
from __future__ import annotations

import math
from typing import Any

import pandas as pd

from data_utils import resample_to_timeframe
from indicator_params import IndicatorParams
from sequence_vova import run_sequence_vova_full, structure_snapshot_from_full


def _emoji_state(state: int) -> str:
    if state == 1:
        return "🟢"
    if state == -1:
        return "🔴"
    return "🟡"


def _seq_display(snap: dict, sma_major_above: bool | None = None) -> int:
    """Price vs critical (primary), SMA200 fallback."""
    crit = snap.get("critical_level")
    close = snap.get("close")
    seq = int(snap.get("seq_state", 0))
    if crit is not None and close is not None:
        if close > crit:
            return 1
        if close < crit:
            return -1
    if crit is None and sma_major_above is not None:
        return 1 if sma_major_above else -1
    return seq


def _struct_display(snap: dict, sma_above: bool | None = None) -> tuple[str, str]:
    """Returns (emoji, suffix label)."""
    invalid = bool(snap.get("struct_invalid"))
    trough_hl = bool(snap.get("last_trough_was_hl"))
    peak_hh = bool(snap.get("last_peak_was_hh"))
    last_peak = snap.get("last_peak")
    close = snap.get("close")
    seq = int(snap.get("seq_state", 0))
    last_lh = snap.get("last_lh")
    seq_high = snap.get("seq_high")

    if close is not None and last_peak is not None:
        has_hl = trough_hl and not invalid
        new_high_above_lh = (
            seq == 1
            and last_lh is not None
            and seq_high is not None
            and seq_high > last_lh
        )
        green = has_hl and (peak_hh or new_high_above_lh)
        yellow = has_hl and not green
        if green:
            return "🟢", " (HL+HH)"
        if yellow:
            return "🟡", " (HL)"
        return "🔴", ""
    if sma_above:
        return "🟡", ""
    return "🔴", ""


def snapshot_for_df(df: pd.DataFrame, params: IndicatorParams) -> dict | None:
    if df is None or len(df) < 2:
        return None
    full = run_sequence_vova_full(df, params=params)
    if full is None:
        return None
    snap = structure_snapshot_from_full(full)
    close = df["Close"].iloc[-1]
    sma_major = full["overlays"]["sma_major"].iloc[-1]
    snap["sma_above"] = bool(close > sma_major) if not math.isnan(sma_major) else None
    snap["sma_major"] = float(sma_major)
    return snap


def build_dwm_lines(
    df_chart: pd.DataFrame,
    df_daily: pd.DataFrame | None,
    params: IndicatorParams,
    *,
    chart_tf: str = "Daily",
    length_major: int | None = None,
) -> dict[str, str]:
    """Build D / W / M status lines from resampled OHLC."""
    length_major = length_major or params.length_major
    daily = df_daily if df_daily is not None and not df_daily.empty else df_chart
    if chart_tf == "Daily":
        daily = df_chart

    def _resample_safe(frame: pd.DataFrame, target_tf: str) -> pd.DataFrame:
        if not isinstance(frame.index, pd.DatetimeIndex):
            return frame
        out = resample_to_timeframe(frame, target_tf)
        return out if out is not None and not out.empty else frame

    weekly = _resample_safe(daily, "Weekly")
    monthly = _resample_safe(daily, "Monthly")

    d_snap = snapshot_for_df(daily, params)
    w_snap = snapshot_for_df(weekly, params)
    m_snap = snapshot_for_df(monthly, params)

    lines: dict[str, str] = {}
    if d_snap:
        d_seq = _seq_display(d_snap, d_snap.get("sma_above"))
        d_struct_e, d_struct_l = _struct_display(d_snap, d_snap.get("sma_above"))
        ma_e = "🟢" if d_snap.get("sma_above") else "🔴"
        lines["daily"] = (
            f"D: Seq {_emoji_state(d_seq)}   Struct {d_struct_e}{d_struct_l}   "
            f"SMA {length_major} {ma_e}"
        )
    if w_snap:
        w_seq = _seq_display(w_snap)
        w_struct_e, w_struct_l = _struct_display(w_snap)
        lines["weekly"] = f"W: Seq {_emoji_state(w_seq)}   Struct {w_struct_e}{w_struct_l}"
    if m_snap:
        m_seq = _seq_display(m_snap)
        m_struct_e, m_struct_l = _struct_display(m_snap)
        lines["monthly"] = f"M: Seq {_emoji_state(m_seq)}   Struct {m_struct_e}{m_struct_l}"
    return lines


def build_trade_line(full: dict, params: IndicatorParams, bar_index_last: int) -> str:
    """Row 8: chart TF trade status."""
    seq_state = full.get("seq_state_final", 0)
    last_trough_hl = full.get("last_trough_was_hl", False)
    last_peak = full.get("last_peak")
    struct_invalid = full.get("struct_invalid_seq_down", False)
    close = full.get("Close", 0)
    struct_ok = (
        last_trough_hl
        or (last_peak is not None and close > last_peak and last_trough_hl)
    ) and not struct_invalid
    seq_ok = seq_state == 1

    crit = full.get("critical_level")
    atr = full.get("ATR", 0)
    sl = min(crit if crit is not None else close - atr, close - atr)
    risk = close - sl
    reward = (last_peak - close) if last_peak is not None else 0.0
    rr = (reward / risk) if risk > 0 else 0.0
    valid = seq_ok and struct_ok and rr >= params.min_rr and risk > 0 and reward > 0

    bear = full.get("bearish_break")
    bearish_break_last = bool(bear[-1]) if bear is not None and len(bear) else False
    sig_idx = full.get("signal_bar_index")

    if valid and bearish_break_last:
        return f"🆕 NEW | R/R: {rr:.2f}"
    if valid:
        bars_since = (bar_index_last - sig_idx) if sig_idx is not None else 0
        return f"✅ VALID | R/R: {rr:.2f} | Bars {bars_since}"
    debug = ""
    if not seq_ok:
        debug += "Seq❌ "
    if not struct_ok:
        debug += "Struct❌ "
    if not debug and rr < params.min_rr:
        return f"❌ R/R too low: {rr:.2f} (need {params.min_rr:.2f})"
    if not debug:
        return "NO SETUP: Risk/Reward Invalid"
    return f"NO SETUP: {debug.strip()}"


def atr_emoji(val: float, low_t: float, high_t: float) -> str:
    if val > high_t:
        return "🔴"
    if val >= low_t:
        return "🟡"
    return "🟢"


def build_watermark_text(
    *,
    fundamentals: dict[str, Any],
    full: dict,
    params: IndicatorParams,
    dwm_lines: dict[str, str],
    chart_tf: str,
    ticker: str,
    trade_line: str,
) -> str:
    """Multi-line watermark block for Plotly annotation."""
    rows: list[str] = []
    rows.append("")
    desc = fundamentals.get("description") or fundamentals.get("company_name") or ticker
    rows.append(str(desc))
    d_chg = fundamentals.get("daily_chg_str", "")
    mcap = fundamentals.get("mcap_str", "N/A")
    rows.append(f"{ticker} ({chart_tf}) | {d_chg} | {mcap}")
    pe = fundamentals.get("pe_str", "N/A")
    earn = fundamentals.get("earn_str", "N/A")
    rows.append(f"PE: {pe} | Earn: {earn}")

    atr_val = full.get("ATR", 0)
    atr_pct = full.get("ATR_pct", 0)
    adx_val = full.get("ADX", 0)
    atr_e = atr_emoji(atr_pct, params.atr_low_thresh, params.atr_high_thresh)
    adx_suffix = f"   ADX: {adx_val:.2f}" if full.get("Valid") else ""
    rows.append(f"ATR: {atr_val:.2f} ({atr_pct:.2f}%) {atr_e}{adx_suffix}")

    if "daily" in dwm_lines:
        rows.append(dwm_lines["daily"])
    if "weekly" in dwm_lines:
        rows.append(dwm_lines["weekly"])
    if "monthly" in dwm_lines:
        rows.append(dwm_lines["monthly"])
    rows.append(trade_line)
    return "<br>".join(rows)
