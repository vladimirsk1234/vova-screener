"""
Sequence Vova logic: exact port of Pine "Sequence Vova Screener".
No UI or I/O dependencies; pure indicator math.
"""
import pandas as pd
import numpy as np


def calc_atr(df, length):
    h, l, c = df['High'], df['Low'], df['Close']
    tr = pd.concat([h-l, (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/length, adjust=False).mean()


def run_sequence_vova_pine(df, atr_len=14, min_rr=1.5, use_last_hl_sl=True, risk_dollars=100):
    """
    Exact port of Pine "Sequence Vova Screener". Returns dict for last bar:
    TP, SL, RR, Valid, New, Strong, position_size, position_value, last_peak, seq_low_prev (for Strong).
    """
    n = len(df)
    if n < 2:
        return None
    atr = calc_atr(df, atr_len)
    c_a = df['Close'].values
    h_a = df['High'].values
    l_a = df['Low'].values
    atr_a = atr.values

    seq_state = 0
    critical_level = np.nan
    seq_high, seq_low = h_a[0], l_a[0]
    last_confirmed_peak = np.nan
    last_confirmed_trough = np.nan
    last_peak_was_hh = False
    last_trough_was_hl = False

    # Store outputs for last bar
    last_crit = np.nan
    last_peak = np.nan
    last_valid = False
    last_new = False
    last_strong = False
    last_sl = np.nan
    last_rr = 0.0
    last_pos_size = np.nan
    last_pos_value = np.nan
    prev_bar_seq_low = l_a[0]  # seq_low of previous bar (for Strong on current bar)

    for i in range(1, n):
        c, h, l = c_a[i], h_a[i], l_a[i]
        cur_atr = atr_a[i]
        prev_state = seq_state
        prev_crit = critical_level
        prev_seq_high = seq_high
        prev_seq_low = seq_low

        is_break = False
        is_bearish_break = False
        if prev_state == 1 and not np.isnan(prev_crit):
            is_break = c < prev_crit
        elif prev_state == -1 and not np.isnan(prev_crit):
            is_break = c > prev_crit
            is_bearish_break = is_break  # bullish break (downtrend broken)

        if is_break:
            if prev_state == 1:
                if h >= seq_high:
                    seq_high = h
                is_current_peak_hh = (np.isnan(last_confirmed_peak) or seq_high > last_confirmed_peak)
                last_peak_was_hh = is_current_peak_hh
                last_confirmed_peak = seq_high
                seq_state = -1
                seq_high, seq_low = h, l
                critical_level = h
            else:
                if l <= seq_low:
                    seq_low = l
                is_current_trough_hl = (np.isnan(last_confirmed_trough) or
                    (seq_low > last_confirmed_trough) or (seq_low == last_confirmed_trough))
                last_trough_was_hl = is_current_trough_hl
                last_confirmed_trough = seq_low
                seq_state = 1
                seq_high, seq_low = h, l
                critical_level = l
        else:
            seq_state = prev_state
            if seq_state == 1:
                if h >= seq_high:
                    seq_high = h
                if h >= prev_seq_high:
                    critical_level = l
                else:
                    critical_level = prev_crit
            elif seq_state == -1:
                if l <= seq_low:
                    seq_low = l
                if l <= prev_seq_low:
                    critical_level = h
                else:
                    critical_level = prev_crit
            else:
                if c > prev_seq_high:
                    seq_state = 1
                    critical_level = l
                elif c < prev_seq_low:
                    seq_state = -1
                    critical_level = h
                else:
                    seq_high = max(prev_seq_high, h)
                    seq_low = min(prev_seq_low, l)

        struct_invalid_seq_down = (seq_state == -1 and last_trough_was_hl and
            not np.isnan(last_confirmed_trough) and seq_low < last_confirmed_trough)
        struct_ok = (last_trough_was_hl or (not np.isnan(last_confirmed_peak) and c > last_confirmed_peak and last_trough_was_hl)) and (not struct_invalid_seq_down)

        sl = c - cur_atr
        if not np.isnan(critical_level) and critical_level < c:
            sl = min(sl, critical_level)
        if use_last_hl_sl and last_trough_was_hl and not np.isnan(last_confirmed_trough) and last_confirmed_trough < c:
            sl = min(sl, last_confirmed_trough)
        risk = c - sl
        reward = last_confirmed_peak - c if not np.isnan(last_confirmed_peak) else 0.0
        rr = (reward / risk) if risk > 0 else 0.0
        position_size = (risk_dollars / risk) if (risk > 0 and risk_dollars > 0) else np.nan
        position_value = position_size * c if not np.isnan(position_size) else np.nan

        valid_signal = (seq_state == 1) and struct_ok and (rr >= min_rr) and (risk > 0) and (reward > 0)
        new_signal = valid_signal and is_bearish_break
        strong_signal = new_signal and (not np.isnan(prev_bar_seq_low)) and (l <= prev_bar_seq_low)

        prev_bar_seq_low = seq_low
        last_crit = critical_level
        last_peak = last_confirmed_peak
        last_valid = valid_signal
        last_new = new_signal
        last_strong = strong_signal
        last_sl = sl
        last_rr = rr
        last_pos_size = position_size
        last_pos_value = position_value

    close_last = c_a[-1]
    atr_last = atr_a[-1]
    return {
        "TP": last_peak,
        "SL": last_sl,
        "RR": last_rr,
        "Valid": last_valid,
        "New": last_new,
        "Strong": last_strong,
        "position_size": last_pos_size,
        "position_value": last_pos_value,
        "Close": close_last,
        "ATR": atr_last,
    }


def run_sequence_vova_full(df, atr_len=14, min_rr=1.5, use_last_hl_sl=True, risk_dollars=100):
    """
    Full-history version of run_sequence_vova_pine: returns per-bar arrays + confirmed
    peak/trough events + final Fibonacci anchors, in addition to the same last-bar
    summary fields. Used by the Plotly preview chart.
    """
    n = len(df)
    if n < 2:
        return None
    atr = calc_atr(df, atr_len)
    c_a = df['Close'].values
    h_a = df['High'].values
    l_a = df['Low'].values
    atr_a = atr.values

    seq_state = 0
    critical_level = np.nan
    seq_high, seq_low = h_a[0], l_a[0]
    seq_high_idx, seq_low_idx = 0, 0

    last_confirmed_peak = np.nan
    last_confirmed_peak_idx = -1
    last_confirmed_trough = np.nan
    last_confirmed_trough_idx = -1
    prev_trough_before_peak = np.nan
    prev_trough_before_peak_idx = -1
    last_peak_was_hh = False
    last_trough_was_hl = False
    last_lh = np.nan

    critical_level_series = np.full(n, np.nan, dtype=float)
    seq_state_series = np.zeros(n, dtype=int)

    peaks: list[dict] = []
    troughs: list[dict] = []

    last_crit = np.nan
    last_peak_val = np.nan
    last_valid = False
    last_new = False
    last_strong = False
    last_sl = np.nan
    last_rr = 0.0
    last_pos_size = np.nan
    last_pos_value = np.nan
    prev_bar_seq_low = l_a[0]

    for i in range(1, n):
        c, h, l = c_a[i], h_a[i], l_a[i]
        cur_atr = atr_a[i]
        prev_state = seq_state
        prev_crit = critical_level
        prev_seq_high = seq_high
        prev_seq_low = seq_low

        is_break = False
        is_bearish_break = False
        if prev_state == 1 and not np.isnan(prev_crit):
            is_break = c < prev_crit
        elif prev_state == -1 and not np.isnan(prev_crit):
            is_break = c > prev_crit
            is_bearish_break = is_break

        if is_break:
            if prev_state == 1:
                if h >= seq_high:
                    seq_high = h
                    seq_high_idx = i
                if np.isnan(last_confirmed_peak):
                    label = "HH"
                    is_current_peak_hh = True
                elif seq_high > last_confirmed_peak:
                    label = "HH"
                    is_current_peak_hh = True
                elif seq_high < last_confirmed_peak:
                    label = "LH"
                    is_current_peak_hh = False
                else:
                    label = "DT"
                    is_current_peak_hh = False
                last_peak_was_hh = is_current_peak_hh
                if not is_current_peak_hh:
                    last_lh = seq_high
                prev_trough_before_peak = last_confirmed_trough
                prev_trough_before_peak_idx = last_confirmed_trough_idx
                peaks.append({"idx": seq_high_idx, "price": seq_high, "label": label})

                last_confirmed_peak = seq_high
                last_confirmed_peak_idx = seq_high_idx
                seq_state = -1
                seq_high, seq_low = h, l
                seq_high_idx, seq_low_idx = i, i
                critical_level = h
            else:
                if l <= seq_low:
                    seq_low = l
                    seq_low_idx = i
                if np.isnan(last_confirmed_trough):
                    label = "LL"
                    is_current_trough_hl = True
                elif seq_low < last_confirmed_trough:
                    label = "LL"
                    is_current_trough_hl = False
                elif seq_low > last_confirmed_trough:
                    label = "HL"
                    is_current_trough_hl = True
                else:
                    label = "DB"
                    is_current_trough_hl = True
                last_trough_was_hl = is_current_trough_hl
                troughs.append({"idx": seq_low_idx, "price": seq_low, "label": label})

                last_confirmed_trough = seq_low
                last_confirmed_trough_idx = seq_low_idx
                seq_state = 1
                seq_high, seq_low = h, l
                seq_high_idx, seq_low_idx = i, i
                critical_level = l
        else:
            seq_state = prev_state
            if seq_state == 1:
                if h >= seq_high:
                    seq_high = h
                    seq_high_idx = i
                if h >= prev_seq_high:
                    critical_level = l
                else:
                    critical_level = prev_crit
            elif seq_state == -1:
                if l <= seq_low:
                    seq_low = l
                    seq_low_idx = i
                if l <= prev_seq_low:
                    critical_level = h
                else:
                    critical_level = prev_crit
            else:
                if c > prev_seq_high:
                    seq_state = 1
                    critical_level = l
                elif c < prev_seq_low:
                    seq_state = -1
                    critical_level = h
                else:
                    seq_high = max(prev_seq_high, h)
                    seq_low = min(prev_seq_low, l)

        critical_level_series[i] = critical_level
        seq_state_series[i] = seq_state

        struct_invalid_seq_down = (seq_state == -1 and last_trough_was_hl and
            not np.isnan(last_confirmed_trough) and seq_low < last_confirmed_trough)
        struct_ok = (last_trough_was_hl or (not np.isnan(last_confirmed_peak) and c > last_confirmed_peak and last_trough_was_hl)) and (not struct_invalid_seq_down)

        sl = c - cur_atr
        if not np.isnan(critical_level) and critical_level < c:
            sl = min(sl, critical_level)
        if use_last_hl_sl and last_trough_was_hl and not np.isnan(last_confirmed_trough) and last_confirmed_trough < c:
            sl = min(sl, last_confirmed_trough)
        risk = c - sl
        reward = last_confirmed_peak - c if not np.isnan(last_confirmed_peak) else 0.0
        rr = (reward / risk) if risk > 0 else 0.0
        position_size = (risk_dollars / risk) if (risk > 0 and risk_dollars > 0) else np.nan
        position_value = position_size * c if not np.isnan(position_size) else np.nan

        valid_signal = (seq_state == 1) and struct_ok and (rr >= min_rr) and (risk > 0) and (reward > 0)
        new_signal = valid_signal and is_bearish_break
        strong_signal = new_signal and (not np.isnan(prev_bar_seq_low)) and (l <= prev_bar_seq_low)

        prev_bar_seq_low = seq_low
        last_crit = critical_level
        last_peak_val = last_confirmed_peak
        last_valid = valid_signal
        last_new = new_signal
        last_strong = strong_signal
        last_sl = sl
        last_rr = rr
        last_pos_size = position_size
        last_pos_value = position_value

    # Fibonacci anchors: drawn when uptrend + valid struct (mirrors Pine)
    fib_levels: dict | None = None
    if (seq_state == 1
            and not np.isnan(last_confirmed_peak)
            and not np.isnan(prev_trough_before_peak)
            and last_trough_was_hl):
        fib_range = last_confirmed_peak - prev_trough_before_peak
        fib_levels = {
            "high": float(last_confirmed_peak),
            "high_idx": int(last_confirmed_peak_idx),
            "low": float(prev_trough_before_peak),
            "low_idx": int(prev_trough_before_peak_idx),
            "fib_382": float(last_confirmed_peak - fib_range * 0.382),
            "fib_500": float(last_confirmed_peak - fib_range * 0.500),
            "fib_618": float(last_confirmed_peak - fib_range * 0.618),
        }

    close_last = c_a[-1]
    atr_last = atr_a[-1]
    return {
        # Summary (same shape as run_sequence_vova_pine)
        "TP": last_peak_val,
        "SL": last_sl,
        "RR": last_rr,
        "Valid": last_valid,
        "New": last_new,
        "Strong": last_strong,
        "position_size": last_pos_size,
        "position_value": last_pos_value,
        "Close": close_last,
        "ATR": atr_last,
        # Per-bar arrays
        "critical_level_series": critical_level_series,
        "seq_state_series": seq_state_series,
        # Event lists (already in display order)
        "peaks": peaks,
        "troughs": troughs,
        # Final structural state
        "seq_state_final": int(seq_state),
        "last_peak": (float(last_confirmed_peak) if not np.isnan(last_confirmed_peak) else None),
        "last_peak_idx": int(last_confirmed_peak_idx) if last_confirmed_peak_idx >= 0 else None,
        "last_trough": (float(last_confirmed_trough) if not np.isnan(last_confirmed_trough) else None),
        "last_trough_idx": int(last_confirmed_trough_idx) if last_confirmed_trough_idx >= 0 else None,
        "last_peak_was_hh": bool(last_peak_was_hh),
        "last_trough_was_hl": bool(last_trough_was_hl),
        "last_lh": (float(last_lh) if not np.isnan(last_lh) else None),
        "critical_level": (float(last_crit) if not np.isnan(last_crit) else None),
        "fib": fib_levels,
    }
