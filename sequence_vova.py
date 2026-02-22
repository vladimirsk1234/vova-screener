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
