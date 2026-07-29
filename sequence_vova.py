"""
Sequence Vova logic: exact port of Pine "Sequence Vova Indicator" / Screener.
No UI or I/O dependencies; pure indicator math.
"""
from __future__ import annotations

import os
from typing import Literal

import pandas as pd
import numpy as np

ScanDirection = Literal["buy", "sell"]

from indicator_params import IndicatorParams

try:
    from numba import njit

    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):  # type: ignore[misc]
        def _wrap(fn):
            return fn

        if args and callable(args[0]):
            return args[0]
        return _wrap


def _is_cloud_runtime() -> bool:
    try:
        from scan_memory import is_streamlit_cloud

        return bool(is_streamlit_cloud())
    except Exception:
        return bool(
            os.environ.get("STREAMLIT_SERVER_PORT")
            or os.path.isdir("/mount/src")
            or os.environ.get("HOME", "").startswith("/home/appuser")
        )


_PINE_USE_NUMBA = (
    os.environ.get("PINE_USE_NUMBA", "1") != "0"
    and _NUMBA_AVAILABLE
    and not _is_cloud_runtime()  # Cloud: avoid native JIT crashes
)


def calc_atr(df: pd.DataFrame, length: int) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False).mean()


def _ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def _sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(length, min_periods=1).mean()


def calc_macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def calc_dmi(df: pd.DataFrame, length: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    """ADX system: +DI, -DI, ADX (Wilder-style EWM)."""
    h, l, c = df["High"], df["Low"], df["Close"]
    up = h.diff()
    down = -l.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / length, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / length, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / length, adjust=False).mean() / atr
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di)).replace([np.inf, -np.inf], np.nan)
    adx = dx.ewm(alpha=1 / length, adjust=False).mean()
    return plus_di, minus_di, adx


def compute_elder_envelope(
    close: pd.Series,
    len_slow: int,
    lookback: int,
    multiplier: float,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_slow = _ema(close, len_slow)
    myvar = (close - ema_slow).abs()
    myvars = myvar * myvar
    mymov = np.sqrt(_sma(myvars, lookback))
    # Pine: max of current + 5 prior (nz)
    newmax = mymov.copy()
    for lag in range(1, 6):
        newmax = np.maximum(newmax, mymov.shift(lag).fillna(0))
    env_upper = ema_slow + newmax * multiplier
    env_lower = ema_slow - newmax * multiplier
    return ema_slow, env_upper, env_lower


def compute_bollinger(
    close: pd.Series,
    length: int,
    mult: float,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    basis = _sma(close, length)
    std = close.rolling(length, min_periods=1).std()
    upper = basis + mult * std
    lower = basis - mult * std
    return basis, upper, lower


def compute_impulse_colors(
    close: pd.Series,
    len_fast: int,
    bull_color: str,
    bear_color: str,
    neut_color: str,
) -> np.ndarray:
    ema_fast = _ema(close, len_fast)
    _, _, macd_hist = calc_macd(close)
    ef = ema_fast.values
    mh = macd_hist.values
    colors = np.empty(len(close), dtype=object)
    for i in range(1, len(close)):
        bulls = ef[i] > ef[i - 1] and mh[i] > mh[i - 1]
        bears = ef[i] < ef[i - 1] and mh[i] < mh[i - 1]
        if bulls:
            colors[i] = bull_color
        elif bears:
            colors[i] = bear_color
        else:
            colors[i] = neut_color
    colors[0] = neut_color
    return colors


def compute_overlays(
    df: pd.DataFrame,
    *,
    len_fast: int = 20,
    len_slow: int = 40,
    length_major: int = 200,
    lookback: int = 100,
    multiplier: float = 2.0,
    bb_length: int = 20,
    bb_mult: float = 2.0,
    elder_bull_color: str = "#00c853",
    elder_bear_color: str = "#ff1744",
    elder_neut_color: str = "#4eadfc",
) -> dict:
    close = df["Close"]
    ema_fast = _ema(close, len_fast)
    ema_slow, env_upper, env_lower = compute_elder_envelope(close, len_slow, lookback, multiplier)
    sma_major = _sma(close, length_major)
    bb_basis, bb_upper, bb_lower = compute_bollinger(close, bb_length, bb_mult)
    impulse_colors = compute_impulse_colors(
        close, len_fast, elder_bull_color, elder_bear_color, elder_neut_color
    )
    return {
        "ema_fast": ema_fast,
        "ema_slow": ema_slow,
        "sma_major": sma_major,
        "env_upper": env_upper,
        "env_lower": env_lower,
        "bb_basis": bb_basis,
        "bb_upper": bb_upper,
        "bb_lower": bb_lower,
        "impulse_colors": impulse_colors,
    }


def _calc_atr_numpy(h: np.ndarray, l: np.ndarray, c: np.ndarray, length: int) -> np.ndarray:
    """Wilder ATR matching pandas ewm(alpha=1/length, adjust=False)."""
    n = len(c)
    tr = np.empty(n, dtype=np.float64)
    tr[0] = h[0] - l[0]
    for i in range(1, n):
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
    atr = np.empty(n, dtype=np.float64)
    atr[0] = tr[0]
    alpha = 1.0 / length
    for i in range(1, n):
        atr[i] = alpha * tr[i] + (1.0 - alpha) * atr[i - 1]
    return atr


def _run_sequence_vova_pine_python(
    c_a: np.ndarray,
    h_a: np.ndarray,
    l_a: np.ndarray,
    atr_a: np.ndarray,
    min_rr: float,
    use_last_hl_sl: bool,
    risk_dollars: float,
    direction_sell: bool,
    no_rr_req: bool = False,
) -> tuple:
    n = len(c_a)
    seq_state = 0
    critical_level = np.nan
    seq_high, seq_low = h_a[0], l_a[0]
    last_confirmed_peak = np.nan
    last_confirmed_trough = np.nan
    last_peak_was_hh = False
    last_trough_was_hl = False

    last_peak = np.nan
    last_valid = False
    last_new = False
    last_strong = False
    last_sl = np.nan
    last_rr = np.nan
    last_pos_size = np.nan
    last_pos_value = np.nan
    prev_bar_seq_low = l_a[0]
    prev_bar_seq_high = h_a[0]

    for i in range(1, n):
        c, h, l = c_a[i], h_a[i], l_a[i]
        cur_atr = atr_a[i]
        prev_state = seq_state
        prev_crit = critical_level
        prev_seq_high = seq_high
        prev_seq_low = seq_low

        is_break = False
        is_bullish_break = False
        is_bearish_break = False
        if prev_state == 1 and not np.isnan(prev_crit):
            is_break = c < prev_crit
            is_bullish_break = is_break
        elif prev_state == -1 and not np.isnan(prev_crit):
            is_break = c > prev_crit
            is_bearish_break = is_break

        if is_break:
            if prev_state == 1:
                if h >= seq_high:
                    seq_high = h
                is_current_peak_hh = np.isnan(last_confirmed_peak) or seq_high > last_confirmed_peak
                last_peak_was_hh = is_current_peak_hh
                last_confirmed_peak = seq_high
                seq_state = -1
                seq_high, seq_low = h, l
                critical_level = h
            else:
                if l <= seq_low:
                    seq_low = l
                is_current_trough_hl = (
                    np.isnan(last_confirmed_trough)
                    or (seq_low > last_confirmed_trough)
                    or (seq_low == last_confirmed_trough)
                )
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

        if direction_sell:
            struct_invalid_seq_up = (
                seq_state == 1
                and last_peak_was_hh
                and not np.isnan(last_confirmed_peak)
                and seq_high > last_confirmed_peak
            )
            last_peak_was_lh = not last_peak_was_hh
            struct_ok_sell = (
                last_peak_was_lh
                or (
                    not np.isnan(last_confirmed_trough)
                    and c < last_confirmed_trough
                    and last_peak_was_lh
                )
            ) and (not struct_invalid_seq_up)

            sl = c + cur_atr
            if not np.isnan(critical_level) and critical_level > c:
                sl = max(sl, critical_level)
            if (
                use_last_hl_sl
                and last_peak_was_lh
                and not np.isnan(last_confirmed_peak)
                and last_confirmed_peak > c
            ):
                sl = max(sl, last_confirmed_peak)
            risk = sl - c
            reward = c - last_confirmed_trough if not np.isnan(last_confirmed_trough) else 0.0
            rr = (reward / risk) if risk > 0 else np.nan
            position_size = (risk_dollars / risk) if (risk > 0 and risk_dollars > 0) else np.nan
            position_value = position_size * c if not np.isnan(position_size) else np.nan

            if no_rr_req:
                valid_signal = (seq_state == -1) and struct_ok_sell
            else:
                valid_signal = (
                    (seq_state == -1)
                    and struct_ok_sell
                    and (rr >= min_rr)
                    and (risk > 0)
                    and (reward > 0)
                )
            new_signal = valid_signal and is_bullish_break
            strong_signal = (
                new_signal and (not np.isnan(prev_bar_seq_high)) and (h >= prev_bar_seq_high)
            )
            last_peak = last_confirmed_trough
        else:
            struct_invalid_seq_down = (
                seq_state == -1
                and last_trough_was_hl
                and not np.isnan(last_confirmed_trough)
                and seq_low < last_confirmed_trough
            )
            struct_ok = (
                last_trough_was_hl
                or (
                    not np.isnan(last_confirmed_peak)
                    and c > last_confirmed_peak
                    and last_trough_was_hl
                )
            ) and last_peak_was_hh and (not struct_invalid_seq_down)

            sl = c - cur_atr
            if not np.isnan(critical_level) and critical_level < c:
                sl = min(sl, critical_level)
            if (
                use_last_hl_sl
                and last_trough_was_hl
                and not np.isnan(last_confirmed_trough)
                and last_confirmed_trough < c
            ):
                sl = min(sl, last_confirmed_trough)
            risk = c - sl
            reward = last_confirmed_peak - c if not np.isnan(last_confirmed_peak) else 0.0
            rr = (reward / risk) if risk > 0 else np.nan
            position_size = (risk_dollars / risk) if (risk > 0 and risk_dollars > 0) else np.nan
            position_value = position_size * c if not np.isnan(position_size) else np.nan

            if no_rr_req:
                valid_signal = (seq_state == 1) and struct_ok
            else:
                valid_signal = (
                    (seq_state == 1) and struct_ok and (rr >= min_rr) and (risk > 0) and (reward > 0)
                )
            new_signal = valid_signal and is_bearish_break
            strong_signal = (
                new_signal and (not np.isnan(prev_bar_seq_low)) and (l <= prev_bar_seq_low)
            )
            last_peak = last_confirmed_peak

        prev_bar_seq_low = seq_low
        prev_bar_seq_high = seq_high
        last_valid = valid_signal
        last_new = new_signal
        last_strong = strong_signal
        last_sl = sl
        last_rr = rr
        last_pos_size = position_size
        last_pos_value = position_value

    return (
        last_peak,
        last_sl,
        last_rr,
        last_valid,
        last_new,
        last_strong,
        last_pos_size,
        last_pos_value,
        c_a[-1],
        atr_a[-1],
    )


if _NUMBA_AVAILABLE:

    @njit(cache=True)
    def _run_sequence_vova_pine_numba(
        c_a,
        h_a,
        l_a,
        atr_a,
        min_rr,
        use_last_hl_sl,
        risk_dollars,
        direction_sell,
        no_rr_req,
    ):
        n = len(c_a)
        seq_state = 0
        critical_level = np.nan
        seq_high = h_a[0]
        seq_low = l_a[0]
        last_confirmed_peak = np.nan
        last_confirmed_trough = np.nan
        last_peak_was_hh = False
        last_trough_was_hl = False

        last_peak = np.nan
        last_valid = False
        last_new = False
        last_strong = False
        last_sl = np.nan
        last_rr = np.nan
        last_pos_size = np.nan
        last_pos_value = np.nan
        prev_bar_seq_low = l_a[0]
        prev_bar_seq_high = h_a[0]

        for i in range(1, n):
            c = c_a[i]
            h = h_a[i]
            l = l_a[i]
            cur_atr = atr_a[i]
            prev_state = seq_state
            prev_crit = critical_level
            prev_seq_high = seq_high
            prev_seq_low = seq_low

            is_break = False
            is_bullish_break = False
            is_bearish_break = False
            if prev_state == 1 and not np.isnan(prev_crit):
                is_break = c < prev_crit
                is_bullish_break = is_break
            elif prev_state == -1 and not np.isnan(prev_crit):
                is_break = c > prev_crit
                is_bearish_break = is_break

            if is_break:
                if prev_state == 1:
                    if h >= seq_high:
                        seq_high = h
                    if np.isnan(last_confirmed_peak) or seq_high > last_confirmed_peak:
                        last_peak_was_hh = True
                    else:
                        last_peak_was_hh = False
                    last_confirmed_peak = seq_high
                    seq_state = -1
                    seq_high = h
                    seq_low = l
                    critical_level = h
                else:
                    if l <= seq_low:
                        seq_low = l
                    if (
                        np.isnan(last_confirmed_trough)
                        or seq_low > last_confirmed_trough
                        or seq_low == last_confirmed_trough
                    ):
                        last_trough_was_hl = True
                    else:
                        last_trough_was_hl = False
                    last_confirmed_trough = seq_low
                    seq_state = 1
                    seq_high = h
                    seq_low = l
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
                        if prev_seq_high > h:
                            seq_high = prev_seq_high
                        else:
                            seq_high = h
                        if prev_seq_low < l:
                            seq_low = prev_seq_low
                        else:
                            seq_low = l

            if direction_sell:
                struct_invalid_seq_up = (
                    seq_state == 1
                    and last_peak_was_hh
                    and not np.isnan(last_confirmed_peak)
                    and seq_high > last_confirmed_peak
                )
                last_peak_was_lh = not last_peak_was_hh
                struct_ok_sell = (
                    last_peak_was_lh
                    or (
                        not np.isnan(last_confirmed_trough)
                        and c < last_confirmed_trough
                        and last_peak_was_lh
                    )
                ) and (not struct_invalid_seq_up)

                sl = c + cur_atr
                if not np.isnan(critical_level) and critical_level > c:
                    sl = max(sl, critical_level)
                if (
                    use_last_hl_sl
                    and last_peak_was_lh
                    and not np.isnan(last_confirmed_peak)
                    and last_confirmed_peak > c
                ):
                    sl = max(sl, last_confirmed_peak)
                risk = sl - c
                if not np.isnan(last_confirmed_trough):
                    reward = c - last_confirmed_trough
                else:
                    reward = 0.0
                if risk > 0:
                    rr = reward / risk
                else:
                    rr = np.nan
                if risk > 0 and risk_dollars > 0:
                    position_size = risk_dollars / risk
                else:
                    position_size = np.nan
                if not np.isnan(position_size):
                    position_value = position_size * c
                else:
                    position_value = np.nan

                if no_rr_req:
                    valid_signal = seq_state == -1 and struct_ok_sell
                else:
                    valid_signal = (
                        seq_state == -1
                        and struct_ok_sell
                        and rr >= min_rr
                        and risk > 0
                        and reward > 0
                    )
                new_signal = valid_signal and is_bullish_break
                strong_signal = (
                    new_signal and (not np.isnan(prev_bar_seq_high)) and h >= prev_bar_seq_high
                )
                last_peak = last_confirmed_trough
            else:
                struct_invalid_seq_down = (
                    seq_state == -1
                    and last_trough_was_hl
                    and not np.isnan(last_confirmed_trough)
                    and seq_low < last_confirmed_trough
                )
                struct_ok = (
                    last_trough_was_hl
                    or (
                        not np.isnan(last_confirmed_peak)
                        and c > last_confirmed_peak
                        and last_trough_was_hl
                    )
                ) and last_peak_was_hh and (not struct_invalid_seq_down)

                sl = c - cur_atr
                if not np.isnan(critical_level) and critical_level < c:
                    sl = min(sl, critical_level)
                if (
                    use_last_hl_sl
                    and last_trough_was_hl
                    and not np.isnan(last_confirmed_trough)
                    and last_confirmed_trough < c
                ):
                    sl = min(sl, last_confirmed_trough)
                risk = c - sl
                if not np.isnan(last_confirmed_peak):
                    reward = last_confirmed_peak - c
                else:
                    reward = 0.0
                if risk > 0:
                    rr = reward / risk
                else:
                    rr = np.nan
                if risk > 0 and risk_dollars > 0:
                    position_size = risk_dollars / risk
                else:
                    position_size = np.nan
                if not np.isnan(position_size):
                    position_value = position_size * c
                else:
                    position_value = np.nan

                if no_rr_req:
                    valid_signal = seq_state == 1 and struct_ok
                else:
                    valid_signal = (
                        seq_state == 1 and struct_ok and rr >= min_rr and risk > 0 and reward > 0
                    )
                new_signal = valid_signal and is_bearish_break
                strong_signal = (
                    new_signal and (not np.isnan(prev_bar_seq_low)) and l <= prev_bar_seq_low
                )
                last_peak = last_confirmed_peak

            prev_bar_seq_low = seq_low
            prev_bar_seq_high = seq_high
            last_valid = valid_signal
            last_new = new_signal
            last_strong = strong_signal
            last_sl = sl
            last_rr = rr
            last_pos_size = position_size
            last_pos_value = position_value

        return (
            last_peak,
            last_sl,
            last_rr,
            last_valid,
            last_new,
            last_strong,
            last_pos_size,
            last_pos_value,
            c_a[-1],
            atr_a[-1],
        )

else:
    _run_sequence_vova_pine_numba = None  # type: ignore[misc, assignment]


def _pine_result_dict(tup: tuple) -> dict:
    return {
        "TP": tup[0],
        "SL": tup[1],
        "RR": tup[2],
        "Valid": tup[3],
        "New": tup[4],
        "Strong": tup[5],
        "position_size": tup[6],
        "position_value": tup[7],
        "Close": tup[8],
        "ATR": tup[9],
    }


def run_sequence_vova_pine(
    df,
    atr_len=14,
    min_rr=1.5,
    use_last_hl_sl=True,
    risk_dollars=100,
    direction: ScanDirection = "buy",
    no_rr_req: bool = False,
):
    """
    Exact port of Pine "Sequence Vova Screener". Returns dict for last bar:
    TP, SL, RR, Valid, New, Strong, position_size, position_value, last_peak, seq_low_prev (for Strong).
    direction: "buy" (long, default) or "sell" (short, mirror logic).
    """
    n = len(df)
    if n < 2:
        return None
    c_a = np.ascontiguousarray(df["Close"].values, dtype=np.float64)
    h_a = np.ascontiguousarray(df["High"].values, dtype=np.float64)
    l_a = np.ascontiguousarray(df["Low"].values, dtype=np.float64)
    atr_a = _calc_atr_numpy(h_a, l_a, c_a, atr_len)

    use_last = bool(use_last_hl_sl)
    min_rr_f = float(min_rr)
    risk_f = float(risk_dollars)
    direction_sell = str(direction).lower() == "sell"
    no_rr = bool(no_rr_req)

    if _PINE_USE_NUMBA and _NUMBA_AVAILABLE and _run_sequence_vova_pine_numba is not None:
        tup = _run_sequence_vova_pine_numba(
            c_a, h_a, l_a, atr_a, min_rr_f, use_last, risk_f, direction_sell, no_rr
        )
    else:
        tup = _run_sequence_vova_pine_python(
            c_a, h_a, l_a, atr_a, min_rr_f, use_last, risk_f, direction_sell, no_rr
        )
    return _pine_result_dict(tup)


def _run_sequence_vova_close_python(
    c_a: np.ndarray,
    h_a: np.ndarray,
    l_a: np.ndarray,
    atr_a: np.ndarray,
    min_rr: float,
    use_last_hl_sl: bool,
    risk_dollars: float,
    no_rr_req: bool = False,
) -> dict:
    """
    Simulate long open (BUY new with min_rr at entry) and close on break SEQ down.
    Opens only when flat (not position_open). Returns close on the last bar only.
    """
    n = len(c_a)
    seq_state = 0
    critical_level = np.nan
    seq_high, seq_low = h_a[0], l_a[0]
    last_confirmed_peak = np.nan
    last_confirmed_trough = np.nan
    last_peak_was_hh = False
    last_trough_was_hl = False

    position_open = False
    entry_price = np.nan
    entry_sl = np.nan
    entry_rr_at_open = np.nan
    position_size = np.nan

    last_valid = False
    last_new = False
    last_entry_price = np.nan
    last_exit_price = np.nan
    last_entry_sl = np.nan
    last_position_size = np.nan
    last_pnl_dollars = np.nan
    last_pnl_pct = np.nan
    last_entry_rr = np.nan
    last_close_rr = np.nan

    for i in range(1, n):
        c, h, l = c_a[i], h_a[i], l_a[i]
        cur_atr = atr_a[i]
        prev_state = seq_state
        prev_crit = critical_level
        prev_seq_high = seq_high
        prev_seq_low = seq_low

        is_bullish_break = False
        is_bearish_break = False
        if prev_state == 1 and not np.isnan(prev_crit):
            is_bullish_break = c < prev_crit
        elif prev_state == -1 and not np.isnan(prev_crit):
            is_bearish_break = c > prev_crit

        if is_bullish_break or is_bearish_break:
            if prev_state == 1:
                if h >= seq_high:
                    seq_high = h
                is_current_peak_hh = np.isnan(last_confirmed_peak) or seq_high > last_confirmed_peak
                last_peak_was_hh = is_current_peak_hh
                last_confirmed_peak = seq_high
                seq_state = -1
                seq_high, seq_low = h, l
                critical_level = h
            else:
                if l <= seq_low:
                    seq_low = l
                is_current_trough_hl = (
                    np.isnan(last_confirmed_trough)
                    or (seq_low > last_confirmed_trough)
                    or (seq_low == last_confirmed_trough)
                )
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

        struct_invalid_seq_down = (
            seq_state == -1
            and last_trough_was_hl
            and not np.isnan(last_confirmed_trough)
            and seq_low < last_confirmed_trough
        )
        struct_ok = (
            last_trough_was_hl
            or (
                not np.isnan(last_confirmed_peak)
                and c > last_confirmed_peak
                and last_trough_was_hl
            )
        ) and last_peak_was_hh and (not struct_invalid_seq_down)

        sl = c - cur_atr
        if not np.isnan(critical_level) and critical_level < c:
            sl = min(sl, critical_level)
        if (
            use_last_hl_sl
            and last_trough_was_hl
            and not np.isnan(last_confirmed_trough)
            and last_confirmed_trough < c
        ):
            sl = min(sl, last_confirmed_trough)
        risk = c - sl
        reward = last_confirmed_peak - c if not np.isnan(last_confirmed_peak) else 0.0
        rr = (reward / risk) if risk > 0 else np.nan

        if no_rr_req:
            valid_signal = (seq_state == 1) and struct_ok
        else:
            valid_signal = (
                (seq_state == 1) and struct_ok and (rr >= min_rr) and (risk > 0) and (reward > 0)
            )
        new_signal = valid_signal and is_bearish_break

        if new_signal and not position_open:
            position_open = True
            entry_price = c
            entry_sl = sl
            entry_rr_at_open = rr
            position_size = (
                (risk_dollars / risk) if (risk > 0 and risk_dollars > 0) else np.nan
            )

        if position_open and is_bullish_break:
            exit_price = c
            psz = position_size
            entry_risk = entry_price - entry_sl
            close_rr = (
                (exit_price - entry_price) / entry_risk
                if entry_risk > 0 and not np.isnan(entry_price)
                else np.nan
            )
            if not np.isnan(psz) and not np.isnan(entry_price):
                pnl = (exit_price - entry_price) * psz
                pnl_pct = (
                    (exit_price - entry_price) / entry_price * 100.0
                    if entry_price > 0
                    else np.nan
                )
            else:
                pnl = np.nan
                pnl_pct = np.nan

            if i == n - 1:
                last_valid = True
                last_new = True
                last_entry_price = entry_price
                last_exit_price = exit_price
                last_entry_sl = entry_sl
                last_position_size = position_size
                last_pnl_dollars = pnl
                last_pnl_pct = pnl_pct
                last_entry_rr = entry_rr_at_open
                last_close_rr = close_rr

            position_open = False
            entry_price = np.nan
            entry_sl = np.nan
            entry_rr_at_open = np.nan
            position_size = np.nan

    return {
        "Valid": last_valid,
        "New": last_new,
        "entry_price": last_entry_price,
        "exit_price": last_exit_price,
        "entry_sl": last_entry_sl,
        "position_size": last_position_size,
        "pnl_dollars": last_pnl_dollars,
        "pnl_pct": last_pnl_pct,
        "entry_rr": last_entry_rr,
        "close_rr": last_close_rr,
        "Close": c_a[-1],
        "ATR": atr_a[-1],
    }


def run_sequence_vova_close_scan(
    df,
    atr_len: int = 14,
    min_rr: float = 1.5,
    use_last_hl_sl: bool = True,
    risk_dollars: float = 100,
    no_rr_req: bool = False,
) -> dict | None:
    """
    Close-scan for SELL TO CLOSE mode: find long positions opened on BUY new (min_rr at entry only),
    close on break SEQ down; return P&L when exit is on the last bar.
    """
    n = len(df)
    if n < 2:
        return None
    c_a = np.ascontiguousarray(df["Close"].values, dtype=np.float64)
    h_a = np.ascontiguousarray(df["High"].values, dtype=np.float64)
    l_a = np.ascontiguousarray(df["Low"].values, dtype=np.float64)
    atr_a = _calc_atr_numpy(h_a, l_a, c_a, atr_len)
    return _run_sequence_vova_close_python(
        c_a,
        h_a,
        l_a,
        atr_a,
        float(min_rr),
        bool(use_last_hl_sl),
        float(risk_dollars),
        bool(no_rr_req),
    )


def run_sequence_vova_full(
    df: pd.DataFrame,
    params: IndicatorParams | None = None,
    *,
    atr_len: int = 14,
    min_rr: float = 1.5,
    use_last_hl_sl: bool = True,
    risk_dollars: float = 100,
    no_rr_req: bool = False,
    len_fast: int = 20,
    len_slow: int = 40,
    length_major: int = 200,
    lookback: int = 100,
    multiplier: float = 2.0,
    bb_length: int = 20,
    bb_mult: float = 2.0,
    elder_bull_color: str = "#00c853",
    elder_bear_color: str = "#ff1744",
    elder_neut_color: str = "#4eadfc",
) -> dict | None:
    """
    Full-history version: per-bar arrays, peaks/troughs, extension lines, overlays, fib.
  Used by Plotly chart preview.
    """
    if params is not None:
        kw = params.to_runner_kwargs()
        atr_len = int(kw["atr_len"])
        min_rr = float(kw["min_rr"])
        use_last_hl_sl = bool(kw["use_last_hl_sl"])
        risk_dollars = float(kw["risk_dollars"])
        no_rr_req = bool(kw.get("no_rr_req", False))
        len_fast = int(kw["len_fast"])
        len_slow = int(kw["len_slow"])
        length_major = int(kw["length_major"])
        lookback = int(kw["lookback"])
        multiplier = float(kw["multiplier"])
        bb_length = params.bb_length
        bb_mult = params.bb_mult
        elder_bull_color = params.elder_bull_color
        elder_bear_color = params.elder_bear_color
        elder_neut_color = params.elder_neut_color

    n = len(df)
    if n < 2:
        return None
    atr = calc_atr(df, atr_len)
    c_a = df["Close"].values
    h_a = df["High"].values
    l_a = df["Low"].values
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
    bullish_break = np.zeros(n, dtype=bool)
    bearish_break = np.zeros(n, dtype=bool)

    peaks: list[dict] = []
    troughs: list[dict] = []
    line_high_ext: dict | None = None
    line_low_ext: dict | None = None

    last_crit = np.nan
    last_peak_val = np.nan
    last_valid = False
    last_new = False
    last_strong = False
    last_sl = np.nan
    last_rr = np.nan
    last_pos_size = np.nan
    last_pos_value = np.nan
    prev_bar_seq_low = l_a[0]
    signal_bar_index: int | None = None

    for i in range(1, n):
        c, h, l = c_a[i], h_a[i], l_a[i]
        cur_atr = atr_a[i]
        prev_state = seq_state
        prev_crit = critical_level
        prev_seq_high = seq_high
        prev_seq_low = seq_low

        is_break = False
        is_bullish_break = False
        is_bearish_break = False
        if prev_state == 1 and not np.isnan(prev_crit):
            is_break = c < prev_crit
            is_bullish_break = is_break
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

                if last_confirmed_peak_idx >= 0 and not np.isnan(last_confirmed_peak):
                    line_high_ext = {
                        "kind": "high",
                        "x0_idx": last_confirmed_peak_idx,
                        "y0": float(last_confirmed_peak),
                        "x1_idx": seq_high_idx,
                        "y1": float(seq_high),
                    }

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

                if last_confirmed_trough_idx >= 0 and not np.isnan(last_confirmed_trough):
                    line_low_ext = {
                        "kind": "low",
                        "x0_idx": last_confirmed_trough_idx,
                        "y0": float(last_confirmed_trough),
                        "x1_idx": seq_low_idx,
                        "y1": float(seq_low),
                    }

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
        bullish_break[i] = is_bullish_break
        bearish_break[i] = is_bearish_break

        struct_invalid_seq_down = (
            seq_state == -1
            and last_trough_was_hl
            and not np.isnan(last_confirmed_trough)
            and seq_low < last_confirmed_trough
        )
        struct_ok = (
            last_trough_was_hl
            or (
                not np.isnan(last_confirmed_peak)
                and c > last_confirmed_peak
                and last_trough_was_hl
            )
        ) and last_peak_was_hh and (not struct_invalid_seq_down)

        cur_cond_seq_ok = seq_state == 1
        if is_bearish_break and cur_cond_seq_ok and struct_ok:
            signal_bar_index = i
        elif is_bearish_break:
            signal_bar_index = None

        sl = c - cur_atr
        if not np.isnan(critical_level) and critical_level < c:
            sl = min(sl, critical_level)
        if (
            use_last_hl_sl
            and last_trough_was_hl
            and not np.isnan(last_confirmed_trough)
            and last_confirmed_trough < c
        ):
            sl = min(sl, last_confirmed_trough)
        risk = c - sl
        reward = last_confirmed_peak - c if not np.isnan(last_confirmed_peak) else 0.0
        rr = (reward / risk) if risk > 0 else np.nan
        position_size = (risk_dollars / risk) if (risk > 0 and risk_dollars > 0) else np.nan
        position_value = position_size * c if not np.isnan(position_size) else np.nan

        if no_rr_req:
            valid_signal = (seq_state == 1) and struct_ok
        else:
            valid_signal = (
                (seq_state == 1) and struct_ok and (rr >= min_rr) and (risk > 0) and (reward > 0)
            )
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

    fib_levels: dict | None = None
    daily_struct_valid_fib = last_trough_was_hl or (
        not np.isnan(last_confirmed_peak)
        and c_a[-1] > last_confirmed_peak
        and last_trough_was_hl
    )
    if (
        seq_state == 1
        and not np.isnan(last_confirmed_peak)
        and not np.isnan(prev_trough_before_peak)
        and daily_struct_valid_fib
    ):
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

    overlays = compute_overlays(
        df,
        len_fast=len_fast,
        len_slow=len_slow,
        length_major=length_major,
        lookback=lookback,
        multiplier=multiplier,
        bb_length=bb_length,
        bb_mult=bb_mult,
        elder_bull_color=elder_bull_color,
        elder_bear_color=elder_bear_color,
        elder_neut_color=elder_neut_color,
    )

    _, _, adx = calc_dmi(df, params.adx_len if params else 14)
    atr_pct = (atr / df["Close"]) * 100

    close_last = c_a[-1]
    atr_last = atr_a[-1]
    return {
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
        "ATR_pct": float(atr_pct.iloc[-1]) if len(atr_pct) else 0.0,
        "ADX": float(adx.iloc[-1]) if len(adx) else 0.0,
        "critical_level_series": critical_level_series,
        "seq_state_series": seq_state_series,
        "bullish_break": bullish_break,
        "bearish_break": bearish_break,
        "peaks": peaks,
        "troughs": troughs,
        "extension_lines": [ln for ln in (line_high_ext, line_low_ext) if ln is not None],
        "seq_state_final": int(seq_state),
        "seq_high_final": float(seq_high),
        "last_peak": float(last_confirmed_peak) if not np.isnan(last_confirmed_peak) else None,
        "last_peak_idx": int(last_confirmed_peak_idx) if last_confirmed_peak_idx >= 0 else None,
        "last_trough": float(last_confirmed_trough) if not np.isnan(last_confirmed_trough) else None,
        "last_trough_idx": int(last_confirmed_trough_idx) if last_confirmed_trough_idx >= 0 else None,
        "last_peak_was_hh": bool(last_peak_was_hh),
        "last_trough_was_hl": bool(last_trough_was_hl),
        "last_lh": float(last_lh) if not np.isnan(last_lh) else None,
        "critical_level": float(last_crit) if not np.isnan(last_crit) else None,
        "struct_invalid_seq_down": bool(
            seq_state == -1
            and last_trough_was_hl
            and not np.isnan(last_confirmed_trough)
            and seq_low < last_confirmed_trough
        ),
        "signal_bar_index": signal_bar_index,
        "fib": fib_levels,
        "overlays": {k: v for k, v in overlays.items() if k != "impulse_colors"},
        "impulse_colors": overlays["impulse_colors"],
    }


def structure_snapshot_from_full(full: dict) -> dict:
    """Last-bar structural state for watermark / HTF helpers."""
    return {
        "seq_state": full.get("seq_state_final", 0),
        "critical_level": full.get("critical_level"),
        "close": full.get("Close"),
        "last_peak_was_hh": full.get("last_peak_was_hh", False),
        "last_trough_was_hl": full.get("last_trough_was_hl", False),
        "last_peak": full.get("last_peak"),
        "last_trough": full.get("last_trough"),
        "last_lh": full.get("last_lh"),
        "seq_high": full.get("seq_high_final"),
        "struct_invalid": full.get("struct_invalid_seq_down", False),
    }


def explain_invalid_buy(full: dict | None, *, min_rr: float = 1.5, no_rr_req: bool = False) -> str:
    """Compact manual-scan reject reason when BUY Valid is false."""
    if full is None:
        return "NO_VALID_SIGNAL"
    seq_ok = int(full.get("seq_state_final", 0) or 0) == 1
    trough_hl = bool(full.get("last_trough_was_hl", False))
    peak_hh = bool(full.get("last_peak_was_hh", False))
    struct_invalid = bool(full.get("struct_invalid_seq_down", False))
    close = full.get("Close")
    last_peak = full.get("last_peak")
    struct_ok = (
        trough_hl
        or (
            last_peak is not None
            and close is not None
            and close > last_peak
            and trough_hl
        )
    ) and peak_hh and not struct_invalid
    rr = full.get("RR")
    try:
        rr_f = float(rr) if rr is not None and not (isinstance(rr, float) and np.isnan(rr)) else 0.0
    except (TypeError, ValueError):
        rr_f = 0.0
    risk_ok = True
    sl = full.get("SL")
    if close is not None and sl is not None:
        try:
            risk_ok = float(close) - float(sl) > 0
        except (TypeError, ValueError):
            risk_ok = False
    reward_ok = last_peak is not None and close is not None and float(last_peak) - float(close) > 0

    if not seq_ok:
        return "NO_SEQ_UP"
    if not trough_hl:
        return "NO_STRUCT_HL"
    if not peak_hh:
        return "NO_STRUCT_HH"
    if struct_invalid:
        return "STRUCT_INVALID"
    if no_rr_req:
        return "NO_VALID_SIGNAL"
    if not reward_ok:
        return "NO_REWARD"
    if not risk_ok:
        return "NO_RISK"
    if rr_f < float(min_rr):
        return f"RR_TOO_LOW:{rr_f:.2f}"
    return "NO_VALID_SIGNAL"
