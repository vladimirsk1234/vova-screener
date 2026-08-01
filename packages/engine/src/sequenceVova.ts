/**
 * Sequence Vova screener engine — TypeScript port of sequence_vova.py
 * (Python _run_sequence_vova_pine_python + close scan).
 */
import type { CloseScanResult, OhlcSeries, PineResult } from './types';

const NaN_ = Number.NaN;
const isNaN_ = (v: number) => Number.isNaN(v);

export function calcAtr(
  highs: Float64Array,
  lows: Float64Array,
  closes: Float64Array,
  length: number,
): Float64Array {
  const n = closes.length;
  const tr = new Float64Array(n);
  const atr = new Float64Array(n);
  tr[0] = highs[0] - lows[0];
  atr[0] = tr[0];
  const alpha = 1.0 / length;
  for (let i = 1; i < n; i++) {
    tr[i] = Math.max(
      highs[i] - lows[i],
      Math.abs(highs[i] - closes[i - 1]),
      Math.abs(lows[i] - closes[i - 1]),
    );
    atr[i] = alpha * tr[i] + (1.0 - alpha) * atr[i - 1];
  }
  return atr;
}

function seriesToArrays(bars: OhlcSeries) {
  const n = bars.length;
  const c = new Float64Array(n);
  const h = new Float64Array(n);
  const l = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    c[i] = bars[i].close;
    h[i] = bars[i].high;
    l[i] = bars[i].low;
  }
  return { c, h, l, n };
}

function pinePython(
  c_a: Float64Array,
  h_a: Float64Array,
  l_a: Float64Array,
  atr_a: Float64Array,
  min_rr: number,
  use_last_hl_sl: boolean,
  risk_dollars: number,
  direction_sell: boolean,
  no_rr_req: boolean,
): PineResult {
  const n = c_a.length;
  let seq_state = 0;
  let critical_level = NaN_;
  let seq_high = h_a[0];
  let seq_low = l_a[0];
  let last_confirmed_peak = NaN_;
  let last_confirmed_trough = NaN_;
  let last_peak_was_hh = false;
  let last_trough_was_hl = false;

  let last_peak = NaN_;
  let last_valid = false;
  let last_new = false;
  let last_strong = false;
  let last_sl = NaN_;
  let last_rr = NaN_;
  let last_pos_size = NaN_;
  let last_pos_value = NaN_;
  let last_seq_state = 0;
  let last_struct_invalid = false;
  let last_crit = NaN_;
  let last_risk = NaN_;
  let last_reward = NaN_;
  let prev_bar_seq_low = l_a[0];
  let prev_bar_seq_high = h_a[0];
  let valid_since_index = -1;

  for (let i = 1; i < n; i++) {
    const c = c_a[i];
    const h = h_a[i];
    const l = l_a[i];
    const cur_atr = atr_a[i];
    const prev_state = seq_state;
    const prev_crit = critical_level;
    const prev_seq_high = seq_high;
    const prev_seq_low = seq_low;

    let is_break = false;
    let is_bullish_break = false;
    let is_bearish_break = false;
    if (prev_state === 1 && !isNaN_(prev_crit)) {
      is_break = c < prev_crit;
      is_bullish_break = is_break;
    } else if (prev_state === -1 && !isNaN_(prev_crit)) {
      is_break = c > prev_crit;
      is_bearish_break = is_break;
    }

    if (is_break) {
      if (prev_state === 1) {
        if (h >= seq_high) seq_high = h;
        last_peak_was_hh = isNaN_(last_confirmed_peak) || seq_high > last_confirmed_peak;
        last_confirmed_peak = seq_high;
        seq_state = -1;
        seq_high = h;
        seq_low = l;
        critical_level = h;
      } else {
        if (l <= seq_low) seq_low = l;
        last_trough_was_hl =
          isNaN_(last_confirmed_trough) ||
          seq_low > last_confirmed_trough ||
          seq_low === last_confirmed_trough;
        last_confirmed_trough = seq_low;
        seq_state = 1;
        seq_high = h;
        seq_low = l;
        critical_level = l;
      }
    } else {
      seq_state = prev_state;
      if (seq_state === 1) {
        if (h >= seq_high) seq_high = h;
        critical_level = h >= prev_seq_high ? l : prev_crit;
      } else if (seq_state === -1) {
        if (l <= seq_low) seq_low = l;
        critical_level = l <= prev_seq_low ? h : prev_crit;
      } else {
        if (c > prev_seq_high) {
          seq_state = 1;
          critical_level = l;
        } else if (c < prev_seq_low) {
          seq_state = -1;
          critical_level = h;
        } else {
          seq_high = Math.max(prev_seq_high, h);
          seq_low = Math.min(prev_seq_low, l);
        }
      }
    }

    let sl: number;
    let rr: number;
    let position_size: number;
    let position_value: number;
    let valid_signal: boolean;
    let new_signal: boolean;
    let strong_signal: boolean;
    let struct_invalid: boolean;
    let risk: number;
    let reward: number;

    if (direction_sell) {
      const struct_invalid_seq_up =
        seq_state === 1 &&
        last_peak_was_hh &&
        !isNaN_(last_confirmed_peak) &&
        seq_high > last_confirmed_peak;
      const last_peak_was_lh = !last_peak_was_hh;
      const struct_ok_sell =
        (last_peak_was_lh ||
          (!isNaN_(last_confirmed_trough) && c < last_confirmed_trough && last_peak_was_lh)) &&
        !struct_invalid_seq_up;

      sl = c + cur_atr;
      if (!isNaN_(critical_level) && critical_level > c) sl = Math.max(sl, critical_level);
      if (
        use_last_hl_sl &&
        last_peak_was_lh &&
        !isNaN_(last_confirmed_peak) &&
        last_confirmed_peak > c
      ) {
        sl = Math.max(sl, last_confirmed_peak);
      }
      struct_invalid = struct_invalid_seq_up;
      risk = sl - c;
      reward = !isNaN_(last_confirmed_trough) ? c - last_confirmed_trough : 0.0;
      rr = risk > 0 ? reward / risk : NaN_;
      position_size = risk > 0 && risk_dollars > 0 ? risk_dollars / risk : NaN_;
      position_value = !isNaN_(position_size) ? position_size * c : NaN_;

      valid_signal = no_rr_req
        ? seq_state === -1 && struct_ok_sell
        : seq_state === -1 && struct_ok_sell && rr >= min_rr && risk > 0 && reward > 0;
      new_signal = valid_signal && is_bullish_break;
      strong_signal =
        new_signal && !isNaN_(prev_bar_seq_high) && h >= prev_bar_seq_high;
      last_peak = last_confirmed_trough;
    } else {
      const struct_invalid_seq_down =
        seq_state === -1 &&
        last_trough_was_hl &&
        !isNaN_(last_confirmed_trough) &&
        seq_low < last_confirmed_trough;
      const struct_ok =
        (last_trough_was_hl ||
          (!isNaN_(last_confirmed_peak) && c > last_confirmed_peak && last_trough_was_hl)) &&
        last_peak_was_hh &&
        !struct_invalid_seq_down;

      sl = c - cur_atr;
      if (!isNaN_(critical_level) && critical_level < c) sl = Math.min(sl, critical_level);
      if (
        use_last_hl_sl &&
        last_trough_was_hl &&
        !isNaN_(last_confirmed_trough) &&
        last_confirmed_trough < c
      ) {
        sl = Math.min(sl, last_confirmed_trough);
      }
      struct_invalid = struct_invalid_seq_down;
      risk = c - sl;
      reward = !isNaN_(last_confirmed_peak) ? last_confirmed_peak - c : 0.0;
      rr = risk > 0 ? reward / risk : NaN_;
      position_size = risk > 0 && risk_dollars > 0 ? risk_dollars / risk : NaN_;
      position_value = !isNaN_(position_size) ? position_size * c : NaN_;

      valid_signal = no_rr_req
        ? seq_state === 1 && struct_ok
        : seq_state === 1 && struct_ok && rr >= min_rr && risk > 0 && reward > 0;
      new_signal = valid_signal && is_bearish_break;
      strong_signal =
        new_signal && !isNaN_(prev_bar_seq_low) && l <= prev_bar_seq_low;
      last_peak = last_confirmed_peak;
    }

    prev_bar_seq_low = seq_low;
    prev_bar_seq_high = seq_high;
    last_seq_state = seq_state;
    last_struct_invalid = struct_invalid;
    last_crit = critical_level;
    last_risk = risk;
    last_reward = reward;
    // `last_valid` still carries the previous bar here, so a false → true flip starts a new run.
    valid_since_index = valid_signal ? (last_valid ? valid_since_index : i) : -1;
    last_valid = valid_signal;
    last_new = new_signal;
    last_strong = strong_signal;
    last_sl = sl;
    last_rr = rr;
    last_pos_size = position_size;
    last_pos_value = position_value;
  }

  return {
    TP: last_peak,
    SL: last_sl,
    RR: last_rr,
    Valid: last_valid,
    New: last_new,
    Strong: last_strong,
    valid_since_index: valid_since_index >= 0 ? valid_since_index : null,
    bars_since_valid: valid_since_index >= 0 ? n - 1 - valid_since_index : null,
    position_size: last_pos_size,
    position_value: last_pos_value,
    Close: c_a[n - 1],
    ATR: atr_a[n - 1],
    last_peak_was_hh,
    last_trough_was_hl,
    seq_state: last_seq_state,
    struct_invalid: last_struct_invalid,
    critical_level: last_crit,
    risk: last_risk,
    reward: last_reward,
  };
}

export function runSequenceVovaPine(
  bars: OhlcSeries,
  opts: {
    atr_len?: number;
    min_rr?: number;
    use_last_hl_sl?: boolean;
    risk_dollars?: number;
    direction?: 'buy' | 'sell';
    no_rr_req?: boolean;
  } = {},
): PineResult | null {
  if (bars.length < 2) return null;
  const { c, h, l } = seriesToArrays(bars);
  const atr = calcAtr(h, l, c, opts.atr_len ?? 14);
  return pinePython(
    c,
    h,
    l,
    atr,
    opts.min_rr ?? 1.5,
    opts.use_last_hl_sl ?? true,
    opts.risk_dollars ?? 100,
    (opts.direction ?? 'buy') === 'sell',
    opts.no_rr_req ?? false,
  );
}

function closePython(
  c_a: Float64Array,
  h_a: Float64Array,
  l_a: Float64Array,
  atr_a: Float64Array,
  min_rr: number,
  use_last_hl_sl: boolean,
  risk_dollars: number,
  no_rr_req: boolean,
): CloseScanResult {
  const n = c_a.length;
  let seq_state = 0;
  let critical_level = NaN_;
  let seq_high = h_a[0];
  let seq_low = l_a[0];
  let last_confirmed_peak = NaN_;
  let last_confirmed_trough = NaN_;
  let last_peak_was_hh = false;
  let last_trough_was_hl = false;

  let position_open = false;
  let entry_price = NaN_;
  let entry_sl = NaN_;
  let entry_rr_at_open = NaN_;
  let position_size = NaN_;

  let last_valid = false;
  let last_new = false;
  let last_entry_price = NaN_;
  let last_exit_price = NaN_;
  let last_entry_sl = NaN_;
  let last_position_size = NaN_;
  let last_pnl_dollars = NaN_;
  let last_pnl_pct = NaN_;
  let last_entry_rr = NaN_;
  let last_close_rr = NaN_;

  for (let i = 1; i < n; i++) {
    const c = c_a[i];
    const h = h_a[i];
    const l = l_a[i];
    const cur_atr = atr_a[i];
    const prev_state = seq_state;
    const prev_crit = critical_level;
    const prev_seq_high = seq_high;
    const prev_seq_low = seq_low;

    let is_bullish_break = false;
    let is_bearish_break = false;
    if (prev_state === 1 && !isNaN_(prev_crit)) {
      is_bullish_break = c < prev_crit;
    } else if (prev_state === -1 && !isNaN_(prev_crit)) {
      is_bearish_break = c > prev_crit;
    }

    if (is_bullish_break || is_bearish_break) {
      if (prev_state === 1) {
        if (h >= seq_high) seq_high = h;
        last_peak_was_hh = isNaN_(last_confirmed_peak) || seq_high > last_confirmed_peak;
        last_confirmed_peak = seq_high;
        seq_state = -1;
        seq_high = h;
        seq_low = l;
        critical_level = h;
      } else {
        if (l <= seq_low) seq_low = l;
        last_trough_was_hl =
          isNaN_(last_confirmed_trough) ||
          seq_low > last_confirmed_trough ||
          seq_low === last_confirmed_trough;
        last_confirmed_trough = seq_low;
        seq_state = 1;
        seq_high = h;
        seq_low = l;
        critical_level = l;
      }
    } else {
      seq_state = prev_state;
      if (seq_state === 1) {
        if (h >= seq_high) seq_high = h;
        critical_level = h >= prev_seq_high ? l : prev_crit;
      } else if (seq_state === -1) {
        if (l <= seq_low) seq_low = l;
        critical_level = l <= prev_seq_low ? h : prev_crit;
      } else {
        if (c > prev_seq_high) {
          seq_state = 1;
          critical_level = l;
        } else if (c < prev_seq_low) {
          seq_state = -1;
          critical_level = h;
        } else {
          seq_high = Math.max(prev_seq_high, h);
          seq_low = Math.min(prev_seq_low, l);
        }
      }
    }

    const struct_invalid_seq_down =
      seq_state === -1 &&
      last_trough_was_hl &&
      !isNaN_(last_confirmed_trough) &&
      seq_low < last_confirmed_trough;
    const struct_ok =
      (last_trough_was_hl ||
        (!isNaN_(last_confirmed_peak) && c > last_confirmed_peak && last_trough_was_hl)) &&
      last_peak_was_hh &&
      !struct_invalid_seq_down;

    let sl = c - cur_atr;
    if (!isNaN_(critical_level) && critical_level < c) sl = Math.min(sl, critical_level);
    if (
      use_last_hl_sl &&
      last_trough_was_hl &&
      !isNaN_(last_confirmed_trough) &&
      last_confirmed_trough < c
    ) {
      sl = Math.min(sl, last_confirmed_trough);
    }
    const risk = c - sl;
    const reward = !isNaN_(last_confirmed_peak) ? last_confirmed_peak - c : 0.0;
    const rr = risk > 0 ? reward / risk : NaN_;

    const valid_signal = no_rr_req
      ? seq_state === 1 && struct_ok
      : seq_state === 1 && struct_ok && rr >= min_rr && risk > 0 && reward > 0;
    const new_signal = valid_signal && is_bearish_break;

    if (new_signal && !position_open) {
      position_open = true;
      entry_price = c;
      entry_sl = sl;
      entry_rr_at_open = rr;
      position_size = risk > 0 && risk_dollars > 0 ? risk_dollars / risk : NaN_;
    }

    if (position_open && is_bullish_break) {
      const exit_price = c;
      const psz = position_size;
      const entry_risk = entry_price - entry_sl;
      const close_rr =
        entry_risk > 0 && !isNaN_(entry_price) ? (exit_price - entry_price) / entry_risk : NaN_;
      let pnl = NaN_;
      let pnl_pct = NaN_;
      if (!isNaN_(psz) && !isNaN_(entry_price)) {
        pnl = (exit_price - entry_price) * psz;
        pnl_pct = entry_price > 0 ? ((exit_price - entry_price) / entry_price) * 100.0 : NaN_;
      }

      if (i === n - 1) {
        last_valid = true;
        last_new = true;
        last_entry_price = entry_price;
        last_exit_price = exit_price;
        last_entry_sl = entry_sl;
        last_position_size = position_size;
        last_pnl_dollars = pnl;
        last_pnl_pct = pnl_pct;
        last_entry_rr = entry_rr_at_open;
        last_close_rr = close_rr;
      }

      position_open = false;
      entry_price = NaN_;
      entry_sl = NaN_;
      entry_rr_at_open = NaN_;
      position_size = NaN_;
    }
  }

  return {
    Valid: last_valid,
    New: last_new,
    entry_price: last_entry_price,
    exit_price: last_exit_price,
    entry_sl: last_entry_sl,
    position_size: last_position_size,
    pnl_dollars: last_pnl_dollars,
    pnl_pct: last_pnl_pct,
    entry_rr: last_entry_rr,
    close_rr: last_close_rr,
    Close: c_a[n - 1],
    ATR: atr_a[n - 1],
  };
}

export function runSequenceVovaCloseScan(
  bars: OhlcSeries,
  opts: {
    atr_len?: number;
    min_rr?: number;
    use_last_hl_sl?: boolean;
    risk_dollars?: number;
    no_rr_req?: boolean;
  } = {},
): CloseScanResult | null {
  if (bars.length < 2) return null;
  const { c, h, l } = seriesToArrays(bars);
  const atr = calcAtr(h, l, c, opts.atr_len ?? 14);
  return closePython(
    c,
    h,
    l,
    atr,
    opts.min_rr ?? 1.5,
    opts.use_last_hl_sl ?? true,
    opts.risk_dollars ?? 100,
    opts.no_rr_req ?? false,
  );
}

/**
 * Reject reason for an invalid BUY setup, in the same order as the Python oracle
 * `sequence_vova.explain_invalid_buy`: the sequence state is checked first, so a
 * down sequence reports NO_SEQ_UP instead of an RR computed off stale structure.
 *
 * The `RR_TOO_LOW` code carries the run threshold as a ` (min x.xx)` suffix;
 * everything before that suffix matches the Python string byte for byte.
 */
export function explainInvalidBuy(
  pine: PineResult | null,
  min_rr = 1.5,
  no_rr_req = false,
): string {
  if (!pine) return 'NO_VALID_SIGNAL';
  if (pine.seq_state !== 1) return 'NO_SEQ_UP';
  if (!pine.last_trough_was_hl) return 'NO_STRUCT_HL';
  if (!pine.last_peak_was_hh) return 'NO_STRUCT_HH';
  if (pine.struct_invalid) return 'STRUCT_INVALID';
  if (no_rr_req) return 'NO_VALID_SIGNAL';
  if (!(pine.reward > 0)) return 'NO_REWARD';
  if (!(pine.risk > 0)) return 'NO_RISK';
  const rr = Number.isFinite(pine.RR) ? pine.RR : 0;
  if (rr < min_rr) return `RR_TOO_LOW:${rr.toFixed(2)} (min ${min_rr.toFixed(2)})`;
  return 'NO_VALID_SIGNAL';
}
