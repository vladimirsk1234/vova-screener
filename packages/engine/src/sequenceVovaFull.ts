/**
 * Full-history Sequence Vova — port of sequence_vova.run_sequence_vova_full.
 * Used by chart overlays (peaks, extensions, fib, BB/EMA/Elder, watermark scalars).
 */
import type { IndicatorParams } from './indicatorParams';
import { defaultIndicatorParams, toRunnerKwargs } from './indicatorParams';
import { calcDmi, computeOverlays, type OverlaySeries } from './indicators';
import { calcAtr } from './sequenceVova';
import type { OhlcSeries } from './types';

const NaN_ = Number.NaN;
const isNaN_ = (v: number) => Number.isNaN(v);

export type PeakLabel = 'HH' | 'LH' | 'DT';
export type TroughLabel = 'HL' | 'LL' | 'DB';

export type StructurePoint = {
  idx: number;
  price: number;
  label: PeakLabel | TroughLabel;
};

export type ExtensionLine = {
  kind: 'high' | 'low';
  x0_idx: number;
  y0: number;
  x1_idx: number;
  y1: number;
};

export type FibLevels = {
  high: number;
  high_idx: number;
  low: number;
  low_idx: number;
  fib_382: number;
  fib_500: number;
  fib_618: number;
};

export type SequenceVovaFullResult = {
  TP: number;
  SL: number;
  RR: number;
  Valid: boolean;
  New: boolean;
  Strong: boolean;
  position_size: number;
  position_value: number;
  Close: number;
  ATR: number;
  ATR_pct: number;
  ADX: number;
  critical_level_series: (number | null)[];
  seq_state_series: number[];
  bullish_break: boolean[];
  bearish_break: boolean[];
  peaks: StructurePoint[];
  troughs: StructurePoint[];
  extension_lines: ExtensionLine[];
  seq_state_final: number;
  seq_high_final: number;
  last_peak: number | null;
  last_peak_idx: number | null;
  last_trough: number | null;
  last_trough_idx: number | null;
  last_peak_was_hh: boolean;
  last_trough_was_hl: boolean;
  last_lh: number | null;
  critical_level: number | null;
  struct_invalid_seq_down: boolean;
  signal_bar_index: number | null;
  fib: FibLevels | null;
  overlays: OverlaySeries;
  impulse_colors: string[];
};

export type StructureSnapshot = {
  seq_state: number;
  critical_level: number | null;
  close: number;
  last_peak_was_hh: boolean;
  last_trough_was_hl: boolean;
  last_peak: number | null;
  last_trough: number | null;
  last_lh: number | null;
  seq_high: number;
  struct_invalid: boolean;
  sma_above?: boolean | null;
  sma_major?: number;
};

export type FullRunnerOpts = {
  atr_len?: number;
  min_rr?: number;
  use_last_hl_sl?: boolean;
  risk_dollars?: number;
  no_rr_req?: boolean;
  len_fast?: number;
  len_slow?: number;
  length_major?: number;
  lookback?: number;
  multiplier?: number;
  bb_length?: number;
  bb_mult?: number;
  elder_bull_color?: string;
  elder_bear_color?: string;
  elder_neut_color?: string;
  adx_len?: number;
  params?: IndicatorParams;
};

function resolveOpts(opts: FullRunnerOpts) {
  const p = opts.params ?? defaultIndicatorParams();
  const kw = toRunnerKwargs(p);
  return {
    atr_len: opts.atr_len ?? kw.atr_len,
    min_rr: opts.min_rr ?? kw.min_rr,
    use_last_hl_sl: opts.use_last_hl_sl ?? kw.use_last_hl_sl,
    risk_dollars: opts.risk_dollars ?? kw.risk_dollars,
    no_rr_req: opts.no_rr_req ?? kw.no_rr_req,
    len_fast: opts.len_fast ?? kw.len_fast,
    len_slow: opts.len_slow ?? kw.len_slow,
    length_major: opts.length_major ?? kw.length_major,
    lookback: opts.lookback ?? kw.lookback,
    multiplier: opts.multiplier ?? kw.multiplier,
    bb_length: opts.bb_length ?? kw.bb_length,
    bb_mult: opts.bb_mult ?? kw.bb_mult,
    elder_bull_color: opts.elder_bull_color ?? kw.elder_bull_color,
    elder_bear_color: opts.elder_bear_color ?? kw.elder_bear_color,
    elder_neut_color: opts.elder_neut_color ?? kw.elder_neut_color,
    adx_len: opts.adx_len ?? kw.adx_len,
  };
}

export function runSequenceVovaFull(
  bars: OhlcSeries,
  opts: FullRunnerOpts = {},
): SequenceVovaFullResult | null {
  const n = bars.length;
  if (n < 2) return null;

  const o = resolveOpts(opts);
  const c_a = new Float64Array(n);
  const h_a = new Float64Array(n);
  const l_a = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    c_a[i] = bars[i].close;
    h_a[i] = bars[i].high;
    l_a[i] = bars[i].low;
  }
  const atr_a = calcAtr(h_a, l_a, c_a, o.atr_len);

  let seq_state = 0;
  let critical_level = NaN_;
  let seq_high = h_a[0];
  let seq_low = l_a[0];
  let seq_high_idx = 0;
  let seq_low_idx = 0;

  let last_confirmed_peak = NaN_;
  let last_confirmed_peak_idx = -1;
  let last_confirmed_trough = NaN_;
  let last_confirmed_trough_idx = -1;
  let prev_trough_before_peak = NaN_;
  let prev_trough_before_peak_idx = -1;
  let last_peak_was_hh = false;
  let last_trough_was_hl = false;
  let last_lh = NaN_;

  const critical_level_series: (number | null)[] = new Array(n).fill(null);
  const seq_state_series: number[] = new Array(n).fill(0);
  const bullish_break: boolean[] = new Array(n).fill(false);
  const bearish_break: boolean[] = new Array(n).fill(false);

  const peaks: StructurePoint[] = [];
  const troughs: StructurePoint[] = [];
  let line_high_ext: ExtensionLine | null = null;
  let line_low_ext: ExtensionLine | null = null;

  let last_crit = NaN_;
  let last_peak_val = NaN_;
  let last_valid = false;
  let last_new = false;
  let last_strong = false;
  let last_sl = NaN_;
  let last_rr = NaN_;
  let last_pos_size = NaN_;
  let last_pos_value = NaN_;
  let prev_bar_seq_low = l_a[0];
  let signal_bar_index: number | null = null;

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
        if (h >= seq_high) {
          seq_high = h;
          seq_high_idx = i;
        }
        let label: PeakLabel;
        let is_current_peak_hh: boolean;
        if (isNaN_(last_confirmed_peak)) {
          label = 'HH';
          is_current_peak_hh = true;
        } else if (seq_high > last_confirmed_peak) {
          label = 'HH';
          is_current_peak_hh = true;
        } else if (seq_high < last_confirmed_peak) {
          label = 'LH';
          is_current_peak_hh = false;
        } else {
          label = 'DT';
          is_current_peak_hh = false;
        }
        last_peak_was_hh = is_current_peak_hh;
        if (!is_current_peak_hh) last_lh = seq_high;
        prev_trough_before_peak = last_confirmed_trough;
        prev_trough_before_peak_idx = last_confirmed_trough_idx;
        peaks.push({ idx: seq_high_idx, price: seq_high, label });

        if (last_confirmed_peak_idx >= 0 && !isNaN_(last_confirmed_peak)) {
          line_high_ext = {
            kind: 'high',
            x0_idx: last_confirmed_peak_idx,
            y0: last_confirmed_peak,
            x1_idx: seq_high_idx,
            y1: seq_high,
          };
        }

        last_confirmed_peak = seq_high;
        last_confirmed_peak_idx = seq_high_idx;
        seq_state = -1;
        seq_high = h;
        seq_low = l;
        seq_high_idx = i;
        seq_low_idx = i;
        critical_level = h;
      } else {
        if (l <= seq_low) {
          seq_low = l;
          seq_low_idx = i;
        }
        let label: TroughLabel;
        let is_current_trough_hl: boolean;
        if (isNaN_(last_confirmed_trough)) {
          label = 'LL';
          is_current_trough_hl = true;
        } else if (seq_low < last_confirmed_trough) {
          label = 'LL';
          is_current_trough_hl = false;
        } else if (seq_low > last_confirmed_trough) {
          label = 'HL';
          is_current_trough_hl = true;
        } else {
          label = 'DB';
          is_current_trough_hl = true;
        }
        last_trough_was_hl = is_current_trough_hl;
        troughs.push({ idx: seq_low_idx, price: seq_low, label });

        if (last_confirmed_trough_idx >= 0 && !isNaN_(last_confirmed_trough)) {
          line_low_ext = {
            kind: 'low',
            x0_idx: last_confirmed_trough_idx,
            y0: last_confirmed_trough,
            x1_idx: seq_low_idx,
            y1: seq_low,
          };
        }

        last_confirmed_trough = seq_low;
        last_confirmed_trough_idx = seq_low_idx;
        seq_state = 1;
        seq_high = h;
        seq_low = l;
        seq_high_idx = i;
        seq_low_idx = i;
        critical_level = l;
      }
    } else {
      seq_state = prev_state;
      if (seq_state === 1) {
        if (h >= seq_high) {
          seq_high = h;
          seq_high_idx = i;
        }
        critical_level = h >= prev_seq_high ? l : prev_crit;
      } else if (seq_state === -1) {
        if (l <= seq_low) {
          seq_low = l;
          seq_low_idx = i;
        }
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

    critical_level_series[i] = isNaN_(critical_level) ? null : critical_level;
    seq_state_series[i] = seq_state;
    bullish_break[i] = is_bullish_break;
    bearish_break[i] = is_bearish_break;

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

    const cur_cond_seq_ok = seq_state === 1;
    if (is_bearish_break && cur_cond_seq_ok && struct_ok) signal_bar_index = i;
    else if (is_bearish_break) signal_bar_index = null;

    let sl = c - cur_atr;
    if (!isNaN_(critical_level) && critical_level < c) sl = Math.min(sl, critical_level);
    if (
      o.use_last_hl_sl &&
      last_trough_was_hl &&
      !isNaN_(last_confirmed_trough) &&
      last_confirmed_trough < c
    ) {
      sl = Math.min(sl, last_confirmed_trough);
    }
    const risk = c - sl;
    const reward = !isNaN_(last_confirmed_peak) ? last_confirmed_peak - c : 0;
    const rr = risk > 0 ? reward / risk : NaN_;
    const position_size = risk > 0 && o.risk_dollars > 0 ? o.risk_dollars / risk : NaN_;
    const position_value = !isNaN_(position_size) ? position_size * c : NaN_;

    const valid_signal = o.no_rr_req
      ? seq_state === 1 && struct_ok
      : seq_state === 1 && struct_ok && rr >= o.min_rr && risk > 0 && reward > 0;
    const new_signal = valid_signal && is_bearish_break;
    const strong_signal =
      new_signal && !isNaN_(prev_bar_seq_low) && l <= prev_bar_seq_low;

    prev_bar_seq_low = seq_low;
    last_crit = critical_level;
    last_peak_val = last_confirmed_peak;
    last_valid = valid_signal;
    last_new = new_signal;
    last_strong = strong_signal;
    last_sl = sl;
    last_rr = rr;
    last_pos_size = position_size;
    last_pos_value = position_value;
  }

  let fib_levels: FibLevels | null = null;
  const daily_struct_valid_fib =
    last_trough_was_hl ||
    (!isNaN_(last_confirmed_peak) && c_a[n - 1] > last_confirmed_peak && last_trough_was_hl);
  if (
    seq_state === 1 &&
    !isNaN_(last_confirmed_peak) &&
    !isNaN_(prev_trough_before_peak) &&
    daily_struct_valid_fib
  ) {
    const fib_range = last_confirmed_peak - prev_trough_before_peak;
    fib_levels = {
      high: last_confirmed_peak,
      high_idx: last_confirmed_peak_idx,
      low: prev_trough_before_peak,
      low_idx: prev_trough_before_peak_idx,
      fib_382: last_confirmed_peak - fib_range * 0.382,
      fib_500: last_confirmed_peak - fib_range * 0.5,
      fib_618: last_confirmed_peak - fib_range * 0.618,
    };
  }

  const { overlays, impulse_colors } = computeOverlays(c_a, {
    len_fast: o.len_fast,
    len_slow: o.len_slow,
    length_major: o.length_major,
    lookback: o.lookback,
    multiplier: o.multiplier,
    bb_length: o.bb_length,
    bb_mult: o.bb_mult,
    elder_bull_color: o.elder_bull_color,
    elder_bear_color: o.elder_bear_color,
    elder_neut_color: o.elder_neut_color,
  });

  const { adx } = calcDmi(h_a, l_a, c_a, o.adx_len);
  const atr_last = atr_a[n - 1];
  const close_last = c_a[n - 1];
  const atr_pct = close_last !== 0 ? (atr_last / close_last) * 100 : 0;

  return {
    TP: last_peak_val,
    SL: last_sl,
    RR: last_rr,
    Valid: last_valid,
    New: last_new,
    Strong: last_strong,
    position_size: last_pos_size,
    position_value: last_pos_value,
    Close: close_last,
    ATR: atr_last,
    ATR_pct: atr_pct,
    ADX: adx[n - 1] ?? 0,
    critical_level_series,
    seq_state_series,
    bullish_break,
    bearish_break,
    peaks,
    troughs,
    extension_lines: [line_high_ext, line_low_ext].filter(
      (ln): ln is ExtensionLine => ln != null,
    ),
    seq_state_final: seq_state,
    seq_high_final: seq_high,
    last_peak: isNaN_(last_confirmed_peak) ? null : last_confirmed_peak,
    last_peak_idx: last_confirmed_peak_idx >= 0 ? last_confirmed_peak_idx : null,
    last_trough: isNaN_(last_confirmed_trough) ? null : last_confirmed_trough,
    last_trough_idx: last_confirmed_trough_idx >= 0 ? last_confirmed_trough_idx : null,
    last_peak_was_hh,
    last_trough_was_hl,
    last_lh: isNaN_(last_lh) ? null : last_lh,
    critical_level: isNaN_(last_crit) ? null : last_crit,
    struct_invalid_seq_down:
      seq_state === -1 &&
      last_trough_was_hl &&
      !isNaN_(last_confirmed_trough) &&
      seq_low < last_confirmed_trough,
    signal_bar_index,
    fib: fib_levels,
    overlays,
    impulse_colors,
  };
}

export function structureSnapshotFromFull(full: SequenceVovaFullResult): StructureSnapshot {
  return {
    seq_state: full.seq_state_final,
    critical_level: full.critical_level,
    close: full.Close,
    last_peak_was_hh: full.last_peak_was_hh,
    last_trough_was_hl: full.last_trough_was_hl,
    last_peak: full.last_peak,
    last_trough: full.last_trough,
    last_lh: full.last_lh,
    seq_high: full.seq_high_final,
    struct_invalid: full.struct_invalid_seq_down,
  };
}

/** Structure + critical level series for chart overlays (subset of full). */
export type StructureOverlay = {
  critical: (number | null)[];
  seq_state: number[];
  bullish_break: boolean[];
  bearish_break: boolean[];
  last_peak: number | null;
  last_trough: number | null;
  last_peak_was_hh: boolean;
  last_trough_was_hl: boolean;
  TP: number | null;
  SL: number | null;
  RR: number;
};

export function runStructureOverlay(
  bars: OhlcSeries,
  opts: FullRunnerOpts = {},
): StructureOverlay | null {
  const full = runSequenceVovaFull(bars, opts);
  if (!full) return null;
  return {
    critical: full.critical_level_series,
    seq_state: full.seq_state_series,
    bullish_break: full.bullish_break,
    bearish_break: full.bearish_break,
    last_peak: full.last_peak,
    last_trough: full.last_trough,
    last_peak_was_hh: full.last_peak_was_hh,
    last_trough_was_hl: full.last_trough_was_hl,
    TP: Number.isFinite(full.TP) ? full.TP : null,
    SL: Number.isFinite(full.SL) ? full.SL : null,
    RR: full.RR,
  };
}
