/** Streamlit IndicatorParams defaults + hardcoded visibility rules. */
import type { ChartSettings } from './api';

export const DEFAULT_CHART_SETTINGS: ChartSettings = {
  len_fast: 20,
  len_slow: 40,
  length_major: 200,
  lookback: 100,
  multiplier: 2,
  bb_length: 20,
  bb_mult: 2,
  min_rr: 1.5,
  no_rr_req: false,
  use_last_hl_sl: true,
  bg_color: '#707585',
  paper_color: '#2a2e39',
  grid_color: '#363a45',
  candle_up: '#089981',
  candle_down: '#f23645',
  candle_border: '#000000',
  candle_wick: '#000000',
  hhll_color: '#000000',
  crit_stop_color_up: '#00c853',
  crit_stop_color_down: '#f44336',
  crit_custom_color: '#000000',
  fib_color: '#000000',
  fib_width: 2,
  short_ema_color: '#2196f3',
  center_ema_color: '#f44336',
  sma_major_color: '#ff9800',
  bb_basis_color: '#2196f3',
  bb_upper_color: '#9e9e9e',
  bb_lower_color: '#9e9e9e',
  bb_fill_color: 'rgba(158,158,158,0.15)',
  env_upper_color: 'rgba(128,128,128,0.5)',
  env_lower_color: 'rgba(128,128,128,0.5)',
  wm_text_color: '#e0e0e0',
  wm_font_size: 11,
  show_crit_level: true,
  show_hhll: true,
  show_extension_lines: true,
  show_fib: false,
  show_short_ema: false,
  show_center_ema: false,
  show_sma_major: false,
  show_elder_envelope: false,
  show_elder_impulse: false,
  show_bb: false,
  show_bb_background: false,
  show_breaks: true,
  show_tp_sl: false,
  show_watermark: true,
};

/** Match Streamlit `_apply_hardcoded_params`. */
export function applyHardcodedSettings(p: ChartSettings): ChartSettings {
  return {
    ...p,
    show_crit_level: true,
    show_hhll: true,
    show_extension_lines: true,
    show_breaks: true,
    show_watermark: true,
    show_elder_envelope: false,
    show_elder_impulse: false,
    bb_length: 20,
    bb_mult: 2,
    show_bb_background: false,
  };
}

export function mergeChartSettings(partial?: Partial<ChartSettings> | null): ChartSettings {
  return applyHardcodedSettings({ ...DEFAULT_CHART_SETTINGS, ...(partial ?? {}) });
}

/** Fundamentals view: weekly candles only. Does not go through applyHardcodedSettings. */
export function stripTaOverlays(settings: ChartSettings): ChartSettings {
  return {
    ...settings,
    show_crit_level: false,
    show_hhll: false,
    show_extension_lines: false,
    show_fib: false,
    show_short_ema: false,
    show_center_ema: false,
    show_sma_major: false,
    show_elder_envelope: false,
    show_elder_impulse: false,
    show_bb: false,
    show_bb_background: false,
    show_breaks: false,
    show_tp_sl: false,
    show_watermark: false,
  };
}

/** Numeric params that require server recompute. */
export function numericChartParams(s: ChartSettings): Partial<ChartSettings> {
  return {
    len_fast: s.len_fast,
    len_slow: s.len_slow,
    length_major: s.length_major,
    lookback: s.lookback,
    multiplier: s.multiplier,
    bb_length: s.bb_length,
    bb_mult: s.bb_mult,
    min_rr: s.min_rr,
    no_rr_req: s.no_rr_req,
    use_last_hl_sl: s.use_last_hl_sl,
  };
}
