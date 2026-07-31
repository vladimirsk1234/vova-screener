/** Port of indicator_params.py — chart + runner defaults. */

export type IndicatorParams = {
  len_fast: number;
  len_slow: number;
  length_major: number;
  lookback: number;
  multiplier: number;
  elder_bull_color: string;
  elder_bear_color: string;
  elder_neut_color: string;
  env_upper_color: string;
  env_lower_color: string;
  short_ema_color: string;
  center_ema_color: string;
  sma_major_color: string;
  hhll_color: string;
  hhll_label_size: number;
  crit_stop_color_up: string;
  crit_stop_color_down: string;
  crit_custom_color: string;
  crit_lbl_offset: number;
  fib_color: string;
  fib_width: number;
  bb_length: number;
  bb_mult: number;
  bb_basis_color: string;
  bb_upper_color: string;
  bb_lower_color: string;
  bb_fill_color: string;
  atr_len: number;
  atr_low_thresh: number;
  atr_high_thresh: number;
  adx_len: number;
  adx_thresh: number;
  min_rr: number;
  no_rr_req: boolean;
  use_last_hl_sl: boolean;
  risk_dollars: number;
  wm_text_color: string;
  wm_font_size: number;
  bg_color: string;
  paper_color: string;
  grid_color: string;
  candle_up: string;
  candle_down: string;
  candle_border: string;
  candle_wick: string;
  show_crit_level: boolean;
  show_hhll: boolean;
  show_extension_lines: boolean;
  show_fib: boolean;
  show_short_ema: boolean;
  show_center_ema: boolean;
  show_sma_major: boolean;
  show_elder_envelope: boolean;
  show_elder_impulse: boolean;
  show_bb: boolean;
  show_bb_background: boolean;
  show_breaks: boolean;
  show_tp_sl: boolean;
  show_watermark: boolean;
};

export function defaultIndicatorParams(): IndicatorParams {
  return {
    len_fast: 20,
    len_slow: 40,
    length_major: 200,
    lookback: 100,
    multiplier: 2.0,
    elder_bull_color: '#00c853',
    elder_bear_color: '#ff1744',
    elder_neut_color: '#4eadfc',
    env_upper_color: 'rgba(128,128,128,0.5)',
    env_lower_color: 'rgba(128,128,128,0.5)',
    short_ema_color: '#2196f3',
    center_ema_color: '#f44336',
    sma_major_color: '#ff9800',
    hhll_color: '#000000',
    hhll_label_size: 12,
    crit_stop_color_up: '#00c853',
    crit_stop_color_down: '#f44336',
    crit_custom_color: '#000000',
    crit_lbl_offset: 10,
    fib_color: '#000000',
    fib_width: 2,
    bb_length: 20,
    bb_mult: 2.0,
    bb_basis_color: '#2196f3',
    bb_upper_color: '#9e9e9e',
    bb_lower_color: '#9e9e9e',
    bb_fill_color: 'rgba(158,158,158,0.15)',
    atr_len: 14,
    atr_low_thresh: 3.0,
    atr_high_thresh: 5.0,
    adx_len: 14,
    adx_thresh: 20,
    min_rr: 1.5,
    no_rr_req: false,
    use_last_hl_sl: true,
    risk_dollars: 100.0,
    wm_text_color: '#e0e0e0',
    wm_font_size: 11,
    bg_color: '#707585',
    paper_color: '#2a2e39',
    grid_color: '#363a45',
    candle_up: '#089981',
    candle_down: '#f23645',
    candle_border: '#000000',
    candle_wick: '#000000',
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
}

/** Match Streamlit `_apply_hardcoded_params` visibility rules. */
export function applyHardcodedChartParams(p: IndicatorParams): IndicatorParams {
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
    bb_mult: 2.0,
    show_bb_background: false,
    atr_len: 14,
    atr_low_thresh: 3.0,
    atr_high_thresh: 5.0,
    adx_len: 14,
    adx_thresh: 20,
    wm_text_color: '#e0e0e0',
  };
}

export function indicatorParamsFromDict(
  d: Partial<IndicatorParams> | Record<string, unknown> | null | undefined,
): IndicatorParams {
  const base = defaultIndicatorParams();
  if (!d) return base;
  const out = { ...base };
  const src = d as Record<string, unknown>;
  for (const key of Object.keys(base) as (keyof IndicatorParams)[]) {
    if (src[key] !== undefined) {
      (out as Record<string, unknown>)[key] = src[key];
    }
  }
  return out;
}

export function toRunnerKwargs(p: IndicatorParams) {
  return {
    atr_len: p.atr_len,
    min_rr: p.min_rr,
    no_rr_req: p.no_rr_req,
    use_last_hl_sl: p.use_last_hl_sl,
    risk_dollars: p.risk_dollars,
    len_fast: p.len_fast,
    len_slow: p.len_slow,
    length_major: p.length_major,
    lookback: p.lookback,
    multiplier: p.multiplier,
    bb_length: p.bb_length,
    bb_mult: p.bb_mult,
    elder_bull_color: p.elder_bull_color,
    elder_bear_color: p.elder_bear_color,
    elder_neut_color: p.elder_neut_color,
    adx_len: p.adx_len,
  };
}
