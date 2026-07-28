/** Chart / strategy params — port of indicator_params.py */

export type IndicatorParams = {
  len_fast: number;
  len_slow: number;
  length_major: number;
  lookback: number;
  multiplier: number;
  atr_len: number;
  min_rr: number;
  no_rr_req: boolean;
  use_last_hl_sl: boolean;
  risk_dollars: number;
  show_crit_level: boolean;
  show_hhll: boolean;
  show_extension_lines: boolean;
  show_fib: boolean;
  show_short_ema: boolean;
  show_center_ema: boolean;
  show_sma_major: boolean;
  show_bb: boolean;
  show_breaks: boolean;
  show_tp_sl: boolean;
  show_watermark: boolean;
  bg_color: string;
  paper_color: string;
  candle_up: string;
  candle_down: string;
  crit_stop_color_up: string;
  crit_stop_color_down: string;
};

export function defaultChartParams(): IndicatorParams {
  return {
    len_fast: 20,
    len_slow: 40,
    length_major: 200,
    lookback: 100,
    multiplier: 2.0,
    atr_len: 14,
    min_rr: 1.5,
    no_rr_req: false,
    use_last_hl_sl: true,
    risk_dollars: 100,
    show_crit_level: true,
    show_hhll: true,
    show_extension_lines: true,
    show_fib: false,
    show_short_ema: false,
    show_center_ema: false,
    show_sma_major: false,
    show_bb: false,
    show_breaks: true,
    show_tp_sl: false,
    show_watermark: true,
    bg_color: '#707585',
    paper_color: '#2a2e39',
    candle_up: '#089981',
    candle_down: '#f23645',
    crit_stop_color_up: '#00c853',
    crit_stop_color_down: '#f44336',
  };
}
