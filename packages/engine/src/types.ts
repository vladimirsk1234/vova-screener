/** Shared domain types for Sequence Vova engine (@vova/engine). */

export type Timeframe = 'Daily' | 'Weekly' | 'Monthly';
export type ScanDirection = 'buy' | 'sell';
export type SourceLabel = 'Stocks' | 'ETF' | 'MANUAL SCAN';

export type OhlcBar = {
  date: string; // ISO date YYYY-MM-DD
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
};

export type OhlcSeries = OhlcBar[];

export type PineResult = {
  TP: number;
  SL: number;
  RR: number;
  Valid: boolean;
  New: boolean;
  Strong: boolean;
  /**
   * Bar the current uninterrupted valid run started on, and its distance from the last bar.
   * `0` means the signal became valid on the last (possibly in-progress) bar; `null` when the
   * last bar is not valid at all.
   *
   * These count the run of *this* call's parameters, so a `min_rr` gate makes them count bars since
   * the ratio last crossed the threshold. Use `signalAge` for the NEW / VALID split — the age of a
   * signal must not depend on RR.
   */
  valid_since_index: number | null;
  bars_since_valid: number | null;
  position_size: number;
  position_value: number;
  Close: number;
  ATR: number;
  last_peak_was_hh: boolean;
  last_trough_was_hl: boolean;
  /** Last-bar sequence state: 1 up, -1 down, 0 undecided. */
  seq_state: number;
  /** Sequence broke below the last confirmed trough (buy) / above the peak (sell). */
  struct_invalid: boolean;
  critical_level: number;
  risk: number;
  reward: number;
};

export type CloseScanResult = {
  Valid: boolean;
  New: boolean;
  entry_price: number;
  exit_price: number;
  entry_sl: number;
  position_size: number;
  pnl_dollars: number;
  pnl_pct: number;
  entry_rr: number;
  close_rr: number;
  Close: number;
  ATR: number;
};

/**
 * One long as the close scan replays it: taken on the bar a buy signal appeared, given up on the
 * bar the sequence broke back down. `exit_index` is null while the position is still running.
 *
 * Prices are the closes of those bars, and the RR and SL are the ones the entry bar produced —
 * the numbers the Streamlit close-scan table shows, not anything measured later.
 */
export type CloseTrade = {
  entry_index: number;
  entry_date: string;
  entry_price: number;
  entry_sl: number;
  entry_rr: number;
  position_size: number;
  exit_index: number | null;
  exit_date: string | null;
  exit_price: number;
  close_rr: number;
  pnl_dollars: number;
  pnl_pct: number;
};

/** Every long the close scan would have taken over a series, oldest first. */
export type CloseLedger = {
  trades: CloseTrade[];
  /** The still-running trade, which is the last entry of `trades` when there is one. */
  open: CloseTrade | null;
  /** Date of the last bar the replay saw. */
  asOf: string;
};

export type BuyRow = {
  Symbol: string;
  tv_symbol: string;
  'Company Name': string;
  TP: number;
  SL: number;
  RR: number | string;
  'Position Size (shares)': number;
  'Position Value ($)': number;
  New: number;
  Valid: number;
  Strong: number;
  yahoo_ticker: string;
};

export type SellRow = {
  Symbol: string;
  tv_symbol: string;
  'Company Name': string;
  Entry: number;
  Exit: number;
  'Position Size (shares)': number;
  'RR at Entry': number | string;
  'RR at Close': number | string;
  'Invested ($)': number;
  'P&L ($)': number;
  'P&L (%)': number;
  yahoo_ticker: string;
  _is_summary?: boolean;
};

export type ResultRow = BuyRow | SellRow;

export type RejectedRow = { Symbol: string; Reason: string };

export type ScanParams = {
  source: SourceLabel;
  manualTickers: string;
  riskPerTrade: number;
  minRr: number;
  noRrReq: boolean;
  scanDirection: ScanDirection;
  useLastHlSl: boolean;
  tf: Timeframe;
  newOnly: boolean;
};

export type OhlcCacheEntry = {
  bars: OhlcSeries;
  tf: Timeframe;
  symbol: string;
  yahoo_ticker: string;
};

export type ScanProgress = {
  phase: 'idle' | 'download' | 'process' | 'done' | 'cancelled';
  downloadPct: number;
  processPct: number;
  message: string;
};

export const NAN = Number.NaN;
export const isFiniteNum = (n: unknown): n is number =>
  typeof n === 'number' && Number.isFinite(n);
