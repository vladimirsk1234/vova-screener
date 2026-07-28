/** Shared domain types for Sequence Vova mobile screener. */

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
  position_size: number;
  position_value: number;
  Close: number;
  ATR: number;
  last_peak_was_hh: boolean;
  last_trough_was_hl: boolean;
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

export type BuyRow = {
  Symbol: string;
  tv_symbol: string;
  'Company Name': string;
  TP: number;
  SL: number;
  RR: number;
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
  'RR at Entry': number;
  'RR at Close': number;
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
