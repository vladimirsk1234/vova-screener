/** On-device screener records. No FMP fundamental fields. */

export type UserTf = 'Weekly' | 'Monthly';
export type HistoryTf = UserTf | 'All';
export type HistoryGroupBy = 'Day' | 'Weekly' | 'Monthly';
export type HistoryRange = 'all' | 'ytd' | '1m' | '3m' | '6m' | '1y' | 'max';
export type Universe = 'Stocks' | 'ETF';
export type Bucket = 'new' | 'valid' | 'closed';
export type Interest = 'interested' | 'not_interested';
export type ExitReason = 'TP' | 'SL' | 'sell_to_close' | 'signal_lost' | 'manual';
export type SortDir = 'asc' | 'desc';
export type ResultSort = 'rr' | 'pnl' | 'interest' | 'symbol';
export type HistoryPeriodSort = 'period' | 'pnl' | 'winRate' | 'trades' | 'rr';
export type HistoryTradeSort = 'date' | 'pnl' | 'r' | 'rr' | 'interest' | 'symbol';

export const TIMEFRAMES = ['Weekly', 'Monthly'] as const satisfies readonly UserTf[];
export const HISTORY_TFS = ['Weekly', 'Monthly', 'All'] as const satisfies readonly HistoryTf[];
export const HISTORY_GROUP_BYS = ['Day', 'Weekly', 'Monthly'] as const satisfies readonly HistoryGroupBy[];
export const HISTORY_RANGES = ['all', 'ytd', '1m', '3m', '6m', '1y'] as const satisfies readonly HistoryRange[];
export const UNIVERSES = ['Stocks', 'ETF'] as const satisfies readonly Universe[];
export const BUCKETS = ['new', 'valid', 'closed'] as const satisfies readonly Bucket[];

export const INTEREST_RANK: Record<Interest | 'none', number> = {
  interested: 2,
  none: 1,
  not_interested: 0,
};

export type AppSettings = {
  maxRiskUsd: number;
  /** 0 shows every signal. NEW/VALID use live RR; CLOSED and History use entry RR. */
  minRr: number;
};

export const DEFAULT_SETTINGS: AppSettings = { maxRiskUsd: 100, minRr: 0 };

export type TrackedSignal = {
  id: string;
  yahooTicker: string;
  symbol: string;
  tvSymbol: string;
  companyName: string;
  universe: Universe;
  tf: UserTf;
  status: 'active' | 'closed';
  provisional: boolean;
  provisionalClose: boolean;
  signalValid: boolean;
  imported: boolean;
  openedPeriodKey: string;
  openedAsOf: string;
  openedAt: string;
  entry: number;
  tp: number | null;
  sl: number | null;
  rrAtEntry: number | null;
  shares: number;
  riskUsd: number;
  lastSeenPeriodKey: string;
  lastSeenAsOf: string;
  lastPrice: number | null;
  lastRr: number | null;
  barsSinceValid: number | null;
  validSinceAsOf: string | null;
  isStrong: boolean;
  unrealizedUsd: number | null;
  unrealizedR: number | null;
  unrealizedPct: number | null;
  closedPeriodKey: string | null;
  closedAt: string | null;
  exitDate: string | null;
  exitPrice: number | null;
  exitReason: ExitReason | null;
  pnlUsd: number | null;
  pnlR: number | null;
  pnlPct: number | null;
  holdPeriods: number | null;
  interest: Interest | null;
  interestRank: number;
};

export type ScanMeta = {
  periodKey: string;
  asOf: string | null;
  newestAsOf: string | null;
  finishedAt: string | null;
  running: boolean;
  status: string | null;
};

export type RejectDetail = {
  barDate: string | null;
  close: number | null;
  criticalLevel: number | null;
  seqState: number | null;
  rr: number | null;
  sl: number | null;
  tp: number | null;
  minRr: number;
};

export type Rejection = {
  id: string;
  symbol: string;
  yahooTicker: string;
  reason: string;
  detail: RejectDetail | null;
};

export type ManualRun = {
  id: string;
  ticker: string;
  tf: UserTf;
  status: 'running' | 'completed' | 'cancelled' | 'failed';
  percent: number;
  message: string;
  signal: TrackedSignal | null;
  rejection: Rejection | null;
  finishedAt: string | null;
};

export type CachedBars = {
  yahooTicker: string;
  tf: UserTf;
  bars: import('../../engine/src/types.ts').OhlcBar[];
  fetchedAt: string;
  companyName: string | null;
};

export type BenchmarkCache = {
  symbol: string;
  bars: Array<{ date: string; close: number }>;
  fetchedAt: string;
};
