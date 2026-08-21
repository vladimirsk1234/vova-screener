/** REST client for @vova/api. Same-origin /api (Vite proxy in dev). */

import { parseApiErrorBody } from './apiError';

export type {
  AnnualFundamentalPoint,
  ValuationMetric,
  ValuationSeriesPoint,
  ValuationSummary,
} from '@vova/engine';

export type Timeframe = 'Daily' | 'Weekly' | 'Monthly';
export type HistoryTf = Timeframe | 'All';
/** Exit-date lookback on History. `max` is accepted by the API as an alias of `all`. */
export type HistoryRange = 'all' | 'ytd' | '1m' | '3m' | '6m' | '1y' | 'max';
export type Universe = 'Stocks' | 'ETF';
export type SourceLabel = Universe | 'MANUAL SCAN';
export type Direction = 'buy' | 'sell';
export type Bucket = 'new' | 'valid' | 'closed';
export type Interest = 'interested' | 'not_interested';
/**
 * A trade taken by this app only ever ends on `sell_to_close`. The rest belong to the imported
 * journal and to records written before the sell-to-close rule was the only one.
 */
export type ExitReason = 'TP' | 'SL' | 'sell_to_close' | 'signal_lost' | 'manual';

export const TIMEFRAMES = ['Daily', 'Weekly', 'Monthly'] as const satisfies readonly Timeframe[];
export const UNIVERSES = ['Stocks', 'ETF'] as const satisfies readonly Universe[];
export const BUCKETS = ['new', 'valid', 'closed'] as const satisfies readonly Bucket[];
export const HISTORY_RANGES = ['all', 'ytd', '1m', '3m', '6m', '1y'] as const satisfies readonly HistoryRange[];

export type ScanParams = {
  source: SourceLabel;
  manualTickers: string;
  tf: Timeframe;
  direction: Direction;
  minRr: number;
  riskPerTrade: number;
  noRrReq: boolean;
  useLastHlSl: boolean;
  newOnly: boolean;
  minAvgVolume?: number;
  maxSymbols?: number;
  forceRefresh?: boolean;
};

export type ScanRun = {
  _id: string;
  params: ScanParams;
  status: 'queued' | 'running' | 'completed' | 'cancelled' | 'failed';
  asOf: string | null;
  barsOldestAt?: string | null;
  periodKey?: string;
  periodTf?: Timeframe;
  trigger?: 'manual' | 'scheduled';
  counters: {
    total: number;
    downloaded: number;
    evaluated: number;
    signals: number;
    /** Sell-to-close breaks a buy pass found alongside the signals. */
    closes?: number;
    rejected: number;
    skipped: number;
    fromCache: number;
  };
  reasonCounts: Record<string, number>;
  timings: { totalMs: number };
  newSymbols: string[];
  error?: string;
  createdAt: string;
};

export type BuySignal = {
  kind: 'buy';
  symbol: string;
  tvSymbol: string;
  yahooTicker: string;
  companyName: string;
  tvUrl: string;
  entry: number;
  tp: number;
  sl: number;
  rr: number | null;
  shares: number;
  positionValue: number;
  isNew: boolean;
  isStrong: boolean;
  /** Bars of the scanned timeframe since the signal appeared: 0 means "on the latest bar". */
  barsSinceValid: number | null;
  validSinceAsOf: string | null;
  atr: number;
  asOf: string;
};

/** One tracked signal, as rendered by Results and History. */
export type ResultRow = {
  id: string;
  symbol: string;
  tvSymbol: string;
  yahooTicker: string;
  companyName: string;
  universe: Universe;
  tf: Timeframe;
  status: 'active' | 'closed';
  provisional: boolean;
  /** Sell-to-close break on the bar still running: in CLOSED now, in History once it finishes. */
  provisionalClose: boolean;
  entry: number;
  tp: number | null;
  sl: number | null;
  rr: number | null;
  currentRr: number | null;
  shares: number;
  positionValue: number;
  riskUsd: number;
  isStrong: boolean;
  openedPeriodKey: string;
  openedAsOf: string | null;
  /** Bars of `tf` since the signal appeared: 0 in NEW, 1 or more in VALID. */
  barsSinceValid: number | null;
  validSinceAsOf: string | null;
  lastPrice: number | null;
  lastSeenAsOf: string | null;
  /** Unrealized while active, realized once closed. */
  pnlUsd: number | null;
  pnlR: number | null;
  pnlPct: number | null;
  realized: boolean;
  closedPeriodKey: string | null;
  exitDate: string | null;
  exitPrice: number | null;
  exitReason: ExitReason | null;
  holdPeriods: number | null;
  interest: Interest | null;
  epsAtEntry: number | null;
  epsPositiveAtEntry: boolean | null;
};

export type ScanMeta = {
  /** Period of the newest scan that produced data — which period CLOSED reports on. */
  periodKey: string;
  asOf: string | null;
  /** Newest bar the scan saw. CLOSED reports on the period this falls in. */
  newestAsOf: string | null;
  finishedAt: string | null;
  running: boolean;
  status: string | null;
};

export type ResultSort = 'rr' | 'pnl' | 'interest' | 'symbol';
export type SortDir = 'asc' | 'desc';

export type ResultsPage = {
  universe: Universe;
  tf: Timeframe;
  bucket: Bucket;
  sort: ResultSort;
  dir: SortDir;
  total: number;
  rows: ResultRow[];
  scan: ScanMeta;
};

export type BucketCounts = { new: number; valid: number; closed: number };
export type ResultsSummary = Record<
  Universe,
  Record<Timeframe, { counts: BucketCounts; scan: ScanMeta }>
>;

export type HistoryPeriod = {
  periodKey: string;
  trades: number;
  wins: number;
  losses: number;
  winRatePct: number;
  pnlUsd: number;
  invested: number;
  avgR: number | null;
  avgRrEntry: number | null;
  avgHold: number | null;
  avgTradeSizeUsd: number | null;
  avgWinPct: number | null;
  avgLossPct: number | null;
  avgPnlPct: number | null;
};

export type EquityPoint = { periodKey: string; equity: number };

/** One timeframe's own record, reported whatever the filter above the page is set to. */
export type HistoryTimeframe = {
  tf: Timeframe;
  closed: number;
  wins: number;
  winRatePct: number;
  pnlUsd: number;
  invested: number;
  avgR: number | null;
  avgTradeSizeUsd: number | null;
  avgWinPct: number | null;
  avgLossPct: number | null;
  avgPnlPct: number | null;
  equity: EquityPoint[];
};

export type HistoryReport = {
  universe: Universe;
  tf: HistoryTf;
  groupBy: Timeframe;
  range: HistoryRange;
  holdUnit: string;
  periods: HistoryPeriod[];
  equity: EquityPoint[];
  timeframes: HistoryTimeframe[];
  exitReasons: Array<{ reason: string; count: number }>;
  totals: {
    closed: number;
    active: number;
    wins: number;
    losses: number;
    winRatePct: number;
    pnlUsd: number;
    invested: number;
    avgR: number | null;
    avgRrEntry: number | null;
    avgHold: number | null;
    avgTradeSizeUsd: number | null;
    avgWinPct: number | null;
    avgLossPct: number | null;
    avgPnlPct: number | null;
    maxRiskUsd: number;
    totalRiskUsd: number;
    profitToRisk: number | null;
  };
};

/** Progress of `POST /history/rebuild` — poll until status is done/failed. */
export type HistoryRebuildStatus = {
  status: 'idle' | 'running' | 'done' | 'failed';
  startedAt: string | null;
  finishedAt: string | null;
  error: string | null;
  progress: {
    universe: Universe | null;
    tf: Timeframe | null;
    symbolsDone: number;
    symbolsTotal: number;
  };
  counts: {
    inserted: number;
    skipped: number;
    noBars: number;
    symbols: number;
  };
};

export type HorizonReturns = {
  y1: number | null;
  y3: number | null;
  y5: number | null;
  y10: number | null;
};

export type EstimateRow = {
  year: number;
  date: string;
  eps: number | null;
  epsChgPct: number | null;
  dividend: number | null;
  analysts: number | null;
  estimated: boolean;
};

export type PerformanceYear = {
  year: number;
  tickerClose: number | null;
  spyClose: number | null;
  tickerRetPct: number | null;
  spyRetPct: number | null;
  eps: number | null;
  epsChgPct: number | null;
};

export type FundamentalsPayload = {
  provider: 'fmp';
  yahooTicker: string;
  fmpSymbol: string;
  tvSymbol: string;
  profile: {
    companyName: string;
    currency: string | null;
    exchange: string | null;
    sector: string | null;
    industry: string | null;
    description: string | null;
    mktCap: number | null;
    price: number | null;
    beta: number | null;
    country: string | null;
    website: string | null;
    image: string | null;
  };
  scale?: {
    version: number;
    reportedCurrency: string | null;
    listingCurrency: string | null;
    fxToListing: number;
    adrRatio: number;
    shareScale: 'ordinary' | 'ads' | 'double_adr' | 'unknown';
    perShareFactor: number;
    reliable: boolean;
  } | null;
  snapshot: {
    peTTM: number | null;
    ttmEps: number | null;
    ttmAsOf?: string | null;
    ttmFcf?: number | null;
    pbTTM: number | null;
    psTTM: number | null;
    pegTTM: number | null;
    roeTTM: number | null;
    roicTTM: number | null;
    dividendYieldTTM: number | null;
    earningsYieldTTM: number | null;
    blendedPe: number | null;
    fwdPe: number | null;
    fwdEps: number | null;
    tev: number | null;
    ltDebtToCapitalTTM: number | null;
    spCreditRating: null;
    estAnnualRorPct: number | null;
    futurePrice: number | null;
    debtToEquityTTM: number | null;
    currentRatioTTM: number | null;
    profitMarginTTM: number | null;
    operatingMarginTTM: number | null;
    fcfYieldTTM: number | null;
    dcf: number | null;
    dcfPremiumPct: number | null;
    altmanZScore: number | null;
    piotroskiScore: number | null;
    nextEarningsDate?: string | null;
  };
  valuation: {
    series: import('@vova/engine').ValuationSeriesPoint[];
    summary: import('@vova/engine').ValuationSummary;
  };
  forecastSeries: import('@vova/engine').ValuationSeriesPoint[];
  estimates: EstimateRow[];
  performance: {
    price: HorizonReturns;
    spy: HorizonReturns;
    eps: HorizonReturns;
    years: PerformanceYear[];
  };
  annual: import('@vova/engine').AnnualFundamentalPoint[];
  quarters?: Array<{ date: string; eps: number | null; fcfPerShare?: number | null }>;
  incomeTrend: Array<{
    year: number;
    date: string;
    revenue: number | null;
    netIncome: number | null;
    eps: number | null;
    epsChgPct: number | null;
    dividend: number | null;
    operatingCashFlow: number | null;
    freeCashFlow: number | null;
  }>;
  asOf: string;
  cached: boolean;
};

export type CustomDcfAssumptions = {
  revenueGrowthPct?: number;
  ebitdaPct?: number;
  operatingCashFlowPct?: number;
  capitalExpenditurePct?: number;
  longTermGrowthRate?: number;
  taxRate?: number;
  riskFreeRate?: number;
  marketRiskPremium?: number;
  costOfEquity?: number;
  costOfDebt?: number;
};

export type CustomDcfYear = {
  year: number;
  revenue: number | null;
  ebitda: number | null;
  ebit: number | null;
  depreciation: number | null;
  capitalExpenditure: number | null;
  ufcf: number | null;
  pvUfcf: number | null;
};

export type CustomDcfPayload = {
  yahooTicker: string;
  fmpSymbol: string;
  model: 'unlevered';
  price: number | null;
  equityValuePerShare: number | null;
  premiumPct: number | null;
  enterpriseValue: number | null;
  equityValue: number | null;
  netDebt: number | null;
  terminalValue: number | null;
  presentTerminalValue: number | null;
  sumPvUfcf: number | null;
  dilutedShares: number | null;
  wacc: number | null;
  beta: number | null;
  costOfEquity: number | null;
  costOfDebt: number | null;
  afterTaxCostOfDebt: number | null;
  taxRate: number | null;
  riskFreeRate: number | null;
  marketRiskPremium: number | null;
  debtWeighting: number | null;
  equityWeighting: number | null;
  longTermGrowthRate: number | null;
  revenueGrowthPct: number | null;
  ebitdaPct: number | null;
  capitalExpenditurePct: number | null;
  operatingCashFlowPct: number | null;
  years: CustomDcfYear[];
  fragile: boolean;
  terminalSharePct: number | null;
  asOf: string;
  cached: boolean;
};

/** Slim valuation fields for Results / History signal cards (FMP). */
export type CardFundamentals = {
  fairValue: number | null;
  /** (price − fairValue) / fairValue × 100. Same number the Settings valuation filter uses. */
  premiumPct: number | null;
  growthRatePct: number | null;
  blendedPe: number | null;
  ltDebtToCapitalTTM: number | null;
};

export type SeqStructStatus = {
  seq: number;
  seqEmoji: string;
  structEmoji: string;
  structLabel: string;
};

export type ValueScreenerRow = {
  yahooTicker: string;
  symbol: string;
  tvSymbol: string;
  companyName: string;
  stars: number;
  epsPremiumPct: number | null;
  fcfPremiumPct: number | null;
  dcfPremiumPct: number | null;
  epsFairValue: number | null;
  fcfFairValue: number | null;
  dcfFairValue: number | null;
  bestPremiumPct: number | null;
  growthRatePct: number | null;
  blendedPe: number | null;
  ltDebtToCapitalTTM: number | null;
  interest: Interest | null;
  interestRank: number;
  ta: {
    daily?: SeqStructStatus | null;
    weekly?: SeqStructStatus | null;
    monthly?: SeqStructStatus | null;
  };
};

export type ValueStarsFilter = 'undervalued' | '0' | '1' | '2' | '3' | 'all';
export type ValueScreenerSort = 'stars' | 'eps' | 'fcf' | 'dcf' | 'symbol' | 'interest';

export type TickerInterest = {
  yahooTicker: string;
  interest: Interest | null;
  interestRank: number;
};

export type ValueScreenerPage = {
  rows: ValueScreenerRow[];
  total: number;
  counts: { all: number; undervalued: number; 0: number; 1: number; 2: number; 3: number };
  coverage: { universe: number; stored: number; reliable: number; complete: number };
  lastFullAt: string | null;
  lastRun: {
    kind: string;
    trigger: string;
    status: string;
    startedAt: string | null;
    finishedAt: string | null;
    total: number;
    done: number;
    ok: number;
    skip: number;
    fail: number;
  } | null;
};

export type HistoryEpsEnrichResult = {
  configured: boolean;
  scanned: number;
  updated: number;
  skipped: number;
  errors: number;
  remaining: number;
};

/** Tracked signals whose ticker has since left its universe list file. */
export type DelistedSummary = {
  symbols: number;
  records: number;
  closed: number;
  active: number;
  sample: string[];
};

export type HistoryPeriodSort = 'period' | 'pnl' | 'winRate' | 'trades' | 'rr';
export type HistoryTradeSort = 'date' | 'pnl' | 'r' | 'rr' | 'interest' | 'symbol';

export type FundamentalsFilter = 'all' | 'undervalued' | 'overvalued';
export const FUNDAMENTALS_FILTERS = [
  'all',
  'undervalued',
  'overvalued',
] as const satisfies readonly FundamentalsFilter[];

export type AppSettings = {
  maxRiskUsd: number;
  minRr: number;
  fundamentalsFilter: FundamentalsFilter;
};

/** What `POST /scans/run-now` answers: the pass is queued, or one was already going. */
export type ScanNowResult = { started: boolean; timeframes: Timeframe[]; reason?: string };

/** Engine numbers behind a reject, for side-by-side comparison with TradingView. */
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
  _id: string;
  symbol: string;
  yahooTicker?: string;
  reason: string;
  detail?: RejectDetail | null;
};

export type Bar = {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
};

export type ChartSettings = {
  len_fast: number;
  len_slow: number;
  length_major: number;
  lookback: number;
  multiplier: number;
  bb_length: number;
  bb_mult: number;
  min_rr: number;
  no_rr_req: boolean;
  use_last_hl_sl: boolean;
  bg_color: string;
  paper_color: string;
  grid_color: string;
  candle_up: string;
  candle_down: string;
  candle_border: string;
  candle_wick: string;
  hhll_color: string;
  crit_stop_color_up: string;
  crit_stop_color_down: string;
  crit_custom_color: string;
  fib_color: string;
  fib_width: number;
  short_ema_color: string;
  center_ema_color: string;
  sma_major_color: string;
  bb_basis_color: string;
  bb_upper_color: string;
  bb_lower_color: string;
  bb_fill_color: string;
  env_upper_color: string;
  env_lower_color: string;
  wm_text_color: string;
  wm_font_size: number;
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

export type ChartDrawing = {
  id: string;
  type: 'trend' | 'ray' | 'hline' | 'vline' | 'fib' | 'text';
  points: Array<{ time: string; price: number }>;
  color?: string;
  text?: string;
};

export type ChartPayload = {
  yahooTicker: string;
  /** Display form, no exchange prefix. `tvSymbol` is only for the TradingView link. */
  symbol: string;
  tvSymbol: string;
  companyName: string;
  tf: Timeframe;
  /** Bar the series was cut at for a trade snapshot, null on the live chart. */
  asOf: string | null;
  bars: Bar[];
  overlay: {
    critical: (number | null)[];
    seqState: number[];
    bullishBreak: boolean[];
    bearishBreak: boolean[];
    lastPeak: number | null;
    lastTrough: number | null;
    tp: number | null;
    sl: number | null;
    rr: number | null;
    peaks: Array<{ idx: number; price: number; label: string }>;
    troughs: Array<{ idx: number; price: number; label: string }>;
    extensionLines: Array<{
      kind: 'high' | 'low';
      x0Idx: number;
      y0: number;
      x1Idx: number;
      y1: number;
      rawX0Idx: number;
      rawX1Idx: number;
    }>;
    fib: {
      high: number;
      highIdx: number;
      low: number;
      lowIdx: number;
      fib382: number;
      fib500: number;
      fib618: number;
    } | null;
    overlays: {
      emaFast: (number | null)[];
      emaSlow: (number | null)[];
      smaMajor: (number | null)[];
      envUpper: (number | null)[];
      envLower: (number | null)[];
      bbBasis: (number | null)[];
      bbUpper: (number | null)[];
      bbLower: (number | null)[];
    };
    impulseColors: string[];
    atrPct: number;
    adx: number;
    signalBarIndex: number | null;
    seqStateFinal: number;
    criticalLevel: number | null;
  } | null;
  pine: {
    valid: boolean;
    isNew: boolean;
    strong: boolean;
    /**
     * Bars since the signal appeared: 0 means "on the latest bar", exactly as in the Results tabs.
     * Measured with the RR requirement off, so `minRr` here cannot change it.
     */
    barsSinceValid: number | null;
    validSinceAsOf: string | null;
    tp: number | null;
    sl: number | null;
    rr: number | null;
    close: number;
    atr: number;
    lastPeakWasHh: boolean;
    lastTroughWasHl: boolean;
  } | null;
  watermark: {
    lines: string[];
    main: string;
    tradeLine: string;
    dwmLines: Partial<Record<'daily' | 'weekly' | 'monthly', string>>;
    description: string | null;
  } | null;
  params?: ChartSettings;
};

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`/api${path}`, {
    ...init,
    headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) },
  });
  if (!res.ok) {
    const text = await res.text().catch(() => '');
    throw new Error(parseApiErrorBody(res.status, text));
  }
  return res.json() as Promise<T>;
}

function query(params: Record<string, string | number | boolean | undefined>): string {
  const q = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== '') q.set(key, String(value));
  }
  const s = q.toString();
  return s ? `?${s}` : '';
}

export const api = {
  // Results — always the latest background scan, which Settings can also ask for by hand.
  results: (opts: {
    universe: Universe;
    tf: Timeframe;
    bucket: Bucket;
    sort?: ResultSort;
    dir?: SortDir;
    limit?: number;
    offset?: number;
  }) => request<ResultsPage>(`/results${query(opts)}`),
  resultsSummary: () => request<ResultsSummary>('/results/summary'),
  lookupSignal: (yahooTicker: string, tf: Timeframe) =>
    request<ResultRow | null>(`/results/lookup${query({ yahooTicker, tf })}`),
  /** One tracked signal by id, closed included — how the chart is opened on a trade from History. */
  signal: (id: string) => request<ResultRow>(`/results/signal/${encodeURIComponent(id)}`),
  setInterest: (id: string, interest: Interest | null) =>
    request<ResultRow>(`/results/${id}/interest`, {
      method: 'PATCH',
      body: JSON.stringify({ interest }),
    }),
  getTickerInterest: (ticker: string) =>
    request<TickerInterest>(`/instruments/${encodeURIComponent(ticker)}/interest`),
  setTickerInterest: (ticker: string, interest: Interest | null) =>
    request<TickerInterest>(`/instruments/${encodeURIComponent(ticker)}/interest`, {
      method: 'PATCH',
      body: JSON.stringify({ interest }),
    }),

  // History — statistics over closed tracked signals.
  history: (opts: {
    universe: Universe;
    tf: HistoryTf;
    groupBy: Timeframe;
    range?: HistoryRange;
    sort?: HistoryPeriodSort;
    dir?: SortDir;
  }) => request<HistoryReport>(`/history${query(opts)}`),
  historyTrades: (opts: {
    universe: Universe;
    tf: HistoryTf;
    groupBy?: Timeframe;
    periodKey?: string;
    range?: HistoryRange;
    sort?: HistoryTradeSort;
    dir?: SortDir;
    limit?: number;
    offset?: number;
    hideUnprofitable?: boolean;
  }) => request<{ total: number; rows: ResultRow[] }>(`/history/trades${query(opts)}`),
  /** Replay close-ledger over the bar cache and insert missing closed trades into History. */
  rebuildHistory: () =>
    request<{ started: boolean; reason?: string }>('/history/rebuild', { method: 'POST' }),
  historyRebuildStatus: () => request<HistoryRebuildStatus>('/history/rebuild'),
  enrichHistoryEps: (limit = 40) =>
    request<HistoryEpsEnrichResult>(`/history/enrich-eps${query({ limit })}`, { method: 'POST' }),

  // Settings
  settings: () => request<AppSettings>('/settings'),
  saveSettings: (patch: Partial<AppSettings>) =>
    request<AppSettings>('/settings', { method: 'PUT', body: JSON.stringify(patch) }),

  /**
   * Rescan the tracked universes now, re-downloading every symbol. Returns once the pass is
   * queued — a full universe takes minutes, and progress shows up in the Results header.
   */
  runScanNow: (tf: Timeframe | 'all' = 'all') =>
    request<ScanNowResult>('/scans/run-now', { method: 'POST', body: JSON.stringify({ tf }) }),

  // Manual scan
  startScan: (params: Partial<ScanParams>) =>
    request<{ runId: string; params: ScanParams }>('/scans', {
      method: 'POST',
      body: JSON.stringify(params),
    }),
  cancelScan: (runId: string) =>
    request<{ ok: boolean }>(`/scans/${runId}/cancel`, { method: 'POST' }),
  run: (runId: string) => request<ScanRun>(`/scans/${runId}`),
  signals: (runId: string, opts: { onlyStrong?: boolean; limit?: number } = {}) =>
    request<{ run: ScanRun; count: number; rows: BuySignal[]; newSymbols: string[] }>(
      `/scans/${runId}/signals${query({
        onlyStrong: opts.onlyStrong ? 'true' : undefined,
        limit: opts.limit ?? 300,
      })}`,
    ),
  rejections: (runId: string, limit = 500) =>
    request<{ rows: Rejection[]; reasonCounts: Record<string, number>; total: number }>(
      `/scans/${runId}/rejections?limit=${limit}`,
    ),
  resetHistory: () =>
    request<{ ok: boolean; deletedRuns: number; deletedSignals: number }>('/scans/history', {
      method: 'DELETE',
    }),
  /** Tracked signals left behind by tickers that are no longer in their universe list file. */
  delistedPreview: () => request<DelistedSummary>('/scans/delisted'),
  purgeDelisted: () =>
    request<DelistedSummary & { ok: boolean; deletedSignals: number }>('/scans/delisted', {
      method: 'DELETE',
    }),

  // Charts / universe / chart presets
  /**
   * `riskUsd` is the global Max risk setting — position size has one source everywhere.
   * `asOf` cuts the series at that bar, which is what makes the chart behind a closed trade a
   * snapshot of the trade rather than a view of today.
   */
  chart: (
    ticker: string,
    tf: Timeframe,
    params?: Partial<ChartSettings>,
    riskUsd?: number,
    asOf?: string | null,
    fullSeries?: boolean,
  ) => {
    const q = new URLSearchParams({ tf });
    if (riskUsd != null && Number.isFinite(riskUsd)) q.set('riskPerTrade', String(riskUsd));
    if (asOf) q.set('asOf', asOf);
    if (fullSeries) q.set('fullSeries', '1');
    if (params) {
      const num: Array<[string, number | undefined]> = [
        ['minRr', params.min_rr],
        ['lenFast', params.len_fast],
        ['lenSlow', params.len_slow],
        ['lengthMajor', params.length_major],
        ['lookback', params.lookback],
        ['multiplier', params.multiplier],
        ['bbLength', params.bb_length],
        ['bbMult', params.bb_mult],
      ];
      for (const [key, value] of num) {
        if (value != null && Number.isFinite(value)) q.set(key, String(value));
      }
      if (params.use_last_hl_sl != null) q.set('useLastHlSl', String(params.use_last_hl_sl));
      if (params.no_rr_req != null) q.set('noRrReq', String(params.no_rr_req));
    }
    return request<ChartPayload>(`/instruments/${encodeURIComponent(ticker)}/chart?${q.toString()}`);
  },
  fundamentals: (ticker: string, metric: import('@vova/engine').ValuationMetric = 'eps') =>
    request<FundamentalsPayload>(
      `/instruments/${encodeURIComponent(ticker)}/fundamentals${query({ metric })}`,
    ),
  /** Unlevered Custom DCF. Rates as decimals (0.08 = 8%). Empty object = FMP defaults. */
  customDcf: (ticker: string, assumptions: CustomDcfAssumptions = {}) =>
    request<CustomDcfPayload>(
      `/instruments/${encodeURIComponent(ticker)}/dcf${query(assumptions)}`,
    ),
  /** Batch slim valuation for signal cards. Empty `{}` when FMP key is missing. */
  fundamentalsCards: (tickers: string[]) => {
    const unique = [...new Set(tickers.map((t) => t.trim().toUpperCase()).filter(Boolean))];
    if (!unique.length) return Promise.resolve({} as Record<string, CardFundamentals>);
    return request<Record<string, CardFundamentals>>(
      `/instruments/fundamentals-cards${query({ tickers: unique.join(',') })}`,
    );
  },
  fundamentalsRefresh: () =>
    request<{
      coverage: ValueScreenerPage['coverage'];
      lastRun: ValueScreenerPage['lastRun'];
      lastFullAt: string | null;
    }>('/instruments/fundamentals-refresh'),
  fundamentalsScreener: (opts: {
    stars?: ValueStarsFilter;
    sort?: ValueScreenerSort;
    dir?: SortDir;
    limit?: number;
    offset?: number;
  }) => request<ValueScreenerPage>(`/instruments/fundamentals-screener${query(opts)}`),
  universeSummary: () => request<{ stocks: number; etf: number; total: number }>('/universe/summary'),
  getPreset: <T>(key: string) => request<T>(`/presets/${key}`),
  putPreset: (key: string, data: unknown) =>
    request<{ ok: boolean }>(`/presets/${key}`, { method: 'PUT', body: JSON.stringify(data) }),
};

export type ScanProgressEvent = {
  runId: string;
  phase: 'queued' | 'resolving' | 'scanning' | 'saving' | 'completed' | 'cancelled' | 'failed';
  percent: number;
  message: string;
  counters?: Record<string, number>;
};
