/** REST client for @vova/api. Same-origin /api (Vite proxy in dev). */

export type Timeframe = 'Daily' | 'Weekly' | 'Monthly';
export type SourceLabel = 'Stocks' | 'ETF' | 'MANUAL SCAN';
export type Direction = 'buy' | 'sell';

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
  periodKey?: string;
  periodTf?: Timeframe;
  trigger?: 'manual' | 'scheduled';
  counters: {
    total: number;
    downloaded: number;
    evaluated: number;
    signals: number;
    rejected: number;
    skipped: number;
    fromCache: number;
  };
  reasonCounts: Record<string, number>;
  timings: { totalMs: number };
  newSymbols: string[];
  summary: SellSummary | null;
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
  atr: number;
  asOf: string;
};

export type SellSignal = {
  kind: 'sell';
  symbol: string;
  tvSymbol: string;
  yahooTicker: string;
  companyName: string;
  tvUrl: string;
  entry: number;
  exit: number;
  shares: number;
  rrAtEntry: number | null;
  rrAtClose: number | null;
  invested: number;
  pnlUsd: number;
  pnlPct: number;
  isNew: boolean;
  asOf: string;
};

export type Signal = BuySignal | SellSignal;

export type SellSummary = {
  count: number;
  winRatePct: number;
  shares: number;
  avgEntryRr: number;
  avgCloseRr: number;
  invested: number;
  pnlUsd: number;
  pnlPct: number;
};

export type Trade = {
  _id: string;
  symbol: string;
  yahooTicker: string;
  companyName?: string;
  tf: Timeframe;
  openedAt: string;
  asOf?: string;
  entry: number;
  tp?: number;
  sl?: number;
  rrAtEntry?: number;
  shares: number;
  riskUsd: number;
  status: 'open' | 'closed' | 'dismissed';
  source?: 'auto' | 'manual';
  periodKey?: string;
  exitPrice?: number;
  exitDate?: string;
  exitReason?: string;
  pnlUsd?: number;
  pnlR?: number;
  currentPrice?: number | null;
  unrealizedUsd?: number | null;
  unrealizedR?: number | null;
};

export type PerformanceReport = {
  tf: Timeframe;
  periods: Array<{
    periodKey: string;
    trades: number;
    wins: number;
    winRatePct: number;
    pnlUsd: number;
    avgR: number | null;
  }>;
  equity: Array<{ date: string; equity: number }>;
  totals: {
    closed: number;
    open: number;
    wins: number;
    winRatePct: number;
    pnlUsd: number;
    avgR: number | null;
  };
};

export type MonthlyReport = {
  months: Array<{
    month: string;
    trades: number;
    wins: number;
    winRatePct: number;
    pnlUsd: number;
    avgR: number | null;
  }>;
  equity: Array<{ date: string; equity: number }>;
  totals: {
    closed: number;
    open: number;
    wins: number;
    winRatePct: number;
    pnlUsd: number;
    avgR: number | null;
  };
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
  risk_dollars: number;
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
  wm_text_color: string;
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
  tvSymbol: string;
  companyName: string;
  tf: Timeframe;
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
    throw new Error(`${res.status} ${res.statusText}${text ? `: ${text}` : ''}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  health: () =>
    request<{ ok: boolean; universe: { stocks: number; etf: number; total: number }; cache: { series: number; bars: number } }>(
      '/health',
    ),
  startScan: (params: Partial<ScanParams>) =>
    request<{ runId: string; params: ScanParams }>('/scans', {
      method: 'POST',
      body: JSON.stringify(params),
    }),
  cancelScan: (runId: string) => request<{ ok: boolean }>(`/scans/${runId}/cancel`, { method: 'POST' }),
  runs: (opts: { limit?: number; tf?: Timeframe } | number = 30) => {
    const limit = typeof opts === 'number' ? opts : (opts.limit ?? 30);
    const tf = typeof opts === 'number' ? undefined : opts.tf;
    const q = new URLSearchParams({ limit: String(limit) });
    if (tf) q.set('tf', tf);
    return request<ScanRun[]>(`/scans?${q.toString()}`);
  },
  resetHistory: () =>
    request<{ ok: boolean; deletedRuns: number }>('/scans/history', { method: 'DELETE' }),
  run: (runId: string) => request<ScanRun>(`/scans/${runId}`),
  signals: (runId: string, opts: { onlyNew?: boolean; onlyStrong?: boolean; limit?: number } = {}) => {
    const q = new URLSearchParams();
    if (opts.onlyNew) q.set('onlyNew', 'true');
    if (opts.onlyStrong) q.set('onlyStrong', 'true');
    q.set('limit', String(opts.limit ?? 300));
    return request<{ run: ScanRun; count: number; rows: Signal[]; newSymbols: string[] }>(
      `/scans/${runId}/signals?${q.toString()}`,
    );
  },
  rejections: (runId: string, limit = 500) =>
    request<{
      rows: Array<{ _id: string; symbol: string; reason: string }>;
      reasonCounts: Record<string, number>;
      total: number;
    }>(`/scans/${runId}/rejections?limit=${limit}`),
  chart: (ticker: string, tf: Timeframe, params?: Partial<ChartSettings>) => {
    const q = new URLSearchParams({ tf });
    if (params) {
      const num: Array<[string, number | undefined]> = [
        ['minRr', params.min_rr],
        ['riskPerTrade', params.risk_dollars],
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
    return request<ChartPayload>(
      `/instruments/${encodeURIComponent(ticker)}/chart?${q.toString()}`,
    );
  },
  status: (ticker: string) =>
    request<{
      yahooTicker: string;
      timeframes: Record<
        string,
        null | {
          asOf: string;
          seqState: number | null;
          lastPeakWasHh: boolean | null;
          lastTroughWasHl: boolean | null;
          valid: boolean;
          rr: number | null;
          close: number;
        }
      >;
    }>(`/instruments/${encodeURIComponent(ticker)}/status`),
  trades: (status?: 'open' | 'closed' | 'dismissed') =>
    request<Trade[]>(`/trades${status ? `?status=${status}` : ''}`),
  createTrade: (body: Record<string, unknown>) =>
    request<Trade>('/trades', { method: 'POST', body: JSON.stringify(body) }),
  closeTrade: (id: string, body: { exitPrice: number; exitReason?: string }) =>
    request<Trade>(`/trades/${id}/close`, { method: 'POST', body: JSON.stringify(body) }),
  dismissTrade: (id: string) =>
    request<Trade>(`/trades/${id}/dismiss`, { method: 'POST' }),
  deleteTrade: (id: string) => request<{ ok: boolean }>(`/trades/${id}`, { method: 'DELETE' }),
  refreshTrades: () =>
    request<{ checked: number; closed: number }>('/trades/refresh', { method: 'POST' }),
  monthly: () => request<MonthlyReport>('/reports/monthly'),
  performance: (tf: Timeframe = 'Daily') =>
    request<PerformanceReport>(`/reports/performance?tf=${encodeURIComponent(tf)}`),
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
