/**
 * Per-symbol scan evaluation — pure, mirrors the Streamlit scan decision path
 * (headless_scanner.run_scan) without any I/O.
 */
import {
  explainInvalidBuy,
  runSequenceVovaCloseScan,
  runSequenceVovaPine,
} from './sequenceVova';
import { buildChartUrl, inferTvSymbol, tfToTvInterval } from './tradingview';
import type { OhlcSeries, ScanDirection, Timeframe } from './types';

export const MIN_BARS = 50;
export const ATR_LEN = 14;

export type EvaluateParams = {
  direction: ScanDirection;
  minRr: number;
  riskPerTrade: number;
  noRrReq: boolean;
  useLastHlSl: boolean;
  newOnly: boolean;
  tf: Timeframe;
  minAvgVolume?: number;
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

export type Evaluation =
  | { status: 'signal'; signal: Signal }
  | { status: 'rejected'; reason: string }
  | { status: 'skipped'; reason: string };

function round2(n: number) {
  return Math.round(n * 100) / 100;
}

function finiteOrNull(n: number): number | null {
  return Number.isFinite(n) ? round2(n) : null;
}

function avgVolume(bars: OhlcSeries, lookback = 20): number {
  const slice = bars.slice(-lookback);
  if (!slice.length) return 0;
  const sum = slice.reduce((acc, b) => acc + (Number.isFinite(b.volume) ? b.volume : 0), 0);
  return sum / slice.length;
}

export function evaluateSymbol(input: {
  bars: OhlcSeries | null | undefined;
  yahooTicker: string;
  tvSymbol?: string;
  companyName?: string;
  params: EvaluateParams;
}): Evaluation {
  const { bars, yahooTicker, params } = input;
  if (!bars || !bars.length) return { status: 'rejected', reason: 'NO_DATA' };
  if (bars.length < MIN_BARS) return { status: 'rejected', reason: 'INSUFFICIENT_DATA' };

  if (params.minAvgVolume && avgVolume(bars) < params.minAvgVolume) {
    return { status: 'rejected', reason: 'LOW_VOL' };
  }

  const tvSymbol = input.tvSymbol || inferTvSymbol(yahooTicker);
  const companyName = input.companyName || yahooTicker;
  const tvUrl = buildChartUrl(tvSymbol, tfToTvInterval(params.tf));
  const asOf = bars[bars.length - 1].date;

  if (params.direction === 'sell') {
    const out = runSequenceVovaCloseScan(bars, {
      atr_len: ATR_LEN,
      min_rr: params.minRr,
      use_last_hl_sl: params.useLastHlSl,
      risk_dollars: params.riskPerTrade,
      no_rr_req: params.noRrReq,
    });
    if (!out || !out.Valid) return { status: 'rejected', reason: 'NO_CLOSE_SIGNAL' };
    if (params.newOnly && !out.New) return { status: 'skipped', reason: 'NOT_NEW' };

    const shares =
      Number.isFinite(out.position_size) && out.position_size >= 1
        ? Math.round(out.position_size)
        : 0;
    const entry = round2(out.entry_price);
    return {
      status: 'signal',
      signal: {
        kind: 'sell',
        symbol: tvSymbol,
        tvSymbol,
        yahooTicker,
        companyName,
        tvUrl,
        entry,
        exit: round2(out.exit_price),
        shares,
        rrAtEntry: finiteOrNull(out.entry_rr),
        rrAtClose: finiteOrNull(out.close_rr),
        invested: shares > 0 ? round2(entry * shares) : 0,
        pnlUsd: round2(out.pnl_dollars),
        pnlPct: round2(out.pnl_pct),
        isNew: Boolean(out.New),
        asOf,
      },
    };
  }

  const out = runSequenceVovaPine(bars, {
    atr_len: ATR_LEN,
    min_rr: params.minRr,
    use_last_hl_sl: params.useLastHlSl,
    risk_dollars: params.riskPerTrade,
    direction: 'buy',
    no_rr_req: params.noRrReq,
  });
  if (!out || !out.Valid) {
    return { status: 'rejected', reason: explainInvalidBuy(out, params.minRr, params.noRrReq) };
  }
  if (params.newOnly && !out.New) return { status: 'skipped', reason: 'NOT_NEW' };
  if (!out.last_peak_was_hh) return { status: 'rejected', reason: 'NO_HH_LAST_PEAK' };

  const shares =
    Number.isFinite(out.position_size) && out.position_size >= 1
      ? Math.round(out.position_size)
      : 0;
  return {
    status: 'signal',
    signal: {
      kind: 'buy',
      symbol: tvSymbol,
      tvSymbol,
      yahooTicker,
      companyName,
      tvUrl,
      entry: round2(out.Close),
      tp: round2(out.TP),
      sl: round2(out.SL),
      rr: finiteOrNull(out.RR),
      shares,
      positionValue: Number.isFinite(out.position_value) ? round2(out.position_value) : 0,
      isNew: Boolean(out.New),
      isStrong: Boolean(out.Strong),
      atr: Number.isFinite(out.ATR) ? round2(out.ATR) : 0,
      asOf,
    },
  };
}

export function buildSellSummary(signals: SellSignal[]): SellSummary | null {
  if (!signals.length) return null;
  const wins = signals.filter((s) => s.pnlUsd > 0).length;
  const invested = signals.reduce((a, s) => a + s.invested, 0);
  const pnlUsd = signals.reduce((a, s) => a + s.pnlUsd, 0);
  const entryRrs = signals.map((s) => s.rrAtEntry).filter((v): v is number => v != null);
  const closeRrs = signals.map((s) => s.rrAtClose).filter((v): v is number => v != null);
  const mean = (arr: number[]) => (arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0);
  return {
    count: signals.length,
    winRatePct: round2((wins / signals.length) * 100),
    shares: signals.reduce((a, s) => a + s.shares, 0),
    avgEntryRr: round2(mean(entryRrs)),
    avgCloseRr: round2(mean(closeRrs)),
    invested: round2(invested),
    pnlUsd: round2(pnlUsd),
    pnlPct: invested > 0 ? round2((pnlUsd / invested) * 100) : 0,
  };
}
