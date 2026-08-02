/**
 * Per-symbol scan evaluation — pure, mirrors the Streamlit scan decision path
 * (headless_scanner.run_scan) without any I/O.
 */
import { explainInvalidBuy, runCloseLedger, runSequenceVovaPine } from './sequenceVova';
import { signalAge } from './signalAge';
import { buildChartUrl, inferTvSymbol, tfToTvInterval } from './tradingview';
import type { CloseTrade, OhlcSeries, PineResult, ScanDirection, Timeframe } from './types';

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
  /**
   * Age of the signal in bars of `tf`: `0` means it appeared on the bar named by `asOf`. This is
   * what splits the NEW and VALID lists, so it holds for Daily, Weekly and Monthly alike and never
   * depends on the RR settings of the run that found the signal — see `signalAge`.
   */
  barsSinceValid: number | null;
  /** Date of the bar the signal appeared on. */
  validSinceAsOf: string | null;
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
  /** Bar the close scan's replay took this long on, and the numbers that bar priced it with. */
  entryAsOf: string;
  entrySl: number | null;
  entryTp: number | null;
  /** Bar the sequence broke back down on. Always the last bar of the series for a close signal. */
  exitAsOf: string;
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

/**
 * Numbers behind a reject, so a symbol can be compared with the live TradingView
 * chart without re-running the scan (the app evaluates a stored bar snapshot,
 * TradingView draws the in-progress bar).
 */
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

export type Evaluation =
  | { status: 'signal'; signal: Signal }
  | { status: 'rejected'; reason: string; detail?: RejectDetail }
  | { status: 'skipped'; reason: string };

function round2(n: number) {
  return Math.round(n * 100) / 100;
}

function finiteOrNull(n: number): number | null {
  return Number.isFinite(n) ? round2(n) : null;
}

function rejectDetail(
  bars: OhlcSeries,
  pine: PineResult | null,
  minRr: number,
): RejectDetail {
  const last = bars.length ? bars[bars.length - 1] : null;
  return {
    barDate: last?.date ?? null,
    close: pine ? finiteOrNull(pine.Close) : (last ? round2(last.close) : null),
    criticalLevel: pine ? finiteOrNull(pine.critical_level) : null,
    seqState: pine ? pine.seq_state : null,
    rr: pine ? finiteOrNull(pine.RR) : null,
    sl: pine ? finiteOrNull(pine.SL) : null,
    tp: pine ? finiteOrNull(pine.TP) : null,
    minRr,
  };
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
  if (bars.length < MIN_BARS) {
    return {
      status: 'rejected',
      reason: 'INSUFFICIENT_DATA',
      detail: rejectDetail(bars, null, params.minRr),
    };
  }

  if (params.minAvgVolume && avgVolume(bars) < params.minAvgVolume) {
    return { status: 'rejected', reason: 'LOW_VOL', detail: rejectDetail(bars, null, params.minRr) };
  }

  const tvSymbol = input.tvSymbol || inferTvSymbol(yahooTicker);
  const companyName = input.companyName || yahooTicker;
  const tvUrl = buildChartUrl(tvSymbol, tfToTvInterval(params.tf));
  const asOf = bars[bars.length - 1].date;

  if (params.direction === 'sell') {
    const closing = closingTrade(bars, params);
    if (!closing) {
      return {
        status: 'rejected',
        reason: 'NO_CLOSE_SIGNAL',
        detail: rejectDetail(bars, null, params.minRr),
      };
    }
    return {
      status: 'signal',
      signal: sellSignal(closing, { tvSymbol, yahooTicker, companyName, tvUrl, asOf }),
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
    return {
      status: 'rejected',
      reason: explainInvalidBuy(out, params.minRr, params.noRrReq),
      detail: rejectDetail(bars, out, params.minRr),
    };
  }
  if (params.newOnly && !out.New) return { status: 'skipped', reason: 'NOT_NEW' };
  if (!out.last_peak_was_hh) {
    return {
      status: 'rejected',
      reason: 'NO_HH_LAST_PEAK',
      detail: rejectDetail(bars, out, params.minRr),
    };
  }

  const shares =
    Number.isFinite(out.position_size) && out.position_size >= 1
      ? Math.round(out.position_size)
      : 0;
  const age = signalAge(bars);
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
      barsSinceValid: age.barsSinceValid,
      validSinceAsOf: age.validSinceAsOf,
      atr: Number.isFinite(out.ATR) ? round2(out.ATR) : 0,
      asOf,
    },
  };
}

/**
 * The long the close scan gives up on the last bar of `bars`, or nothing.
 *
 * This is the whole of "SELL TO CLOSE": the replay is over the bars alone, so a position is found
 * and priced from its own history rather than from whatever this app happened to have recorded
 * about the symbol.
 */
function closingTrade(bars: OhlcSeries, params: EvaluateParams): CloseTrade | null {
  const ledger = runCloseLedger(bars, {
    atr_len: ATR_LEN,
    min_rr: params.minRr,
    use_last_hl_sl: params.useLastHlSl,
    risk_dollars: params.riskPerTrade,
    no_rr_req: params.noRrReq,
  });
  const last = ledger?.trades[ledger.trades.length - 1];
  return last && last.exit_index === bars.length - 1 ? last : null;
}

function sellSignal(
  trade: CloseTrade,
  ids: {
    tvSymbol: string;
    yahooTicker: string;
    companyName: string;
    tvUrl: string;
    asOf: string;
  },
): SellSignal {
  const shares =
    Number.isFinite(trade.position_size) && trade.position_size >= 1
      ? Math.round(trade.position_size)
      : 0;
  const entry = round2(trade.entry_price);
  return {
    kind: 'sell',
    symbol: ids.tvSymbol,
    tvSymbol: ids.tvSymbol,
    yahooTicker: ids.yahooTicker,
    companyName: ids.companyName,
    tvUrl: ids.tvUrl,
    entry,
    exit: round2(trade.exit_price),
    shares,
    rrAtEntry: finiteOrNull(trade.entry_rr),
    rrAtClose: finiteOrNull(trade.close_rr),
    invested: shares > 0 ? round2(entry * shares) : 0,
    pnlUsd: round2(trade.pnl_dollars),
    pnlPct: round2(trade.pnl_pct),
    isNew: true,
    entryAsOf: trade.entry_date,
    entrySl: finiteOrNull(trade.entry_sl),
    entryTp: finiteOrNull(trade.entry_tp),
    exitAsOf: trade.exit_date ?? ids.asOf,
    asOf: ids.asOf,
  };
}

/**
 * The close signal for a symbol a buy scan is looking at anyway.
 *
 * A symbol closing today is by definition not a buy today — the break puts the sequence down — so
 * it leaves a buy scan as a reject and would never be heard of again. Running the close scan over
 * the same bars in the same pass is what lets the tracker see the trade end.
 */
export function evaluateClose(input: {
  bars: OhlcSeries | null | undefined;
  yahooTicker: string;
  tvSymbol?: string;
  companyName?: string;
  params: EvaluateParams;
}): SellSignal | null {
  const { bars, yahooTicker, params } = input;
  if (!bars || bars.length < MIN_BARS) return null;
  const trade = closingTrade(bars, params);
  if (!trade) return null;
  const tvSymbol = input.tvSymbol || inferTvSymbol(yahooTicker);
  return sellSignal(trade, {
    tvSymbol,
    yahooTicker,
    companyName: input.companyName || yahooTicker,
    tvUrl: buildChartUrl(tvSymbol, tfToTvInterval(params.tf)),
    asOf: bars[bars.length - 1].date,
  });
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
