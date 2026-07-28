/** Run full scan orchestration on device. */
import {
  explainInvalidBuy,
  runSequenceVovaCloseScan,
  runSequenceVovaPine,
} from '../engine/sequenceVova';
import type {
  BuyRow,
  OhlcCacheEntry,
  RejectedRow,
  ResultRow,
  ScanParams,
  SellRow,
  Timeframe,
} from '../types';
import { resolveSourceTickers } from '../tickers/lists';
import { buildChartUrl, inferTvSymbol, tfToTvInterval } from '../tradingview';
import { downloadBatch } from '../yahoo/client';

const MIN_BARS = 50;
const ATR_LEN = 14;

export type ScanOutcome = {
  rows: ResultRow[];
  rejected: RejectedRow[];
  asOf: string | null;
  ohlcCache: Record<string, OhlcCacheEntry>;
  cancelled: boolean;
};

function round2(n: number) {
  return Math.round(n * 100) / 100;
}

function buildSellSummary(rows: SellRow[]): SellRow | null {
  const data = rows.filter((r) => !r._is_summary);
  if (!data.length) return null;
  const pnls = data.map((r) => r['P&L ($)']);
  const wins = pnls.filter((p) => p > 0).length;
  const totalPnl = pnls.reduce((a, b) => a + b, 0);
  const invested = data.reduce((a, r) => a + r['Invested ($)'], 0);
  const totalPnlPct = invested > 0 ? (totalPnl / invested) * 100 : 0;
  const avgEntryRr =
    data.reduce((a, r) => a + r['RR at Entry'], 0) / data.length;
  const avgCloseRr =
    data.reduce((a, r) => a + r['RR at Close'], 0) / data.length;
  return {
    Symbol: 'TOTAL',
    tv_symbol: 'TOTAL',
    'Company Name': `Win rate: ${((wins / data.length) * 100).toFixed(0)}%`,
    Entry: 0,
    Exit: 0,
    'Position Size (shares)': data.reduce((a, r) => a + r['Position Size (shares)'], 0),
    'RR at Entry': round2(avgEntryRr),
    'RR at Close': round2(avgCloseRr),
    'Invested ($)': round2(invested),
    'P&L ($)': round2(totalPnl),
    'P&L (%)': round2(totalPnlPct),
    yahoo_ticker: '',
    _is_summary: true,
  };
}

export async function runScan(
  params: ScanParams,
  opts: {
    signal?: AbortSignal;
    onProgress?: (p: {
      phase: 'download' | 'process';
      pct: number;
      message: string;
    }) => void;
  } = {},
): Promise<ScanOutcome> {
  const { tickers, tvByYahoo, nameByYahoo } = await resolveSourceTickers(
    params.source,
    params.manualTickers,
  );
  const isManual = params.source === 'MANUAL SCAN';
  const rows: ResultRow[] = [];
  const rejected: RejectedRow[] = [];
  const ohlcCache: Record<string, OhlcCacheEntry> = {};

  if (!tickers.length) {
    return { rows, rejected, asOf: null, ohlcCache, cancelled: false };
  }

  opts.onProgress?.({
    phase: 'download',
    pct: 0,
    message: `Downloading ${tickers.length} tickers…`,
  });

  const ohlcMap = await downloadBatch(tickers, params.tf, {
    chunkSize: 8,
    signal: opts.signal,
    onProgress: (done, total) => {
      opts.onProgress?.({
        phase: 'download',
        pct: Math.round((done / total) * 100),
        message: `Download ${done}/${total}`,
      });
    },
  });

  if (opts.signal?.aborted) {
    return { rows, rejected, asOf: null, ohlcCache, cancelled: true };
  }

  let asOf: string | null = null;
  let processed = 0;
  const total = tickers.length;

  for (const t of tickers) {
    if (opts.signal?.aborted) {
      return { rows, rejected, asOf, ohlcCache, cancelled: true };
    }
    processed += 1;
    opts.onProgress?.({
      phase: 'process',
      pct: Math.round((processed / total) * 100),
      message: `Process ${processed}/${total}`,
    });

    const bars = ohlcMap.get(t);
    if (!bars || bars.length < MIN_BARS) {
      if (isManual) rejected.push({ Symbol: t, Reason: bars ? 'INSUFFICIENT_DATA' : 'NO_DATA' });
      continue;
    }
    const lastDate = bars[bars.length - 1].date;
    if (!asOf || lastDate < asOf) asOf = lastDate;

    const tvSym = tvByYahoo[t] || inferTvSymbol(t);
    const company = nameByYahoo[t] || t;
    const interval = tfToTvInterval(params.tf);
    const tvUrl = buildChartUrl(tvSym, interval);

    try {
      if (params.scanDirection === 'sell') {
        const out = runSequenceVovaCloseScan(bars, {
          atr_len: ATR_LEN,
          min_rr: params.minRr,
          use_last_hl_sl: params.useLastHlSl,
          risk_dollars: params.riskPerTrade,
        });
        if (!out || !out.Valid) {
          if (isManual) rejected.push({ Symbol: t, Reason: 'NO_CLOSE_SIGNAL' });
          continue;
        }
        if (params.newOnly && !out.New) continue;
        let posSize = out.position_size;
        if (!Number.isFinite(posSize) || posSize < 1) posSize = 0;
        else posSize = Math.round(posSize);
        const entryPx = round2(out.entry_price);
        const invested = posSize > 0 ? round2(entryPx * posSize) : 0;
        const row: SellRow = {
          Symbol: tvUrl,
          tv_symbol: tvSym,
          'Company Name': company,
          Entry: entryPx,
          Exit: round2(out.exit_price),
          'Position Size (shares)': posSize,
          'RR at Entry': Number.isFinite(out.entry_rr) ? round2(out.entry_rr) : 0,
          'RR at Close': Number.isFinite(out.close_rr) ? round2(out.close_rr) : 0,
          'Invested ($)': invested,
          'P&L ($)': round2(out.pnl_dollars),
          'P&L (%)': round2(out.pnl_pct),
          yahoo_ticker: t,
        };
        rows.push(row);
        ohlcCache[tvSym] = { bars, tf: params.tf, symbol: tvSym, yahoo_ticker: t };
      } else {
        const out = runSequenceVovaPine(bars, {
          atr_len: ATR_LEN,
          min_rr: params.minRr,
          use_last_hl_sl: params.useLastHlSl,
          risk_dollars: params.riskPerTrade,
          direction: 'buy',
        });
        if (!out || !out.Valid) {
          if (isManual) {
            rejected.push({
              Symbol: t,
              Reason: explainInvalidBuy(out, params.minRr),
            });
          }
          continue;
        }
        if (params.newOnly && !out.New) continue;
        if (!out.last_peak_was_hh) {
          if (isManual) rejected.push({ Symbol: t, Reason: 'NO_HH_LAST_PEAK' });
          continue;
        }
        let posSize = out.position_size;
        if (!Number.isFinite(posSize) || posSize < 1) posSize = 0;
        else posSize = Math.round(posSize);
        const posValue = Number.isFinite(out.position_value) ? out.position_value : 0;
        const row: BuyRow = {
          Symbol: tvUrl,
          tv_symbol: tvSym,
          'Company Name': company,
          TP: round2(out.TP),
          SL: round2(out.SL),
          RR: round2(out.RR),
          'Position Size (shares)': posSize,
          'Position Value ($)': round2(posValue),
          New: out.New ? 1 : 0,
          Valid: out.Valid ? 1 : 0,
          Strong: out.Strong ? 1 : 0,
          yahoo_ticker: t,
        };
        rows.push(row);
        ohlcCache[tvSym] = { bars, tf: params.tf, symbol: tvSym, yahoo_ticker: t };
      }
    } catch (e) {
      const msg = e instanceof Error ? `${e.name}: ${e.message}` : String(e);
      rejected.push({
        Symbol: t,
        Reason: `ERROR: ${msg.slice(0, 197)}`,
      });
    }
  }

  if (params.scanDirection === 'sell') {
    const summary = buildSellSummary(rows as SellRow[]);
    if (summary) rows.push(summary);
  }

  return { rows, rejected, asOf, ohlcCache, cancelled: false };
}

export function defaultScanParams(): ScanParams {
  return {
    source: 'MANUAL SCAN',
    manualTickers: 'AAPL, TSLA, NVDA',
    riskPerTrade: 100,
    minRr: 1.5,
    scanDirection: 'buy',
    useLastHlSl: true,
    tf: 'Weekly' as Timeframe,
    newOnly: true,
  };
}
