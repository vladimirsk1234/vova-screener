/** Yahoo Finance chart API client (same endpoints yfinance uses). */
import type { OhlcSeries, Timeframe } from '../types';
import { dropIncompleteBars, fillLastBarOhlc, intervalAndPeriod, toOhlcSeries } from '../engine/dataUtils';

const UA =
  'Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1';

type ChartResponse = {
  chart?: {
    result?: Array<{
      timestamp?: number[];
      indicators?: {
        quote?: Array<{
          open?: (number | null)[];
          high?: (number | null)[];
          low?: (number | null)[];
          close?: (number | null)[];
          volume?: (number | null)[];
        }>;
      };
      meta?: { shortName?: string; exchangeName?: string; symbol?: string };
    }>;
    error?: { description?: string } | null;
  };
};

function numArr(a: (number | null)[] | undefined, n: number): number[] {
  const out = new Array(n).fill(NaN);
  if (!a) return out;
  for (let i = 0; i < n; i++) {
    const v = a[i];
    out[i] = v == null ? NaN : Number(v);
  }
  return out;
}

export async function fetchYahooOhlc(
  yahooTicker: string,
  tf: Timeframe,
  signal?: AbortSignal,
): Promise<OhlcSeries | null> {
  const { interval, range } = intervalAndPeriod(tf);
  const url =
    `https://query1.finance.yahoo.com/v8/finance/chart/${encodeURIComponent(yahooTicker)}` +
    `?interval=${interval}&range=${range}&includePrePost=false&events=div%2Csplit`;

  const res = await fetch(url, {
    signal,
    headers: {
      'User-Agent': UA,
      Accept: 'application/json',
    },
  });
  if (!res.ok) return null;
  const json = (await res.json()) as ChartResponse;
  const result = json.chart?.result?.[0];
  if (!result?.timestamp?.length) return null;
  const quote = result.indicators?.quote?.[0];
  if (!quote) return null;
  const n = result.timestamp.length;
  let bars = toOhlcSeries(
    result.timestamp,
    numArr(quote.open, n),
    numArr(quote.high, n),
    numArr(quote.low, n),
    numArr(quote.close, n),
    numArr(quote.volume, n),
  );
  bars = fillLastBarOhlc(bars);
  bars = dropIncompleteBars(bars);
  return bars.length ? bars : null;
}

export async function fetchYahooMeta(
  yahooTicker: string,
  signal?: AbortSignal,
): Promise<{ companyName: string; exchange?: string } | null> {
  const url =
    `https://query1.finance.yahoo.com/v8/finance/chart/${encodeURIComponent(yahooTicker)}` +
    `?interval=1d&range=5d`;
  try {
    const res = await fetch(url, {
      signal,
      headers: { 'User-Agent': UA, Accept: 'application/json' },
    });
    if (!res.ok) return null;
    const json = (await res.json()) as ChartResponse;
    const meta = json.chart?.result?.[0]?.meta;
    if (!meta) return null;
    return {
      companyName: meta.shortName || yahooTicker,
      exchange: meta.exchangeName,
    };
  } catch {
    return null;
  }
}

/** Chunked sequential downloads with progress callback (phone-friendly). */
export async function downloadBatch(
  tickers: string[],
  tf: Timeframe,
  opts: {
    chunkSize?: number;
    signal?: AbortSignal;
    onProgress?: (done: number, total: number) => void;
  } = {},
): Promise<Map<string, OhlcSeries>> {
  const chunkSize = opts.chunkSize ?? 8;
  const out = new Map<string, OhlcSeries>();
  const total = tickers.length;
  let done = 0;
  for (let i = 0; i < tickers.length; i += chunkSize) {
    if (opts.signal?.aborted) break;
    const chunk = tickers.slice(i, i + chunkSize);
    await Promise.all(
      chunk.map(async (t) => {
        try {
          const bars = await fetchYahooOhlc(t, tf, opts.signal);
          if (bars && bars.length >= 50) out.set(t, bars);
        } catch {
          /* skip */
        } finally {
          done += 1;
          opts.onProgress?.(done, total);
        }
      }),
    );
    // gentle rate limit
    await new Promise((r) => setTimeout(r, 120));
  }
  return out;
}
