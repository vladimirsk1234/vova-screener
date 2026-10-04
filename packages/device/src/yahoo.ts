/** Yahoo chart API — the same OHLC source the server uses. No FMP. */
import {
  collapseInProgressPeriodBars,
  dropIncompleteBars,
  fillLastBarOhlc,
  intervalAndPeriod,
  toOhlcSeries,
  type OhlcSeries,
  type Timeframe,
} from '../../engine/src/index.ts';

const UA =
  'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36';

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
        adjclose?: Array<{ adjclose?: (number | null)[] }>;
      };
      meta?: { shortName?: string; longName?: string };
    }>;
    error?: { description?: string } | null;
  };
};

function numArr(a: (number | null)[] | undefined, n: number): number[] {
  const out = new Array<number>(n).fill(Number.NaN);
  if (!a) return out;
  for (let i = 0; i < n; i++) {
    const v = a[i];
    out[i] = v == null ? Number.NaN : Number(v);
  }
  return out;
}

export function parseYahooChart(json: ChartResponse, tf: Timeframe): {
  bars: OhlcSeries | null;
  companyName: string | null;
} {
  const result = json.chart?.result?.[0];
  if (!result?.timestamp?.length) return { bars: null, companyName: null };
  const quote = result.indicators?.quote?.[0];
  if (!quote) return { bars: null, companyName: null };
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
  bars = collapseInProgressPeriodBars(bars, tf);
  return {
    bars: bars.length ? bars : null,
    companyName: result.meta?.longName || result.meta?.shortName || null,
  };
}

export function parseYahooDailyCloses(json: ChartResponse): Array<{ date: string; close: number }> | null {
  const result = json.chart?.result?.[0];
  if (!result?.timestamp?.length) return null;
  const adj = result.indicators?.adjclose?.[0]?.adjclose;
  const raw = result.indicators?.quote?.[0]?.close;
  const series = adj?.some((v) => v != null && Number.isFinite(Number(v))) ? adj : raw;
  if (!series) return null;
  const out: Array<{ date: string; close: number }> = [];
  for (let i = 0; i < result.timestamp.length; i++) {
    const close = series[i];
    if (close == null || !Number.isFinite(Number(close)) || Number(close) <= 0) continue;
    const date = new Date(result.timestamp[i] * 1000).toISOString().slice(0, 10);
    out.push({ date, close: Number(close) });
  }
  return out.length ? out : null;
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

async function getJson(
  url: string,
  opts: { signal?: AbortSignal; retries?: number; fetchImpl?: typeof fetch } = {},
): Promise<ChartResponse | null> {
  const retries = opts.retries ?? 2;
  const doFetch = opts.fetchImpl ?? fetch;
  for (let attempt = 0; attempt <= retries; attempt++) {
    if (opts.signal?.aborted) return null;
    try {
      const res = await doFetch(url, {
        signal: opts.signal,
        headers: { 'User-Agent': UA, Accept: 'application/json' },
      });
      if (res.status === 429 || res.status >= 500) {
        await sleep(400 * (attempt + 1));
        continue;
      }
      if (!res.ok) return null;
      return (await res.json()) as ChartResponse;
    } catch {
      if (opts.signal?.aborted) return null;
      if (attempt === retries) return null;
      await sleep(300 * (attempt + 1));
    }
  }
  return null;
}

export async function fetchYahooOhlc(
  yahooTicker: string,
  tf: Timeframe,
  opts: { signal?: AbortSignal; fetchImpl?: typeof fetch } = {},
): Promise<{ bars: OhlcSeries | null; companyName: string | null }> {
  const { interval, range } = intervalAndPeriod(tf);
  const url =
    `https://query1.finance.yahoo.com/v8/finance/chart/${encodeURIComponent(yahooTicker)}` +
    `?interval=${interval}&range=${range}&includePrePost=false&events=div%2Csplit`;
  const json = await getJson(url, opts);
  if (!json) return { bars: null, companyName: null };
  return parseYahooChart(json, tf);
}

/** Daily adjusted closes for the History S&P compare. Yahoo only — no FMP fallback. */
export async function fetchYahooDailyCloses(
  yahooTicker: string,
  opts: { range?: string; signal?: AbortSignal; fetchImpl?: typeof fetch } = {},
): Promise<Array<{ date: string; close: number }> | null> {
  const range = opts.range ?? 'max';
  const url =
    `https://query1.finance.yahoo.com/v8/finance/chart/${encodeURIComponent(yahooTicker)}` +
    `?interval=1d&range=${encodeURIComponent(range)}&includePrePost=false&events=div%2Csplit`;
  const json = await getJson(url, opts);
  if (!json) return null;
  return parseYahooDailyCloses(json);
}
