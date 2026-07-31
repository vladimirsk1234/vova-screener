/** Yahoo Finance chart API client (same endpoints yfinance uses). */
import { Injectable, Logger } from '@nestjs/common';
import {
  collapseInProgressPeriodBars,
  dropIncompleteBars,
  fillLastBarOhlc,
  intervalAndPeriod,
  toOhlcSeries,
  type OhlcSeries,
  type Timeframe,
} from '@vova/engine';

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
      };
      meta?: { shortName?: string; longName?: string; exchangeName?: string; symbol?: string };
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

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

@Injectable()
export class YahooClient {
  private readonly log = new Logger(YahooClient.name);

  async fetchOhlc(
    yahooTicker: string,
    tf: Timeframe,
    opts: { signal?: AbortSignal; retries?: number } = {},
  ): Promise<{ bars: OhlcSeries | null; meta?: { companyName?: string; exchange?: string } }> {
    const { interval, range } = intervalAndPeriod(tf);
    const url =
      `https://query1.finance.yahoo.com/v8/finance/chart/${encodeURIComponent(yahooTicker)}` +
      `?interval=${interval}&range=${range}&includePrePost=false&events=div%2Csplit`;

    const retries = opts.retries ?? 2;
    for (let attempt = 0; attempt <= retries; attempt++) {
      if (opts.signal?.aborted) return { bars: null };
      try {
        const res = await fetch(url, {
          signal: opts.signal,
          headers: { 'User-Agent': UA, Accept: 'application/json' },
        });
        if (res.status === 429 || res.status >= 500) {
          await sleep(400 * (attempt + 1));
          continue;
        }
        if (!res.ok) return { bars: null };
        const json = (await res.json()) as ChartResponse;
        const result = json.chart?.result?.[0];
        if (!result?.timestamp?.length) return { bars: null };
        const quote = result.indicators?.quote?.[0];
        if (!quote) return { bars: null };
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
          meta: {
            companyName: result.meta?.longName || result.meta?.shortName,
            exchange: result.meta?.exchangeName,
          },
        };
      } catch (err) {
        if (opts.signal?.aborted) return { bars: null };
        if (attempt === retries) {
          this.log.debug(`${yahooTicker}: ${(err as Error).message}`);
          return { bars: null };
        }
        await sleep(300 * (attempt + 1));
      }
    }
    return { bars: null };
  }
}
