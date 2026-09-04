/** SPY / ^GSPC period return for History. Yahoo first, FMP fallback; never throws. */
import { Injectable, Logger } from '@nestjs/common';
import { FmpClient } from '../market/fmp.client';
import { YahooClient } from '../market/yahoo.client';
import {
  BENCHMARK_SYMBOL,
  periodReturnPct,
  type BenchmarkReturn,
  type DailyClose,
} from './benchmark';

const CACHE_TTL_MS = 60 * 60 * 1000;
const FETCH_TIMEOUT_MS = 8_000;
const CANDIDATES = [BENCHMARK_SYMBOL, '^GSPC'] as const;

type Series = { symbol: string; bars: DailyClose[] };

@Injectable()
export class BenchmarkService {
  private readonly log = new Logger(BenchmarkService.name);
  private cache: { at: number; series: Series | null } | null = null;
  private inflight: Promise<Series | null> | null = null;

  constructor(
    private readonly yahoo: YahooClient,
    private readonly fmp: FmpClient,
  ) {}

  async periodReturn(from: string | null, to: string | null): Promise<BenchmarkReturn | null> {
    if (!from || !to || from > to) return null;
    try {
      const series = await this.series();
      if (!series?.bars.length) return null;
      const returnPct = periodReturnPct(series.bars, from, to);
      if (returnPct == null) return null;
      return { symbol: series.symbol, returnPct };
    } catch (err) {
      this.log.debug(`benchmark ${from}→${to}: ${(err as Error).message}`);
      return null;
    }
  }

  private async series(): Promise<Series | null> {
    if (this.cache && Date.now() - this.cache.at < CACHE_TTL_MS) return this.cache.series;
    if (this.inflight) return this.inflight;
    this.inflight = this.load()
      .catch((err) => {
        this.log.debug(`benchmark series: ${(err as Error).message}`);
        return null;
      })
      .then((series) => {
        this.cache = { at: Date.now(), series };
        return series;
      })
      .finally(() => {
        this.inflight = null;
      });
    return this.inflight;
  }

  private async load(): Promise<Series | null> {
    for (const symbol of CANDIDATES) {
      const bars = await this.withTimeout((signal) =>
        this.yahoo.fetchDailyCloses(symbol, { range: 'max', signal }),
      );
      if (bars?.length) return { symbol, bars };
    }
    if (this.fmp.configured()) {
      const bars = await this.fmp.historicalCloses(BENCHMARK_SYMBOL, '1993-01-22');
      if (bars.length) return { symbol: BENCHMARK_SYMBOL, bars };
    }
    return null;
  }

  private async withTimeout<T>(fn: (signal: AbortSignal) => Promise<T>): Promise<T | null> {
    const ac = new AbortController();
    const timer = setTimeout(() => ac.abort(), FETCH_TIMEOUT_MS);
    try {
      return await fn(ac.signal);
    } catch {
      return null;
    } finally {
      clearTimeout(timer);
    }
  }
}
