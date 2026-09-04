/** SPY / ^GSPC period return for History. Yahoo first, FMP fallback; never throws. */
import { Injectable, Logger } from '@nestjs/common';
import { FmpClient } from '../market/fmp.client';
import { YahooClient } from '../market/yahoo.client';
import {
  benchmarkFromSeries,
  loadBenchmarkSeries,
  type BenchmarkReturn,
  type BenchmarkSeries,
} from './benchmark';

const CACHE_TTL_MS = 60 * 60 * 1000;
const FETCH_TIMEOUT_MS = 8_000;

@Injectable()
export class BenchmarkService {
  private readonly log = new Logger(BenchmarkService.name);
  private cache: { at: number; series: BenchmarkSeries | null } | null = null;
  private inflight: Promise<BenchmarkSeries | null> | null = null;
  private readonly yahoo: YahooClient;
  private readonly fmp: FmpClient;

  constructor(yahoo: YahooClient, fmp: FmpClient) {
    this.yahoo = yahoo;
    this.fmp = fmp;
  }

  async periodReturn(from: string | null, to: string | null): Promise<BenchmarkReturn | null> {
    try {
      const series = await this.series();
      return benchmarkFromSeries(series, from, to);
    } catch (err) {
      this.log.debug(`benchmark ${from}→${to}: ${(err as Error).message}`);
      return null;
    }
  }

  private async series(): Promise<BenchmarkSeries | null> {
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

  private async load(): Promise<BenchmarkSeries | null> {
    const yahoo = this.yahoo;
    const fmp = this.fmp;
    return loadBenchmarkSeries({
      fetchYahoo: (symbol, signal) =>
        this.withTimeout((abort) =>
          yahoo.fetchDailyCloses(symbol, { range: 'max', signal: signal ?? abort }),
        ),
      fmpConfigured: () => fmp.configured(),
      fetchFmp: (symbol, from) => fmp.historicalCloses(symbol, from),
    });
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
