/** Nightly price refresh + weekly full FMP pull into instrumentFundamentals. */
import { Injectable, Logger, type OnApplicationBootstrap } from '@nestjs/common';
import { Cron } from '@nestjs/schedule';
import { FmpClient } from '../market/fmp.client';
import { MARKET_TZ } from '../scans/period';
import { UniverseService } from '../universe/universe.service';
import { FundamentalsService } from './fundamentals.service';

const PRICE_CRON = process.env.VOVA_FUNDAMENTALS_CRON || '15 2 * * *';
const FULL_CRON = process.env.VOVA_FUNDAMENTALS_FULL_CRON || '15 3 * * 0';
const GAP_MS = 80;

function sleep(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

@Injectable()
export class FundamentalsRefreshService implements OnApplicationBootstrap {
  private readonly log = new Logger(FundamentalsRefreshService.name);
  private busy = false;

  constructor(
    private readonly fundamentals: FundamentalsService,
    private readonly universe: UniverseService,
    private readonly fmp: FmpClient,
  ) {}

  onApplicationBootstrap() {
    if (!this.fmp.configured()) {
      this.log.warn('FMP_API_KEY is not set — fundamentals refresh is off');
      return;
    }
    void this.fundamentals.invalidateUnscaledStore().catch((err) => {
      this.log.warn(
        `Unscaled fundamentals purge failed: ${err instanceof Error ? err.message : String(err)}`,
      );
    });
    void this.catchUpIfEmpty();
  }

  /** First deploy: fill Mongo once so ticker pages do not each hit FMP. */
  private async catchUpIfEmpty() {
    try {
      const tickers = await this.universe.listActiveYahooTickers();
      if (!tickers.length) return;
      const sample = await this.fundamentals.storedCount();
      if (sample > 0) return;
      this.log.log(`instrumentFundamentals empty — starting full backfill of ${tickers.length} names`);
      await this.runFull();
    } catch (err) {
      this.log.warn(
        `Fundamentals catch-up failed: ${err instanceof Error ? err.message : String(err)}`,
      );
    }
  }

  @Cron(PRICE_CRON, { timeZone: MARKET_TZ })
  dailyPrice() {
    if (this.busy) {
      this.log.warn('Price refresh skipped — a fundamentals job is still running');
      return;
    }
    void this.runPrice();
  }

  @Cron(FULL_CRON, { timeZone: MARKET_TZ })
  weeklyFull() {
    if (this.busy) {
      this.log.warn('Full refresh skipped — a fundamentals job is still running');
      return;
    }
    void this.runFull();
  }

  async runPrice(): Promise<{ ok: number; skip: number; fail: number }> {
    return this.walk('price', (ticker) => this.fundamentals.refreshPrice(ticker));
  }

  async runFull(): Promise<{ ok: number; skip: number; fail: number }> {
    return this.walk('full', (ticker) => this.fundamentals.refreshFull(ticker));
  }

  private async walk(
    kind: 'price' | 'full',
    fn: (ticker: string) => Promise<boolean>,
  ): Promise<{ ok: number; skip: number; fail: number }> {
    if (!this.fmp.configured()) return { ok: 0, skip: 0, fail: 0 };
    if (this.busy) return { ok: 0, skip: 0, fail: 0 };
    this.busy = true;
    const counts = { ok: 0, skip: 0, fail: 0 };
    try {
      const tickers = await this.universe.listActiveYahooTickers();
      this.log.log(`${kind} refresh starting for ${tickers.length} tickers`);
      for (const ticker of tickers) {
        try {
          const did = await fn(ticker);
          if (did) counts.ok += 1;
          else counts.skip += 1;
        } catch (err) {
          counts.fail += 1;
          this.log.warn(
            `${kind} refresh failed for ${ticker}: ${err instanceof Error ? err.message : String(err)}`,
          );
        }
        await sleep(GAP_MS);
      }
      this.log.log(`${kind} refresh done: ok=${counts.ok} skip=${counts.skip} fail=${counts.fail}`);
    } finally {
      this.busy = false;
    }
    return counts;
  }
}
