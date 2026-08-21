/**
 * Weekday EOD full FMP pull into instrumentFundamentals + boot catch-up when coverage is
 * incomplete or today's EOD slot was missed (cron does not fire retroactively).
 */
import { Injectable, Logger, type OnApplicationBootstrap } from '@nestjs/common';
import { InjectModel } from '@nestjs/mongoose';
import { Cron } from '@nestjs/schedule';
import type { Model } from 'mongoose';
import { FUNDAMENTALS_REFRESH_RUN } from '../db/schemas';
import { FmpClient } from '../market/fmp.client';
import {
  isFullRunAfterTodaysClose,
  isPastFundamentalsEodSlot,
  MARKET_TZ,
} from '../scans/period';
import { UniverseService } from '../universe/universe.service';
import { FundamentalsService } from './fundamentals.service';

const FULL_CRON = process.env.VOVA_FUNDAMENTALS_FULL_CRON || '15 18 * * 1-5';
const GAP_MS = 80;
const PROGRESS_EVERY = 10;

function sleep(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

@Injectable()
export class FundamentalsRefreshService implements OnApplicationBootstrap {
  private readonly log = new Logger(FundamentalsRefreshService.name);
  private busy = false;

  constructor(
    @InjectModel(FUNDAMENTALS_REFRESH_RUN) private readonly runs: Model<any>,
    private readonly fundamentals: FundamentalsService,
    private readonly universe: UniverseService,
    private readonly fmp: FmpClient,
  ) {}

  onApplicationBootstrap() {
    if (!this.fmp.configured()) {
      this.log.warn('FMP_API_KEY is not set — fundamentals refresh is off');
      return;
    }
    // Wipe must finish before coverage is measured — otherwise catch-up sees a full
    // store, skips, then invalidate empties every payload and nothing refills.
    void this.startBootRefresh();
  }

  private async startBootRefresh() {
    try {
      await this.fundamentals.invalidateUnscaledStore();
    } catch (err) {
      this.log.warn(
        `Unscaled fundamentals purge failed: ${err instanceof Error ? err.message : String(err)}`,
      );
    }
    await this.catchUpOnBoot();
  }

  /**
   * Empty/partial Mongo or a missed weekday EOD after 18:15 ET → run now, do not wait for
   * tomorrow's cron (Nest does not fire missed slots).
   */
  private async catchUpOnBoot() {
    try {
      const decision = await this.shouldRunFullNow();
      if (!decision) {
        this.log.log('Fundamentals catch-up skipped — coverage ok and today EOD already done');
        return;
      }
      this.log.log(`Fundamentals boot catch-up: ${decision}`);
      if (decision === 'missing') await this.runMissing({ trigger: 'boot' });
      else await this.runFull({ trigger: 'boot' });
    } catch (err) {
      this.log.warn(
        `Fundamentals catch-up failed: ${err instanceof Error ? err.message : String(err)}`,
      );
    }
  }

  async shouldRunFullNow(now: Date = new Date()): Promise<'missing' | 'full' | null> {
    if (!this.fmp.configured() || this.busy) return null;
    const coverage = await this.fundamentals.coverageStats();
    if (coverage.universe > 0 && coverage.complete < coverage.universe) return 'missing';

    if (!isPastFundamentalsEodSlot(now)) return null;
    const last = await this.latestCompletedFullAt();
    if (isFullRunAfterTodaysClose(last, now)) return null;
    return 'full';
  }

  private async latestCompletedFullAt(): Promise<Date | null> {
    const doc = await this.runs
      .findOne({ status: 'completed', kind: { $in: ['full', 'missing'] } })
      .sort({ finishedAt: -1 })
      .select('finishedAt')
      .lean<{ finishedAt?: Date }>()
      .exec();
    return doc?.finishedAt ? new Date(doc.finishedAt) : null;
  }

  /** Weekday after cash close: full pull of Stocks + ETF. */
  @Cron(FULL_CRON, { timeZone: MARKET_TZ })
  weekdayFull() {
    if (this.busy) {
      this.log.warn('Full refresh skipped — a fundamentals job is still running');
      return;
    }
    void this.runFull({ trigger: 'cron' });
  }

  async runFull(opts: { trigger?: 'cron' | 'boot' | 'catch-up' } = {}): Promise<{
    ok: number;
    skip: number;
    fail: number;
  }> {
    return this.walk('full', opts.trigger ?? 'cron', async () => {
      const tickers = await this.universe.listActiveYahooTickers();
      return tickers;
    });
  }

  /** Only names that still lack a full scaled payload. */
  async runMissing(opts: { trigger?: 'cron' | 'boot' | 'catch-up' } = {}): Promise<{
    ok: number;
    skip: number;
    fail: number;
  }> {
    return this.walk('missing', opts.trigger ?? 'catch-up', async () => {
      const tickers = await this.universe.listActiveYahooTickers();
      const complete = await this.fundamentals.completeTickerSet(tickers);
      return tickers.filter((t) => !complete.has(t));
    });
  }

  private async walk(
    kind: 'full' | 'missing',
    trigger: 'cron' | 'boot' | 'catch-up',
    resolveTickers: () => Promise<string[]>,
  ): Promise<{ ok: number; skip: number; fail: number }> {
    if (!this.fmp.configured()) return { ok: 0, skip: 0, fail: 0 };
    if (this.busy) return { ok: 0, skip: 0, fail: 0 };
    this.busy = true;
    const counts = { ok: 0, skip: 0, fail: 0 };
    let runId: string | null = null;
    try {
      const tickers = await resolveTickers();
      this.log.log(`${kind} refresh starting for ${tickers.length} tickers (trigger=${trigger})`);
      const startedAt = new Date();
      const created = await this.runs.create({
        kind,
        trigger,
        status: 'running',
        startedAt,
        total: tickers.length,
        done: 0,
        ok: 0,
        skip: 0,
        fail: 0,
      });
      runId = String(created._id);

      for (let i = 0; i < tickers.length; i++) {
        const ticker = tickers[i];
        try {
          const did = await this.fundamentals.refreshFull(ticker);
          if (did) counts.ok += 1;
          else counts.skip += 1;
        } catch (err) {
          counts.fail += 1;
          this.log.warn(
            `${kind} refresh failed for ${ticker}: ${err instanceof Error ? err.message : String(err)}`,
          );
        }
        if ((i + 1) % PROGRESS_EVERY === 0 || i + 1 === tickers.length) {
          await this.runs
            .updateOne(
              { _id: runId },
              {
                $set: {
                  done: i + 1,
                  ok: counts.ok,
                  skip: counts.skip,
                  fail: counts.fail,
                },
              },
            )
            .exec();
        }
        await sleep(GAP_MS);
      }

      await this.runs
        .updateOne(
          { _id: runId },
          {
            $set: {
              status: 'completed',
              finishedAt: new Date(),
              done: tickers.length,
              ok: counts.ok,
              skip: counts.skip,
              fail: counts.fail,
            },
          },
        )
        .exec();
      this.log.log(
        `${kind} refresh done: ok=${counts.ok} skip=${counts.skip} fail=${counts.fail}`,
      );
    } catch (err) {
      if (runId) {
        await this.runs
          .updateOne(
            { _id: runId },
            {
              $set: {
                status: 'failed',
                finishedAt: new Date(),
                ok: counts.ok,
                skip: counts.skip,
                fail: counts.fail,
              },
            },
          )
          .exec()
          .catch(() => undefined);
      }
      throw err;
    } finally {
      this.busy = false;
    }
    return counts;
  }
}
