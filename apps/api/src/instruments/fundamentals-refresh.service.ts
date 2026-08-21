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
  partsInNy,
} from '../scans/period';
import { UniverseService } from '../universe/universe.service';
import { fundamentalsCatchUpKind } from './fundamentals-catchup';
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

  /** Value / Fundamentals screens call this so a skipped boot catch-up still starts. */
  kickIfNeeded() {
    void this.kickIfNeededAsync();
  }

  private async startBootRefresh() {
    try {
      await this.failInterruptedRuns();
      await this.fundamentals.invalidateUnscaledStore();
    } catch (err) {
      this.log.warn(
        `Unscaled fundamentals purge failed: ${err instanceof Error ? err.message : String(err)}`,
      );
    }
    await this.waitForUniverse();
    const ran = await this.catchUpOnBoot();
    await this.retryCatchUpAfterBoot(ran);
  }

  /** STOCK-TICKERS import is fire-and-forget — wait so we do not skip or complete 0 names. */
  private async waitForUniverse(maxMs = 90_000): Promise<number> {
    const started = Date.now();
    while (Date.now() - started < maxMs) {
      const n = (await this.universe.listActiveYahooTickers()).length;
      if (n > 0) return n;
      await sleep(2000);
    }
    return (await this.universe.listActiveYahooTickers()).length;
  }

  private async failInterruptedRuns() {
    const res = await this.runs
      .updateMany(
        { status: 'running' },
        { $set: { status: 'failed', finishedAt: new Date() } },
      )
      .exec();
    const n = res.modifiedCount ?? 0;
    if (n) this.log.warn(`Marked ${n} interrupted fundamentals runs as failed`);
  }

  /**
   * Empty/partial Mongo or a missed weekday EOD after 18:15 ET → run now, do not wait for
   * tomorrow's cron (Nest does not fire missed slots).
   */
  private async catchUpOnBoot(): Promise<boolean> {
    try {
      const decision = await this.shouldRunFullNow();
      if (!decision) {
        this.log.log('Fundamentals catch-up skipped — coverage ok and today EOD already done');
        return false;
      }
      this.log.log(`Fundamentals boot catch-up: ${decision}`);
      if (decision === 'missing') await this.runMissing({ trigger: 'boot' });
      else await this.runFull({ trigger: 'boot' });
      return true;
    } catch (err) {
      this.log.warn(
        `Fundamentals catch-up failed: ${err instanceof Error ? err.message : String(err)}`,
      );
      return false;
    }
  }

  private async kickIfNeededAsync() {
    try {
      if (this.busy || !this.fmp.configured()) return;
      const decision = await this.shouldRunFullNow();
      if (!decision) return;
      this.log.log(`Fundamentals catch-up kick: ${decision}`);
      if (decision === 'missing') await this.runMissing({ trigger: 'catch-up' });
      else await this.runFull({ trigger: 'catch-up' });
    } catch (err) {
      this.log.warn(
        `Fundamentals catch-up kick failed: ${err instanceof Error ? err.message : String(err)}`,
      );
    }
  }

  /** Universe import can finish after the first wait; do not start a second walk after one already ran. */
  private async retryCatchUpAfterBoot(alreadyRan: boolean) {
    if (alreadyRan) return;
    for (const delay of [15_000, 45_000, 120_000]) {
      await sleep(delay);
      if (this.busy) return;
      const decision = await this.shouldRunFullNow();
      if (!decision) continue;
      this.log.log(`Fundamentals delayed catch-up: ${decision}`);
      if (decision === 'missing') await this.runMissing({ trigger: 'catch-up' });
      else await this.runFull({ trigger: 'catch-up' });
      return;
    }
  }

  async shouldRunFullNow(now: Date = new Date()): Promise<'missing' | 'full' | null> {
    const [coverage, last, latest] = await Promise.all([
      this.fundamentals.coverageStats(),
      this.latestCompletedFullAt(),
      this.fundamentals.latestRefreshRun(),
    ]);
    const jobOpen = this.busy || latest?.status === 'running';
    const completedPassToday = Boolean(
      last && partsInNy(last).dateStr === partsInNy(now).dateStr,
    );
    return fundamentalsCatchUpKind({
      fmpConfigured: this.fmp.configured(),
      busy: jobOpen,
      universe: coverage.universe,
      complete: coverage.complete,
      pastEodSlot: isPastFundamentalsEodSlot(now),
      todayFullDone: isFullRunAfterTodaysClose(last, now),
      completedPassToday,
    });
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
      if (!tickers.length) {
        this.log.warn(`${kind} refresh skipped — ticker list is empty (trigger=${trigger})`);
        return counts;
      }
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
