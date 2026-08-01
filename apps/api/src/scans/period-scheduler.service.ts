/**
 * Background scans. Results always show the latest of these — nothing in the UI starts a
 * universe scan.
 *
 * Two cadences: a mid-session refresh that surfaces signals appearing during the day, and a
 * scan right after each period closes. Only the latter confirms and closes tracked signals.
 * A catch-up runs at boot when the newest scan predates the current period.
 *
 * One pass over Stocks + ETF is ~3k Yahoo requests, so the cadence is deliberately low. Both
 * crons can be retuned with VOVA_SESSION_SCAN_CRON / VOVA_*_CLOSE_CRON, and background scanning
 * turns off entirely with VOVA_BACKGROUND_SCANS=off.
 */
import { Injectable, Logger, type OnApplicationBootstrap } from '@nestjs/common';
import { InjectModel } from '@nestjs/mongoose';
import { Cron } from '@nestjs/schedule';
import type { Model } from 'mongoose';
import type { Timeframe } from '@vova/engine';
import { SCAN_RUN } from '../db/schemas';
import { SettingsService } from '../settings/settings.module';
import { isLastTradingDayOfMonth, MARKET_TZ, periodKey } from './period';
import { ScansService } from './scans.service';
import type { ScanParamsApi } from './scan-runner.service';

const UNIVERSES = ['Stocks', 'ETF'] as const;
const TIMEFRAMES: readonly Timeframe[] = ['Daily', 'Weekly', 'Monthly'];

const SESSION_CRON = process.env.VOVA_SESSION_SCAN_CRON || '5 12 * * 1-5';
const DAILY_CLOSE_CRON = process.env.VOVA_DAILY_CLOSE_CRON || '15 16 * * 1-5';
const WEEKLY_CLOSE_CRON = process.env.VOVA_WEEKLY_CLOSE_CRON || '20 16 * * 5';
const MONTHLY_CLOSE_CRON = process.env.VOVA_MONTHLY_CLOSE_CRON || '25 16 * * 1-5';

/**
 * `newOnly: false` matters: the tracker needs every still-valid signal to know which of its
 * positions are alive. `noRrReq: true` is the "MIN RR = any" rule — RR is a sort key, not a filter.
 */
const BASE_PARAMS: Omit<ScanParamsApi, 'source' | 'tf' | 'riskPerTrade'> = {
  direction: 'buy',
  minRr: 0,
  noRrReq: true,
  useLastHlSl: true,
  newOnly: false,
};

@Injectable()
export class PeriodSchedulerService implements OnApplicationBootstrap {
  private readonly log = new Logger(PeriodSchedulerService.name);
  private readonly enabled = process.env.VOVA_BACKGROUND_SCANS !== 'off';
  private queue: Promise<void> = Promise.resolve();

  constructor(
    @InjectModel(SCAN_RUN) private readonly runs: Model<any>,
    private readonly scans: ScansService,
    private readonly settings: SettingsService,
  ) {}

  onApplicationBootstrap() {
    if (!this.enabled) {
      this.log.warn('Background scans disabled (VOVA_BACKGROUND_SCANS=off)');
      return;
    }
    void this.enqueue(() => this.catchUp(), 'catch-up');
  }

  /** Mid-session pass: picks up signals that appear during the day as provisional. */
  @Cron(SESSION_CRON, { timeZone: MARKET_TZ })
  sessionRefresh() {
    void this.enqueue(() => this.scanAll(TIMEFRAMES, false), 'session refresh');
  }

  @Cron(DAILY_CLOSE_CRON, { timeZone: MARKET_TZ })
  dailyClose() {
    void this.enqueue(() => this.scanAll(['Daily'], true), 'daily close');
  }

  @Cron(WEEKLY_CLOSE_CRON, { timeZone: MARKET_TZ })
  weeklyClose() {
    void this.enqueue(() => this.scanAll(['Weekly'], true), 'weekly close');
  }

  @Cron(MONTHLY_CLOSE_CRON, { timeZone: MARKET_TZ })
  monthlyClose() {
    if (!isLastTradingDayOfMonth()) return;
    void this.enqueue(() => this.scanAll(['Monthly'], true), 'monthly close');
  }

  /** Scans are serialised: two full universe passes at once would only fight over Yahoo. */
  private enqueue(job: () => Promise<void>, label: string) {
    if (!this.enabled) return this.queue;
    this.queue = this.queue
      .then(job)
      .catch((err) => this.log.error(`${label} failed: ${(err as Error).message}`));
    return this.queue;
  }

  private async scanAll(timeframes: readonly Timeframe[], atPeriodClose: boolean) {
    const { maxRiskUsd } = await this.settings.get();
    for (const tf of timeframes) {
      for (const source of UNIVERSES) {
        const started = Date.now();
        const { runId } = await this.scans.start(
          {
            ...BASE_PARAMS,
            source,
            tf,
            riskPerTrade: maxRiskUsd,
            forceRefresh: atPeriodClose,
            barsMaxAgeHours: atPeriodClose ? 0 : 3,
          },
          { trigger: 'scheduled', wait: true },
        );
        this.log.log(
          `${tf} ${source}: run ${runId} took ${Math.round((Date.now() - started) / 1000)}s`,
        );
      }
    }
  }

  /** Only scans the timeframes whose newest completed run is older than the current period. */
  private async catchUp() {
    const stale: Timeframe[] = [];
    for (const tf of TIMEFRAMES) {
      const key = periodKey(tf);
      for (const source of UNIVERSES) {
        const latest = await this.runs
          .findOne({ 'params.source': source, periodTf: tf, lastCompletedAt: { $exists: true } })
          .sort({ periodKey: -1, lastCompletedAt: -1 })
          .select('periodKey')
          .lean<any>()
          .exec();
        if (latest?.periodKey !== key) {
          stale.push(tf);
          break;
        }
      }
    }
    if (!stale.length) {
      this.log.log('Catch-up skipped — every universe is scanned for the current period');
      return;
    }
    this.log.log(`Catch-up scanning ${stale.join(', ')}`);
    await this.scanAll(stale, false);
  }
}
