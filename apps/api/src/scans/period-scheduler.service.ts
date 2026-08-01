/**
 * Background scans. Results always show the latest of these — nothing in the UI starts a
 * universe scan.
 *
 * Two cadences: an hourly session refresh that surfaces signals appearing during the day, and a
 * scan right after each period closes. Only the latter confirms and closes tracked signals.
 * A catch-up runs at boot when the newest scan predates the current period.
 *
 * One hourly pass re-downloads Stocks + ETF across all three timeframes, so it is the heaviest
 * thing this service does — see `scanAll` for how throttling shows up. Session passes are skipped
 * rather than queued when the previous one is still going, so a slow hour cannot snowball. Every
 * cron can be retuned with VOVA_SESSION_SCAN_CRON / VOVA_*_CLOSE_CRON, and background scanning
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

/** Every hour of the cash session, 10:05 through 15:05 ET; the 16:15 close scan covers the last hour. */
const SESSION_CRON = process.env.VOVA_SESSION_SCAN_CRON || '5 10-15 * * 1-5';
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
  private busy = false;

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

  /** Hourly session pass: picks up signals that appear during the day as provisional. */
  @Cron(SESSION_CRON, { timeZone: MARKET_TZ })
  sessionRefresh() {
    if (this.busy) {
      this.log.warn('Session refresh skipped — the previous pass is still running');
      return;
    }
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
    this.busy = true;
    this.queue = this.queue
      .then(job)
      .catch((err) => this.log.error(`${label} failed: ${(err as Error).message}`))
      .finally(() => {
        this.busy = false;
      });
    return this.queue;
  }

  /**
   * `barsMaxAgeHours: 0.5` is what makes an hourly cadence real: it is short enough that every
   * symbol is genuinely re-downloaded each pass, and long enough that two passes running back to
   * back do not fetch the same series twice.
   *
   * A throttled pass degrades instead of failing — `BarsService` falls back to the cached series
   * when Yahoo answers 429 — so the cache/download split is logged as the signal to watch. If
   * `cached` climbs towards the symbol count hour after hour, the cadence is too aggressive for
   * this IP and `VOVA_SESSION_SCAN_CRON` should be widened.
   */
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
            barsMaxAgeHours: atPeriodClose ? 0 : 0.5,
          },
          { trigger: 'scheduled', wait: true },
        );
        const run = await this.runs.findById(runId).select('counters status').lean<any>().exec();
        const counters = run?.counters ?? {};
        this.log.log(
          `${tf} ${source}: ${run?.status} in ${Math.round((Date.now() - started) / 1000)}s · ` +
            `${counters.signals ?? 0} signals · ${counters.fromCache ?? 0}/${counters.total ?? 0} cached`,
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
