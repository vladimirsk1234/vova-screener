/**
 * Background scans. Results always show the latest of these; `runNow` is the one way the UI can
 * ask for another, and it starts the same pass the cron does.
 *
 * One hourly pass covers Stocks + ETF across Daily, Weekly and Monthly. Whether the tracker treats
 * a run as a period close is decided from the clock in `ScansService` (`isPeriodClosed`), so the
 * 16:05 / 17:05 ticks after the cash close are themselves the close scans — no separate close
 * crons. A catch-up runs at boot when the newest scan predates the current period.
 *
 * Passes are skipped rather than queued when the previous one is still going, so a slow hour
 * cannot snowball. The cadence can be retuned with `VOVA_SESSION_SCAN_CRON`, and background
 * scanning turns off entirely with `VOVA_BACKGROUND_SCANS=off`.
 */
import { Injectable, Logger, type OnApplicationBootstrap } from '@nestjs/common';
import { InjectModel } from '@nestjs/mongoose';
import { Cron } from '@nestjs/schedule';
import type { Model } from 'mongoose';
import type { Timeframe } from '@vova/engine';
import { SCAN_RUN } from '../db/schemas';
import { SettingsService } from '../settings/settings.module';
import { MARKET_TZ, periodKey } from './period';
import { ScansService } from './scans.service';
import type { ScanParamsApi } from './scan-runner.service';

const UNIVERSES = ['Stocks', 'ETF'] as const;
export const SCAN_TIMEFRAMES: readonly Timeframe[] = ['Daily', 'Weekly', 'Monthly'];

/** Every hour from 09:05 through 17:05 ET, Mon–Fri. Post-close ticks are the period-close scans. */
const PASS_CRON = process.env.VOVA_SESSION_SCAN_CRON || '5 9-17 * * 1-5';

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

export type ScanNowResult = {
  started: boolean;
  timeframes: Timeframe[];
  reason?: string;
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

  /** Hourly pass: all three timeframes, both universes. Post-close ticks confirm and close. */
  @Cron(PASS_CRON, { timeZone: MARKET_TZ })
  hourlyPass() {
    if (this.busy) {
      this.log.warn('Hourly pass skipped — the previous pass is still running');
      return;
    }
    void this.enqueue(() => this.scanAll(SCAN_TIMEFRAMES), 'hourly pass');
  }

  /**
   * A scan asked for from the Settings sheet. It re-downloads every symbol rather than reusing
   * the hourly cache — the reason to press it is that the screen disagrees with the market — and
   * it runs even when the cron is switched off, because it was asked for by hand.
   *
   * Returns as soon as the pass is queued: a full universe takes minutes, and the Results header
   * already reports a scan in progress.
   */
  runNow(timeframes: readonly Timeframe[] = SCAN_TIMEFRAMES): ScanNowResult {
    if (this.busy) return { started: false, timeframes: [], reason: 'A scan is already running' };
    const tfs = timeframes.length ? [...timeframes] : [...SCAN_TIMEFRAMES];
    void this.enqueue(() => this.scanAll(tfs, true), 'manual rescan', true);
    return { started: true, timeframes: tfs };
  }

  /** Scans are serialised: two full universe passes at once would only fight over Yahoo. */
  private enqueue(job: () => Promise<void>, label: string, force = false) {
    if (!this.enabled && !force) return this.queue;
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
   * `barsMaxAgeHours: 0.5` is what makes an hourly cadence real: short enough that every symbol is
   * genuinely re-downloaded each pass, long enough that two passes running back to back do not
   * fetch the same series twice. Manual rescans pass `forceRefresh` so they never reuse the cache.
   * Whether the tracker treats the result as a period close is decided from the clock in
   * `ScansService`, not here.
   *
   * Universe-major order keeps the three timeframes of one universe updating together, so Results
   * for Stocks Daily / Weekly / Monthly do not lag each other by half a pass.
   *
   * A throttled pass degrades instead of failing — `BarsService` falls back to the cached series
   * when Yahoo answers 429 — so the cache/download split is logged as the signal to watch. If
   * `cached` climbs towards the symbol count hour after hour, the cadence is too aggressive for
   * this IP and `VOVA_SESSION_SCAN_CRON` should be widened.
   */
  private async scanAll(timeframes: readonly Timeframe[], forceRefresh = false) {
    const passStarted = Date.now();
    const { maxRiskUsd } = await this.settings.get();
    for (const source of UNIVERSES) {
      for (const tf of timeframes) {
        const started = Date.now();
        const { runId } = await this.scans.start(
          {
            ...BASE_PARAMS,
            source,
            tf,
            riskPerTrade: maxRiskUsd,
            forceRefresh,
            barsMaxAgeHours: forceRefresh ? 0 : 0.5,
          },
          { trigger: 'scheduled', wait: true },
        );
        const run = await this.runs.findById(runId).select('counters status').lean<any>().exec();
        const counters = run?.counters ?? {};
        this.log.log(
          `${source} ${tf}: ${run?.status} in ${Math.round((Date.now() - started) / 1000)}s · ` +
            `${counters.signals ?? 0} signals · ${counters.closes ?? 0} closes · ` +
            `${counters.fromCache ?? 0}/${counters.total ?? 0} cached`,
        );
      }
    }
    this.log.log(
      `Pass done in ${Math.round((Date.now() - passStarted) / 1000)}s · ` +
        `${UNIVERSES.length} universes × ${timeframes.length} timeframes`,
    );
  }

  /** Only scans the timeframes whose newest completed run is older than the current period. */
  private async catchUp() {
    const stale: Timeframe[] = [];
    for (const tf of SCAN_TIMEFRAMES) {
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
    await this.scanAll(stale);
  }
}
