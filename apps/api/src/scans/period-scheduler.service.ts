/** End-of-period scheduled scans: always re-run at close, journal new buys, close trades. */
import { Injectable, Logger } from '@nestjs/common';
import { Cron } from '@nestjs/schedule';
import type { Timeframe } from '@vova/engine';
import { TradesService } from '../trades/trades.service';
import { isLastTradingDayOfMonth, MARKET_TZ } from './period';
import { ScansService } from './scans.service';
import type { ScanParamsApi } from './scan-runner.service';

const SCHEDULED_BASE: Omit<ScanParamsApi, 'source' | 'tf'> = {
  direction: 'buy',
  minRr: 1.5,
  riskPerTrade: 100,
  noRrReq: false,
  useLastHlSl: true,
  newOnly: true,
  forceRefresh: true,
};

@Injectable()
export class PeriodSchedulerService {
  private readonly log = new Logger(PeriodSchedulerService.name);
  private queue: Promise<void> = Promise.resolve();

  constructor(
    private readonly scans: ScansService,
    private readonly trades: TradesService,
  ) {}

  @Cron('15 16 * * 1-5', { timeZone: MARKET_TZ })
  dailyEod() {
    void this.enqueue('Daily');
  }

  @Cron('20 16 * * 5', { timeZone: MARKET_TZ })
  weeklyEow() {
    void this.enqueue('Weekly');
  }

  @Cron('25 16 * * 1-5', { timeZone: MARKET_TZ })
  monthlyEom() {
    if (!isLastTradingDayOfMonth()) {
      this.log.debug('Skipping monthly job — not last trading day');
      return;
    }
    void this.enqueue('Monthly');
  }

  private enqueue(tf: Timeframe) {
    this.queue = this.queue
      .then(() => this.runPeriod(tf))
      .catch((err) => this.log.error(`Period job ${tf} failed: ${(err as Error).message}`));
    return this.queue;
  }

  async runPeriod(tf: Timeframe) {
    this.log.log(`Starting end-of-period ${tf} scans`);
    for (const source of ['Stocks', 'ETF'] as const) {
      const { runId } = await this.scans.start(
        { ...SCHEDULED_BASE, source, tf },
        { trigger: 'scheduled', wait: true },
      );
      const journaled = await this.trades.journalNewBuySignals(runId);
      this.log.log(`${tf} ${source}: run ${runId}, auto trades ${journaled.created}`);
    }
    const closed = await this.trades.refresh({ tf });
    this.log.log(`${tf} close-check: checked ${closed.checked}, closed ${closed.closed}`);
  }
}
