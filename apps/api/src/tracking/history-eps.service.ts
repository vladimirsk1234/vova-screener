/** Tag History trades with FMP EPS on the entry bar (`openedAsOf`). */
import { Injectable, Logger, ServiceUnavailableException } from '@nestjs/common';
import { InjectModel } from '@nestjs/mongoose';
import type { Model } from 'mongoose';
import { TRACKED_SIGNAL } from '../db/schemas';
import { FmpClient } from '../market/fmp.client';
import { enrichHistoryEps, type HistoryEpsEnrichResult } from './history-eps';

export type { HistoryEpsEnrichResult, EpsHit } from './history-eps';
export {
  EPS_UNKNOWN_STAMP,
  FMP_EPS_ENRICH_TICKER_GAP_MS,
  enrichRemaining,
  epsStampFromHit,
} from './history-eps';

@Injectable()
export class HistoryEpsService {
  private readonly log = new Logger(HistoryEpsService.name);
  private readonly tracked: Model<any>;
  private readonly fmp: FmpClient;

  constructor(
    @InjectModel(TRACKED_SIGNAL) tracked: Model<any>,
    fmp: FmpClient,
  ) {
    this.tracked = tracked;
    this.fmp = fmp;
  }

  async enrich(limit = 40, opts?: { tickerGapMs?: number }): Promise<HistoryEpsEnrichResult> {
    if (!this.fmp.configured()) {
      throw new ServiceUnavailableException(
        'FMP_API_KEY is not set. Add your Financial Modeling Prep key to tag History EPS at entry.',
      );
    }
    return enrichHistoryEps(this.tracked, this.fmp, {
      limit,
      tickerGapMs: opts?.tickerGapMs,
      log: this.log,
    });
  }
}
