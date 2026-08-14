/** Tag History trades with FMP EPS on the entry bar (`openedAsOf`). */
import { Injectable, Logger, ServiceUnavailableException } from '@nestjs/common';
import { InjectModel } from '@nestjs/mongoose';
import type { Model } from 'mongoose';
import { TRACKED_SIGNAL } from '../db/schemas';
import { FmpClient, yahooToFmpSymbol } from '../market/fmp.client';

export type HistoryEpsEnrichResult = {
  configured: boolean;
  scanned: number;
  updated: number;
  skipped: number;
  errors: number;
  remaining: number;
};

@Injectable()
export class HistoryEpsService {
  private readonly log = new Logger(HistoryEpsService.name);

  constructor(
    @InjectModel(TRACKED_SIGNAL) private readonly tracked: Model<any>,
    private readonly fmp: FmpClient,
  ) {}

  async enrich(limit = 40): Promise<HistoryEpsEnrichResult> {
    if (!this.fmp.configured()) {
      throw new ServiceUnavailableException(
        'FMP_API_KEY is not set. Add your Financial Modeling Prep key to tag History EPS at entry.',
      );
    }
    const cap = Math.min(Math.max(limit, 1), 200);
    const pending = await this.tracked
      .find({
        openedAsOf: { $type: 'string', $ne: '' },
        epsPositiveAtEntry: { $exists: false },
      })
      .select('_id yahooTicker openedAsOf')
      .limit(cap)
      .lean<Array<{ _id: unknown; yahooTicker: string; openedAsOf: string }>>()
      .exec();

    const remainingBefore = await this.tracked
      .countDocuments({
        openedAsOf: { $type: 'string', $ne: '' },
        epsPositiveAtEntry: { $exists: false },
      })
      .exec();

    let updated = 0;
    let skipped = 0;
    let errors = 0;
    const epsCache = new Map<string, { eps: number | null; date: string | null }>();

    for (const doc of pending) {
      const asOf = doc.openedAsOf;
      const ticker = String(doc.yahooTicker || '').trim();
      if (!ticker || !/^\d{4}-\d{2}-\d{2}$/.test(asOf)) {
        skipped += 1;
        continue;
      }
      const cacheKey = `${ticker.toUpperCase()}|${asOf}`;
      try {
        let hit = epsCache.get(cacheKey);
        if (!hit) {
          hit = await this.fmp.epsAsOf(yahooToFmpSymbol(ticker), asOf);
          epsCache.set(cacheKey, hit);
        }
        const eps = hit.eps;
        await this.tracked.updateOne(
          { _id: doc._id },
          {
            $set: {
              epsAtEntry: eps,
              epsPositiveAtEntry: eps == null ? null : eps > 0,
              epsAtEntryAsOf: hit.date,
            },
          },
        );
        updated += 1;
      } catch (err) {
        errors += 1;
        this.log.warn(
          `EPS-at-entry failed for ${ticker} @ ${asOf}: ${(err as Error).message}`,
        );
      }
    }

    const remaining = Math.max(0, remainingBefore - updated - skipped);
    return {
      configured: true,
      scanned: pending.length,
      updated,
      skipped,
      errors,
      remaining,
    };
  }
}
