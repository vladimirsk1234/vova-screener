/** Tag trades with card fair-value premium as of `openedAsOf` (History UV/OV filter). */
import { Injectable, Logger } from '@nestjs/common';
import { InjectModel } from '@nestjs/mongoose';
import type { Model } from 'mongoose';
import { TRACKED_SIGNAL } from '../db/schemas';
import { FundamentalsService } from '../instruments/fundamentals.service';
import { stampFromResolved, stampIfResolvable } from '@vova/engine';

export type HistoryPremiumEnrichResult = {
  scanned: number;
  updated: number;
  skipped: number;
  remaining: number;
};

const PENDING = {
  openedAsOf: { $type: 'string', $ne: '' },
  premiumPctAtEntry: { $exists: false },
};

@Injectable()
export class HistoryPremiumService {
  private readonly log = new Logger(HistoryPremiumService.name);

  constructor(
    @InjectModel(TRACKED_SIGNAL) private readonly tracked: Model<any>,
    private readonly fundamentals: FundamentalsService,
  ) {}

  /**
   * Fill `premiumPctAtEntry` from stored instrumentFundamentals payload + entry price.
   * No payload or no as-of FV → write null (do not invent). Open-path stamp still
   * leaves the field unset so NEW/VALID can fall back to live until this job runs.
   */
  async enrich(limit = 80): Promise<HistoryPremiumEnrichResult> {
    const cap = Math.min(Math.max(limit, 1), 400);
    const pending = await this.tracked
      .find(PENDING)
      .select('_id yahooTicker openedAsOf entry')
      .limit(cap)
      .lean<Array<{ _id: unknown; yahooTicker: string; openedAsOf: string; entry?: number }>>()
      .exec();

    const remainingBefore = await this.tracked.countDocuments(PENDING).exec();
    const payloads = await this.fundamentals.loadPayloads(pending.map((d) => d.yahooTicker));

    let updated = 0;
    let skipped = 0;

    for (const doc of pending) {
      const asOf = doc.openedAsOf;
      const ticker = String(doc.yahooTicker || '').trim();
      if (!ticker || !/^\d{4}-\d{2}-\d{2}$/.test(asOf)) {
        await this.tracked.updateOne(
          { _id: doc._id },
          {
            $set: {
              premiumPctAtEntry: null,
              undervaluedAtEntry: null,
              premiumPctAtEntryAsOf: null,
            },
          },
        );
        updated += 1;
        continue;
      }
      const stamp =
        stampIfResolvable(asOf, doc.entry, payloads.get(ticker.toUpperCase()) ?? null) ??
        stampFromResolved({ premiumPct: null, asOf: null });
      if (stamp.premiumPctAtEntry == null) skipped += 1;
      await this.tracked.updateOne({ _id: doc._id }, { $set: stamp });
      updated += 1;
    }

    const remaining = Math.max(0, remainingBefore - updated);
    this.log.log(
      `premium-at-entry: scanned ${pending.length}, updated ${updated}, skipped ${skipped}, remaining ~${remaining}`,
    );
    return {
      scanned: pending.length,
      updated,
      skipped,
      remaining,
    };
  }
}
