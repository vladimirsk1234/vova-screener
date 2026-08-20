/**
 * Fills in the age of tracked signals that were written before the app recorded it.
 *
 * The NEW / VALID tabs split on `barsSinceValid`, and a record without it can only fall on the
 * VALID side — a whole timeframe of records written by an older build therefore shows up as "no new
 * signals at all" until a scan happens to touch it, which for Weekly and Monthly can be days away.
 * This runs on boot instead, off the bar cache, so the split is right immediately after a deploy.
 *
 * Safe to repeat: only records missing the age are read, and only the two age fields are written.
 * `lastSeenPeriodKey` is deliberately left alone — it says which scan priced the record, and a
 * migration is not a scan.
 */
import { Injectable, Logger, OnModuleInit } from '@nestjs/common';
import { InjectModel } from '@nestjs/mongoose';
import { Types, type Model } from 'mongoose';
import { signalAge, type OhlcSeries, type Timeframe } from '@vova/engine';
import { TRACKED_SIGNAL } from '../db/schemas';
import { BarsService } from '../market/bars.service';

const BATCH = 500;
const CONCURRENCY = 8;

type StaleDoc = {
  _id: Types.ObjectId;
  yahooTicker: string;
  tf: Timeframe;
  lastSeenAsOf?: string;
};

export type AgeBackfillReport = {
  /** Records that got an age from the cached bars. */
  filled: number;
  /** No cached bars, or no signal on the bar the record was last priced at. */
  skipped: number;
};

@Injectable()
export class SignalAgeBackfill implements OnModuleInit {
  private readonly log = new Logger(SignalAgeBackfill.name);

  constructor(
    @InjectModel(TRACKED_SIGNAL) private readonly tracked: Model<any>,
    private readonly bars: BarsService,
  ) {}

  onModuleInit() {
    void this.run()
      .then((report) => {
        if (report.filled || report.skipped) {
          this.log.log(
            `signal age backfill: ${report.filled} filled, ${report.skipped} left without an age`,
          );
        }
      })
      .catch((err) => {
        this.log.error(`signal age backfill failed: ${(err as Error).message}`);
      });
  }

  async run(): Promise<AgeBackfillReport> {
    const docs = await this.tracked
      .find({
        status: 'active',
        $or: [{ barsSinceValid: { $exists: false } }, { barsSinceValid: null }],
      })
      .select('yahooTicker tf lastSeenAsOf')
      .lean<StaleDoc[]>()
      .exec();
    if (!docs.length) return { filled: 0, skipped: 0 };

    const report: AgeBackfillReport = { filled: 0, skipped: 0 };
    const ops: any[] = [];
    const queue = [...docs];

    const worker = async () => {
      while (queue.length) {
        const doc = queue.shift();
        if (!doc) return;
        const op = await this.ageOp(doc);
        if (op) {
          ops.push(op);
          report.filled += 1;
        } else {
          report.skipped += 1;
        }
      }
    };
    await Promise.all(Array.from({ length: CONCURRENCY }, () => worker()));

    for (let i = 0; i < ops.length; i += BATCH) {
      await this.tracked.bulkWrite(ops.slice(i, i + BATCH), { ordered: false });
    }
    return report;
  }

  private async ageOp(doc: StaleDoc) {
    const cached = await this.bars.getCached(doc.yahooTicker, doc.tf);
    if (!cached?.length) return null;

    const age = signalAge(seriesAsScanned(cached, doc.lastSeenAsOf));
    // No signal on that bar means the record predates the bars we still hold, or its structure has
    // since broken. Either way it is not a signal that appeared on the current bar, so leaving it
    // without an age keeps it in VALID rather than guessing.
    if (age.barsSinceValid == null) return null;

    return {
      updateOne: {
        filter: { _id: doc._id },
        update: { $set: { barsSinceValid: age.barsSinceValid, validSinceAsOf: age.validSinceAsOf } },
      },
    };
  }
}

/**
 * The age has to be counted up to the bar the record was last priced at, not to whatever the cache
 * holds now: a record last seen on Friday would otherwise gain a bar of age the moment a newer bar
 * lands in the cache, and the tab would disagree with the price shown on the card.
 */
function seriesAsScanned(bars: OhlcSeries, lastSeenAsOf?: string): OhlcSeries {
  if (!lastSeenAsOf) return bars;
  const at = bars.findIndex((bar) => bar.date === lastSeenAsOf);
  return at >= 0 ? bars.slice(0, at + 1) : bars;
}
