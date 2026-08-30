/**
 * Removes Daily as a product timeframe from Mongo so History/Results cannot resurrect it.
 *
 * Deletes `trackedSignals` with tf:'Daily', Daily scan runs and their per-run signals/rejections,
 * and unsets leftover Value TA daily snapshots. Does not touch Daily barSeries — Weekly/Monthly
 * charts still use those bars for the watermark daily change and D/W/M overlays.
 *
 * Safe to repeat: after a pass nothing matches, and a later scan/rebuild cannot insert Daily
 * because TIMEFRAMES no longer includes it.
 */
import { Injectable, Logger, OnModuleInit } from '@nestjs/common';
import { InjectModel } from '@nestjs/mongoose';
import type { Model } from 'mongoose';
import {
  INSTRUMENT_FUNDAMENTALS,
  REJECTION,
  SCAN_RUN,
  SIGNAL,
  TRACKED_SIGNAL,
} from '../db/schemas';

export type DropDailyReport = {
  trackedSignals: number;
  scanRuns: number;
  signals: number;
  rejections: number;
  taSnapshots: number;
};

const DAILY_RUN = {
  $or: [{ periodTf: 'Daily' }, { 'params.tf': 'Daily' }],
};

@Injectable()
export class DropDailyTimeframe implements OnModuleInit {
  private readonly log = new Logger(DropDailyTimeframe.name);

  constructor(
    @InjectModel(TRACKED_SIGNAL) private readonly tracked: Model<any>,
    @InjectModel(SCAN_RUN) private readonly runs: Model<any>,
    @InjectModel(SIGNAL) private readonly signals: Model<any>,
    @InjectModel(REJECTION) private readonly rejections: Model<any>,
    @InjectModel(INSTRUMENT_FUNDAMENTALS) private readonly fundamentals: Model<any>,
  ) {}

  onModuleInit() {
    void this.run()
      .then((report) => {
        const n =
          report.trackedSignals +
          report.scanRuns +
          report.signals +
          report.rejections +
          report.taSnapshots;
        if (n) {
          this.log.log(
            `dropped Daily timeframe: ${report.trackedSignals} tracked signals, ` +
              `${report.scanRuns} scan runs, ${report.signals} signals, ` +
              `${report.rejections} rejections, ${report.taSnapshots} TA snapshots`,
          );
        }
      })
      .catch((err) => {
        this.log.error(`Daily timeframe cleanup failed: ${(err as Error).message}`);
      });
  }

  async run(): Promise<DropDailyReport> {
    const dailyRuns = await this.runs.find(DAILY_RUN).select('_id').lean<{ _id: unknown }[]>().exec();
    const runIds = dailyRuns.map((r) => r._id);

    const [tracked, signals, rejections, runs, ta] = await Promise.all([
      this.tracked.deleteMany({ tf: 'Daily' }).exec(),
      runIds.length
        ? this.signals.deleteMany({ runId: { $in: runIds } }).exec()
        : Promise.resolve({ deletedCount: 0 }),
      runIds.length
        ? this.rejections.deleteMany({ runId: { $in: runIds } }).exec()
        : Promise.resolve({ deletedCount: 0 }),
      this.runs.deleteMany(DAILY_RUN).exec(),
      this.fundamentals
        .updateMany(
          { 'taSnapshot.daily': { $exists: true } },
          { $unset: { 'taSnapshot.daily': 1 } },
        )
        .exec(),
    ]);

    return {
      trackedSignals: tracked.deletedCount ?? 0,
      scanRuns: runs.deletedCount ?? 0,
      signals: signals.deletedCount ?? 0,
      rejections: rejections.deletedCount ?? 0,
      taSnapshots: ta.modifiedCount ?? 0,
    };
  }
}
