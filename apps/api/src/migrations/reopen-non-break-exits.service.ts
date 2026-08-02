/**
 * Re-opens tracked signals that older builds closed on something other than a sell-to-close break.
 *
 * A trade now ends one way only: the close of a bar falling back through the critical level. TP and
 * SL are entry-time numbers, and a buy setup going quiet says nothing about a position that is
 * already on — but earlier builds closed on all three, so History carries realized P&L for trades
 * that were never actually exited. Left alone those rows keep skewing win rate and net P&L, and the
 * positions behind them stay invisible instead of running until they break.
 *
 * Imported journal trades ('manual') are somebody's real record of a real exit and are left alone.
 *
 * Safe to repeat: after a pass no record matches, and a symbol that has since been re-opened keeps
 * the newer record — the unique index allows one active signal per (ticker, timeframe, universe).
 */
import { Injectable, Logger, OnModuleInit } from '@nestjs/common';
import { InjectModel } from '@nestjs/mongoose';
import { Types, type Model } from 'mongoose';
import type { Timeframe } from '@vova/engine';
import { TRACKED_SIGNAL } from '../db/schemas';

const BATCH = 500;

/** Exits this app no longer takes. Everything closed on one of these was still an open trade. */
const NOT_A_BREAK = ['TP', 'SL', 'signal_lost'];

type ClosedDoc = {
  _id: Types.ObjectId;
  yahooTicker: string;
  tf: Timeframe;
  universe: string;
  openedAt?: Date;
};

export type ReopenReport = {
  /** Back to running, waiting for a break. */
  reopened: number;
  /** Superseded by a signal that re-opened after the bogus close, so the older row is dropped. */
  superseded: number;
};

@Injectable()
export class ReopenNonBreakExits implements OnModuleInit {
  private readonly log = new Logger(ReopenNonBreakExits.name);

  constructor(@InjectModel(TRACKED_SIGNAL) private readonly tracked: Model<any>) {}

  async onModuleInit() {
    try {
      const report = await this.run();
      if (report.reopened || report.superseded) {
        this.log.log(
          `non-break exits: ${report.reopened} trades re-opened, ${report.superseded} superseded`,
        );
      }
    } catch (err) {
      // The rows stay exactly where they were, so a failure here must not stop boot.
      this.log.error(`re-opening non-break exits failed: ${(err as Error).message}`);
    }
  }

  async run(): Promise<ReopenReport> {
    const docs = await this.tracked
      .find({ status: 'closed', exitReason: { $in: NOT_A_BREAK } })
      .select('yahooTicker tf universe openedAt')
      .sort({ openedAt: -1 })
      .lean<ClosedDoc[]>()
      .exec();
    if (!docs.length) return { reopened: 0, superseded: 0 };

    const taken = new Set(
      (
        await this.tracked
          .find({ status: 'active' })
          .select('yahooTicker tf universe')
          .lean<ClosedDoc[]>()
          .exec()
      ).map(key),
    );

    const report: ReopenReport = { reopened: 0, superseded: 0 };
    const ops: any[] = [];
    for (const doc of docs) {
      if (taken.has(key(doc))) {
        ops.push({ deleteOne: { filter: { _id: doc._id } } });
        report.superseded += 1;
        continue;
      }
      taken.add(key(doc));
      ops.push({
        updateOne: {
          filter: { _id: doc._id },
          update: {
            $set: { status: 'active', provisional: false, provisionalClose: false },
            $unset: {
              closedPeriodKey: '',
              closedAt: '',
              exitDate: '',
              exitPrice: '',
              exitReason: '',
              pnlUsd: '',
              pnlR: '',
              pnlPct: '',
              holdPeriods: '',
            },
          },
        },
      });
      report.reopened += 1;
    }

    for (let i = 0; i < ops.length; i += BATCH) {
      await this.tracked.bulkWrite(ops.slice(i, i + BATCH), { ordered: false });
    }
    return report;
  }
}

function key(doc: Pick<ClosedDoc, 'yahooTicker' | 'tf' | 'universe'>): string {
  return `${doc.yahooTicker}|${doc.tf}|${doc.universe}`;
}
