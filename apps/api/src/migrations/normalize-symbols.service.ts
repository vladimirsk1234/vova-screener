/**
 * Puts every tracked signal on one ticker format, and leaves one record per trade.
 *
 * A record used to be named by whichever code wrote it: a live scan stored the TradingView symbol
 * (`NASDAQ:LMAT`), the History rebuild stored the short one (`LMAT`) and the trade journal stored
 * whatever string the build of its day happened to use, along with a Yahoo company name. The lists
 * print that field as it stands, so one position could appear twice under two names and read as two
 * different tickers.
 *
 * The instrument list is the authority: `tvSymbol` is what TradingView needs, its short form is what
 * every screen shows, and `companyName` comes from the same line of the list file.
 *
 * The duplicates behind those pairs are real: the one-active-signal-per-symbol index only guards
 * `status: 'active'`, so a journal copy could be created next to a closed scan record of the same
 * trade. They are matched on (ticker, timeframe, universe, opening period) and the losing copy is
 * deleted — logged one line each, because these carry realized P&L.
 *
 * Safe to repeat: a second pass finds nothing to rename and nothing to drop.
 */
import { Injectable, Logger, OnModuleInit } from '@nestjs/common';
import { InjectModel } from '@nestjs/mongoose';
import { Types, type Model } from 'mongoose';
import { shortSymbol, type Timeframe } from '@vova/engine';
import { INSTRUMENT, TRACKED_SIGNAL } from '../db/schemas';

const BATCH = 500;

type TrackedDoc = {
  _id: Types.ObjectId;
  yahooTicker: string;
  symbol?: string;
  tvSymbol?: string;
  companyName?: string;
  tf: Timeframe;
  universe: string;
  status: 'active' | 'closed';
  imported?: boolean;
  openedAsOf?: string;
  entry?: number;
  exitDate?: string;
  pnlUsd?: number;
  updatedAt?: Date;
};

type ListedInstrument = {
  tvSymbol?: string;
  companyName?: string;
};

export type NormalizeSymbolsReport = {
  /** Records whose symbol, TradingView symbol or company name was rewritten. */
  normalized: number;
  /** Second copies of a trade that were deleted. */
  deduped: number;
};

@Injectable()
export class NormalizeSymbols implements OnModuleInit {
  private readonly log = new Logger(NormalizeSymbols.name);

  constructor(
    @InjectModel(TRACKED_SIGNAL) private readonly tracked: Model<any>,
    @InjectModel(INSTRUMENT) private readonly instruments: Model<any>,
  ) {}

  onModuleInit() {
    void this.run()
      .then((report) => {
      if (report.normalized || report.deduped) {
        this.log.log(
          `symbols: ${report.normalized} records renamed, ${report.deduped} duplicate trades dropped`,
        );
      }
      })
      .catch((err) => {
        this.log.error(`symbol normalization failed: ${(err as Error).message}`);
      });
  }

  async run(): Promise<NormalizeSymbolsReport> {
    const docs = await this.tracked
      .find({})
      .select(
        'yahooTicker symbol tvSymbol companyName tf universe status imported openedAsOf entry exitDate pnlUsd updatedAt',
      )
      .lean<TrackedDoc[]>()
      .exec();
    const report: NormalizeSymbolsReport = { normalized: 0, deduped: 0 };
    if (!docs.length) return report;

    const listed = await this.listedInstruments();
    const drop = duplicates(docs);
    const ops: any[] = [];

    for (const doc of docs) {
      if (drop.has(String(doc._id))) {
        ops.push({ deleteOne: { filter: { _id: doc._id } } });
        report.deduped += 1;
        this.log.warn(
          `duplicate dropped ${doc.yahooTicker}/${doc.tf}/${doc.universe} ` +
            `opened ${doc.openedAsOf ?? '—'} at ${doc.entry ?? '—'} ` +
            `exit ${doc.exitDate ?? '—'} pnl ${doc.pnlUsd ?? '—'} imported ${Boolean(doc.imported)}`,
        );
        continue;
      }
      const set = rename(doc, listed.get(doc.yahooTicker));
      if (!set) continue;
      ops.push({ updateOne: { filter: { _id: doc._id }, update: { $set: set } } });
      report.normalized += 1;
    }

    for (let i = 0; i < ops.length; i += BATCH) {
      await this.tracked.bulkWrite(ops.slice(i, i + BATCH), { ordered: false });
    }
    return report;
  }

  private async listedInstruments(): Promise<Map<string, ListedInstrument>> {
    const rows = await this.instruments
      .find({})
      .select('yahooTicker tvSymbol companyName')
      .lean<Array<{ yahooTicker: string; tvSymbol?: string; companyName?: string }>>()
      .exec();
    return new Map(
      rows.map((row) => [row.yahooTicker, { tvSymbol: row.tvSymbol, companyName: row.companyName }]),
    );
  }
}

/** Only the fields that actually change, so a second pass reports nothing to do. */
function rename(doc: TrackedDoc, listed: ListedInstrument | undefined): Record<string, string> | null {
  // A ticker that has left the list files keeps its own strings; all it needs is the short form.
  const tvSymbol = listed?.tvSymbol ?? doc.tvSymbol ?? doc.symbol ?? doc.yahooTicker;
  const symbol = shortSymbol(tvSymbol);
  const companyName = listed?.companyName ?? doc.companyName ?? symbol;

  const set: Record<string, string> = {};
  if (doc.symbol !== symbol) set.symbol = symbol;
  if (doc.tvSymbol !== tvSymbol) set.tvSymbol = tvSymbol;
  if (doc.companyName !== companyName) set.companyName = companyName;
  return Object.keys(set).length ? set : null;
}

function updatedMs(doc: TrackedDoc): number {
  const ms = doc.updatedAt ? new Date(doc.updatedAt).getTime() : 0;
  return Number.isFinite(ms) ? ms : 0;
}

/**
 * Ids of the copies to delete.
 *
 * A position is held one at a time per ticker and timeframe, so a trade is named by the bar it
 * started on — and, when it has ended, by the bar it ended on. Two records sharing either bar are
 * two copies of one trade, whichever code wrote them and whatever period each filed itself under.
 * The bars, rather than the periods, because a Monthly trade can close and another open inside one
 * month and those are two trades.
 *
 * The record a scan wrote wins over a journal copy — its entry and exit come from the bars the
 * close-scan replay reads — then a realized close wins over one still settling on the bar in
 * progress, then the one written most recently.
 */
function duplicates(docs: TrackedDoc[]): Set<string> {
  const drop = new Set<string>();
  for (const bar of ['openedAsOf', 'exitDate'] as const) {
    const groups = new Map<string, TrackedDoc[]>();
    for (const doc of docs) {
      const date = doc[bar];
      if (!date || drop.has(String(doc._id))) continue;
      const key = `${doc.yahooTicker}|${doc.tf}|${doc.universe}|${date}`;
      const group = groups.get(key);
      if (group) group.push(doc);
      else groups.set(key, [doc]);
    }

    for (const group of groups.values()) {
      if (group.length < 2) continue;
      const ranked = [...group].sort(
        (a, b) =>
          Number(Boolean(a.imported)) - Number(Boolean(b.imported)) ||
          Number(a.status !== 'closed') - Number(b.status !== 'closed') ||
          updatedMs(b) - updatedMs(a),
      );
      for (const doc of ranked.slice(1)) drop.add(String(doc._id));
    }
  }
  return drop;
}
