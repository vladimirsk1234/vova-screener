/**
 * Rebuild History from the close-scan ledger over cached bars.
 *
 * Live scans only adopt a break on the newest bar. This pass walks every closed trade
 * `runCloseLedger` finds in the Yahoo window already in `barSeries` and inserts missing closes
 * into `trackedSignals`, so the History tab can show the full replay rather than only what the
 * app happened to catch while running.
 */
import { Injectable, Logger } from '@nestjs/common';
import { InjectModel } from '@nestjs/mongoose';
import type { Model } from 'mongoose';
import {
  runCloseLedger,
  shortSymbol,
  stampIfResolvable,
  type CloseTrade,
  type Timeframe,
} from '@vova/engine';
import { TRACKED_SIGNAL } from '../db/schemas';
import type { FundamentalsPayload } from '../instruments/fundamentals.service';
import { FundamentalsService } from '../instruments/fundamentals.service';
import { BarsService } from '../market/bars.service';
import { barPeriodKey } from '../scans/period';
import { SettingsService } from '../settings/settings.module';
import { UniverseService } from '../universe/universe.service';
import {
  TIMEFRAMES,
  UNIVERSES,
  computePnl,
  holdPeriods,
  round2,
  sharesFromRisk,
  type TrackedUniverse,
} from './tracked-signal';

const LEDGER_CONCURRENCY = 8;
const BULK_CHUNK = 500;

export type RebuildStatus = {
  status: 'idle' | 'running' | 'done' | 'failed';
  startedAt: string | null;
  finishedAt: string | null;
  error: string | null;
  /** How far the job has got, for the Settings sheet. */
  progress: {
    universe: TrackedUniverse | null;
    tf: Timeframe | null;
    symbolsDone: number;
    symbolsTotal: number;
  };
  counts: {
    inserted: number;
    skipped: number;
    noBars: number;
    symbols: number;
  };
};

type SymbolRow = {
  yahooTicker: string;
  symbol: string;
  tvSymbol: string;
  companyName: string;
};

@Injectable()
export class HistoryRebuildService {
  private readonly log = new Logger(HistoryRebuildService.name);
  private busy = false;
  private state: RebuildStatus = emptyStatus();

  constructor(
    @InjectModel(TRACKED_SIGNAL) private readonly tracked: Model<any>,
    private readonly bars: BarsService,
    private readonly universe: UniverseService,
    private readonly settings: SettingsService,
    private readonly fundamentals: FundamentalsService,
  ) {}

  status(): RebuildStatus {
    return { ...this.state, progress: { ...this.state.progress }, counts: { ...this.state.counts } };
  }

  /**
   * Kick off a background rebuild. Returns immediately; poll `status()` for progress.
   * Does not delete existing rows — only fills gaps from the ledger.
   */
  start(): { started: boolean; reason?: string } {
    if (this.busy) {
      return { started: false, reason: 'A history rebuild is already running' };
    }
    this.busy = true;
    this.state = {
      ...emptyStatus(),
      status: 'running',
      startedAt: new Date().toISOString(),
    };
    void this.run()
      .then(() => {
        this.state = {
          ...this.state,
          status: 'done',
          finishedAt: new Date().toISOString(),
          progress: { universe: null, tf: null, symbolsDone: 0, symbolsTotal: 0 },
        };
        this.log.log(
          `history rebuild done: +${this.state.counts.inserted} inserted, ` +
            `${this.state.counts.skipped} skipped, ${this.state.counts.noBars} no bars ` +
            `(${this.state.counts.symbols} symbols)`,
        );
      })
      .catch((err: Error) => {
        this.state = {
          ...this.state,
          status: 'failed',
          finishedAt: new Date().toISOString(),
          error: err.message,
        };
        this.log.error(`history rebuild failed: ${err.message}`);
      })
      .finally(() => {
        this.busy = false;
      });
    return { started: true };
  }

  private async run() {
    const { maxRiskUsd } = await this.settings.get();
    const ledgerOpts = {
      min_rr: 0,
      no_rr_req: true,
      use_last_hl_sl: true,
      risk_dollars: maxRiskUsd,
    };

    for (const universe of UNIVERSES) {
      const entries = await this.universe.resolveEntries(universe);
      const symbols: SymbolRow[] = entries.map((e) => {
        const tv = e.tv || e.yahoo;
        return {
          yahooTicker: e.yahoo,
          symbol: shortSymbol(tv),
          tvSymbol: tv,
          companyName: e.name ?? e.yahoo,
        };
      });

      for (const tf of TIMEFRAMES) {
        this.state.progress = {
          universe,
          tf,
          symbolsDone: 0,
          symbolsTotal: symbols.length,
        };
        await this.rebuildUniverseTf(universe, tf, symbols, maxRiskUsd, ledgerOpts);
      }
    }
  }

  private async rebuildUniverseTf(
    universe: TrackedUniverse,
    tf: Timeframe,
    symbols: SymbolRow[],
    maxRiskUsd: number,
    ledgerOpts: {
      min_rr: number;
      no_rr_req: boolean;
      use_last_hl_sl: boolean;
      risk_dollars: number;
    },
  ) {
    const recorded = await this.loadRecorded(universe, tf);
    const payloads = await this.fundamentals.loadPayloads(symbols.map((s) => s.yahooTicker));
    const queue = [...symbols];

    const worker = async () => {
      const ops: any[] = [];
      while (queue.length) {
        const row = queue.shift();
        if (!row) return;
        this.state.counts.symbols += 1;
        this.state.progress.symbolsDone += 1;

        const bars = await this.bars.getCached(row.yahooTicker, tf);
        if (!bars?.length) {
          this.state.counts.noBars += 1;
          continue;
        }

        const ledger = runCloseLedger(bars, ledgerOpts);
        if (!ledger) continue;

        for (const trade of ledger.trades) {
          if (trade.exit_index == null || !trade.exit_date) continue;
          const entered = entryKey(row.yahooTicker, trade.entry_date);
          const exited = exitKey(row.yahooTicker, trade.exit_date);
          // Set.add is sync; two workers can race a duplicate key only if the same ticker were
          // queued twice — the queue is unique per yahooTicker, so this stays safe.
          if (recorded.has(entered) || recorded.has(exited)) {
            this.state.counts.skipped += 1;
            continue;
          }
          recorded.add(entered);
          recorded.add(exited);
          ops.push({
            insertOne: {
              document: closedDocument(row, trade, universe, tf, maxRiskUsd, payloads),
            },
          });
          this.state.counts.inserted += 1;
        }

        if (ops.length >= BULK_CHUNK) {
          await this.flush(ops.splice(0, ops.length));
        }
      }
      if (ops.length) await this.flush(ops);
    };

    await Promise.all(Array.from({ length: LEDGER_CONCURRENCY }, () => worker()));

    this.log.log(
      `history rebuild ${universe}/${tf}: ` +
        `+${this.state.counts.inserted} inserted so far, ` +
        `${this.state.counts.skipped} skipped, ${this.state.counts.noBars} no bars`,
    );
  }

  /**
   * Trades this universe/tf already has on record, keyed by the bar they started on and the bar
   * they ended on.
   *
   * Both keys, and every record rather than the closed ones: a position the app is still carrying —
   * open, or breaking on the bar in progress — is the same trade the replay is about to find, and
   * inserting a close beside it is what used to put one trade on screen twice under two exits.
   */
  private async loadRecorded(universe: TrackedUniverse, tf: Timeframe): Promise<Set<string>> {
    const rows = await this.tracked
      .find({ universe, tf })
      .select('yahooTicker exitDate openedAsOf')
      .lean<Array<{ yahooTicker: string; exitDate?: string; openedAsOf?: string }>>()
      .exec();
    const keys = new Set<string>();
    for (const row of rows) {
      if (row.openedAsOf) keys.add(entryKey(row.yahooTicker, row.openedAsOf));
      if (row.exitDate) keys.add(exitKey(row.yahooTicker, row.exitDate));
    }
    return keys;
  }

  private async flush(ops: any[]) {
    for (let i = 0; i < ops.length; i += BULK_CHUNK) {
      await this.tracked.bulkWrite(ops.slice(i, i + BULK_CHUNK), { ordered: false });
    }
  }
}

function closedDocument(
  row: SymbolRow,
  trade: CloseTrade,
  universe: TrackedUniverse,
  tf: Timeframe,
  maxRiskUsd: number,
  payloads: Map<string, FundamentalsPayload> = new Map(),
) {
  const entry = round2(trade.entry_price);
  const sl = Number.isFinite(trade.entry_sl) ? round2(trade.entry_sl) : null;
  const tp = Number.isFinite(trade.entry_tp) ? round2(trade.entry_tp) : null;
  const rrAtEntry = Number.isFinite(trade.entry_rr) ? round2(trade.entry_rr) : null;
  const exitPrice = round2(trade.exit_price);
  const shares = sharesFromRisk(entry, sl, maxRiskUsd);
  const pnl = computePnl(entry, sl, shares, exitPrice);
  const exitDate = trade.exit_date as string;
  const premium =
    stampIfResolvable(
      trade.entry_date,
      entry,
      payloads.get(row.yahooTicker.toUpperCase()) ?? payloads.get(row.yahooTicker) ?? null,
    ) ?? {};
  return {
    yahooTicker: row.yahooTicker,
    symbol: row.symbol,
    tvSymbol: row.tvSymbol,
    companyName: row.companyName,
    universe,
    tf,
    status: 'closed' as const,
    provisional: false,
    provisionalClose: false,
    signalValid: false,
    imported: false,
    backfilled: true,
    openedPeriodKey: barPeriodKey(tf, trade.entry_date),
    openedAsOf: trade.entry_date,
    openedAt: atNoon(trade.entry_date),
    entry,
    tp,
    sl,
    rrAtEntry,
    shares,
    riskUsd: maxRiskUsd,
    lastSeenPeriodKey: barPeriodKey(tf, exitDate),
    lastSeenAsOf: exitDate,
    lastSeenAt: new Date(),
    lastPrice: exitPrice,
    lastRr: null,
    barsSinceValid: null,
    validSinceAsOf: null,
    isStrong: false,
    closedPeriodKey: barPeriodKey(tf, exitDate),
    closedAt: new Date(),
    exitDate,
    exitPrice,
    exitReason: 'sell_to_close' as const,
    pnlUsd: pnl.usd,
    pnlR: pnl.r,
    pnlPct: pnl.pct,
    holdPeriods: holdPeriods(tf, trade.entry_date, exitDate),
    interest: null,
    interestRank: 1,
    ...premium,
  };
}

function atNoon(date: string): Date {
  return new Date(`${date}T12:00:00Z`);
}

/** One position per ticker at a time, so the bar a trade started on names it. */
function entryKey(yahooTicker: string, entryDate: string): string {
  return `${yahooTicker}#${entryDate}`;
}

function exitKey(yahooTicker: string, exitDate: string): string {
  return `${yahooTicker}@${exitDate}`;
}

function emptyStatus(): RebuildStatus {
  return {
    status: 'idle',
    startedAt: null,
    finishedAt: null,
    error: null,
    progress: { universe: null, tf: null, symbolsDone: 0, symbolsTotal: 0 },
    counts: { inserted: 0, skipped: 0, noBars: 0, symbols: 0 },
  };
}
