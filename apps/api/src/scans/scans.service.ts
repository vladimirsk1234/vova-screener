import { Injectable, Logger, NotFoundException } from '@nestjs/common';
import { InjectModel } from '@nestjs/mongoose';
import { Types, type Model } from 'mongoose';
import type { Timeframe } from '@vova/engine';
import { REJECTION, SCAN_RUN, SIGNAL, TRACKED_SIGNAL } from '../db/schemas';
import { SignalTrackerService } from '../tracking/signal-tracker.service';
import { UniverseService } from '../universe/universe.service';
import { isPeriodClosed, periodKey } from './period';
import { ScanRunnerService, type ScanParamsApi } from './scan-runner.service';

const DEFAULTS: ScanParamsApi = {
  source: 'MANUAL SCAN',
  manualTickers: 'AAPL, TSLA, NVDA',
  tf: 'Daily',
  direction: 'buy',
  minRr: 1.5,
  riskPerTrade: 100,
  noRrReq: true,
  useLastHlSl: true,
  newOnly: false,
};

export type StartOpts = {
  trigger?: 'manual' | 'scheduled';
  wait?: boolean;
};

const TRACKED_UNIVERSES = ['Stocks', 'ETF'] as const;
type TrackedUniverseName = (typeof TRACKED_UNIVERSES)[number];

type DelistedGroup = { universe: TrackedUniverseName; symbols: string[] };

export type DelistedSummary = {
  symbols: number;
  records: number;
  closed: number;
  active: number;
  sample: string[];
};

const EMPTY_COUNTERS = {
  total: 0,
  downloaded: 0,
  evaluated: 0,
  signals: 0,
  closes: 0,
  rejected: 0,
  skipped: 0,
  fromCache: 0,
};

@Injectable()
export class ScansService {
  private readonly log = new Logger(ScansService.name);

  constructor(
    @InjectModel(SCAN_RUN) private readonly runs: Model<any>,
    @InjectModel(SIGNAL) private readonly signals: Model<any>,
    @InjectModel(REJECTION) private readonly rejections: Model<any>,
    @InjectModel(TRACKED_SIGNAL) private readonly tracked: Model<any>,
    private readonly runner: ScanRunnerService,
    private readonly tracker: SignalTrackerService,
    private readonly universe: UniverseService,
  ) {}

  async start(input: Partial<ScanParamsApi>, opts: StartOpts = {}) {
    const params: ScanParamsApi = { ...DEFAULTS, ...input };
    const trigger = opts.trigger ?? 'manual';
    const key = periodKey(params.tf);
    const periodTf = params.tf;
    const periodClose = isPeriodClosed(params.tf);

    let run = await this.runs
      .findOne({ periodKey: key, periodTf, 'params.source': params.source })
      .exec();

    if (run) {
      const existingId = String(run._id);
      if (this.runner.isRunning(existingId)) {
        this.runner.cancel(existingId);
        for (let i = 0; i < 50; i++) {
          if (!this.runner.isRunning(existingId)) break;
          await new Promise((r) => setTimeout(r, 200));
        }
      }
      await this.signals.deleteMany({ runId: run._id }).exec();
      await this.rejections.deleteMany({ runId: run._id }).exec();
      run.params = params;
      run.status = 'queued';
      run.trigger = trigger;
      run.periodKey = key;
      run.periodTf = periodTf;
      run.periodClose = periodClose;
      run.counters = { ...EMPTY_COUNTERS };
      run.reasonCounts = {};
      run.newSymbols = [];
      run.summary = null;
      run.error = undefined;
      run.cancelRequested = false;
      run.startedAt = undefined;
      // asOf / barsOldestAt / finishedAt keep describing the last completed pass over this
      // period until the new one overwrites them. Results derives its bucket boundary from
      // them, and clearing them would move every signal one bucket while a rescan is running.
      run.timings = { downloadMs: 0, processMs: 0, totalMs: 0 };
      await run.save();
    } else {
      run = await this.runs.create({
        params,
        status: 'queued',
        periodKey: key,
        periodTf,
        periodClose,
        trigger,
      });
    }

    const runId = String(run._id);
    const finish = async () => {
      await this.runner.execute(runId);
      await this.afterScanComplete(runId);
    };
    if (opts.wait) {
      await finish();
    } else {
      void finish().catch((err) =>
        this.log.error(`post-scan for ${runId} failed: ${(err as Error).message}`),
      );
    }
    return { runId, params };
  }

  /** Feed a finished universe scan into the signal tracker (manual scans are ignored there). */
  async afterScanComplete(runId: string) {
    await this.tracker.applyRun(runId);
  }

  async list(opts: { limit?: number; tf?: Timeframe } = {}) {
    const filter: Record<string, unknown> = {};
    if (opts.tf) {
      filter.$or = [
        { periodTf: opts.tf },
        { periodTf: { $exists: false }, 'params.tf': opts.tf },
      ];
    }
    return this.runs
      .find(filter)
      .sort({ periodKey: -1, createdAt: -1 })
      .limit(Math.min(opts.limit ?? 30, 100))
      .lean()
      .exec();
  }

  async resetHistory() {
    const [, , tracked] = await Promise.all([
      this.signals.deleteMany({}).exec(),
      this.rejections.deleteMany({}).exec(),
      this.tracked.deleteMany({}).exec(),
    ]);
    const result = await this.runs.deleteMany({}).exec();
    return {
      ok: true,
      deletedRuns: result.deletedCount ?? 0,
      deletedSignals: tracked.deletedCount ?? 0,
    };
  }

  /**
   * Tracked signals whose ticker has left its universe list file — e.g. a symbol dropped by the
   * EPS rebuild of STOCK-TICKERS.txt. Scans stop covering them, but their rows stay in History
   * because History only filters on status / universe / timeframe.
   */
  private async findDelisted(): Promise<DelistedGroup[]> {
    const groups: DelistedGroup[] = [];
    for (const universe of TRACKED_UNIVERSES) {
      const entries = await this.universe.resolveEntries(universe);
      // An unreadable or empty list file would otherwise condemn the whole universe.
      if (!entries.length) {
        this.log.warn(`Skipping ${universe}: universe resolved to no tickers`);
        continue;
      }
      const live = new Set(entries.map((e) => e.yahoo));
      const seen: string[] = await this.tracked.distinct('yahooTicker', { universe }).exec();
      const symbols = seen.filter((t) => !live.has(t)).sort();
      if (symbols.length) groups.push({ universe, symbols });
    }
    return groups;
  }

  private async summarizeDelisted(groups: DelistedGroup[]): Promise<DelistedSummary> {
    let symbols = 0;
    let records = 0;
    let closed = 0;
    const sample: string[] = [];
    for (const group of groups) {
      const filter = { universe: group.universe, yahooTicker: { $in: group.symbols } };
      const [total, closedCount] = await Promise.all([
        this.tracked.countDocuments(filter).exec(),
        this.tracked.countDocuments({ ...filter, status: 'closed' }).exec(),
      ]);
      symbols += group.symbols.length;
      records += total;
      closed += closedCount;
      sample.push(...group.symbols);
    }
    return { symbols, records, closed, active: records - closed, sample: sample.slice(0, 20) };
  }

  async delistedPreview(): Promise<DelistedSummary> {
    return this.summarizeDelisted(await this.findDelisted());
  }

  async purgeDelisted() {
    const groups = await this.findDelisted();
    const summary = await this.summarizeDelisted(groups);
    let deletedSignals = 0;
    for (const group of groups) {
      const res = await this.tracked
        .deleteMany({ universe: group.universe, yahooTicker: { $in: group.symbols } })
        .exec();
      deletedSignals += res.deletedCount ?? 0;
    }
    if (deletedSignals) {
      this.log.log(`Purged ${deletedSignals} tracked signals across ${summary.symbols} delisted tickers`);
    }
    return { ok: true, deletedSignals, ...summary };
  }

  async get(id: string) {
    if (!Types.ObjectId.isValid(id)) throw new NotFoundException('bad run id');
    const run = await this.runs.findById(id).lean<any>().exec();
    if (!run) throw new NotFoundException('run not found');
    return run;
  }

  async listSignals(
    id: string,
    opts: { limit?: number; offset?: number; onlyNew?: boolean; onlyStrong?: boolean } = {},
  ) {
    const run = await this.get(id);
    // A buy run also records the sell-to-close breaks it found for the tracker. They are a
    // different table with different columns, so the scan screen only ever shows its own kind.
    const filter: Record<string, unknown> = {
      runId: new Types.ObjectId(id),
      kind: run.params?.direction === 'sell' ? 'sell' : 'buy',
    };
    if (opts.onlyNew) filter.isNew = true;
    if (opts.onlyStrong) filter.isStrong = true;
    const rows = await this.signals
      .find(filter)
      .sort({ isStrong: -1, rr: -1 })
      .skip(opts.offset ?? 0)
      .limit(Math.min(opts.limit ?? 200, 500))
      .lean()
      .exec();

    const payloads = rows.map((r: any) => r.payload);
    return { run, count: payloads.length, rows: payloads, newSymbols: run.newSymbols ?? [] };
  }

  async listRejections(id: string, limit = 300) {
    const rows = await this.rejections
      .find({ runId: new Types.ObjectId(id) })
      .limit(Math.min(limit, 2000))
      .lean()
      .exec();
    const run = await this.get(id);
    return { rows, reasonCounts: run.reasonCounts ?? {}, total: run.counters?.rejected ?? 0 };
  }

  async cancel(id: string) {
    await this.runs.findByIdAndUpdate(id, { cancelRequested: true }).exec();
    this.runner.cancel(id);
    return { ok: true };
  }

  defaults() {
    return DEFAULTS;
  }
}
