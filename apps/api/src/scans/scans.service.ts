import { Injectable, Logger, NotFoundException } from '@nestjs/common';
import { InjectModel } from '@nestjs/mongoose';
import { Types, type Model } from 'mongoose';
import type { Timeframe } from '@vova/engine';
import { REJECTION, SCAN_RUN, SIGNAL, TRACKED_SIGNAL } from '../db/schemas';
import { SignalTrackerService } from '../tracking/signal-tracker.service';
import { isPeriodClosed, periodKey } from './period';
import { ScanRunnerService, type ScanParamsApi } from './scan-runner.service';

const DEFAULTS: ScanParamsApi = {
  source: 'MANUAL SCAN',
  manualTickers: 'AAPL, TSLA, NVDA',
  tf: 'Daily',
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

const EMPTY_COUNTERS = {
  total: 0,
  downloaded: 0,
  evaluated: 0,
  signals: 0,
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
    const filter: Record<string, unknown> = { runId: new Types.ObjectId(id) };
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
