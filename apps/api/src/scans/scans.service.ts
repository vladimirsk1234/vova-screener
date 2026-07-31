import { Injectable, NotFoundException } from '@nestjs/common';
import { InjectModel } from '@nestjs/mongoose';
import { Types, type Model } from 'mongoose';
import type { Timeframe } from '@vova/engine';
import { REJECTION, SCAN_RUN, SIGNAL } from '../db/schemas';
import { periodKey } from './period';
import { ScanRunnerService, type ScanParamsApi } from './scan-runner.service';

const DEFAULTS: ScanParamsApi = {
  source: 'MANUAL SCAN',
  manualTickers: 'AAPL, TSLA, NVDA',
  tf: 'Daily',
  direction: 'buy',
  minRr: 1.5,
  riskPerTrade: 100,
  noRrReq: false,
  useLastHlSl: true,
  newOnly: true,
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
  constructor(
    @InjectModel(SCAN_RUN) private readonly runs: Model<any>,
    @InjectModel(SIGNAL) private readonly signals: Model<any>,
    @InjectModel(REJECTION) private readonly rejections: Model<any>,
    private readonly runner: ScanRunnerService,
  ) {}

  async start(input: Partial<ScanParamsApi>, opts: StartOpts = {}) {
    const params: ScanParamsApi = { ...DEFAULTS, ...input };
    const trigger = opts.trigger ?? 'manual';
    const key = periodKey(params.tf);
    const periodTf = params.tf;

    let run = await this.runs
      .findOne({ periodKey: key, periodTf, 'params.source': params.source })
      .exec();

    if (run) {
      const runId = String(run._id);
      if (this.runner.isRunning(runId)) {
        this.runner.cancel(runId);
        for (let i = 0; i < 50; i++) {
          if (!this.runner.isRunning(runId)) break;
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
      run.counters = { ...EMPTY_COUNTERS };
      run.reasonCounts = {};
      run.newSymbols = [];
      run.summary = null;
      run.error = undefined;
      run.cancelRequested = false;
      run.asOf = undefined;
      run.startedAt = undefined;
      run.finishedAt = undefined;
      run.timings = { downloadMs: 0, processMs: 0, totalMs: 0 };
      await run.save();
    } else {
      run = await this.runs.create({
        params,
        status: 'queued',
        periodKey: key,
        periodTf,
        trigger,
      });
    }

    const runId = String(run._id);
    if (opts.wait) {
      await this.runner.execute(runId);
    } else {
      void this.runner.execute(runId);
    }
    return { runId, params };
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
    await Promise.all([
      this.signals.deleteMany({}).exec(),
      this.rejections.deleteMany({}).exec(),
    ]);
    const result = await this.runs.deleteMany({}).exec();
    return { ok: true, deletedRuns: result.deletedCount ?? 0 };
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
    const [rows, count] = await Promise.all([
      this.signals
        .find(filter)
        .sort({ isStrong: -1, rr: -1 })
        .skip(opts.offset ?? 0)
        .limit(Math.min(opts.limit ?? 200, 500))
        .lean()
        .exec(),
      this.signals.countDocuments(filter).exec(),
    ]);
    return {
      run,
      count,
      rows: rows.map((r: any) => r.payload),
      newSymbols: run.newSymbols ?? [],
    };
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
