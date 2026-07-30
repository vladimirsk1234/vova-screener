import { Injectable, NotFoundException } from '@nestjs/common';
import { InjectModel } from '@nestjs/mongoose';
import { Types, type Model } from 'mongoose';
import { REJECTION, SCAN_RUN, SIGNAL } from '../db/schemas';
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
  newOnly: false,
};

@Injectable()
export class ScansService {
  constructor(
    @InjectModel(SCAN_RUN) private readonly runs: Model<any>,
    @InjectModel(SIGNAL) private readonly signals: Model<any>,
    @InjectModel(REJECTION) private readonly rejections: Model<any>,
    private readonly runner: ScanRunnerService,
  ) {}

  async start(input: Partial<ScanParamsApi>) {
    const params: ScanParamsApi = { ...DEFAULTS, ...input };
    const run = await this.runs.create({ params, status: 'queued' });
    const runId = String(run._id);
    void this.runner.execute(runId);
    return { runId, params };
  }

  async list(limit = 30) {
    return this.runs
      .find()
      .sort({ createdAt: -1 })
      .limit(Math.min(limit, 100))
      .lean()
      .exec();
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
