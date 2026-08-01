/**
 * Scan execution. Runs out-of-request (fire and forget) so HTTP/SSE stay responsive.
 * Locally this is an in-process runner; on Railway the same body moves into the
 * BullMQ worker without changing the logic.
 */
import { Injectable, Logger } from '@nestjs/common';
import { InjectModel } from '@nestjs/mongoose';
import { Types, type Model } from 'mongoose';
import {
  evaluateSymbol,
  type BuySignal,
  type EvaluateParams,
  type Timeframe,
} from '@vova/engine';
import { REJECTION, SCAN_RUN, SIGNAL } from '../db/schemas';
import { BarsService } from '../market/bars.service';
import { UniverseService, type SourceLabelApi } from '../universe/universe.service';
import { ProgressBus } from './progress.bus';

const CONCURRENCY = 12;
const FLUSH_EVERY = 100;

export type ScanParamsApi = {
  source: SourceLabelApi;
  manualTickers?: string;
  tf: Timeframe;
  minRr: number;
  riskPerTrade: number;
  noRrReq: boolean;
  useLastHlSl: boolean;
  newOnly: boolean;
  minAvgVolume?: number;
  maxSymbols?: number;
  barsMaxAgeHours?: number;
  forceRefresh?: boolean;
};

@Injectable()
export class ScanRunnerService {
  private readonly log = new Logger(ScanRunnerService.name);
  private readonly aborts = new Map<string, AbortController>();

  constructor(
    @InjectModel(SCAN_RUN) private readonly runs: Model<any>,
    @InjectModel(SIGNAL) private readonly signals: Model<any>,
    @InjectModel(REJECTION) private readonly rejections: Model<any>,
    private readonly bars: BarsService,
    private readonly universe: UniverseService,
    private readonly bus: ProgressBus,
  ) {}

  cancel(runId: string) {
    this.aborts.get(runId)?.abort();
  }

  isRunning(runId: string) {
    return this.aborts.has(runId);
  }

  async execute(runId: string) {
    const controller = new AbortController();
    this.aborts.set(runId, controller);
    const startedAt = Date.now();

    try {
      const run = await this.runs.findById(runId).exec();
      if (!run) return;
      const params = run.params as ScanParamsApi;

      this.publish(runId, 'resolving', 0, 'Resolving universe...');
      let entries = await this.universe.resolveEntries(params.source, params.manualTickers ?? '');
      if (params.maxSymbols && params.maxSymbols > 0) entries = entries.slice(0, params.maxSymbols);

      const total = entries.length;
      run.status = 'running';
      run.startedAt = new Date();
      run.counters = { ...run.counters, total };
      await run.save();

      if (!total) {
        await this.finish(run, 'completed', { startedAt, asOf: null });
        this.publish(runId, 'completed', 100, 'Nothing to scan');
        return;
      }

      const evalParams: EvaluateParams = {
        minRr: params.minRr,
        riskPerTrade: params.riskPerTrade,
        noRrReq: params.noRrReq,
        useLastHlSl: params.useLastHlSl,
        newOnly: params.newOnly,
        tf: params.tf,
        minAvgVolume: params.minAvgVolume ?? 0,
      };

      const counters = {
        total,
        downloaded: 0,
        evaluated: 0,
        signals: 0,
        rejected: 0,
        skipped: 0,
        fromCache: 0,
      };
      const reasonCounts: Record<string, number> = {};
      const collectedSignals: BuySignal[] = [];
      let signalBuffer: any[] = [];
      let rejectionBuffer: any[] = [];
      let asOf: string | null = null;
      /** Oldest bar date seen, so a run without signals still reports what it looked at. */
      let evaluatedAsOf: string | null = null;
      /** Oldest Yahoo pull across the universe — the run's worst-case data age. */
      let barsOldestAt: Date | null = null;
      let lastPublish = 0;

      const queue = [...entries];
      const runObjectId = new Types.ObjectId(runId);

      const flush = async (force = false) => {
        if (force || signalBuffer.length >= FLUSH_EVERY) {
          if (signalBuffer.length) {
            await this.signals.insertMany(signalBuffer, { ordered: false });
            signalBuffer = [];
          }
        }
        if (force || rejectionBuffer.length >= FLUSH_EVERY) {
          if (rejectionBuffer.length) {
            await this.rejections.insertMany(rejectionBuffer, { ordered: false });
            rejectionBuffer = [];
          }
        }
      };

      const worker = async () => {
        while (queue.length) {
          if (controller.signal.aborted) return;
          const entry = queue.shift();
          if (!entry) return;

          const result = await this.bars.getBars(entry.yahoo, params.tf, {
            maxAgeHours: params.barsMaxAgeHours ?? 12,
            force: Boolean(params.forceRefresh),
            signal: controller.signal,
          });
          counters.downloaded += 1;
          if (result.fromCache) counters.fromCache += 1;
          if (result.fetchedAt && (!barsOldestAt || result.fetchedAt < barsOldestAt)) {
            barsOldestAt = result.fetchedAt;
          }
          if (result.bars?.length) {
            const barDate = result.bars[result.bars.length - 1].date;
            if (!evaluatedAsOf || barDate < evaluatedAsOf) evaluatedAsOf = barDate;
          }

          const evaluation = evaluateSymbol({
            bars: result.bars,
            yahooTicker: entry.yahoo,
            tvSymbol: entry.tv,
            companyName: entry.name ?? undefined,
            params: evalParams,
          });
          counters.evaluated += 1;

          if (evaluation.status === 'signal') {
            const signal = evaluation.signal;
            counters.signals += 1;
            collectedSignals.push(signal);
            if (!asOf || signal.asOf < asOf) asOf = signal.asOf;
            signalBuffer.push({
              runId: runObjectId,
              kind: signal.kind,
              symbol: signal.symbol,
              yahooTicker: signal.yahooTicker,
              companyName: signal.companyName,
              isNew: signal.isNew,
              isStrong: signal.isStrong,
              rr: signal.rr,
              payload: signal,
            });
          } else {
            const bucket = evaluation.status === 'rejected' ? 'rejected' : 'skipped';
            counters[bucket] += 1;
            reasonCounts[evaluation.reason] = (reasonCounts[evaluation.reason] ?? 0) + 1;
            if (evaluation.status === 'rejected') {
              rejectionBuffer.push({
                runId: runObjectId,
                symbol: entry.yahoo,
                reason: evaluation.reason,
                detail: evaluation.detail ?? null,
              });
            }
          }

          await flush();

          const now = Date.now();
          if (now - lastPublish > 400 || counters.evaluated === total) {
            lastPublish = now;
            this.publish(
              runId,
              'scanning',
              Math.round((counters.evaluated / total) * 100),
              `Scanned ${counters.evaluated}/${total} · ${counters.signals} signals`,
              { ...counters },
            );
          }

          if (counters.evaluated % 200 === 0) {
            const fresh = await this.runs
              .findById(runId)
              .select('cancelRequested')
              .lean<any>()
              .exec();
            if (fresh?.cancelRequested) controller.abort();
          }
        }
      };

      await Promise.all(Array.from({ length: CONCURRENCY }, () => worker()));
      await flush(true);

      const cancelled = controller.signal.aborted;
      this.publish(runId, cancelled ? 'cancelled' : 'saving', 99, 'Finalising...', {
        ...counters,
      });

      const newSymbols = await this.computeNewSymbols(run, collectedSignals);

      run.counters = counters;
      run.reasonCounts = reasonCounts;
      run.newSymbols = newSymbols;
      run.barsOldestAt = barsOldestAt;
      await this.finish(run, cancelled ? 'cancelled' : 'completed', {
        startedAt,
        asOf: asOf ?? evaluatedAsOf,
      });

      this.publish(
        runId,
        cancelled ? 'cancelled' : 'completed',
        100,
        cancelled
          ? `Cancelled after ${counters.evaluated}/${total}`
          : `Done · ${counters.signals} signals`,
        { ...counters },
      );
    } catch (err) {
      this.log.error(`run ${runId} failed: ${(err as Error).message}`);
      await this.runs
        .findByIdAndUpdate(runId, {
          status: 'failed',
          error: (err as Error).message,
          finishedAt: new Date(),
        })
        .exec();
      this.publish(runId, 'failed', 100, (err as Error).message);
    } finally {
      this.aborts.delete(runId);
    }
  }

  /** Symbols that were not present in the previous completed run of the same shape. */
  private async computeNewSymbols(run: any, signals: BuySignal[]): Promise<string[]> {
    const symbols = signals.map((s) => s.symbol);
    if (!symbols.length) return [];
    const prev = await this.runs
      .findOne({
        _id: { $ne: run._id },
        status: 'completed',
        'params.source': run.params.source,
        'params.tf': run.params.tf,
      })
      .sort({ createdAt: -1 })
      .select('_id')
      .lean<any>()
      .exec();
    if (!prev) return symbols;
    const prevSignals = await this.signals.find({ runId: prev._id }).select('symbol').lean().exec();
    const seen = new Set(prevSignals.map((s: any) => s.symbol));
    return symbols.filter((s) => !seen.has(s));
  }

  private async finish(
    run: any,
    status: 'completed' | 'cancelled',
    ctx: { startedAt: number; asOf: string | null },
  ) {
    run.status = status;
    run.asOf = ctx.asOf;
    run.timings = { ...run.timings, totalMs: Date.now() - ctx.startedAt };
    run.finishedAt = new Date();
    if (status === 'completed') run.lastCompletedAt = run.finishedAt;
    await run.save();
  }

  private publish(
    runId: string,
    phase: 'resolving' | 'scanning' | 'saving' | 'completed' | 'cancelled' | 'failed',
    percent: number,
    message: string,
    counters?: Record<string, number>,
  ) {
    this.bus.publish({ runId, phase, percent, message, counters });
  }
}
