/**
 * Scan execution. Runs out-of-request (fire and forget) so HTTP/SSE stay responsive.
 * Locally this is an in-process runner; on Railway the same body moves into the
 * BullMQ worker without changing the logic.
 */
import { Injectable, Logger } from '@nestjs/common';
import { InjectModel } from '@nestjs/mongoose';
import { Types, type Model } from 'mongoose';
import {
  buildSellSummary,
  evaluateClose,
  evaluateSymbol,
  inferTvSymbol,
  canadianYahooCandidatesIfBare,
  seqStructFromBars,
  shortSymbol,
  type EvaluateParams,
  type ParsedEntry,
  type SellSignal,
  type SeqStructStatus,
  type Signal,
  type Timeframe,
} from '@vova/engine';
import { REJECTION, SCAN_RUN, SIGNAL } from '../db/schemas';
import { FundamentalsService } from '../instruments/fundamentals.service';
import { BarsService } from '../market/bars.service';
import { UniverseService, type SourceLabelApi } from '../universe/universe.service';
import { barPeriodKey } from './period';
import { ProgressBus } from './progress.bus';

const CONCURRENCY = 12;
const FLUSH_EVERY = 100;

export type ScanParamsApi = {
  source: SourceLabelApi;
  manualTickers?: string;
  tf: Timeframe;
  direction: 'buy' | 'sell';
  minRr: number;
  riskPerTrade: number;
  noRrReq: boolean;
  useLastHlSl: boolean;
  newOnly: boolean;
  minAvgVolume?: number;
  maxSymbols?: number;
  barsMaxAgeHours?: number;
  forceRefresh?: boolean;
  /** Listed Manual: never hit Yahoo for bars. */
  barsCacheOnly?: boolean;
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
    private readonly fundamentals: FundamentalsService,
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

      let fundWarm: Promise<unknown> | null = null;
      if (params.source === 'MANUAL SCAN' && entries[0]) {
        const known = await this.universe.isInTrackedUniverse(entries[0].yahoo);
        if (known) {
          // Listed: Mongo barSeries + instrumentFundamentals only — never force Yahoo/FMP.
          params.forceRefresh = false;
          params.barsCacheOnly = true;
          params.barsMaxAgeHours = 24 * 7;
        } else {
          params.forceRefresh = true;
          params.barsCacheOnly = false;
          params.barsMaxAgeHours = 0;
          this.publish(runId, 'resolving', 0, `Downloading ${entries[0].yahoo} from Yahoo + FMP…`);
          fundWarm = this.fundamentals.get(entries[0].yahoo).catch((err) => {
            this.log.warn(`FMP prefetch ${entries[0].yahoo} failed: ${(err as Error).message}`);
          });
        }
      }

      const total = entries.length;
      run.status = 'running';
      run.startedAt = new Date();
      run.counters = { ...run.counters, total };
      await run.save();

      if (!total) {
        await this.finish(run, 'completed', { startedAt, asOf: null, summary: null });
        this.publish(runId, 'completed', 100, 'Nothing to scan');
        return;
      }

      const evalParams: EvaluateParams = {
        direction: params.direction,
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
        closes: 0,
        rejected: 0,
        skipped: 0,
        fromCache: 0,
      };
      const reasonCounts: Record<string, number> = {};
      const collectedSignals: Signal[] = [];
      let signalBuffer: any[] = [];
      let rejectionBuffer: any[] = [];
      let taBuffer: Array<{ yahooTicker: string; tf: Timeframe; status: SeqStructStatus }> = [];
      let asOf: string | null = null;
      /** Oldest bar date seen, so a run without signals still reports what it looked at. */
      let evaluatedAsOf: string | null = null;
      /**
       * How many symbols' newest bar landed in each period, and the newest bar inside it. This is
       * how the run answers "which period am I reporting on", and neither extreme of the range
       * can answer it: the oldest bar is one halted ticker away from putting CLOSED days behind
       * the market, and the newest is one off-grid bar away from putting it a period ahead, where
       * nothing has closed yet. Yahoo does hand out the odd bar a day off the grid — a handful of
       * weekly series are stamped Tuesday while the rest of the market is stamped Monday — so
       * across three thousand symbols both are a matter of time. What the market is on is what
       * nearly every symbol agrees it is on.
       */
      const periods = new Map<string, { count: number; newest: string }>();
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
        if (force || taBuffer.length >= FLUSH_EVERY) {
          if (taBuffer.length) {
            try {
              await this.fundamentals.mergeTaSnapshots(taBuffer);
            } catch (err) {
              this.log.warn(
                `TA snapshot flush failed: ${err instanceof Error ? err.message : String(err)}`,
              );
            }
            taBuffer = [];
          }
        }
      };

      const worker = async () => {
        while (queue.length) {
          if (controller.signal.aborted) return;
          const entry = queue.shift();
          if (!entry) return;

          const resolved = await this.loadEntryBars(entry, params, controller.signal);
          const { bars: series, fromCache, fetchedAt, yahoo, tv, name } = resolved;
          counters.downloaded += 1;
          if (fromCache) counters.fromCache += 1;
          if (fetchedAt && (!barsOldestAt || fetchedAt < barsOldestAt)) {
            barsOldestAt = fetchedAt;
          }
          if (series?.length) {
            const barDate = series[series.length - 1].date;
            if (!evaluatedAsOf || barDate < evaluatedAsOf) evaluatedAsOf = barDate;
            const key = barPeriodKey(params.tf, barDate);
            const slot = periods.get(key);
            if (!slot) periods.set(key, { count: 1, newest: barDate });
            else {
              slot.count += 1;
              if (barDate > slot.newest) slot.newest = barDate;
            }
          }

          const evaluation = evaluateSymbol({
            bars: series,
            yahooTicker: yahoo,
            tvSymbol: tv,
            companyName: name ?? undefined,
            params: evalParams,
          });
          counters.evaluated += 1;

          if (series?.length && params.source !== 'MANUAL SCAN') {
            const status = seqStructFromBars(series, params.tf);
            if (status) taBuffer.push({ yahooTicker: yahoo, tf: params.tf, status });
          }

          // A buy pass also asks the close scan, because the two never report the same symbol:
          // the break that ends a trade puts the sequence down, so a symbol closing today is a
          // reject here and the tracker would never hear about it. Same bars, same pass.
          if (params.direction === 'buy') {
            const close = evaluateClose({
              bars: series,
              yahooTicker: yahoo,
              tvSymbol: tv,
              companyName: name ?? undefined,
              params: evalParams,
            });
            if (close) {
              counters.closes += 1;
              signalBuffer.push({
                runId: runObjectId,
                kind: 'sell',
                symbol: close.symbol,
                yahooTicker: close.yahooTicker,
                companyName: close.companyName,
                isNew: true,
                isStrong: false,
                rr: close.rrAtEntry,
                payload: close,
              });
            }
          }

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
              isStrong: signal.kind === 'buy' ? signal.isStrong : false,
              rr: signal.kind === 'buy' ? signal.rr : signal.rrAtEntry,
              payload: signal,
            });
          } else {
            const bucket = evaluation.status === 'rejected' ? 'rejected' : 'skipped';
            counters[bucket] += 1;
            reasonCounts[evaluation.reason] = (reasonCounts[evaluation.reason] ?? 0) + 1;
            if (evaluation.status === 'rejected') {
              rejectionBuffer.push({
                runId: runObjectId,
                // `symbol` is the display form the Rejected tab prints; `yahooTicker` is the key
                // the tracker matches a record against.
                symbol: shortSymbol(tv || yahoo),
                yahooTicker: yahoo,
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
      if (fundWarm) await fundWarm;

      const cancelled = controller.signal.aborted;
      this.publish(runId, cancelled ? 'cancelled' : 'saving', 99, 'Finalising...', {
        ...counters,
      });

      const summary =
        params.direction === 'sell'
          ? buildSellSummary(collectedSignals.filter((s): s is SellSignal => s.kind === 'sell'))
          : null;

      const newSymbols = await this.computeNewSymbols(run, collectedSignals);

      run.counters = counters;
      run.reasonCounts = reasonCounts;
      run.newSymbols = newSymbols;
      run.barsOldestAt = barsOldestAt;
      run.newestAsOf = this.consensusAsOf(periods);
      await this.finish(run, cancelled ? 'cancelled' : 'completed', {
        startedAt,
        asOf: asOf ?? evaluatedAsOf,
        summary,
      });

      this.publish(
        runId,
        cancelled ? 'cancelled' : 'completed',
        100,
        cancelled
          ? `Cancelled after ${counters.evaluated}/${total}`
          : `Done · ${counters.signals} signals · ${counters.closes} closes`,
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

  /**
   * Manual scans type a short ticker. If Yahoo has no bars for the bare US symbol, try the
   * Canadian listings — RBY is empty on YHD, RBY.TO is Rubellite on TSX.
   */
  private async loadEntryBars(
    entry: ParsedEntry,
    params: ScanParamsApi,
    signal: AbortSignal,
  ): Promise<{
    bars: Awaited<ReturnType<BarsService['getBars']>>['bars'];
    fromCache: boolean;
    fetchedAt: Date | null;
    yahoo: string;
    tv: string;
    name: string | null;
  }> {
    const opts = {
      maxAgeHours: params.barsMaxAgeHours ?? 12,
      force: Boolean(params.forceRefresh),
      cacheOnly: Boolean(params.barsCacheOnly),
      signal,
    };
    const result = await this.bars.getBars(entry.yahoo, params.tf, opts);
    const passthrough = {
      bars: result.bars,
      fromCache: result.fromCache,
      fetchedAt: result.fetchedAt,
      yahoo: entry.yahoo,
      tv: entry.tv,
      name: entry.name,
    };
    if (result.bars?.length || params.source !== 'MANUAL SCAN') return passthrough;
    const alts = canadianYahooCandidatesIfBare(entry.yahoo);
    if (!alts) return passthrough;
    for (const alt of alts) {
      if (signal.aborted) break;
      const altResult = await this.bars.getBars(alt, params.tf, opts);
      if (!altResult.bars?.length) continue;
      return {
        bars: altResult.bars,
        fromCache: altResult.fromCache,
        fetchedAt: altResult.fetchedAt,
        yahoo: alt,
        tv: inferTvSymbol(alt, altResult.exchange),
        name: entry.name ?? altResult.companyName ?? null,
      };
    }
    return passthrough;
  }

  /** Symbols that were not present in the previous completed run of the same shape. */
  private async computeNewSymbols(run: any, signals: Signal[]): Promise<string[]> {
    const symbols = signals.map((s) => s.symbol);
    if (!symbols.length) return [];
    const prev = await this.runs
      .findOne({
        _id: { $ne: run._id },
        status: 'completed',
        'params.source': run.params.source,
        'params.tf': run.params.tf,
        'params.direction': run.params.direction,
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
    ctx: { startedAt: number; asOf: string | null; summary: unknown },
  ) {
    run.status = status;
    run.asOf = ctx.asOf;
    run.summary = ctx.summary;
    run.timings = { ...run.timings, totalMs: Date.now() - ctx.startedAt };
    run.finishedAt = new Date();
    if (status === 'completed') run.lastCompletedAt = run.finishedAt;
    await run.save();
  }

  /**
   * The newest bar of the period most of the universe is in — the bar the screens should read
   * this run as being about. A tie goes to the later period, which is the market moving on while
   * the scan ran rather than a stray series.
   */
  private consensusAsOf(periods: Map<string, { count: number; newest: string }>): string | null {
    let best: { key: string; count: number; newest: string } | null = null;
    for (const [key, slot] of periods) {
      if (!best || slot.count > best.count || (slot.count === best.count && key > best.key)) {
        best = { key, ...slot };
      }
    }
    return best?.newest ?? null;
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
