/** Trade journal with mark-to-market, TP/SL, sell-to-close, and auto-journal from scheduled scans. */
import { Injectable, NotFoundException } from '@nestjs/common';
import { InjectModel } from '@nestjs/mongoose';
import { Types, type Model } from 'mongoose';
import { runStructureOverlay, type Timeframe } from '@vova/engine';
import { SCAN_RUN, SIGNAL, TRADE } from '../db/schemas';
import { BarsService } from '../market/bars.service';

export type CreateTradeDto = {
  symbol: string;
  yahooTicker: string;
  companyName?: string;
  tf?: Timeframe;
  entry: number;
  tp?: number;
  sl?: number;
  rrAtEntry?: number;
  shares?: number;
  riskUsd?: number;
  asOf?: string;
  runId?: string;
  source?: 'auto' | 'manual';
  periodKey?: string;
};

function round2(n: number) {
  return Math.round(n * 100) / 100;
}

@Injectable()
export class TradesService {
  constructor(
    @InjectModel(TRADE) private readonly trades: Model<any>,
    @InjectModel(SCAN_RUN) private readonly runs: Model<any>,
    @InjectModel(SIGNAL) private readonly signals: Model<any>,
    private readonly bars: BarsService,
  ) {}

  async create(dto: CreateTradeDto) {
    return this.trades.create({
      ...dto,
      tf: dto.tf ?? 'Daily',
      shares: dto.shares ?? 0,
      riskUsd: dto.riskUsd ?? 0,
      status: 'open',
      source: dto.source ?? 'manual',
      openedAt: new Date(),
      runId: dto.runId && Types.ObjectId.isValid(dto.runId) ? new Types.ObjectId(dto.runId) : undefined,
    });
  }

  async list(status?: 'open' | 'closed' | 'dismissed') {
    const filter = status ? { status } : { status: { $in: ['open', 'closed'] } };
    const rows = await this.trades.find(filter).sort({ openedAt: -1 }).lean().exec();
    const marked = await Promise.all(
      rows.map(async (t: any) => {
        if (t.status !== 'open') return t;
        const price = await this.bars.lastClose(t.yahooTicker, (t.tf as Timeframe) ?? 'Daily');
        if (price == null) return { ...t, currentPrice: null, unrealizedUsd: null };
        const unrealizedUsd = round2((price - t.entry) * (t.shares || 0));
        const risk = t.entry - (t.sl ?? t.entry);
        return {
          ...t,
          currentPrice: round2(price),
          unrealizedUsd,
          unrealizedR: risk > 0 ? round2((price - t.entry) / risk) : null,
        };
      }),
    );
    return marked;
  }

  async close(
    id: string,
    dto: { exitPrice: number; exitDate?: string; exitReason?: string },
  ) {
    const trade = await this.trades.findById(id).exec();
    if (!trade) throw new NotFoundException('trade not found');
    if (trade.status !== 'open') throw new NotFoundException('trade not open');
    const risk = trade.entry - (trade.sl ?? trade.entry);
    const pnlUsd = round2((dto.exitPrice - trade.entry) * (trade.shares || 0));
    trade.status = 'closed';
    trade.exitPrice = dto.exitPrice;
    trade.exitDate = dto.exitDate ?? new Date().toISOString().slice(0, 10);
    trade.exitReason = dto.exitReason ?? 'manual';
    trade.pnlUsd = pnlUsd;
    trade.pnlR = risk > 0 ? round2((dto.exitPrice - trade.entry) / risk) : null;
    await trade.save();
    return trade.toObject();
  }

  async dismiss(id: string) {
    const trade = await this.trades.findById(id).exec();
    if (!trade) throw new NotFoundException('trade not found');
    trade.status = 'dismissed';
    await trade.save();
    return trade.toObject();
  }

  /**
   * Auto-open journal rows from New buy signals of a scheduled end-of-period run.
   * Mid-period manual runs must not call this (caller checks trigger).
   */
  async journalNewBuySignals(runId: string) {
    if (!Types.ObjectId.isValid(runId)) return { created: 0 };
    const run = await this.runs.findById(runId).lean<any>().exec();
    if (!run || run.trigger !== 'scheduled') return { created: 0 };
    if (run.params?.direction !== 'buy') return { created: 0 };
    if (run.status !== 'completed') return { created: 0 };

    const tf = (run.params.tf as Timeframe) ?? 'Daily';
    const periodKey = run.periodKey as string | undefined;
    const rows = await this.signals
      .find({ runId: new Types.ObjectId(runId), kind: 'buy', isNew: true })
      .lean<any>()
      .exec();

    let created = 0;
    for (const row of rows) {
      const signal = row.payload;
      if (!signal?.symbol) continue;

      const open = await this.trades
        .findOne({ symbol: signal.symbol, tf, status: 'open' })
        .lean()
        .exec();
      if (open) continue;

      if (periodKey) {
        const dismissed = await this.trades
          .findOne({ symbol: signal.symbol, tf, status: 'dismissed', periodKey })
          .lean()
          .exec();
        if (dismissed) continue;
      }

      await this.create({
        symbol: signal.symbol,
        yahooTicker: signal.yahooTicker,
        companyName: signal.companyName,
        tf,
        entry: signal.entry,
        tp: signal.tp,
        sl: signal.sl,
        rrAtEntry: signal.rr ?? undefined,
        shares: signal.shares,
        riskUsd: run.params.riskPerTrade ?? 100,
        asOf: signal.asOf,
        runId,
        source: 'auto',
        periodKey,
      });
      created += 1;
    }
    return { created };
  }

  /** Auto-close open trades: TP/SL first, then Sequence Vova sell-to-close (bullish break). */
  async refresh(opts: { tf?: Timeframe } = {}) {
    const filter: Record<string, unknown> = { status: 'open' };
    if (opts.tf) filter.tf = opts.tf;
    const open = await this.trades.find(filter).exec();
    let closed = 0;
    for (const trade of open) {
      const tf = (trade.tf as Timeframe) ?? 'Daily';
      const bars = await this.bars.getCached(trade.yahooTicker, tf);
      if (!bars?.length) continue;
      const since = trade.asOf ?? new Date(trade.openedAt).toISOString().slice(0, 10);
      const overlay = runStructureOverlay(bars);

      for (let i = 0; i < bars.length; i++) {
        const bar = bars[i];
        if (bar.date <= since) continue;

        const hitSl = trade.sl != null && bar.low <= trade.sl;
        const hitTp = trade.tp != null && bar.high >= trade.tp;
        if (hitSl || hitTp) {
          const exitPrice = hitSl ? trade.sl : trade.tp;
          await this.close(String(trade._id), {
            exitPrice,
            exitDate: bar.date,
            exitReason: hitSl ? 'SL' : 'TP',
          });
          closed += 1;
          break;
        }

        if (overlay?.bullish_break[i]) {
          await this.close(String(trade._id), {
            exitPrice: bar.close,
            exitDate: bar.date,
            exitReason: 'sell_to_close',
          });
          closed += 1;
          break;
        }
      }
    }
    return { checked: open.length, closed };
  }

  async remove(id: string) {
    await this.trades.findByIdAndDelete(id).exec();
    return { ok: true };
  }
}
