/** Trade journal with mark-to-market and TP/SL auto-close from cached bars. */
import { Injectable, NotFoundException } from '@nestjs/common';
import { InjectModel } from '@nestjs/mongoose';
import { Types, type Model } from 'mongoose';
import type { Timeframe } from '@vova/engine';
import { TRADE } from '../db/schemas';
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
};

function round2(n: number) {
  return Math.round(n * 100) / 100;
}

@Injectable()
export class TradesService {
  constructor(
    @InjectModel(TRADE) private readonly trades: Model<any>,
    private readonly bars: BarsService,
  ) {}

  async create(dto: CreateTradeDto) {
    return this.trades.create({
      ...dto,
      tf: dto.tf ?? 'Daily',
      shares: dto.shares ?? 0,
      riskUsd: dto.riskUsd ?? 0,
      status: 'open',
      openedAt: new Date(),
      runId: dto.runId && Types.ObjectId.isValid(dto.runId) ? new Types.ObjectId(dto.runId) : undefined,
    });
  }

  async list(status?: 'open' | 'closed') {
    const filter = status ? { status } : {};
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

  /** Auto-close open trades whose TP/SL was touched after entry (uses cached bars). */
  async refresh() {
    const open = await this.trades.find({ status: 'open' }).exec();
    let closed = 0;
    for (const trade of open) {
      const bars = await this.bars.getCached(trade.yahooTicker, (trade.tf as Timeframe) ?? 'Daily');
      if (!bars?.length) continue;
      const since = trade.asOf ?? new Date(trade.openedAt).toISOString().slice(0, 10);
      const after = bars.filter((b) => b.date > since);
      for (const bar of after) {
        const hitSl = trade.sl != null && bar.low <= trade.sl;
        const hitTp = trade.tp != null && bar.high >= trade.tp;
        if (!hitSl && !hitTp) continue;
        // Conservative: if both touched in one bar, assume stop first.
        const exitPrice = hitSl ? trade.sl : trade.tp;
        await this.close(String(trade._id), {
          exitPrice,
          exitDate: bar.date,
          exitReason: hitSl ? 'SL' : 'TP',
        });
        closed += 1;
        break;
      }
    }
    return { checked: open.length, closed };
  }

  async remove(id: string) {
    await this.trades.findByIdAndDelete(id).exec();
    return { ok: true };
  }
}
