/** Trade journal with mark-to-market, TP/SL, sell-to-close, and interest → open at period end. */
import { Injectable, NotFoundException } from '@nestjs/common';
import { InjectModel } from '@nestjs/mongoose';
import { Types, type Model } from 'mongoose';
import { runStructureOverlay, type Timeframe } from '@vova/engine';
import { SCAN_RUN, SIGNAL, TRADE } from '../db/schemas';
import { BarsService } from '../market/bars.service';

export type TradeStatus = 'interested' | 'not_interested' | 'open' | 'closed' | 'dismissed';

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
  status?: TradeStatus;
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
    const tf = dto.tf ?? 'Daily';
    const status: TradeStatus = dto.status ?? 'open';
    const runId =
      dto.runId && Types.ObjectId.isValid(dto.runId) ? new Types.ObjectId(dto.runId) : undefined;

    // Upsert interest marks for the same ticker + tf + period.
    if (
      (status === 'interested' || status === 'not_interested') &&
      dto.periodKey &&
      dto.yahooTicker
    ) {
      const existing = await this.trades
        .findOne({
          yahooTicker: dto.yahooTicker,
          tf,
          periodKey: dto.periodKey,
          status: { $in: ['interested', 'not_interested'] },
        })
        .exec();
      if (existing) {
        existing.symbol = dto.symbol;
        existing.companyName = dto.companyName ?? existing.companyName;
        existing.entry = dto.entry;
        existing.tp = dto.tp;
        existing.sl = dto.sl;
        existing.rrAtEntry = dto.rrAtEntry;
        existing.shares = dto.shares ?? 0;
        existing.riskUsd = dto.riskUsd ?? 0;
        existing.asOf = dto.asOf;
        existing.status = status;
        existing.source = dto.source ?? 'manual';
        existing.runId = runId ?? existing.runId;
        existing.openedAt = new Date();
        await existing.save();
        return existing.toObject();
      }
    }

    return this.trades.create({
      ...dto,
      tf,
      shares: dto.shares ?? 0,
      riskUsd: dto.riskUsd ?? 0,
      status,
      source: dto.source ?? 'manual',
      openedAt: new Date(),
      runId,
    });
  }

  async list(status?: TradeStatus, tf?: Timeframe) {
    const filter: Record<string, unknown> = status
      ? { status }
      : { status: { $in: ['open', 'closed'] } };
    if (tf) filter.tf = tf;

    const rows = await this.trades.find(filter).sort({ openedAt: -1 }).lean().exec();
    const marked = await Promise.all(
      rows.map(async (t: any) => {
        const tradeTf = (t.tf as Timeframe) ?? 'Daily';
        const investedUsd = round2((t.entry ?? 0) * (t.shares || 0));
        if (t.status !== 'open') {
          return { ...t, investedUsd };
        }

        const result = await this.bars.getBars(t.yahooTicker, tradeTf, { maxAgeHours: 6 });
        const price = result.bars?.length ? result.bars[result.bars.length - 1].close : null;
        if (price == null) {
          return {
            ...t,
            investedUsd,
            currentPrice: null,
            unrealizedUsd: null,
            unrealizedR: null,
            unrealizedPct: null,
          };
        }
        const unrealizedUsd = round2((price - t.entry) * (t.shares || 0));
        const risk = t.entry - (t.sl ?? t.entry);
        return {
          ...t,
          investedUsd,
          currentPrice: round2(price),
          unrealizedUsd,
          unrealizedR: risk > 0 ? round2((price - t.entry) / risk) : null,
          unrealizedPct:
            investedUsd > 0 ? round2((unrealizedUsd / investedUsd) * 100) : null,
        };
      }),
    );
    return marked;
  }

  /** Interest marks for a calendar period (Results filtering / badges). */
  async interestMarks(tf: Timeframe, periodKey: string) {
    if (!periodKey) return { interested: [] as string[], notInterested: [] as string[] };
    const rows = await this.trades
      .find({
        tf,
        periodKey,
        status: { $in: ['interested', 'not_interested'] },
      })
      .lean<any>()
      .exec();
    const interested: string[] = [];
    const notInterested: string[] = [];
    const pushKeys = (arr: string[], row: any) => {
      if (row.yahooTicker) arr.push(row.yahooTicker);
      if (row.symbol && row.symbol !== row.yahooTicker) arr.push(row.symbol);
    };
    for (const row of rows) {
      if (row.status === 'interested') pushKeys(interested, row);
      else pushKeys(notInterested, row);
    }
    return { interested, notInterested };
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
   * At period end: promote interested marks that still have a valid buy signal → open
   * with updated prices; dismiss interested that no longer qualify.
   */
  async promoteInterested(runId: string) {
    if (!Types.ObjectId.isValid(runId)) return { promoted: 0, dismissed: 0 };
    const run = await this.runs.findById(runId).lean<any>().exec();
    if (!run) return { promoted: 0, dismissed: 0 };
    if (run.params?.direction !== 'buy') return { promoted: 0, dismissed: 0 };
    if (run.status !== 'completed') return { promoted: 0, dismissed: 0 };

    const tf = (run.params.tf as Timeframe) ?? 'Daily';
    const periodKeyVal = run.periodKey as string | undefined;
    if (!periodKeyVal) return { promoted: 0, dismissed: 0 };

    const interested = await this.trades
      .find({ tf, periodKey: periodKeyVal, status: 'interested' })
      .exec();

    const buySignals = await this.signals
      .find({ runId: new Types.ObjectId(runId), kind: 'buy' })
      .lean<any>()
      .exec();
    const byTicker = new Map<string, any>();
    for (const row of buySignals) {
      const payload = row.payload;
      const keys = [
        payload?.yahooTicker,
        row.yahooTicker,
        payload?.symbol,
        row.symbol,
      ].filter(Boolean);
      for (const k of keys) byTicker.set(k, payload ?? row);
    }

    let promoted = 0;
    let dismissed = 0;

    for (const trade of interested) {
      const signal =
        byTicker.get(trade.yahooTicker) || byTicker.get(trade.symbol);

      const open = await this.trades
        .findOne({ symbol: trade.symbol, tf, status: 'open' })
        .lean()
        .exec();
      if (open) {
        trade.status = 'dismissed';
        trade.exitReason = 'already_open';
        await trade.save();
        dismissed += 1;
        continue;
      }

      if (signal?.entry != null && Number.isFinite(signal.entry)) {
        trade.entry = signal.entry;
        trade.tp = signal.tp;
        trade.sl = signal.sl;
        trade.rrAtEntry = signal.rr ?? undefined;
        trade.shares = signal.shares ?? trade.shares;
        trade.riskUsd = run.params.riskPerTrade ?? trade.riskUsd;
        trade.asOf = signal.asOf ?? trade.asOf;
        trade.companyName = signal.companyName ?? trade.companyName;
        trade.status = 'open';
        trade.source = 'auto';
        trade.runId = new Types.ObjectId(runId);
        trade.openedAt = new Date();
        await trade.save();
        promoted += 1;
      } else {
        trade.status = 'dismissed';
        trade.exitReason = 'signal_invalid';
        await trade.save();
        dismissed += 1;
      }
    }

    return { promoted, dismissed };
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
