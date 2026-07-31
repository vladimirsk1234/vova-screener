import { Controller, Get, Injectable, Module, Query } from '@nestjs/common';
import { InjectModel } from '@nestjs/mongoose';
import type { Model } from 'mongoose';
import type { Timeframe } from '@vova/engine';
import { TRADE } from '../db/schemas';
import { isoWeekKey } from '../scans/period';

function round2(n: number) {
  return Math.round(n * 100) / 100;
}

function bucketKey(tf: Timeframe, trade: { periodKey?: string; exitDate?: string; openedAt?: Date | string }): string {
  if (trade.periodKey) return trade.periodKey;
  const raw = trade.exitDate || (trade.openedAt ? String(trade.openedAt).slice(0, 10) : '');
  if (!raw || raw === 'unknown') return 'unknown';
  if (tf === 'Daily') return raw.slice(0, 10);
  if (tf === 'Monthly') return raw.slice(0, 7);
  const [y, m, d] = raw.slice(0, 10).split('-').map(Number);
  if (!y || !m || !d) return raw;
  return isoWeekKey(y, m, d);
}

type PeriodBucket = {
  periodKey: string;
  trades: number;
  wins: number;
  pnlUsd: number;
  rSum: number;
  rCount: number;
};

@Injectable()
class ReportsService {
  constructor(@InjectModel(TRADE) private readonly trades: Model<any>) {}

  async performance(tf: Timeframe = 'Daily') {
    const closed = await this.trades
      .find({ status: 'closed', tf })
      .sort({ exitDate: 1, openedAt: 1 })
      .lean()
      .exec();

    const byPeriod = new Map<string, PeriodBucket>();
    let cumulative = 0;
    const equity: Array<{ date: string; equity: number }> = [];

    for (const t of closed as any[]) {
      const key = bucketKey(tf, t);
      const bucket =
        byPeriod.get(key) ??
        { periodKey: key, trades: 0, wins: 0, pnlUsd: 0, rSum: 0, rCount: 0 };
      bucket.trades += 1;
      if ((t.pnlUsd ?? 0) > 0) bucket.wins += 1;
      bucket.pnlUsd += t.pnlUsd ?? 0;
      if (t.pnlR != null) {
        bucket.rSum += t.pnlR;
        bucket.rCount += 1;
      }
      byPeriod.set(key, bucket);
      cumulative += t.pnlUsd ?? 0;
      equity.push({ date: t.exitDate ?? key, equity: round2(cumulative) });
    }

    const periods = [...byPeriod.values()]
      .map((p) => ({
        periodKey: p.periodKey,
        trades: p.trades,
        wins: p.wins,
        winRatePct: p.trades ? round2((p.wins / p.trades) * 100) : 0,
        pnlUsd: round2(p.pnlUsd),
        avgR: p.rCount ? round2(p.rSum / p.rCount) : null,
      }))
      .sort((a, b) => b.periodKey.localeCompare(a.periodKey));

    const openCount = await this.trades.countDocuments({ status: 'open', tf }).exec();
    const totalTrades = closed.length;
    const wins = closed.filter((t: any) => (t.pnlUsd ?? 0) > 0).length;
    const rValues = closed.filter((t: any) => t.pnlR != null).map((t: any) => t.pnlR as number);

    return {
      tf,
      periods,
      equity,
      totals: {
        closed: totalTrades,
        open: openCount,
        wins,
        winRatePct: totalTrades ? round2((wins / totalTrades) * 100) : 0,
        pnlUsd: round2(cumulative),
        avgR: rValues.length
          ? round2(rValues.reduce((a, b) => a + b, 0) / rValues.length)
          : null,
      },
    };
  }

  /** @deprecated prefer performance(tf) — kept for older clients */
  async monthly() {
    const report = await this.performance('Monthly');
    return {
      months: report.periods.map((p) => ({
        month: p.periodKey,
        trades: p.trades,
        wins: p.wins,
        winRatePct: p.winRatePct,
        pnlUsd: p.pnlUsd,
        avgR: p.avgR,
      })),
      equity: report.equity,
      totals: report.totals,
    };
  }
}

@Controller('reports')
class ReportsController {
  constructor(private readonly reports: ReportsService) {}

  @Get('performance')
  performance(@Query('tf') tf?: Timeframe) {
    const allowed: Timeframe[] = ['Daily', 'Weekly', 'Monthly'];
    const selected = allowed.includes(tf as Timeframe) ? (tf as Timeframe) : 'Daily';
    return this.reports.performance(selected);
  }

  @Get('monthly')
  monthly() {
    return this.reports.monthly();
  }
}

@Module({
  controllers: [ReportsController],
  providers: [ReportsService],
})
export class ReportsModule {}
