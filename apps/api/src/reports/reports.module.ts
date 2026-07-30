import { Controller, Get, Injectable, Module } from '@nestjs/common';
import { InjectModel } from '@nestjs/mongoose';
import type { Model } from 'mongoose';
import { TRADE } from '../db/schemas';

function round2(n: number) {
  return Math.round(n * 100) / 100;
}

@Injectable()
class ReportsService {
  constructor(@InjectModel(TRADE) private readonly trades: Model<any>) {}

  async monthly() {
    const closed = await this.trades
      .find({ status: 'closed' })
      .sort({ exitDate: 1 })
      .lean()
      .exec();

    const byMonth = new Map<
      string,
      { month: string; trades: number; wins: number; pnlUsd: number; rSum: number; rCount: number }
    >();
    let cumulative = 0;
    const equity: Array<{ date: string; equity: number }> = [];

    for (const t of closed) {
      const month = (t.exitDate ?? '').slice(0, 7) || 'unknown';
      const bucket =
        byMonth.get(month) ??
        { month, trades: 0, wins: 0, pnlUsd: 0, rSum: 0, rCount: 0 };
      bucket.trades += 1;
      if ((t.pnlUsd ?? 0) > 0) bucket.wins += 1;
      bucket.pnlUsd += t.pnlUsd ?? 0;
      if (t.pnlR != null) {
        bucket.rSum += t.pnlR;
        bucket.rCount += 1;
      }
      byMonth.set(month, bucket);
      cumulative += t.pnlUsd ?? 0;
      equity.push({ date: t.exitDate ?? '', equity: round2(cumulative) });
    }

    const months = [...byMonth.values()].map((m) => ({
      month: m.month,
      trades: m.trades,
      wins: m.wins,
      winRatePct: m.trades ? round2((m.wins / m.trades) * 100) : 0,
      pnlUsd: round2(m.pnlUsd),
      avgR: m.rCount ? round2(m.rSum / m.rCount) : null,
    }));

    const openCount = await this.trades.countDocuments({ status: 'open' }).exec();
    const totalTrades = closed.length;
    const wins = closed.filter((t: any) => (t.pnlUsd ?? 0) > 0).length;
    const rValues = closed.filter((t: any) => t.pnlR != null).map((t: any) => t.pnlR);

    return {
      months,
      equity,
      totals: {
        closed: totalTrades,
        open: openCount,
        wins,
        winRatePct: totalTrades ? round2((wins / totalTrades) * 100) : 0,
        pnlUsd: round2(cumulative),
        avgR: rValues.length
          ? round2(rValues.reduce((a: number, b: number) => a + b, 0) / rValues.length)
          : null,
      },
    };
  }
}

@Controller('reports')
class ReportsController {
  constructor(private readonly reports: ReportsService) {}

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
