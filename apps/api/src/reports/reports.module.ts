import { Controller, Get, Injectable, Module, Query, Res } from '@nestjs/common';
import { InjectModel } from '@nestjs/mongoose';
import type { Response } from 'express';
import type { Model } from 'mongoose';
import type { Timeframe } from '@vova/engine';
import { TRADE } from '../db/schemas';
import { isoWeekKey } from '../scans/period';

function round2(n: number) {
  return Math.round(n * 100) / 100;
}

function round4(n: number) {
  return Math.round(n * 10_000) / 10_000;
}

function toDateStr(value: unknown): string | null {
  if (!value) return null;
  if (typeof value === 'string') return value.slice(0, 10);
  if (value instanceof Date && !Number.isNaN(value.getTime())) {
    return value.toISOString().slice(0, 10);
  }
  const d = new Date(String(value));
  return Number.isNaN(d.getTime()) ? null : d.toISOString().slice(0, 10);
}

function bucketKey(
  tf: Timeframe,
  trade: { periodKey?: string; exitDate?: string; openedAt?: Date | string; asOf?: string },
): string {
  if (trade.periodKey) return trade.periodKey;
  const raw = trade.exitDate || trade.asOf || toDateStr(trade.openedAt) || '';
  if (!raw || raw === 'unknown') return 'unknown';
  if (tf === 'Daily') return raw.slice(0, 10);
  if (tf === 'Monthly') return raw.slice(0, 7);
  const [y, m, d] = raw.slice(0, 10).split('-').map(Number);
  if (!y || !m || !d) return raw;
  return isoWeekKey(y, m, d);
}

/** Holding length in TF units: days / weeks / months. */
function holdUnits(
  tf: Timeframe,
  trade: { asOf?: string; openedAt?: Date | string; exitDate?: string },
): number | null {
  const start = trade.asOf || toDateStr(trade.openedAt);
  const end = trade.exitDate;
  if (!start || !end) return null;
  const ms = Date.parse(`${end}T12:00:00Z`) - Date.parse(`${start}T12:00:00Z`);
  if (!Number.isFinite(ms) || ms < 0) return null;
  const days = ms / 86_400_000;
  if (tf === 'Daily') return days;
  if (tf === 'Weekly') return days / 7;
  return days / 30.4375;
}

function holdUnitLabel(tf: Timeframe) {
  if (tf === 'Daily') return 'days';
  if (tf === 'Weekly') return 'weeks';
  return 'months';
}

type PeriodBucket = {
  periodKey: string;
  trades: number;
  wins: number;
  pnlUsd: number;
  rSum: number;
  rCount: number;
  entryRrSum: number;
  entryRrCount: number;
  exitRrSum: number;
  exitRrCount: number;
  holdSum: number;
  holdCount: number;
};

function emptyBucket(periodKey: string): PeriodBucket {
  return {
    periodKey,
    trades: 0,
    wins: 0,
    pnlUsd: 0,
    rSum: 0,
    rCount: 0,
    entryRrSum: 0,
    entryRrCount: 0,
    exitRrSum: 0,
    exitRrCount: 0,
    holdSum: 0,
    holdCount: 0,
  };
}

function finalizePeriod(p: PeriodBucket) {
  return {
    periodKey: p.periodKey,
    trades: p.trades,
    wins: p.wins,
    winRatePct: p.trades ? round2((p.wins / p.trades) * 100) : 0,
    pnlUsd: round2(p.pnlUsd),
    avgR: p.rCount ? round2(p.rSum / p.rCount) : null,
    avgRrEntry: p.entryRrCount ? round2(p.entryRrSum / p.entryRrCount) : null,
    avgRrExit: p.exitRrCount ? round2(p.exitRrSum / p.exitRrCount) : null,
    avgHold: p.holdCount ? round2(p.holdSum / p.holdCount) : null,
  };
}

function accumulateTrade(bucket: PeriodBucket, t: any, tf: Timeframe) {
  bucket.trades += 1;
  if ((t.pnlUsd ?? 0) > 0) bucket.wins += 1;
  bucket.pnlUsd += t.pnlUsd ?? 0;
  if (t.pnlR != null) {
    bucket.rSum += t.pnlR;
    bucket.rCount += 1;
    bucket.exitRrSum += t.pnlR;
    bucket.exitRrCount += 1;
  }
  if (t.rrAtEntry != null && Number.isFinite(t.rrAtEntry)) {
    bucket.entryRrSum += t.rrAtEntry;
    bucket.entryRrCount += 1;
  }
  const hold = holdUnits(tf, t);
  if (hold != null) {
    bucket.holdSum += hold;
    bucket.holdCount += 1;
  }
}

function csvEscape(value: unknown): string {
  const s = value == null ? '' : String(value);
  if (/[",\n\r]/.test(s)) return `"${s.replace(/"/g, '""')}"`;
  return s;
}

@Injectable()
class ReportsService {
  constructor(@InjectModel(TRADE) private readonly trades: Model<any>) {}

  private async closedForTf(tf: Timeframe) {
    return this.trades
      .find({ status: 'closed', tf })
      .sort({ exitDate: 1, openedAt: 1 })
      .lean()
      .exec();
  }

  async performance(tf: Timeframe = 'Daily') {
    const closed = (await this.closedForTf(tf)) as any[];
    const byPeriod = new Map<string, PeriodBucket>();
    let cumulative = 0;
    const equity: Array<{ date: string; equity: number }> = [];
    const totalsBucket = emptyBucket('_totals');

    for (const t of closed) {
      const key = bucketKey(tf, t);
      const bucket = byPeriod.get(key) ?? emptyBucket(key);
      accumulateTrade(bucket, t, tf);
      byPeriod.set(key, bucket);
      accumulateTrade(totalsBucket, t, tf);
      cumulative += t.pnlUsd ?? 0;
      equity.push({ date: t.exitDate ?? key, equity: round2(cumulative) });
    }

    const periods = [...byPeriod.values()]
      .map(finalizePeriod)
      .sort((a, b) => b.periodKey.localeCompare(a.periodKey));

    const openCount = await this.trades.countDocuments({ status: 'open', tf }).exec();
    const finalized = finalizePeriod(totalsBucket);

    return {
      tf,
      holdUnit: holdUnitLabel(tf),
      periods,
      equity,
      totals: {
        closed: finalized.trades,
        open: openCount,
        wins: finalized.wins,
        winRatePct: finalized.winRatePct,
        pnlUsd: finalized.pnlUsd,
        avgR: finalized.avgR,
        avgRrEntry: finalized.avgRrEntry,
        avgRrExit: finalized.avgRrExit,
        avgHold: finalized.avgHold,
      },
    };
  }

  async exportCsv(tf: Timeframe = 'Daily') {
    const closed = (await this.closedForTf(tf)) as any[];
    const report = await this.performance(tf);
    const unit = holdUnitLabel(tf);
    const lines: string[] = [];

    lines.push(`# Performance report — ${tf}`);
    lines.push(
      [
        'section',
        'periodKey',
        'trades',
        'wins',
        'winRatePct',
        'pnlUsd',
        'avgR',
        'avgRrEntry',
        'avgRrExit',
        `avgHold_${unit}`,
      ].join(','),
    );
    lines.push(
      [
        'totals',
        '',
        report.totals.closed,
        report.totals.wins,
        report.totals.winRatePct,
        report.totals.pnlUsd,
        report.totals.avgR ?? '',
        report.totals.avgRrEntry ?? '',
        report.totals.avgRrExit ?? '',
        report.totals.avgHold ?? '',
      ]
        .map(csvEscape)
        .join(','),
    );
    for (const p of report.periods) {
      lines.push(
        [
          'period',
          p.periodKey,
          p.trades,
          p.wins,
          p.winRatePct,
          p.pnlUsd,
          p.avgR ?? '',
          p.avgRrEntry ?? '',
          p.avgRrExit ?? '',
          p.avgHold ?? '',
        ]
          .map(csvEscape)
          .join(','),
      );
    }

    lines.push('');
    lines.push(
      [
        'section',
        'periodKey',
        'symbol',
        'yahooTicker',
        'companyName',
        'tf',
        'status',
        'source',
        'entry',
        'exitPrice',
        'shares',
        'rrAtEntry',
        'pnlR',
        'pnlUsd',
        'asOf',
        'exitDate',
        'exitReason',
        `hold_${unit}`,
        'openedAt',
      ].join(','),
    );

    const tradesSorted = [...closed].sort((a, b) => {
      const ka = bucketKey(tf, a);
      const kb = bucketKey(tf, b);
      return kb.localeCompare(ka) || String(a.symbol).localeCompare(String(b.symbol));
    });

    for (const t of tradesSorted) {
      const hold = holdUnits(tf, t);
      lines.push(
        [
          'trade',
          bucketKey(tf, t),
          t.symbol,
          t.yahooTicker,
          t.companyName ?? '',
          t.tf,
          t.status,
          t.source ?? '',
          t.entry,
          t.exitPrice ?? '',
          t.shares ?? '',
          t.rrAtEntry ?? '',
          t.pnlR ?? '',
          t.pnlUsd ?? '',
          t.asOf ?? '',
          t.exitDate ?? '',
          t.exitReason ?? '',
          hold != null ? round4(hold) : '',
          toDateStr(t.openedAt) ?? '',
        ]
          .map(csvEscape)
          .join(','),
      );
    }

    return lines.join('\n');
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

function parseTf(tf?: string): Timeframe {
  const allowed: Timeframe[] = ['Daily', 'Weekly', 'Monthly'];
  return allowed.includes(tf as Timeframe) ? (tf as Timeframe) : 'Daily';
}

@Controller('reports')
class ReportsController {
  constructor(private readonly reports: ReportsService) {}

  @Get('performance')
  performance(@Query('tf') tf?: Timeframe) {
    return this.reports.performance(parseTf(tf));
  }

  @Get('export')
  async export(@Query('tf') tf: string | undefined, @Res() res: Response) {
    const selected = parseTf(tf);
    const csv = await this.reports.exportCsv(selected);
    const filename = `vova-pnl-${selected.toLowerCase()}-${new Date().toISOString().slice(0, 10)}.csv`;
    res.setHeader('Content-Type', 'text/csv; charset=utf-8');
    res.setHeader('Content-Disposition', `attachment; filename="${filename}"`);
    res.send(csv);
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
