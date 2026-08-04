import { Body, Controller, Get, Param, Patch, Post, Query } from '@nestjs/common';
import type { Timeframe } from '@vova/engine';
import { HistoryRebuildService } from './history-rebuild.service';
import {
  HistoryService,
  type HistoryRange,
  type HistoryTf,
  type PeriodSort,
  type SortDir,
  type TradeSort,
} from './history.service';
import { ResultsService, type SortKey } from './results.service';
import { BUCKETS, TIMEFRAMES, UNIVERSES, type Bucket, type Interest, type TrackedUniverse } from './tracked-signal';

const HISTORY_RANGES: readonly HistoryRange[] = ['all', 'ytd', '1m', '3m', '6m', '1y', 'max'];

function parseUniverse(value?: string): TrackedUniverse {
  return UNIVERSES.includes(value as TrackedUniverse) ? (value as TrackedUniverse) : 'Stocks';
}

function parseTf(value?: string): Timeframe {
  return TIMEFRAMES.includes(value as Timeframe) ? (value as Timeframe) : 'Daily';
}

function parseHistoryTf(value?: string): HistoryTf {
  if (value === 'All') return 'All';
  return parseTf(value);
}

function parseHistoryRange(value?: string): HistoryRange {
  return HISTORY_RANGES.includes(value as HistoryRange) ? (value as HistoryRange) : 'all';
}

function parseBucket(value?: string): Bucket {
  return BUCKETS.includes(value as Bucket) ? (value as Bucket) : 'new';
}

function parseDir(value?: string): SortDir {
  return value === 'asc' ? 'asc' : 'desc';
}

function parseInt0(value?: string): number | undefined {
  const n = Number(value);
  return Number.isFinite(n) ? n : undefined;
}

@Controller('results')
export class ResultsController {
  constructor(private readonly results: ResultsService) {}

  @Get()
  list(
    @Query('universe') universe?: string,
    @Query('tf') tf?: string,
    @Query('bucket') bucket?: string,
    @Query('sort') sort?: string,
    @Query('dir') dir?: string,
    @Query('limit') limit?: string,
    @Query('offset') offset?: string,
  ) {
    return this.results.list({
      universe: parseUniverse(universe),
      tf: parseTf(tf),
      bucket: parseBucket(bucket),
      sort: (['rr', 'pnl', 'interest', 'symbol'] as SortKey[]).includes(sort as SortKey)
        ? (sort as SortKey)
        : 'rr',
      dir: parseDir(dir),
      limit: parseInt0(limit),
      offset: parseInt0(offset),
    });
  }

  @Get('summary')
  summary() {
    return this.results.summary();
  }

  @Get('lookup')
  lookup(@Query('yahooTicker') yahooTicker: string, @Query('tf') tf?: string) {
    return this.results.lookup(yahooTicker, parseTf(tf));
  }

  @Get('signal/:id')
  byId(@Param('id') id: string) {
    return this.results.byId(id);
  }

  @Patch(':id/interest')
  setInterest(@Param('id') id: string, @Body() body: { interest?: Interest | null }) {
    const value = body?.interest;
    const interest = value === 'interested' || value === 'not_interested' ? value : null;
    return this.results.setInterest(id, interest);
  }
}

@Controller('history')
export class HistoryController {
  constructor(
    private readonly history: HistoryService,
    private readonly rebuild: HistoryRebuildService,
  ) {}

  @Get('rebuild')
  rebuildStatus() {
    return this.rebuild.status();
  }

  @Post('rebuild')
  startRebuild() {
    return this.rebuild.start();
  }

  @Get()
  report(
    @Query('universe') universe?: string,
    @Query('tf') tf?: string,
    @Query('groupBy') groupBy?: string,
    @Query('range') range?: string,
    @Query('sort') sort?: string,
    @Query('dir') dir?: string,
  ) {
    return this.history.report({
      universe: parseUniverse(universe),
      tf: parseHistoryTf(tf),
      groupBy: parseTf(groupBy ?? tf),
      range: parseHistoryRange(range),
      sort: (['period', 'pnl', 'winRate', 'trades', 'rr'] as PeriodSort[]).includes(
        sort as PeriodSort,
      )
        ? (sort as PeriodSort)
        : 'period',
      dir: parseDir(dir),
    });
  }

  @Get('trades')
  trades(
    @Query('universe') universe?: string,
    @Query('tf') tf?: string,
    @Query('periodKey') periodKey?: string,
    @Query('groupBy') groupBy?: string,
    @Query('range') range?: string,
    @Query('sort') sort?: string,
    @Query('dir') dir?: string,
    @Query('limit') limit?: string,
    @Query('offset') offset?: string,
  ) {
    return this.history.trades({
      universe: parseUniverse(universe),
      tf: parseHistoryTf(tf),
      periodKey,
      groupBy: parseTf(groupBy ?? tf),
      range: parseHistoryRange(range),
      sort: (['date', 'pnl', 'r', 'rr', 'interest', 'symbol'] as TradeSort[]).includes(
        sort as TradeSort,
      )
        ? (sort as TradeSort)
        : 'date',
      dir: parseDir(dir),
      limit: parseInt0(limit),
      offset: parseInt0(offset),
    });
  }
}
