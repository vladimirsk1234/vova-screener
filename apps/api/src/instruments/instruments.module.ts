import { Body, Controller, Get, Module, Param, Patch, Query } from '@nestjs/common';
import { ScheduleModule } from '@nestjs/schedule';
import type { IndicatorParams, Timeframe, ValuationMetric } from '@vova/engine';
import { MarketModule } from '../market/market.module';
import { UniverseModule } from '../universe/universe.module';
import { FundamentalsRefreshService } from './fundamentals-refresh.service';
import { FundamentalsService } from './fundamentals.service';
import { InstrumentsService } from './instruments.service';

@Controller('instruments')
class InstrumentsController {
  constructor(
    private readonly instruments: InstrumentsService,
    private readonly fundamentalsSvc: FundamentalsService,
    private readonly fundamentalsRefresh: FundamentalsRefreshService,
  ) {}

  /**
   * Batch slim valuation for Results / History cards.
   * Must be registered above `:ticker/...` or Nest treats the path as a ticker.
   */
  @Get('fundamentals-cards')
  fundamentalsCards(@Query('tickers') tickers?: string) {
    const list = String(tickers ?? '')
      .split(',')
      .map((t) => t.trim())
      .filter(Boolean);
    return this.fundamentalsSvc.getCardMetrics(list);
  }

  /**
   * Fundamental Value screener: STOCK-TICKERS ranked by how many of EPS/FCF/DCF/LT D/C score a star.
   */
  @Get('fundamentals-screener')
  fundamentalsScreener(
    @Query('stars') stars?: string,
    @Query('sort') sort?: string,
    @Query('dir') dir?: string,
    @Query('limit') limit?: string,
    @Query('offset') offset?: string,
  ) {
    const starFilters = ['undervalued', '0', '1', '2', '3', '4', 'all', 'garp'] as const;
    const sorts = ['stars', 'eps', 'fcf', 'dcf', 'symbol', 'interest'] as const;
    const dirs = ['asc', 'desc'] as const;
    const star = starFilters.includes(stars as (typeof starFilters)[number])
      ? (stars as (typeof starFilters)[number])
      : 'undervalued';
    const s = sorts.includes(sort as (typeof sorts)[number])
      ? (sort as (typeof sorts)[number])
      : 'stars';
    const d = dirs.includes(dir as (typeof dirs)[number])
      ? (dir as (typeof dirs)[number])
      : 'desc';
    const lim = Number(limit);
    const off = Number(offset);
    return this.fundamentalsSvc.listScreener({
      stars: star,
      sort: s,
      dir: d,
      limit: Number.isFinite(lim) ? lim : undefined,
      offset: Number.isFinite(off) ? off : undefined,
    });
  }

  /** EOD / catch-up progress — same numbers the Value tab shows. Do not kick from this poll. */
  @Get('fundamentals-refresh')
  fundamentalsRefreshStatus() {
    return this.fundamentalsSvc.refreshStatus();
  }

  @Get(':ticker/chart')
  chart(
    @Param('ticker') ticker: string,
    @Query('tf') tf?: string,
    @Query('minRr') minRr?: string,
    @Query('useLastHlSl') useLastHlSl?: string,
    @Query('riskPerTrade') riskPerTrade?: string,
    @Query('noRrReq') noRrReq?: string,
    @Query('asOf') asOf?: string,
    @Query('lenFast') lenFast?: string,
    @Query('lenSlow') lenSlow?: string,
    @Query('lengthMajor') lengthMajor?: string,
    @Query('lookback') lookback?: string,
    @Query('multiplier') multiplier?: string,
    @Query('bbLength') bbLength?: string,
    @Query('bbMult') bbMult?: string,
    @Query('fullSeries') fullSeries?: string,
  ) {
    const chartParams: Partial<IndicatorParams> = {};
    const num = (raw: string | undefined) => {
      if (raw == null || raw === '') return undefined;
      const v = Number(raw);
      return Number.isFinite(v) ? v : undefined;
    };
    const assign = <K extends keyof IndicatorParams>(key: K, raw: string | undefined) => {
      const v = num(raw);
      if (v !== undefined) chartParams[key] = v as IndicatorParams[K];
    };
    assign('len_fast', lenFast);
    assign('len_slow', lenSlow);
    assign('length_major', lengthMajor);
    assign('lookback', lookback);
    assign('multiplier', multiplier);
    assign('bb_length', bbLength);
    assign('bb_mult', bbMult);

    return this.instruments.chart(ticker, (tf as Timeframe) || 'Daily', {
      minRr: num(minRr),
      useLastHlSl: useLastHlSl ? useLastHlSl === 'true' : undefined,
      riskPerTrade: num(riskPerTrade),
      noRrReq: noRrReq ? noRrReq === 'true' : undefined,
      asOf: /^\d{4}-\d{2}-\d{2}$/.test(asOf ?? '') ? asOf : undefined,
      chartParams,
      fullSeries: fullSeries === '1' || fullSeries === 'true',
    });
  }

  @Get(':ticker/status')
  status(@Param('ticker') ticker: string) {
    return this.instruments.status(ticker);
  }

  /** Ticker-level Interested / Not Interested for the Value tab (not a tracked signal). */
  @Get(':ticker/interest')
  getTickerInterest(@Param('ticker') ticker: string) {
    return this.fundamentalsSvc.getTickerInterest(ticker);
  }

  @Patch(':ticker/interest')
  setTickerInterest(
    @Param('ticker') ticker: string,
    @Body() body: { interest?: 'interested' | 'not_interested' | null },
  ) {
    const value = body?.interest;
    const interest = value === 'interested' || value === 'not_interested' ? value : null;
    return this.fundamentalsSvc.setTickerInterest(ticker, interest);
  }

  /** Fast Graphs–style fundamental valuation. Reads Mongo; FMP only for unknown Manual tickers. */
  @Get(':ticker/fundamentals')
  fundamentals(@Param('ticker') ticker: string, @Query('metric') metric?: string) {
    const allowed: ValuationMetric[] = ['eps', 'operatingEps', 'revenue', 'fcf', 'ownerEarnings'];
    const m = allowed.includes(metric as ValuationMetric) ? (metric as ValuationMetric) : 'eps';
    this.fundamentalsRefresh.kickIfNeeded();
    return this.fundamentalsSvc.get(ticker, m);
  }

  /**
   * Unlevered Custom DCF. Hits FMP on miss of a 1h in-memory cache keyed by ticker+assumptions.
   * Not stored in Mongo and not part of the scheduled fundamentals refresh.
   */
  @Get(':ticker/dcf')
  customDcf(@Param('ticker') ticker: string, @Query() query: Record<string, string>) {
    return this.fundamentalsSvc.getCustomDcf(ticker, query);
  }
}

@Module({
  imports: [MarketModule, UniverseModule, ScheduleModule.forRoot()],
  controllers: [InstrumentsController],
  providers: [InstrumentsService, FundamentalsService, FundamentalsRefreshService],
  exports: [FundamentalsService],
})
export class InstrumentsModule {}
