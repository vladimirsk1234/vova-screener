import { Controller, Get, Module, Param, Query } from '@nestjs/common';
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

  /** Fast Graphs–style fundamental valuation. Reads Mongo; FMP only on a first miss. */
  @Get(':ticker/fundamentals')
  fundamentals(@Param('ticker') ticker: string, @Query('metric') metric?: string) {
    const allowed: ValuationMetric[] = ['eps', 'revenue', 'fcf', 'ownerEarnings'];
    const m = allowed.includes(metric as ValuationMetric) ? (metric as ValuationMetric) : 'eps';
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
