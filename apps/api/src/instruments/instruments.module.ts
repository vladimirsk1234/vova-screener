import { Controller, Get, Module, Param, Query } from '@nestjs/common';
import type { IndicatorParams, Timeframe } from '@vova/engine';
import { MarketModule } from '../market/market.module';
import { UniverseModule } from '../universe/universe.module';
import { InstrumentsService } from './instruments.service';

@Controller('instruments')
class InstrumentsController {
  constructor(private readonly instruments: InstrumentsService) {}

  @Get(':ticker/chart')
  chart(
    @Param('ticker') ticker: string,
    @Query('tf') tf?: string,
    @Query('minRr') minRr?: string,
    @Query('useLastHlSl') useLastHlSl?: string,
    @Query('riskPerTrade') riskPerTrade?: string,
    @Query('noRrReq') noRrReq?: string,
    @Query('lenFast') lenFast?: string,
    @Query('lenSlow') lenSlow?: string,
    @Query('lengthMajor') lengthMajor?: string,
    @Query('lookback') lookback?: string,
    @Query('multiplier') multiplier?: string,
    @Query('bbLength') bbLength?: string,
    @Query('bbMult') bbMult?: string,
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
      chartParams,
    });
  }

  @Get(':ticker/status')
  status(@Param('ticker') ticker: string) {
    return this.instruments.status(ticker);
  }
}

@Module({
  imports: [MarketModule, UniverseModule],
  controllers: [InstrumentsController],
  providers: [InstrumentsService],
})
export class InstrumentsModule {}
