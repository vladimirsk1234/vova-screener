import { Module } from '@nestjs/common';
import { BarsService } from './bars.service';
import { FmpClient } from './fmp.client';
import { YahooClient } from './yahoo.client';

@Module({
  providers: [YahooClient, FmpClient, BarsService],
  exports: [YahooClient, FmpClient, BarsService],
})
export class MarketModule {}
