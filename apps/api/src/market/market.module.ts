import { Module } from '@nestjs/common';
import { BarsService } from './bars.service';
import { YahooClient } from './yahoo.client';

@Module({
  providers: [YahooClient, BarsService],
  exports: [YahooClient, BarsService],
})
export class MarketModule {}
