import { Module } from '@nestjs/common';
import { ScheduleModule } from '@nestjs/schedule';
import { MarketModule } from '../market/market.module';
import { TradesModule } from '../trades/trades.module';
import { UniverseModule } from '../universe/universe.module';
import { PeriodSchedulerService } from './period-scheduler.service';
import { ProgressBus } from './progress.bus';
import { ScanRunnerService } from './scan-runner.service';
import { ScansController } from './scans.controller';
import { ScansService } from './scans.service';

@Module({
  imports: [MarketModule, UniverseModule, TradesModule, ScheduleModule.forRoot()],
  controllers: [ScansController],
  providers: [ScansService, ScanRunnerService, ProgressBus, PeriodSchedulerService],
  exports: [ScansService],
})
export class ScansModule {}
