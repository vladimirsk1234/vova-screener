import { Module } from '@nestjs/common';
import { ScheduleModule } from '@nestjs/schedule';
import { MarketModule } from '../market/market.module';
import { SettingsModule } from '../settings/settings.module';
import { TrackingModule } from '../tracking/tracking.module';
import { UniverseModule } from '../universe/universe.module';
import { PeriodSchedulerService } from './period-scheduler.service';
import { ProgressBus } from './progress.bus';
import { ScanRunnerService } from './scan-runner.service';
import { ScansController } from './scans.controller';
import { ScansService } from './scans.service';

@Module({
  imports: [
    MarketModule,
    UniverseModule,
    TrackingModule,
    SettingsModule,
    ScheduleModule.forRoot(),
  ],
  controllers: [ScansController],
  providers: [ScansService, ScanRunnerService, ProgressBus, PeriodSchedulerService],
  exports: [ScansService],
})
export class ScansModule {}
