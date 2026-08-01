import { Module } from '@nestjs/common';
import { MarketModule } from '../market/market.module';
import { SettingsModule } from '../settings/settings.module';
import { HistoryService } from './history.service';
import { ResultsService } from './results.service';
import { SignalTrackerService } from './signal-tracker.service';
import { HistoryController, ResultsController } from './tracking.controller';

@Module({
  imports: [MarketModule, SettingsModule],
  controllers: [ResultsController, HistoryController],
  providers: [SignalTrackerService, ResultsService, HistoryService],
  exports: [SignalTrackerService, ResultsService],
})
export class TrackingModule {}
