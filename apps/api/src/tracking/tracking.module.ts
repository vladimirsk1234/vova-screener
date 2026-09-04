import { Module } from '@nestjs/common';
import { InstrumentsModule } from '../instruments/instruments.module';
import { MarketModule } from '../market/market.module';
import { SettingsModule } from '../settings/settings.module';
import { UniverseModule } from '../universe/universe.module';
import { BenchmarkService } from './benchmark.service';
import { HistoryRebuildService } from './history-rebuild.service';
import { HistoryEpsService } from './history-eps.service';
import { HistoryPremiumService } from './history-premium.service';
import { HistoryService } from './history.service';
import { ResultsService } from './results.service';
import { SignalTrackerService } from './signal-tracker.service';
import { HistoryController, ResultsController } from './tracking.controller';

@Module({
  imports: [MarketModule, SettingsModule, UniverseModule, InstrumentsModule],
  controllers: [ResultsController, HistoryController],
  providers: [
    SignalTrackerService,
    ResultsService,
    HistoryService,
    HistoryRebuildService,
    HistoryEpsService,
    HistoryPremiumService,
    BenchmarkService,
  ],
  exports: [SignalTrackerService, ResultsService],
})
export class TrackingModule {}
