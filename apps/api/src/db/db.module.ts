import { Global, Module } from '@nestjs/common';
import { MongooseModule } from '@nestjs/mongoose';
import { resolveMongoUri } from './local-mongo';
import {
  BAR_SERIES,
  BarSeriesSchema,
  FUNDAMENTALS_REFRESH_RUN,
  FundamentalsRefreshRunSchema,
  INSTRUMENT,
  INSTRUMENT_FUNDAMENTALS,
  InstrumentFundamentalsSchema,
  InstrumentSchema,
  PRESET,
  PresetSchema,
  REJECTION,
  RejectionSchema,
  SCAN_RUN,
  ScanRunSchema,
  SIGNAL,
  SignalSchema,
  TRACKED_SIGNAL,
  TrackedSignalSchema,
} from './schemas';

const models = MongooseModule.forFeature([
  { name: INSTRUMENT, schema: InstrumentSchema },
  { name: INSTRUMENT_FUNDAMENTALS, schema: InstrumentFundamentalsSchema },
  { name: FUNDAMENTALS_REFRESH_RUN, schema: FundamentalsRefreshRunSchema },
  { name: BAR_SERIES, schema: BarSeriesSchema },
  { name: SCAN_RUN, schema: ScanRunSchema },
  { name: SIGNAL, schema: SignalSchema },
  { name: REJECTION, schema: RejectionSchema },
  { name: TRACKED_SIGNAL, schema: TrackedSignalSchema },
  { name: PRESET, schema: PresetSchema },
]);

@Global()
@Module({
  imports: [
    MongooseModule.forRootAsync({
      useFactory: async () => ({ uri: await resolveMongoUri() }),
    }),
    models,
  ],
  exports: [models],
})
export class DbModule {}
