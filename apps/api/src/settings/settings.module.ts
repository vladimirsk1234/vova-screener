/** The app's user-facing settings: risk per signal and the RR floor for lists/stats. */
import { Body, Controller, Get, Injectable, Module, Put } from '@nestjs/common';
import { InjectModel } from '@nestjs/mongoose';
import type { Model } from 'mongoose';
import { PRESET } from '../db/schemas';

export type AppSettings = {
  maxRiskUsd: number;
  /** Floor on live RR for NEW/VALID (and History active); CLOSED/History closed use entry RR. 0 = no filter. */
  minRr: number;
};

export type SettingsListener = (next: AppSettings, prev: AppSettings) => Promise<void> | void;

export const DEFAULT_SETTINGS: AppSettings = { maxRiskUsd: 100, minRr: 0 };

const SETTINGS_KEY = 'app';

function sanitize(patch: Partial<AppSettings>): Partial<AppSettings> {
  const out: Partial<AppSettings> = {};
  const risk = Number(patch.maxRiskUsd);
  if (Number.isFinite(risk) && risk > 0) out.maxRiskUsd = Math.round(risk * 100) / 100;
  if (patch.minRr !== undefined) {
    const minRr = Number(patch.minRr);
    if (Number.isFinite(minRr) && minRr >= 0) out.minRr = Math.round(minRr * 100) / 100;
  }
  return out;
}

@Injectable()
export class SettingsService {
  private readonly listeners: SettingsListener[] = [];

  constructor(@InjectModel(PRESET) private readonly presets: Model<any>) {}

  /**
   * Lets tracking re-size open positions when the risk changes, without SettingsModule having to
   * depend on TrackingModule — the dependency already runs the other way.
   */
  onChange(listener: SettingsListener) {
    this.listeners.push(listener);
  }

  async get(): Promise<AppSettings> {
    const doc = await this.presets.findOne({ key: SETTINGS_KEY }).lean<any>().exec();
    return { ...DEFAULT_SETTINGS, ...(doc?.data ?? {}) };
  }

  async put(patch: Partial<AppSettings>): Promise<AppSettings> {
    const prev = await this.get();
    const next = { ...prev, ...sanitize(patch) };
    await this.presets
      .updateOne({ key: SETTINGS_KEY }, { $set: { key: SETTINGS_KEY, data: next } }, { upsert: true })
      .exec();
    // Awaited, so the response the UI refetches against already reflects the new sizes.
    for (const listener of this.listeners) await listener(next, prev);
    return next;
  }
}

@Controller('settings')
class SettingsController {
  constructor(private readonly settings: SettingsService) {}

  @Get()
  get() {
    return this.settings.get();
  }

  @Put()
  put(@Body() body: Partial<AppSettings>) {
    return this.settings.put(body ?? {});
  }
}

@Module({
  controllers: [SettingsController],
  providers: [SettingsService],
  exports: [SettingsService],
})
export class SettingsModule {}
