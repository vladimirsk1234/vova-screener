/** The app's only user-facing setting: how much money one signal is allowed to risk. */
import { Body, Controller, Get, Injectable, Module, Put } from '@nestjs/common';
import { InjectModel } from '@nestjs/mongoose';
import type { Model } from 'mongoose';
import { PRESET } from '../db/schemas';

export type AppSettings = {
  maxRiskUsd: number;
};

export const DEFAULT_SETTINGS: AppSettings = { maxRiskUsd: 100 };

const SETTINGS_KEY = 'app';

function sanitize(patch: Partial<AppSettings>): Partial<AppSettings> {
  const out: Partial<AppSettings> = {};
  const risk = Number(patch.maxRiskUsd);
  if (Number.isFinite(risk) && risk > 0) out.maxRiskUsd = Math.round(risk * 100) / 100;
  return out;
}

@Injectable()
export class SettingsService {
  constructor(@InjectModel(PRESET) private readonly presets: Model<any>) {}

  async get(): Promise<AppSettings> {
    const doc = await this.presets.findOne({ key: SETTINGS_KEY }).lean<any>().exec();
    return { ...DEFAULT_SETTINGS, ...(doc?.data ?? {}) };
  }

  async put(patch: Partial<AppSettings>): Promise<AppSettings> {
    const next = { ...(await this.get()), ...sanitize(patch) };
    await this.presets
      .updateOne({ key: SETTINGS_KEY }, { $set: { key: SETTINGS_KEY, data: next } }, { upsert: true })
      .exec();
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
