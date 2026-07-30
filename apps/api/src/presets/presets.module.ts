import { Body, Controller, Get, Injectable, Module, Param, Put } from '@nestjs/common';
import { InjectModel } from '@nestjs/mongoose';
import type { Model } from 'mongoose';
import { PRESET } from '../db/schemas';

@Injectable()
class PresetsService {
  constructor(@InjectModel(PRESET) private readonly presets: Model<any>) {}

  async get(key: string) {
    const doc = await this.presets.findOne({ key }).lean<any>().exec();
    return doc?.data ?? {};
  }

  async put(key: string, data: Record<string, unknown>) {
    await this.presets.updateOne({ key }, { $set: { key, data } }, { upsert: true }).exec();
    return { ok: true, key, data };
  }
}

@Controller('presets')
class PresetsController {
  constructor(private readonly presets: PresetsService) {}

  @Get(':key')
  get(@Param('key') key: string) {
    return this.presets.get(key);
  }

  @Put(':key')
  put(@Param('key') key: string, @Body() body: Record<string, unknown>) {
    return this.presets.put(key, body ?? {});
  }
}

@Module({
  controllers: [PresetsController],
  providers: [PresetsService],
})
export class PresetsModule {}
