/** Universe: imports the root ticker text files into the instruments collection. */
import { Injectable, Logger, type OnModuleInit } from '@nestjs/common';
import { InjectModel } from '@nestjs/mongoose';
import type { Model } from 'mongoose';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { parseListText, parseManualTickers, type ParsedEntry } from '@vova/engine';
import { INSTRUMENT } from '../db/schemas';
import { REPO_ROOT } from '../db/local-mongo';

const SOURCES = {
  Stocks: { file: 'STOCK-TICKERS.txt', universe: 'stocks', assetType: 'stock' },
  ETF: { file: 'TV-LIST-ETF.txt', universe: 'etf', assetType: 'etf' },
} as const;

export type SourceLabelApi = 'Stocks' | 'ETF' | 'MANUAL SCAN';

@Injectable()
export class UniverseService implements OnModuleInit {
  private readonly log = new Logger(UniverseService.name);

  constructor(@InjectModel(INSTRUMENT) private readonly instruments: Model<any>) {}

  async onModuleInit() {
    const count = await this.instruments.countDocuments().exec();
    if (count === 0) {
      const res = await this.importFromFiles();
      this.log.log(`Universe imported: ${JSON.stringify(res)}`);
    }
  }

  async importFromFiles() {
    const out: Record<string, number> = {};
    for (const [label, cfg] of Object.entries(SOURCES)) {
      const file = path.join(REPO_ROOT, cfg.file);
      if (!fs.existsSync(file)) {
        out[label] = 0;
        continue;
      }
      const parsed = parseListText(fs.readFileSync(file, 'utf8'));
      if (!parsed.entries.length) {
        out[label] = 0;
        continue;
      }
      await this.instruments.bulkWrite(
        parsed.entries.map((e) => ({
          updateOne: {
            filter: { yahooTicker: e.yahoo },
            update: {
              $set: {
                yahooTicker: e.yahoo,
                tvSymbol: e.tv,
                companyName: e.name ?? e.yahoo,
                assetType: cfg.assetType,
                active: true,
              },
              $addToSet: { universes: cfg.universe },
            },
            upsert: true,
          },
        })),
      );
      out[label] = parsed.entries.length;
    }
    return out;
  }

  async resolveEntries(source: SourceLabelApi, manualTickers = ''): Promise<ParsedEntry[]> {
    if (source === 'MANUAL SCAN') return parseManualTickers(manualTickers).entries;
    const universe = source === 'ETF' ? 'etf' : 'stocks';
    let docs = await this.instruments.find({ universes: universe, active: true }).lean().exec();
    if (!docs.length) {
      await this.importFromFiles();
      docs = await this.instruments.find({ universes: universe, active: true }).lean().exec();
    }
    return docs.map((d: any) => ({
      yahoo: d.yahooTicker,
      tv: d.tvSymbol || d.yahooTicker,
      name: d.companyName ?? null,
    }));
  }

  async summary() {
    const [stocks, etf, total] = await Promise.all([
      this.instruments.countDocuments({ universes: 'stocks' }).exec(),
      this.instruments.countDocuments({ universes: 'etf' }).exec(),
      this.instruments.countDocuments().exec(),
    ]);
    return { stocks, etf, total };
  }

  async findOne(yahooTicker: string) {
    return this.instruments.findOne({ yahooTicker }).lean<any>().exec();
  }
}
