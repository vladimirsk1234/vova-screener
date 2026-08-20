/** Universe: imports the root ticker text files into the instruments collection. */
import { Injectable, Logger, type OnModuleInit } from '@nestjs/common';
import { InjectModel } from '@nestjs/mongoose';
import type { Model } from 'mongoose';
import * as fs from 'node:fs';
import * as path from 'node:path';
import {
  canadianYahooCandidates,
  parseListText,
  parseManualTickers,
  resolveManualAgainstListings,
  shortSymbol,
  stripCanadianYahooSuffix,
  type ParsedEntry,
} from '@vova/engine';
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
    // Re-import when DB is empty OR out of sync with list files (e.g. after
    // STOCK-TICKERS.txt grows via gap-scan / deploy). Previously only empty DB
    // triggered import, so the UI kept showing the old symbol count.
    if (await this.needsImport()) {
      const res = await this.importFromFiles();
      this.log.log(`Universe imported: ${JSON.stringify(res)}`);
    }
  }

  private fileEntryCount(filename: string): number {
    const file = path.join(REPO_ROOT, filename);
    if (!fs.existsSync(file)) return 0;
    return parseListText(fs.readFileSync(file, 'utf8')).entries.length;
  }

  private async needsImport(): Promise<boolean> {
    const total = await this.instruments.countDocuments().exec();
    if (total === 0) return true;
    for (const cfg of Object.values(SOURCES)) {
      const fileCount = this.fileEntryCount(cfg.file);
      if (fileCount <= 0) continue;
      const dbCount = await this.instruments
        .countDocuments({ universes: cfg.universe, active: true })
        .exec();
      if (dbCount !== fileCount) {
        this.log.log(
          `Universe out of sync for ${cfg.universe}: db=${dbCount} file=${fileCount}`,
        );
        return true;
      }
    }
    return false;
  }

  async importFromFiles() {
    const out: Record<string, number> = {};
    for (const [label, cfg] of Object.entries(SOURCES)) {
      const file = path.join(REPO_ROOT, cfg.file);
      if (!fs.existsSync(file)) {
        this.log.warn(`Universe file missing: ${file} (${label} will be empty)`);
        out[label] = 0;
        continue;
      }
      const parsed = parseListText(fs.readFileSync(file, 'utf8'));
      if (!parsed.entries.length) {
        out[label] = 0;
        continue;
      }
      const yahooInFile = new Set(parsed.entries.map((e) => e.yahoo));
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
      // Drop stale membership so /universe/summary matches the list file.
      await this.instruments.updateMany(
        {
          universes: cfg.universe,
          yahooTicker: { $nin: [...yahooInFile] },
        },
        { $pull: { universes: cfg.universe } },
      );
      out[label] = parsed.entries.length;
    }
    return out;
  }

  async resolveEntries(source: SourceLabelApi, manualTickers = ''): Promise<ParsedEntry[]> {
    if (source === 'MANUAL SCAN') return this.resolveManualEntries(manualTickers);
    const universe = source === 'ETF' ? 'etf' : 'stocks';
    let docs = await this.instruments.find({ universes: universe, active: true }).lean().exec();
    if (!docs.length) {
      await this.importFromFiles();
      docs = await this.instruments.find({ universes: universe, active: true }).lean().exec();
    }
    return docs.map((d: any) => this.toEntry(d));
  }

  /**
   * Bare `RBY` is Rubellite on TSX (`RBY.TO`) in the list file. Manual scans used to send the
   * typed symbol straight to Yahoo, which resolves the empty YHD mutual fund instead.
   */
  private async resolveManualEntries(manualTickers: string): Promise<ParsedEntry[]> {
    const parsed = parseManualTickers(manualTickers).entries.slice(0, 1);
    const entry = parsed[0];
    if (!entry) return [];
    const firstPart =
      String(manualTickers || '')
        .split(/[,\n]/)
        .map((s) => s.trim())
        .filter(Boolean)[0] ?? '';
    const listings = await this.findShortSymbolListings(entry.yahoo);
    const resolved = resolveManualAgainstListings(firstPart, listings);
    return [resolved ?? entry];
  }

  private async findShortSymbolListings(yahoo: string): Promise<ParsedEntry[]> {
    const short = stripCanadianYahooSuffix(yahoo);
    if (!short) return [];
    const yahooCandidates = [short, ...canadianYahooCandidates(short)];
    const tvRe = new RegExp(`(?:^|:)${escapeRegex(short)}$`, 'i');
    const docs = await this.instruments
      .find({
        active: true,
        $or: [{ yahooTicker: { $in: yahooCandidates } }, { tvSymbol: tvRe }],
      })
      .lean()
      .exec();
    return docs.map((d: any) => this.toEntry(d));
  }

  private toEntry(d: { yahooTicker?: string; tvSymbol?: string; companyName?: string }): ParsedEntry {
    const yahoo = String(d.yahooTicker || '').trim();
    return {
      yahoo,
      tv: d.tvSymbol || yahoo,
      name: d.companyName ?? null,
    };
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

  /** Active Stocks list (STOCK-TICKERS.txt) with display names for the Value tab. */
  async listStockEntries(): Promise<
    Array<{ yahooTicker: string; symbol: string; tvSymbol: string; companyName: string }>
  > {
    const docs = await this.instruments
      .find({ universes: 'stocks', active: true })
      .select('yahooTicker tvSymbol companyName')
      .lean<Array<{ yahooTicker?: string; tvSymbol?: string; companyName?: string }>>()
      .exec();
    const out: Array<{
      yahooTicker: string;
      symbol: string;
      tvSymbol: string;
      companyName: string;
    }> = [];
    const seen = new Set<string>();
    for (const d of docs) {
      const yahoo = String(d.yahooTicker || '')
        .trim()
        .toUpperCase();
      if (!yahoo || seen.has(yahoo)) continue;
      seen.add(yahoo);
      const tv = d.tvSymbol || yahoo;
      out.push({
        yahooTicker: yahoo,
        symbol: shortSymbol(tv),
        tvSymbol: tv,
        companyName: d.companyName || yahoo,
      });
    }
    return out;
  }

  /** True when the symbol is in the Stocks or ETF list (STOCK-TICKERS / TV-LIST-ETF). */
  async isInTrackedUniverse(yahooTicker: string): Promise<boolean> {
    const doc = await this.instruments
      .findOne({
        yahooTicker,
        active: true,
        universes: { $in: ['stocks', 'etf'] },
      })
      .select('_id')
      .lean()
      .exec();
    return Boolean(doc);
  }

  /** Active Stocks + ETF yahoo tickers — the daily/weekly fundamentals refresh set. */
  async listActiveYahooTickers(): Promise<string[]> {
    const docs = await this.instruments
      .find({ active: true, universes: { $in: ['stocks', 'etf'] } })
      .select('yahooTicker')
      .lean<Array<{ yahooTicker: string }>>()
      .exec();
    const out: string[] = [];
    const seen = new Set<string>();
    for (const d of docs) {
      const t = String(d.yahooTicker || '')
        .trim()
        .toUpperCase();
      if (!t || seen.has(t)) continue;
      seen.add(t);
      out.push(t);
    }
    return out;
  }
}

function escapeRegex(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}
