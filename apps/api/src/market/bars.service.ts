/** Bar cache: MongoDB barSeries is the market-data cache in front of Yahoo. */
import { Injectable } from '@nestjs/common';
import { InjectModel } from '@nestjs/mongoose';
import type { Model } from 'mongoose';
import {
  decodeSeries,
  encodeSeries,
  intervalAndPeriod,
  type OhlcSeries,
  type Timeframe,
} from '@vova/engine';
import { BAR_SERIES } from '../db/schemas';
import { YahooClient } from './yahoo.client';

export type BarsResult = {
  bars: OhlcSeries | null;
  fromCache: boolean;
  companyName?: string;
  exchange?: string;
};

@Injectable()
export class BarsService {
  constructor(
    @InjectModel(BAR_SERIES) private readonly barSeries: Model<any>,
    private readonly yahoo: YahooClient,
  ) {}

  private intervalOf(tf: Timeframe) {
    return intervalAndPeriod(tf).interval;
  }

  async getCached(yahooTicker: string, tf: Timeframe): Promise<OhlcSeries | null> {
    const doc = await this.barSeries
      .findOne({ yahooTicker, interval: this.intervalOf(tf) })
      .lean<any>()
      .exec();
    if (!doc || !doc.barCount) return null;
    return this.decode(doc);
  }

  async getBars(
    yahooTicker: string,
    tf: Timeframe,
    opts: { maxAgeHours?: number; force?: boolean; signal?: AbortSignal } = {},
  ): Promise<BarsResult> {
    const interval = this.intervalOf(tf);
    const maxAgeMs = (opts.maxAgeHours ?? 12) * 3_600_000;

    if (!opts.force) {
      const doc = await this.barSeries.findOne({ yahooTicker, interval }).lean<any>().exec();
      if (doc?.barCount && Date.now() - new Date(doc.updatedAt).getTime() < maxAgeMs) {
        return { bars: this.decode(doc), fromCache: true };
      }
    }

    const { bars, meta } = await this.yahoo.fetchOhlc(yahooTicker, tf, { signal: opts.signal });
    if (!bars) {
      const stale = await this.barSeries.findOne({ yahooTicker, interval }).lean<any>().exec();
      if (stale?.barCount) return { bars: this.decode(stale), fromCache: true };
      return { bars: null, fromCache: false };
    }

    await this.save(yahooTicker, interval, bars);
    return { bars, fromCache: false, companyName: meta?.companyName, exchange: meta?.exchange };
  }

  private decode(doc: any): OhlcSeries {
    return decodeSeries({
      barCount: doc.barCount,
      dates: new Uint8Array(doc.dates.buffer ?? doc.dates),
      open: new Uint8Array(doc.open.buffer ?? doc.open),
      high: new Uint8Array(doc.high.buffer ?? doc.high),
      low: new Uint8Array(doc.low.buffer ?? doc.low),
      close: new Uint8Array(doc.close.buffer ?? doc.close),
      volume: new Uint8Array(doc.volume.buffer ?? doc.volume),
    });
  }

  private async save(yahooTicker: string, interval: string, bars: OhlcSeries) {
    const enc = encodeSeries(bars);
    await this.barSeries.updateOne(
      { yahooTicker, interval },
      {
        $set: {
          yahooTicker,
          interval,
          firstDate: enc.firstDate,
          lastDate: enc.lastDate,
          barCount: enc.barCount,
          dates: Buffer.from(enc.dates),
          open: Buffer.from(enc.open),
          high: Buffer.from(enc.high),
          low: Buffer.from(enc.low),
          close: Buffer.from(enc.close),
          volume: Buffer.from(enc.volume),
          updatedAt: new Date(),
        },
      },
      { upsert: true },
    );
  }

  async lastClose(yahooTicker: string, tf: Timeframe = 'Daily'): Promise<number | null> {
    const bars = await this.getCached(yahooTicker, tf);
    if (!bars?.length) return null;
    return bars[bars.length - 1].close;
  }

  async stats() {
    const [count, agg] = await Promise.all([
      this.barSeries.countDocuments().exec(),
      this.barSeries.aggregate([{ $group: { _id: null, bars: { $sum: '$barCount' } } }]).exec(),
    ]);
    return { series: count, bars: agg[0]?.bars ?? 0 };
  }
}
