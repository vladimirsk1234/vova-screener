/**
 * Binary column encoding for barSeries documents.
 * Float64 for prices/volume (parity-safe), Int32 day offsets for dates.
 */
import type { OhlcSeries } from './types';

const MS_PER_DAY = 86_400_000;

export type EncodedSeries = {
  barCount: number;
  firstDate: string;
  lastDate: string;
  dates: Uint8Array;
  open: Uint8Array;
  high: Uint8Array;
  low: Uint8Array;
  close: Uint8Array;
  volume: Uint8Array;
};

function dateToDays(iso: string): number {
  return Math.round(Date.parse(`${iso}T00:00:00Z`) / MS_PER_DAY);
}

function daysToDate(days: number): string {
  return new Date(days * MS_PER_DAY).toISOString().slice(0, 10);
}

function f64(values: number[]): Uint8Array {
  const arr = new Float64Array(values);
  return new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength);
}

function readF64(bin: Uint8Array, count: number): Float64Array {
  const copy = new Uint8Array(bin.byteLength);
  copy.set(bin);
  return new Float64Array(copy.buffer, 0, count);
}

export function encodeSeries(bars: OhlcSeries): EncodedSeries {
  const dates = new Int32Array(bars.length);
  const open: number[] = [];
  const high: number[] = [];
  const low: number[] = [];
  const close: number[] = [];
  const volume: number[] = [];
  bars.forEach((bar, i) => {
    dates[i] = dateToDays(bar.date);
    open.push(bar.open);
    high.push(bar.high);
    low.push(bar.low);
    close.push(bar.close);
    volume.push(bar.volume ?? 0);
  });
  return {
    barCount: bars.length,
    firstDate: bars.length ? bars[0].date : '',
    lastDate: bars.length ? bars[bars.length - 1].date : '',
    dates: new Uint8Array(dates.buffer, dates.byteOffset, dates.byteLength),
    open: f64(open),
    high: f64(high),
    low: f64(low),
    close: f64(close),
    volume: f64(volume),
  };
}

export function decodeSeries(enc: {
  barCount: number;
  dates: Uint8Array;
  open: Uint8Array;
  high: Uint8Array;
  low: Uint8Array;
  close: Uint8Array;
  volume: Uint8Array;
}): OhlcSeries {
  const n = enc.barCount;
  const dateCopy = new Uint8Array(enc.dates.byteLength);
  dateCopy.set(enc.dates);
  const dates = new Int32Array(dateCopy.buffer, 0, n);
  const open = readF64(enc.open, n);
  const high = readF64(enc.high, n);
  const low = readF64(enc.low, n);
  const close = readF64(enc.close, n);
  const volume = readF64(enc.volume, n);
  const bars: OhlcSeries = [];
  for (let i = 0; i < n; i++) {
    bars.push({
      date: daysToDate(dates[i]),
      open: open[i],
      high: high[i],
      low: low[i],
      close: close[i],
      volume: volume[i],
    });
  }
  return bars;
}
