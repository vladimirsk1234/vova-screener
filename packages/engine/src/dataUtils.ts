/** OHLC helpers — port of data_utils.py (native Yahoo TF bars, no daily resample). */
import type { OhlcBar, OhlcSeries, Timeframe } from './types';

export function intervalAndPeriod(tf: Timeframe): { interval: string; range: string } {
  if (tf === 'Weekly') return { interval: '1wk', range: '5y' };
  if (tf === 'Monthly') return { interval: '1mo', range: '5y' };
  return { interval: '1d', range: '2y' };
}

export function fillLastBarOhlc(bars: OhlcSeries): OhlcSeries {
  if (bars.length < 2) return bars;
  const out = bars.map((b) => ({ ...b }));
  const last = out[out.length - 1];
  const prev = out[out.length - 2];
  const fill = (v: number | undefined | null, fallback: number) =>
    v == null || Number.isNaN(v) ? fallback : v;
  last.close = fill(last.close, prev.close);
  last.open = fill(last.open, last.close);
  last.high = fill(last.high, last.close);
  last.low = fill(last.low, last.close);
  return out;
}

export function dropIncompleteBars(bars: OhlcSeries): OhlcSeries {
  return bars.filter(
    (b) =>
      Number.isFinite(b.open) &&
      Number.isFinite(b.high) &&
      Number.isFinite(b.low) &&
      Number.isFinite(b.close),
  );
}

export function maxBarsForTf(tf: Timeframe): number {
  if (tf === 'Weekly') return 80;
  if (tf === 'Monthly') return 36;
  return 180;
}

export function trimBars(bars: OhlcSeries, n: number): OhlcSeries {
  if (bars.length <= n) return bars;
  return bars.slice(bars.length - n);
}

export function barDateFromUnix(sec: number): string {
  const d = new Date(sec * 1000);
  const y = d.getUTCFullYear();
  const m = String(d.getUTCMonth() + 1).padStart(2, '0');
  const day = String(d.getUTCDate()).padStart(2, '0');
  return `${y}-${m}-${day}`;
}

export function toOhlcSeries(
  timestamps: number[],
  open: number[],
  high: number[],
  low: number[],
  close: number[],
  volume: number[],
): OhlcSeries {
  const bars: OhlcBar[] = [];
  const n = Math.min(
    timestamps.length,
    open.length,
    high.length,
    low.length,
    close.length,
    volume.length || timestamps.length,
  );
  for (let i = 0; i < n; i++) {
    bars.push({
      date: barDateFromUnix(timestamps[i]),
      open: open[i],
      high: high[i],
      low: low[i],
      close: close[i],
      volume: volume[i] ?? 0,
    });
  }
  return bars;
}
