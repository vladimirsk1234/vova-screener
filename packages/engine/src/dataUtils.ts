/** OHLC helpers — port of data_utils.py (native Yahoo TF bars, no daily resample). */
import type { OhlcBar, OhlcSeries, Timeframe } from './types';

/**
 * Yahoo interval + history range, matching `data_utils.interval_and_period`
 * (non low-memory branch: 10y for Weekly/Monthly).
 *
 * History length is part of engine parity, not a perf knob: the sequence walk is
 * path-dependent, so a shorter window can keep a stale confirmed trough/peak and
 * change SL, RR and reject reasons (YMM Monthly: 5y -> SL 4.12 / RR 0.80,
 * 10y -> SL 6.66 / RR 1.51).
 *
 * `range=max` is not usable here: Yahoo returns an irregular grid for `1mo&range=max`,
 * and `period1=0` requests drop the in-progress bar the scan needs.
 */
export function intervalAndPeriod(tf: Timeframe): { interval: string; range: string } {
  if (tf === 'Weekly') return { interval: '1wk', range: '10y' };
  if (tf === 'Monthly') return { interval: '1mo', range: '10y' };
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

/** Monday (UTC) YYYY-MM-DD for the week containing `dateStr`. */
export function weekStartMonday(dateStr: string): string {
  const [y, m, d] = dateStr.split('-').map(Number);
  const dt = new Date(Date.UTC(y, m - 1, d));
  const day = dt.getUTCDay(); // 0=Sun … 6=Sat
  const diff = day === 0 ? -6 : 1 - day;
  dt.setUTCDate(dt.getUTCDate() + diff);
  const yy = dt.getUTCFullYear();
  const mm = String(dt.getUTCMonth() + 1).padStart(2, '0');
  const dd = String(dt.getUTCDate()).padStart(2, '0');
  return `${yy}-${mm}-${dd}`;
}

function periodKey(dateStr: string, tf: Timeframe): string {
  if (tf === 'Weekly') return weekStartMonday(dateStr);
  if (tf === 'Monthly') return dateStr.slice(0, 7);
  return dateStr;
}

/**
 * Yahoo chart API often appends an extra mid-period stamp (e.g. Thursday)
 * on top of the native Weekly/Monthly candle (Monday / month-start).
 * yfinance keeps a single bar per period — without this collapse, `New` flips
 * false because the break already happened on the previous (real) period bar.
 */
export function collapseInProgressPeriodBars(bars: OhlcSeries, tf: Timeframe): OhlcSeries {
  if (tf === 'Daily' || bars.length < 2) return bars;
  const out = bars.map((b) => ({ ...b }));
  while (
    out.length >= 2 &&
    periodKey(out[out.length - 1].date, tf) === periodKey(out[out.length - 2].date, tf)
  ) {
    const last = out.pop()!;
    const prev = out[out.length - 1];
    prev.high = Math.max(prev.high, last.high);
    prev.low = Math.min(prev.low, last.low);
    if (Number.isFinite(last.close)) prev.close = last.close;
    prev.volume = (Number.isFinite(prev.volume) ? prev.volume : 0) + (Number.isFinite(last.volume) ? last.volume : 0);
  }
  return out;
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
