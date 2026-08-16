import {
  AreaSeries,
  CandlestickSeries,
  LineSeries,
  LineType,
  createChart,
  createSeriesMarkers,
  type IChartApi,
  type ISeriesApi,
  type SeriesMarker,
  type Time,
} from 'lightweight-charts';
import {
  firstForecastDate,
  firstNonForecastDate,
  lastSeriesDate,
  valuationChartLogicalRange,
  valuationChartRange,
  type ValuationWindowYears,
} from '@vova/engine';
import type { ChartDrawing, ChartPayload, ChartSettings, ValuationSeriesPoint } from '../lib/api';

type LinePoint = { time: Time; value: number; color?: string; year?: number };

const FV_FORECAST_COLOR = '#1565c0';
const NORMAL_PE_COLOR = '#0d47a1';
const DIVIDEND_COLOR = '#ffd54f';

function toLine(
  bars: ChartPayload['bars'],
  values: (number | null)[] | undefined,
): LinePoint[] {
  if (!values) return [];
  const out: LinePoint[] = [];
  for (let i = 0; i < bars.length; i++) {
    const v = values[i];
    if (v == null || !Number.isFinite(v)) continue;
    out.push({ time: bars[i].date as Time, value: v });
  }
  return out;
}

/** Continuous critical trail with per-bar color (Pine plot.style_stepline). */
function criticalStepline(
  bars: ChartPayload['bars'],
  critical: (number | null)[],
  seqState: number[],
  colorUp: string,
  colorDown: string,
): LinePoint[] {
  const out: LinePoint[] = [];
  for (let i = 0; i < bars.length; i++) {
    const v = critical[i];
    if (v == null || !Number.isFinite(v)) continue;
    out.push({
      time: bars[i].date as Time,
      value: v,
      color: seqState[i] === 1 ? colorUp : colorDown,
    });
  }
  return out;
}

const RIGHT_EXTEND_BARS = 8;
const DAY_MS = 24 * 60 * 60 * 1000;

function parseBarTimeMs(date: string): number {
  const [y, m, d] = date.split('-').map(Number);
  return Date.UTC(y, (m ?? 1) - 1, d ?? 1);
}

function formatBarTime(ms: number): string {
  const dt = new Date(ms);
  const y = dt.getUTCFullYear();
  const m = String(dt.getUTCMonth() + 1).padStart(2, '0');
  const d = String(dt.getUTCDate()).padStart(2, '0');
  return `${y}-${m}-${d}`;
}

function barStepMs(bars: ChartPayload['bars']): number {
  if (bars.length < 2) return DAY_MS;
  const a = parseBarTimeMs(bars[bars.length - 2].date);
  const b = parseBarTimeMs(bars[bars.length - 1].date);
  const step = b - a;
  return step > 0 ? step : DAY_MS;
}

function extendLine(
  bars: ChartPayload['bars'],
  x0: number,
  y0: number,
  x1: number,
  y1: number,
  /** Extra bars past the last candle (fills timeScale rightOffset). */
  extraBars = 0,
): LinePoint[] {
  const n = bars.length;
  if (n === 0) return [];
  const dx = x1 - x0;
  const slope = dx !== 0 ? (y1 - y0) / dx : 0;
  const end = n - 1;
  const points: LinePoint[] = [];
  // Segment from clamped x0→x1, then extend to last bar (+ optional future).
  const a = Math.max(0, Math.min(n - 1, x0));
  const b = Math.max(0, Math.min(n - 1, x1));
  const yAt = (x: number) => y0 + slope * (x - x0);
  points.push({ time: bars[a].date as Time, value: yAt(a) });
  if (b !== a) points.push({ time: bars[b].date as Time, value: yAt(b) });
  if (end !== b && end !== a) {
    points.push({ time: bars[end].date as Time, value: yAt(end) });
  }
  if (extraBars > 0) {
    // Ensure a point on the last bar before projecting into whitespace.
    if (points[points.length - 1]?.time !== (bars[end].date as Time)) {
      points.push({ time: bars[end].date as Time, value: yAt(end) });
    }
    const step = barStepMs(bars);
    const lastMs = parseBarTimeMs(bars[end].date);
    for (let i = 1; i <= extraBars; i++) {
      points.push({
        time: formatBarTime(lastMs + step * i) as Time,
        value: yAt(end + i),
      });
    }
  }
  return points;
}

export type SequenceChartHandle = {
  fitContent: () => void;
  chart: IChartApi | null;
};

/**
 * The trade the chart is being read for. Its levels win over the ones the engine reports for the
 * latest bar: on a closed trade those are two different things, and the card, the header and the
 * chart all have to be showing the same trade.
 */
export type ChartTrade = {
  entry: number;
  tp: number | null;
  sl: number | null;
  openedAsOf: string | null;
  exitDate: string | null;
  exitPrice: number | null;
};

export type ChartMountMode = 'ta' | 'fundamentals';

function valuationLinePoints(valuationSeries: ValuationSeriesPoint[]): {
  fairPts: LinePoint[];
  forecastPts: LinePoint[];
  forecastOnly: LinePoint[];
  normalPts: LinePoint[];
  normalForecastPts: LinePoint[];
  dividendPts: LinePoint[];
} {
  const fairPts: LinePoint[] = [];
  const forecastOnly: LinePoint[] = [];
  const normalPts: LinePoint[] = [];
  const normalForecastOnly: LinePoint[] = [];
  const dividendPts: LinePoint[] = [];
  let lastSolid: LinePoint | null = null;
  let lastSolidNormal: LinePoint | null = null;
  for (const p of valuationSeries) {
    const time = p.date.slice(0, 10) as Time;
    if (p.fairValue != null && Number.isFinite(p.fairValue) && p.fairValue > 0) {
      const pt = { time, value: p.fairValue, year: p.forecast ? p.year : undefined };
      if (p.forecast) {
        forecastOnly.push(pt);
      } else {
        fairPts.push(pt);
        lastSolid = pt;
      }
    }
    if (p.normalValue != null && Number.isFinite(p.normalValue) && p.normalValue > 0) {
      if (p.forecast) {
        normalForecastOnly.push({ time, value: p.normalValue, year: p.year });
      } else {
        const npt = { time, value: p.normalValue };
        normalPts.push(npt);
        lastSolidNormal = npt;
      }
    }
    if (
      !p.forecast &&
      !p.estimated &&
      p.dividend != null &&
      Number.isFinite(p.dividend) &&
      p.dividend > 0
    ) {
      dividendPts.push({ time, value: p.dividend });
    }
  }
  const forecastPts =
    lastSolid && forecastOnly.length ? [lastSolid, ...forecastOnly] : forecastOnly;
  const normalForecastPts =
    lastSolidNormal && normalForecastOnly.length
      ? [lastSolidNormal, ...normalForecastOnly]
      : normalForecastOnly;
  return { fairPts, forecastPts, forecastOnly, normalPts, normalForecastPts, dividendPts };
}

function seriesToFairPoints(series: ValuationSeriesPoint[]): LinePoint[] {
  const pts: LinePoint[] = [];
  for (const p of series) {
    if (p.fairValue == null || !Number.isFinite(p.fairValue) || p.fairValue <= 0) continue;
    pts.push({ time: p.date.slice(0, 10) as Time, value: p.fairValue });
  }
  return pts;
}

/** Green fill under fair value — added before candles so price stays on top. */
function addFairValueFill(chart: IChartApi, fairPts: LinePoint[]) {
  if (!fairPts.length) return;
  const fill = chart.addSeries(AreaSeries, {
    lineColor: 'rgba(76, 175, 80, 0)',
    topColor: 'rgba(76, 175, 80, 0.55)',
    bottomColor: 'rgba(46, 125, 50, 0.20)',
    lineWidth: 1,
    lineType: LineType.Simple,
    priceLineVisible: false,
    lastValueVisible: false,
    crosshairMarkerVisible: false,
  });
  fill.setData(fairPts);
}

export type ValuationSeriesRefs = {
  fair?: ISeriesApi<'Line'>;
  fairForecast?: ISeriesApi<'Line'>;
  normal?: ISeriesApi<'Line'>;
  normalForecast?: ISeriesApi<'Line'>;
  dividend?: ISeriesApi<'Line'>;
  dcf?: ISeriesApi<'Line'>;
};

function addDashedLine(
  chart: IChartApi,
  lines: ISeriesApi<'Line'>[],
  pts: LinePoint[],
  color: string,
  opts: {
    width?: 2 | 3 | 4;
    markYears?: boolean;
  } = {},
): ISeriesApi<'Line'> | undefined {
  if (!pts.length) return undefined;
  const line = chart.addSeries(LineSeries, {
    color,
    lineWidth: opts.width ?? 2,
    lineStyle: 2,
    lineType: LineType.Simple,
    priceLineVisible: false,
    lastValueVisible: !opts.markYears,
    crosshairMarkerVisible: true,
  });
  lines.push(line);
  line.setData(pts);
  if (!opts.markYears) return line;
  const markers: SeriesMarker<Time>[] = pts
    .filter((p) => p.year != null)
    .map((p) => ({
      time: p.time,
      position: 'atPriceMiddle' as const,
      price: p.value,
      color,
      shape: 'circle' as const,
    }));
  if (markers.length) createSeriesMarkers(line, markers);
  return line;
}

function addValuationLines(
  chart: IChartApi,
  lines: ISeriesApi<'Line'>[],
  fairPts: LinePoint[],
  forecastPts: LinePoint[],
  normalPts: LinePoint[],
  normalForecastPts: LinePoint[],
  dcfPts: LinePoint[],
  dividendPts: LinePoint[],
): ValuationSeriesRefs {
  const refs: ValuationSeriesRefs = {};
  if (fairPts.length) {
    const fair = chart.addSeries(LineSeries, {
      color: '#ff9800',
      lineWidth: 2,
      lineStyle: 0,
      lineType: LineType.Simple,
      priceLineVisible: false,
      lastValueVisible: true,
      crosshairMarkerVisible: true,
      pointMarkersVisible: true,
      pointMarkersRadius: 5,
    });
    lines.push(fair);
    fair.setData(fairPts);
    refs.fair = fair;
  }
  refs.fairForecast = addDashedLine(chart, lines, forecastPts, FV_FORECAST_COLOR, {
    width: 3,
    markYears: true,
  });
  if (normalPts.length) {
    const normal = chart.addSeries(LineSeries, {
      color: NORMAL_PE_COLOR,
      lineWidth: 2,
      lineStyle: 0,
      lineType: LineType.Simple,
      priceLineVisible: false,
      lastValueVisible: true,
      crosshairMarkerVisible: true,
    });
    lines.push(normal);
    normal.setData(normalPts);
    refs.normal = normal;
  }
  refs.normalForecast = addDashedLine(chart, lines, normalForecastPts, NORMAL_PE_COLOR, {
    width: 4,
    markYears: true,
  });
  if (dividendPts.length) {
    const dividend = chart.addSeries(LineSeries, {
      color: DIVIDEND_COLOR,
      lineWidth: 2,
      lineStyle: 0,
      lineType: LineType.Simple,
      priceLineVisible: false,
      lastValueVisible: true,
      crosshairMarkerVisible: true,
    });
    lines.push(dividend);
    dividend.setData(dividendPts);
    refs.dividend = dividend;
  }
  refs.dcf = addDashedLine(chart, lines, dcfPts, '#ab47bc');
  return refs;
}

function lastSeriesDateMs(series: ValuationSeriesPoint[]): number | null {
  let last: string | null = null;
  for (const p of series) {
    const d = p.date.slice(0, 10);
    if (!last || d > last) last = d;
  }
  return last ? parseBarTimeMs(last) : null;
}

function futureWhitespace(
  bars: ChartPayload['bars'],
  series: ValuationSeriesPoint[],
  denseUntilIso?: string | null,
): { time: Time }[] {
  if (!bars.length) return [];
  const lastMs = parseBarTimeMs(bars[bars.length - 1]!.date);
  const lastValMs = lastSeriesDateMs(series);
  if (lastValMs == null || lastValMs <= lastMs) return [];
  const step = barStepMs(bars);
  const denseUntilMs = denseUntilIso
    ? Math.min(parseBarTimeMs(denseUntilIso) + step * 2, lastValMs)
    : lastValMs;
  const times = new Set<string>();
  for (let t = lastMs + step; t <= denseUntilMs + step; t += step) {
    times.add(formatBarTime(t));
  }
  for (const p of series) {
    const d = p.date.slice(0, 10);
    if (parseBarTimeMs(d) > lastMs) times.add(d);
  }
  return [...times]
    .filter((d) => parseBarTimeMs(d) > lastMs)
    .sort()
    .map((time) => ({ time: time as Time }));
}

function valuationRangeInput(
  bars: ChartPayload['bars'],
  valuationSeries: ValuationSeriesPoint[],
  windowYears: ValuationWindowYears | undefined,
  extraSeries: ValuationSeriesPoint[] = [],
) {
  return valuationChartRange({
    firstBarDate: bars[0]!.date,
    lastBarDate: bars[bars.length - 1]!.date,
    windowYears: windowYears === undefined ? null : windowYears,
    firstHistoricalDate: firstNonForecastDate(valuationSeries),
    firstForecastDate: firstForecastDate(valuationSeries),
    lastExtraDate: extraSeries.length ? lastSeriesDate(extraSeries) : null,
  });
}

function bindValuationVisibleRange(
  container: HTMLElement,
  chart: IChartApi,
  timesMs: number[],
  range: { from: string; to: string },
  padMs: number,
): { apply: () => void; detach: () => void } {
  const apply = () => {
    if (!timesMs.length) {
      chart.timeScale().fitContent();
      return;
    }
    const { fromIdx, toIdx } = valuationChartLogicalRange(timesMs, range, padMs);
    chart.timeScale().setVisibleLogicalRange({ from: fromIdx, to: toIdx });
  };
  let lastW = 0;
  const onResize = () => {
    const w = container.clientWidth;
    if (w < 8 || Math.abs(w - lastW) < 2) return;
    lastW = w;
    apply();
  };
  apply();
  let cancelled = false;
  const raf = requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      if (!cancelled) onResize();
    });
  });
  const ro = new ResizeObserver(onResize);
  ro.observe(container);
  return {
    apply,
    detach: () => {
      cancelled = true;
      cancelAnimationFrame(raf);
      ro.disconnect();
    },
  };
}

export function mountSequenceChart(
  container: HTMLElement,
  payload: ChartPayload,
  settings: ChartSettings,
  drawings: ChartDrawing[] = [],
  trade: ChartTrade | null = null,
  valuationSeries: ValuationSeriesPoint[] = [],
  mode: ChartMountMode = 'ta',
  windowYears?: ValuationWindowYears,
  dcfForecastSeries: ValuationSeriesPoint[] = [],
): { destroy: () => void; fitContent: () => void; chart: IChartApi; valuation: ValuationSeriesRefs } {
  const chart = createChart(container, {
    autoSize: true,
    layout: {
      background: { color: settings.bg_color },
      textColor: '#d1d4dc',
      attributionLogo: true,
    },
    grid: {
      vertLines: { color: settings.grid_color },
      horzLines: { color: settings.grid_color },
    },
    rightPriceScale: { borderColor: settings.grid_color },
    timeScale: {
      borderColor: settings.grid_color,
      rightOffset: 8,
      ...(mode === 'fundamentals' ? { minBarSpacing: 0.2 } : {}),
    },
    crosshair: { mode: 0 },
    handleScroll: true,
    handleScale: true,
  });

  const { fairPts, forecastPts, forecastOnly, normalPts, normalForecastPts, dividendPts } =
    valuationLinePoints(valuationSeries);
  const dcfPts = seriesToFairPoints(dcfForecastSeries);
  if (mode === 'fundamentals') {
    addFairValueFill(chart, [...fairPts, ...forecastOnly]);
  }

  const candle = chart.addSeries(CandlestickSeries, {
    upColor: settings.candle_up,
    downColor: settings.candle_down,
    borderUpColor: settings.candle_border || settings.candle_up,
    borderDownColor: settings.candle_border || settings.candle_down,
    wickUpColor: settings.candle_wick || settings.candle_up,
    wickDownColor: settings.candle_wick || settings.candle_down,
  });

  const candleData: Array<
    | { time: Time; open: number; high: number; low: number; close: number }
    | { time: Time }
  > = payload.bars.map((b) => ({
    time: b.date as Time,
    open: b.open,
    high: b.high,
    low: b.low,
    close: b.close,
  }));
  const fundRange =
    mode === 'fundamentals' && payload.bars.length
      ? valuationRangeInput(payload.bars, valuationSeries, windowYears, dcfForecastSeries)
      : null;
  if (mode === 'fundamentals') {
    candleData.push(
      ...futureWhitespace(
        payload.bars,
        [...valuationSeries, ...dcfForecastSeries],
        fundRange?.to,
      ),
    );
  }
  candle.setData(candleData);

  const lines: ISeriesApi<'Line'>[] = [];

  if (mode === 'fundamentals') {
    const valuation = addValuationLines(
      chart,
      lines,
      fairPts,
      forecastPts,
      normalPts,
      normalForecastPts,
      dcfPts,
      dividendPts,
    );
    const timesMs = candleData.map((b) => parseBarTimeMs(String(b.time)));
    const zoom = fundRange
      ? bindValuationVisibleRange(
          container,
          chart,
          timesMs,
          fundRange,
          barStepMs(payload.bars) * 2,
        )
      : null;
    if (!zoom) chart.timeScale().fitContent();
    return {
      chart,
      valuation,
      fitContent: () => (zoom ? zoom.apply() : chart.timeScale().fitContent()),
      destroy: () => {
        zoom?.detach();
        chart.remove();
      },
    };
  }

  const addLine = (
    color: string,
    width = 2,
    style: 0 | 1 | 2 = 0,
    lineType: LineType = LineType.Simple,
  ) => {
    const s = chart.addSeries(LineSeries, {
      color,
      lineWidth: width as 1 | 2 | 3 | 4,
      lineStyle: style,
      lineType,
      priceLineVisible: false,
      lastValueVisible: false,
      crosshairMarkerVisible: false,
    });
    lines.push(s);
    return s;
  };

  const ov = payload.overlay;

  if (settings.show_crit_level && ov) {
    const crit = addLine(
      settings.crit_stop_color_up,
      2,
      1,
      LineType.WithSteps,
    );
    crit.setData(
      criticalStepline(
        payload.bars,
        ov.critical,
        ov.seqState,
        settings.crit_stop_color_up,
        settings.crit_stop_color_down,
      ),
    );
    if (ov.criticalLevel != null) {
      candle.createPriceLine({
        price: ov.criticalLevel,
        color: ov.seqStateFinal === 1 ? settings.crit_stop_color_up : settings.crit_stop_color_down,
        lineWidth: 1,
        lineStyle: 2,
        axisLabelVisible: true,
        title: `Critical ${ov.criticalLevel.toFixed(2)}`,
      });
    }
  }

  if (settings.show_short_ema && ov) {
    addLine(settings.short_ema_color, 2).setData(toLine(payload.bars, ov.overlays.emaFast));
  }
  if (settings.show_center_ema && ov) {
    addLine(settings.center_ema_color, 2).setData(toLine(payload.bars, ov.overlays.emaSlow));
  }
  if (settings.show_sma_major && ov) {
    addLine(settings.sma_major_color, 3).setData(toLine(payload.bars, ov.overlays.smaMajor));
  }
  if (settings.show_elder_envelope && ov) {
    addLine(settings.env_upper_color, 2).setData(toLine(payload.bars, ov.overlays.envUpper));
    addLine(settings.env_lower_color, 2).setData(toLine(payload.bars, ov.overlays.envLower));
  }
  if (settings.show_bb && ov) {
    addLine(settings.bb_upper_color, 2).setData(toLine(payload.bars, ov.overlays.bbUpper));
    addLine(settings.bb_lower_color, 2).setData(toLine(payload.bars, ov.overlays.bbLower));
    addLine(settings.bb_basis_color, 2).setData(toLine(payload.bars, ov.overlays.bbBasis));
  }

  if (settings.show_extension_lines && ov) {
    for (const ln of ov.extensionLines) {
      const pts = extendLine(
        payload.bars,
        ln.rawX0Idx,
        ln.y0,
        ln.rawX1Idx,
        ln.y1,
        RIGHT_EXTEND_BARS,
      );
      addLine(settings.hhll_color, 2, 0).setData(pts);
    }
  }

  if (settings.show_fib && ov?.fib) {
    for (const [price, title] of [
      [ov.fib.fib382, '0.382'],
      [ov.fib.fib500, '0.5'],
      [ov.fib.fib618, '0.618'],
    ] as const) {
      candle.createPriceLine({
        price,
        color: settings.fib_color,
        lineWidth: Math.min(4, Math.max(1, settings.fib_width)) as 1 | 2 | 3 | 4,
        lineStyle: 2,
        axisLabelVisible: true,
        title,
      });
    }
  }

  // The indicator's TP and SL are a suggestion the reader can switch off. A trade's own are the
  // potential and the risk it was actually taken on, so a chart opened on one always draws them.
  if (settings.show_tp_sl || trade) {
    const tp = trade ? trade.tp : (payload.pine?.tp ?? ov?.tp);
    const sl = trade ? trade.sl : (payload.pine?.sl ?? ov?.sl);
    if (tp != null) {
      candle.createPriceLine({
        price: tp,
        color: settings.candle_up,
        lineWidth: 1,
        lineStyle: 2,
        axisLabelVisible: true,
        title: 'TP',
      });
    }
    if (sl != null) {
      candle.createPriceLine({
        price: sl,
        color: settings.candle_down,
        lineWidth: 1,
        lineStyle: 2,
        axisLabelVisible: true,
        title: 'SL',
      });
    }
  }

  const markers: SeriesMarker<Time>[] = [];

  if (trade) {
    candle.createPriceLine({
      price: trade.entry,
      color: '#2962ff',
      lineWidth: 1,
      lineStyle: 0,
      axisLabelVisible: true,
      title: `Entry ${trade.entry.toFixed(2)}`,
    });
    // A date the window does not reach would be dropped by the chart anyway; skipping it here
    // keeps the marker list in step with what is actually drawn.
    const onChart = (date: string | null) =>
      Boolean(date) && payload.bars.some((bar) => bar.date === date);
    if (onChart(trade.openedAsOf)) {
      markers.push({
        time: trade.openedAsOf as Time,
        position: 'belowBar',
        color: '#2962ff',
        shape: 'arrowUp',
        text: `BUY ${trade.entry.toFixed(2)}`,
      });
    }
    if (onChart(trade.exitDate) && trade.exitPrice != null) {
      markers.push({
        time: trade.exitDate as Time,
        position: 'aboveBar',
        color: trade.exitPrice >= trade.entry ? settings.candle_up : settings.candle_down,
        shape: 'arrowDown',
        text: `SELL ${trade.exitPrice.toFixed(2)}`,
      });
    }
  }

  if (settings.show_hhll && ov) {
    for (const p of ov.peaks) {
      const bar = payload.bars[p.idx];
      if (!bar) continue;
      markers.push({
        time: bar.date as Time,
        position: 'aboveBar',
        color: settings.hhll_color,
        shape: 'arrowDown',
        text: p.label,
      });
    }
    for (const t of ov.troughs) {
      const bar = payload.bars[t.idx];
      if (!bar) continue;
      markers.push({
        time: bar.date as Time,
        position: 'belowBar',
        color: settings.hhll_color,
        shape: 'arrowUp',
        text: t.label,
      });
    }
  }
  if (settings.show_breaks && ov) {
    for (let i = 0; i < payload.bars.length; i++) {
      if (ov.bearishBreak[i]) {
        markers.push({
          time: payload.bars[i].date as Time,
          position: 'belowBar',
          color: '#000000',
          shape: 'arrowUp',
        });
      }
      if (ov.bullishBreak[i]) {
        markers.push({
          time: payload.bars[i].date as Time,
          position: 'aboveBar',
          color: '#000000',
          shape: 'arrowDown',
        });
      }
    }
  }
  markers.sort((a, b) => String(a.time).localeCompare(String(b.time)));

  // User drawings (markers collected below, applied once at end)
  for (const d of drawings) {
    if (d.type === 'hline' && d.points[0]) {
      candle.createPriceLine({
        price: d.points[0].price,
        color: d.color ?? '#2962ff',
        lineWidth: 1,
        lineStyle: 0,
        axisLabelVisible: true,
        title: d.text ?? '',
      });
    } else if (d.type === 'text' && d.points[0]) {
      markers.push({
        time: d.points[0].time as Time,
        position: 'aboveBar',
        color: d.color ?? '#e0e0e0',
        shape: 'circle',
        text: d.text ?? 'Note',
      });
    } else if ((d.type === 'trend' || d.type === 'ray' || d.type === 'fib') && d.points.length >= 2) {
      const s = addLine(d.color ?? '#2962ff', 1, d.type === 'fib' ? 2 : 0);
      const pts = d.points.map((p) => ({ time: p.time as Time, value: p.price }));
      if (d.type === 'ray' || d.type === 'fib') {
        const a = d.points[0];
        const b = d.points[1];
        const i0 = payload.bars.findIndex((x) => x.date === a.time);
        const i1 = payload.bars.findIndex((x) => x.date === b.time);
        if (i0 >= 0 && i1 >= 0) {
          s.setData(extendLine(payload.bars, i0, a.price, i1, b.price));
        } else {
          s.setData(pts);
        }
      } else {
        s.setData(pts);
      }
      if (d.type === 'fib') {
        const hi = Math.max(d.points[0].price, d.points[1].price);
        const lo = Math.min(d.points[0].price, d.points[1].price);
        const range = hi - lo;
        for (const r of [0.382, 0.5, 0.618]) {
          candle.createPriceLine({
            price: hi - range * r,
            color: d.color ?? settings.fib_color,
            lineWidth: 1,
            lineStyle: 2,
            axisLabelVisible: true,
            title: String(r),
          });
        }
      }
    } else if (d.type === 'vline' && d.points[0]) {
      markers.push({
        time: d.points[0].time as Time,
        position: 'inBar',
        color: d.color ?? '#2962ff',
        shape: 'circle',
        text: d.text ?? '|',
      });
    }
  }

  markers.sort((a, b) => String(a.time).localeCompare(String(b.time)));
  createSeriesMarkers(candle, markers);

  chart.timeScale().fitContent();

  return {
    chart,
    valuation: {},
    fitContent: () => chart.timeScale().fitContent(),
    destroy: () => chart.remove(),
  };
}
