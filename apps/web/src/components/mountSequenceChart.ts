import {
  CandlestickSeries,
  LineSeries,
  createChart,
  createSeriesMarkers,
  type IChartApi,
  type ISeriesApi,
  type SeriesMarker,
  type Time,
} from 'lightweight-charts';
import type { ChartDrawing, ChartPayload, ChartSettings } from '../lib/api';

type LinePoint = { time: Time; value: number };

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

/** Split critical into up/down stepped-ish series by seqState. */
function criticalSeries(
  bars: ChartPayload['bars'],
  critical: (number | null)[],
  seqState: number[],
  want: 1 | -1,
): LinePoint[] {
  const out: LinePoint[] = [];
  for (let i = 0; i < bars.length; i++) {
    if (seqState[i] !== want) continue;
    const v = critical[i];
    if (v == null || !Number.isFinite(v)) continue;
    out.push({ time: bars[i].date as Time, value: v });
  }
  return out;
}

function extendLine(
  bars: ChartPayload['bars'],
  x0: number,
  y0: number,
  x1: number,
  y1: number,
): LinePoint[] {
  const n = bars.length;
  if (n === 0) return [];
  const dx = x1 - x0;
  const slope = dx !== 0 ? (y1 - y0) / dx : 0;
  const start = Math.max(0, Math.min(x0, x1, n - 1));
  const end = n - 1;
  const points: LinePoint[] = [];
  // Segment from clamped x0→x1, then extend to last bar.
  const a = Math.max(0, Math.min(n - 1, x0));
  const b = Math.max(0, Math.min(n - 1, x1));
  const yAt = (x: number) => y0 + slope * (x - x0);
  points.push({ time: bars[a].date as Time, value: yAt(a) });
  if (b !== a) points.push({ time: bars[b].date as Time, value: yAt(b) });
  if (end !== b && end !== a) {
    points.push({ time: bars[end].date as Time, value: yAt(end) });
  }
  void start;
  return points;
}

export type SequenceChartHandle = {
  fitContent: () => void;
  chart: IChartApi | null;
};

export function mountSequenceChart(
  container: HTMLElement,
  payload: ChartPayload,
  settings: ChartSettings,
  drawings: ChartDrawing[] = [],
): { destroy: () => void; fitContent: () => void; chart: IChartApi } {
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
    timeScale: { borderColor: settings.grid_color, rightOffset: 8 },
    crosshair: { mode: 0 },
    handleScroll: true,
    handleScale: true,
  });

  const candle = chart.addSeries(CandlestickSeries, {
    upColor: settings.candle_up,
    downColor: settings.candle_down,
    borderUpColor: settings.candle_border || settings.candle_up,
    borderDownColor: settings.candle_border || settings.candle_down,
    wickUpColor: settings.candle_wick || settings.candle_up,
    wickDownColor: settings.candle_wick || settings.candle_down,
  });

  candle.setData(
    payload.bars.map((b) => ({
      time: b.date as Time,
      open: b.open,
      high: b.high,
      low: b.low,
      close: b.close,
    })),
  );

  const lines: ISeriesApi<'Line'>[] = [];
  const addLine = (color: string, width = 2, style: 0 | 1 | 2 = 0) => {
    const s = chart.addSeries(LineSeries, {
      color,
      lineWidth: width as 1 | 2 | 3 | 4,
      lineStyle: style,
      priceLineVisible: false,
      lastValueVisible: false,
      crosshairMarkerVisible: false,
    });
    lines.push(s);
    return s;
  };

  const ov = payload.overlay;

  if (settings.show_crit_level && ov) {
    const up = addLine(settings.crit_stop_color_up, 2, 1);
    const down = addLine(settings.crit_stop_color_down, 2, 1);
    up.setData(criticalSeries(payload.bars, ov.critical, ov.seqState, 1));
    down.setData(criticalSeries(payload.bars, ov.critical, ov.seqState, -1));
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
    addLine(settings.sma_major_color, 2).setData(toLine(payload.bars, ov.overlays.smaMajor));
  }
  if (settings.show_elder_envelope && ov) {
    addLine('rgba(128,128,128,0.7)', 1).setData(toLine(payload.bars, ov.overlays.envUpper));
    addLine('rgba(128,128,128,0.7)', 1).setData(toLine(payload.bars, ov.overlays.envLower));
  }
  if (settings.show_bb && ov) {
    addLine(settings.bb_upper_color, 1).setData(toLine(payload.bars, ov.overlays.bbUpper));
    addLine(settings.bb_lower_color, 1).setData(toLine(payload.bars, ov.overlays.bbLower));
    addLine(settings.bb_basis_color, 1).setData(toLine(payload.bars, ov.overlays.bbBasis));
  }

  if (settings.show_extension_lines && ov) {
    for (const ln of ov.extensionLines) {
      const pts = extendLine(payload.bars, ln.rawX0Idx, ln.y0, ln.rawX1Idx, ln.y1);
      addLine(settings.hhll_color, 1, 2).setData(pts);
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

  if (settings.show_tp_sl) {
    const tp = payload.pine?.tp ?? ov?.tp;
    const sl = payload.pine?.sl ?? ov?.sl;
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
          text: 'Bull',
        });
      }
      if (ov.bullishBreak[i]) {
        markers.push({
          time: payload.bars[i].date as Time,
          position: 'aboveBar',
          color: '#000000',
          shape: 'arrowDown',
          text: 'Bear',
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
    fitContent: () => chart.timeScale().fitContent(),
    destroy: () => chart.remove(),
  };
}
