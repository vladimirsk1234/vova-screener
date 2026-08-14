/** Fast Graphs–style annual valuation chart (price + fair value + Normal P/E + earnings). */
import {
  ColorType,
  HistogramSeries,
  LineSeries,
  createChart,
  type IChartApi,
  type Time,
} from 'lightweight-charts';
import type { ValuationSeriesPoint } from '../lib/api';

export function mountValuationChart(
  el: HTMLElement,
  series: ValuationSeriesPoint[],
): { destroy: () => void; chart: IChartApi } {
  const chart = createChart(el, {
    autoSize: true,
    layout: {
      background: { type: ColorType.Solid, color: '#12151c' },
      textColor: '#d1d4dc',
      fontFamily: '"Segoe UI", "Helvetica Neue", system-ui, sans-serif',
    },
    grid: {
      vertLines: { color: 'rgba(42, 46, 57, 0.7)' },
      horzLines: { color: 'rgba(42, 46, 57, 0.7)' },
    },
    rightPriceScale: { borderColor: '#2a2e39' },
    timeScale: { borderColor: '#2a2e39', timeVisible: false },
    crosshair: { mode: 0 },
  });

  const power = chart.addSeries(HistogramSeries, {
    color: 'rgba(8, 153, 129, 0.55)',
    priceFormat: { type: 'price', precision: 2, minMove: 0.01 },
    priceScaleId: 'right',
  });

  const price = chart.addSeries(LineSeries, {
    color: '#d1d4dc',
    lineWidth: 2,
    priceLineVisible: false,
    lastValueVisible: true,
  });

  const fair = chart.addSeries(LineSeries, {
    color: '#ff9800',
    lineWidth: 2,
    priceLineVisible: false,
    lastValueVisible: true,
  });

  const normal = chart.addSeries(LineSeries, {
    color: '#42a5f5',
    lineWidth: 2,
    priceLineVisible: false,
    lastValueVisible: true,
  });

  const hist: { time: Time; value: number; color?: string }[] = [];
  const pricePts: { time: Time; value: number }[] = [];
  const fairPts: { time: Time; value: number }[] = [];
  const normalPts: { time: Time; value: number }[] = [];

  for (const p of series) {
    const time = p.date.slice(0, 10) as Time;
    if (p.earningsPower != null && Number.isFinite(p.earningsPower) && p.earningsPower > 0) {
      hist.push({
        time,
        value: p.earningsPower,
        color: p.estimated ? 'rgba(8, 153, 129, 0.28)' : 'rgba(8, 153, 129, 0.55)',
      });
    }
    if (p.price != null && Number.isFinite(p.price)) {
      pricePts.push({ time, value: p.price });
    }
    if (p.fairValue != null && Number.isFinite(p.fairValue) && p.fairValue > 0) {
      fairPts.push({ time, value: p.fairValue });
    }
    if (p.normalValue != null && Number.isFinite(p.normalValue) && p.normalValue > 0) {
      normalPts.push({ time, value: p.normalValue });
    }
  }

  power.setData(hist);
  price.setData(pricePts);
  fair.setData(fairPts);
  normal.setData(normalPts);
  chart.timeScale().fitContent();

  return {
    chart,
    destroy: () => {
      chart.remove();
    },
  };
}
