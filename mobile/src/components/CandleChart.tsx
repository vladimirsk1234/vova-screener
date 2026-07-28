import React from 'react';
import { View } from 'react-native';
import Svg, { Line, Rect } from 'react-native-svg';
import type { IndicatorParams } from '../indicatorParams';
import { runStructureOverlay } from '../engine/sequenceVova';
import { maxBarsForTf, trimBars } from '../engine/dataUtils';
import type { OhlcSeries, Timeframe } from '../types';

type Props = {
  bars: OhlcSeries;
  tf: Timeframe;
  params: IndicatorParams;
  width: number;
  height: number;
};

export function CandleChart({ bars, tf, params, width, height }: Props) {
  const windowed = trimBars(bars, maxBarsForTf(tf));
  const overlay = runStructureOverlay(windowed, {
    atr_len: params.atr_len,
    min_rr: params.min_rr,
    use_last_hl_sl: params.use_last_hl_sl,
    risk_dollars: params.risk_dollars,
  });
  if (!windowed.length) return <View style={{ width, height, backgroundColor: params.paper_color }} />;

  const padL = 8;
  const padR = 8;
  const padT = 12;
  const padB = 20;
  const w = width - padL - padR;
  const h = height - padT - padB;
  const highs = windowed.map((b) => b.high);
  const lows = windowed.map((b) => b.low);
  let minY = Math.min(...lows);
  let maxY = Math.max(...highs);
  if (params.show_tp_sl && overlay?.TP != null) maxY = Math.max(maxY, overlay.TP);
  if (params.show_tp_sl && overlay?.SL != null) minY = Math.min(minY, overlay.SL);
  if (params.show_crit_level && overlay) {
    for (const c of overlay.critical) {
      if (c != null) {
        minY = Math.min(minY, c);
        maxY = Math.max(maxY, c);
      }
    }
  }
  const span = maxY - minY || 1;
  const n = windowed.length;
  const slot = w / n;
  const bodyW = Math.max(1, slot * 0.6);
  const yScale = (price: number) => padT + ((maxY - price) / span) * h;
  const xAt = (i: number) => padL + i * slot + slot / 2;

  return (
    <Svg width={width} height={height} style={{ backgroundColor: params.bg_color }}>
      <Rect x={0} y={0} width={width} height={height} fill={params.paper_color} />
      {params.show_crit_level &&
        overlay?.critical.map((c, i) => {
          if (c == null || i === 0) return null;
          const prev = overlay.critical[i - 1];
          if (prev == null) return null;
          const color =
            (overlay.seq_state[i] ?? 0) >= 0
              ? params.crit_stop_color_up
              : params.crit_stop_color_down;
          return (
            <Line
              key={`c-${i}`}
              x1={xAt(i - 1)}
              y1={yScale(prev)}
              x2={xAt(i)}
              y2={yScale(c)}
              stroke={color}
              strokeWidth={1.5}
            />
          );
        })}
      {params.show_tp_sl && overlay?.TP != null && (
        <Line
          x1={padL}
          y1={yScale(overlay.TP)}
          x2={padL + w}
          y2={yScale(overlay.TP)}
          stroke="#4caf50"
          strokeDasharray="4 3"
          strokeWidth={1}
        />
      )}
      {params.show_tp_sl && overlay?.SL != null && (
        <Line
          x1={padL}
          y1={yScale(overlay.SL)}
          x2={padL + w}
          y2={yScale(overlay.SL)}
          stroke="#f44336"
          strokeDasharray="4 3"
          strokeWidth={1}
        />
      )}
      {windowed.map((b, i) => {
        const up = b.close >= b.open;
        const color = up ? params.candle_up : params.candle_down;
        const x = xAt(i);
        const yHigh = yScale(b.high);
        const yLow = yScale(b.low);
        const yO = yScale(b.open);
        const yC = yScale(b.close);
        const top = Math.min(yO, yC);
        const bodyH = Math.max(1, Math.abs(yC - yO));
        return (
          <React.Fragment key={b.date + i}>
            <Line x1={x} y1={yHigh} x2={x} y2={yLow} stroke={color} strokeWidth={1} />
            <Rect x={x - bodyW / 2} y={top} width={bodyW} height={bodyH} fill={color} />
          </React.Fragment>
        );
      })}
    </Svg>
  );
}
