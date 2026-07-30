import { useEffect, useMemo, useRef, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { api, type ChartDrawing, type ChartSettings, type Timeframe } from '../lib/api';
import { Chips } from '../components/Chips';
import { ChartSettingsPanel } from '../components/ChartSettingsPanel';
import {
  DrawingToolbar,
  newDrawingId,
  pushDrawingHistory,
  type DrawingTool,
} from '../components/DrawingToolbar';
import { mountSequenceChart } from '../components/mountSequenceChart';
import {
  DEFAULT_CHART_SETTINGS,
  mergeChartSettings,
  numericChartParams,
} from '../lib/chartSettings';

export function ChartPage() {
  const { ticker = '' } = useParams();
  const navigate = useNavigate();
  const qc = useQueryClient();
  const [tf, setTf] = useState<Timeframe>('Daily');
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [settings, setSettings] = useState<ChartSettings>(DEFAULT_CHART_SETTINGS);
  const [settingsReady, setSettingsReady] = useState(false);
  const [tool, setTool] = useState<DrawingTool>('cursor');
  const [magnet, setMagnet] = useState(true);
  const [drawings, setDrawings] = useState<ChartDrawing[]>([]);
  const [past, setPast] = useState<ChartDrawing[][]>([]);
  const [future, setFuture] = useState<ChartDrawing[][]>([]);
  const [pendingPoint, setPendingPoint] = useState<{ time: string; price: number } | null>(null);
  const [crosshair, setCrosshair] = useState<string>('');

  const containerRef = useRef<HTMLDivElement | null>(null);
  const destroyRef = useRef<(() => void) | null>(null);
  const fitRef = useRef<(() => void) | null>(null);

  const presetQ = useQuery({
    queryKey: ['preset', 'chart'],
    queryFn: () => api.getPreset<Partial<ChartSettings>>('chart'),
  });

  useEffect(() => {
    if (presetQ.data && !settingsReady) {
      setSettings(mergeChartSettings(presetQ.data));
      setSettingsReady(true);
    } else if (presetQ.isFetched && !settingsReady) {
      setSettingsReady(true);
    }
  }, [presetQ.data, presetQ.isFetched, settingsReady]);

  const drawingsKey = `drawings:${ticker}:${tf}`;
  const drawingsQ = useQuery({
    queryKey: ['preset', drawingsKey],
    queryFn: () => api.getPreset<{ items?: ChartDrawing[] }>(drawingsKey),
    enabled: Boolean(ticker),
  });

  useEffect(() => {
    const items = drawingsQ.data?.items;
    if (items) {
      setDrawings(items);
      setPast([]);
      setFuture([]);
      setPendingPoint(null);
    }
  }, [drawingsQ.data, ticker, tf]);

  const numeric = useMemo(() => numericChartParams(settings), [settings]);
  const chart = useQuery({
    queryKey: ['chart', ticker, tf, numeric],
    queryFn: () => api.chart(ticker, tf, numeric),
    enabled: Boolean(ticker) && settingsReady,
  });

  const savePreset = useMutation({
    mutationFn: () => api.putPreset('chart', settings),
  });
  const saveDrawings = useMutation({
    mutationFn: (items: ChartDrawing[]) => api.putPreset(drawingsKey, { items }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['preset', drawingsKey] }),
  });

  useEffect(() => {
    if (!containerRef.current || !chart.data) return;
    destroyRef.current?.();
    const mounted = mountSequenceChart(containerRef.current, chart.data, settings, drawings);
    destroyRef.current = mounted.destroy;
    fitRef.current = mounted.fitContent;

    const chartApi = mounted.chart;
    const handler = (param: { time?: unknown; seriesData?: Map<unknown, unknown> }) => {
      if (!param.time) {
        setCrosshair('');
        return;
      }
      const candle = [...(param.seriesData?.values() ?? [])][0] as
        | { open?: number; high?: number; low?: number; close?: number; value?: number }
        | undefined;
      if (candle && candle.open != null) {
        setCrosshair(
          `${String(param.time)}  O ${candle.open.toFixed(2)}  H ${candle.high?.toFixed(2)}  L ${candle.low?.toFixed(2)}  C ${candle.close?.toFixed(2)}`,
        );
      } else if (candle?.value != null) {
        setCrosshair(`${String(param.time)}  ${candle.value.toFixed(2)}`);
      } else {
        setCrosshair(String(param.time));
      }
    };
    chartApi.subscribeCrosshairMove(handler as never);

    return () => {
      chartApi.unsubscribeCrosshairMove(handler as never);
      destroyRef.current?.();
      destroyRef.current = null;
    };
  }, [chart.data, settings, drawings]);

  const applyDrawings = (next: ChartDrawing[]) => {
    const hist = pushDrawingHistory(drawings, past, future, next);
    setDrawings(hist.drawings);
    setPast(hist.past);
    setFuture(hist.future);
    saveDrawings.mutate(hist.drawings);
  };

  const onChartClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (tool === 'cursor' || !chart.data) return;
    const host = containerRef.current;
    if (!host) return;
    // Approximate: map click X to nearest bar by ratio; Y ignored for hline uses last close.
    const rect = host.getBoundingClientRect();
    const ratio = Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width));
    const idx = Math.min(
      chart.data.bars.length - 1,
      Math.max(0, Math.round(ratio * (chart.data.bars.length - 1))),
    );
    let bar = chart.data.bars[idx];
    if (magnet) {
      // already snapped to bar
    }
    const price = bar.close;
    const point = { time: bar.date, price };

    if (tool === 'erase') {
      if (drawings.length) applyDrawings(drawings.slice(0, -1));
      return;
    }
    if (tool === 'hline') {
      applyDrawings([...drawings, { id: newDrawingId(), type: 'hline', points: [point], color: '#2962ff' }]);
      setTool('cursor');
      return;
    }
    if (tool === 'vline') {
      applyDrawings([...drawings, { id: newDrawingId(), type: 'vline', points: [point], color: '#2962ff' }]);
      setTool('cursor');
      return;
    }
    if (tool === 'text') {
      const text = window.prompt('Text note', 'Note');
      if (text) {
        applyDrawings([
          ...drawings,
          { id: newDrawingId(), type: 'text', points: [point], text, color: '#e0e0e0' },
        ]);
      }
      setTool('cursor');
      return;
    }
    if (tool === 'trend' || tool === 'ray' || tool === 'fib') {
      if (!pendingPoint) {
        setPendingPoint(point);
        return;
      }
      applyDrawings([
        ...drawings,
        {
          id: newDrawingId(),
          type: tool,
          points: [pendingPoint, point],
          color: tool === 'fib' ? settings.fib_color : '#2962ff',
        },
      ]);
      setPendingPoint(null);
      setTool('cursor');
    }
  };

  const undo = () => {
    if (!past.length) return;
    const prev = past[past.length - 1];
    setFuture([drawings, ...future]);
    setPast(past.slice(0, -1));
    setDrawings(prev);
    saveDrawings.mutate(prev);
  };
  const redo = () => {
    if (!future.length) return;
    const next = future[0];
    setPast([...past, drawings]);
    setFuture(future.slice(1));
    setDrawings(next);
    saveDrawings.mutate(next);
  };

  const pine = chart.data?.pine;
  const wm = chart.data?.watermark;

  return (
    <div className="chart-page">
      <div className="chart-head">
        <button type="button" className="btn-sm ghost" onClick={() => navigate(-1)}>
          Back
        </button>
        <div>
          <strong>{chart.data?.tvSymbol ?? ticker}</strong>
          <span className="muted small block ellipsis">{chart.data?.companyName ?? ''}</span>
        </div>
        <div className="chart-head-actions">
          <button type="button" className="btn-sm" onClick={() => fitRef.current?.()}>
            Fit
          </button>
          <button type="button" className="btn-sm" onClick={() => setSettingsOpen(true)}>
            Settings
          </button>
        </div>
      </div>

      <Chips value={tf} options={['Daily', 'Weekly', 'Monthly'] as const} onChange={setTf} />

      <DrawingToolbar
        tool={tool}
        onTool={(t) => {
          setTool(t);
          setPendingPoint(null);
        }}
        onUndo={undo}
        onRedo={redo}
        canUndo={past.length > 0}
        canRedo={future.length > 0}
        magnet={magnet}
        onMagnet={setMagnet}
        count={drawings.length}
      />

      {pendingPoint ? (
        <p className="muted small">Click second point for {tool}…</p>
      ) : null}

      <div className="chart-stage">
        <div
          className={`chart-host ${tool !== 'cursor' ? 'drawing' : ''}`}
          ref={containerRef}
          onClick={onChartClick}
        />
        {settings.show_watermark && wm?.lines?.length ? (
          <div className="chart-watermark" style={{ color: settings.wm_text_color }}>
            {wm.lines.map((line) => (
              <div key={line}>{line}</div>
            ))}
          </div>
        ) : null}
      </div>

      {crosshair ? <p className="chart-legend muted small">{crosshair}</p> : null}
      {chart.isLoading ? <p className="muted small">Loading bars…</p> : null}
      {chart.error ? <p className="error">{(chart.error as Error).message}</p> : null}

      <ChartSettingsPanel
        open={settingsOpen}
        value={settings}
        onChange={setSettings}
        onClose={() => setSettingsOpen(false)}
        onSave={() => savePreset.mutate()}
        onReset={() => setSettings(DEFAULT_CHART_SETTINGS)}
      />

      {pine ? (
        <section className="card">
          <div className="chip-row">
            <span className={`badge ${pine.valid ? 'up' : 'down'}`}>
              {pine.valid ? 'VALID' : 'NO SIGNAL'}
            </span>
            {pine.isNew ? <span className="badge up">NEW</span> : null}
            {pine.strong ? <span className="badge">STRONG</span> : null}
          </div>
          <div className="meta-grid">
            <div>
              <span>Close</span>
              {pine.close?.toFixed(2)}
            </div>
            <div>
              <span>RR</span>
              {pine.rr != null ? pine.rr.toFixed(2) : 'n/a'}
            </div>
            <div>
              <span>TP</span>
              {pine.tp != null ? pine.tp.toFixed(2) : 'n/a'}
            </div>
            <div>
              <span>SL</span>
              {pine.sl != null ? pine.sl.toFixed(2) : 'n/a'}
            </div>
          </div>
        </section>
      ) : null}

      {wm?.description ? (
        <section className="card">
          <h3>About</h3>
          <p className="muted small">{wm.description}</p>
        </section>
      ) : null}

      <a
        className="btn btn-accent"
        href={`https://www.tradingview.com/chart/?symbol=${encodeURIComponent(
          chart.data?.tvSymbol ?? ticker,
        )}&interval=${tf === 'Weekly' ? 'W' : tf === 'Monthly' ? 'M' : 'D'}`}
        target="_blank"
        rel="noreferrer"
      >
        Open in TradingView
      </a>
    </div>
  );
}
