import { useEffect, useMemo, useRef, useState } from 'react';
import { useLocation, useNavigate, useParams } from 'react-router-dom';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import {
  api,
  type BuySignal,
  type ChartDrawing,
  type ChartSettings,
  type Timeframe,
} from '../lib/api';
import { Chips } from '../components/Chips';
import { ChartSettingsPanel } from '../components/ChartSettingsPanel';
import { mountSequenceChart } from '../components/mountSequenceChart';
import {
  DEFAULT_CHART_SETTINGS,
  mergeChartSettings,
  numericChartParams,
} from '../lib/chartSettings';

type ChartNavState = {
  signal?: BuySignal;
  runId?: string;
  riskUsd?: number;
  tf?: Timeframe;
  periodKey?: string;
};

function sharesFromRisk(entry: number, sl: number | null | undefined, riskUsd: number) {
  if (sl == null || !Number.isFinite(sl) || !Number.isFinite(entry)) return 0;
  const risk = entry - sl;
  if (risk <= 0 || riskUsd <= 0) return 0;
  return Math.max(0, Math.floor(riskUsd / risk));
}

export function ChartPage() {
  const { ticker = '' } = useParams();
  const navigate = useNavigate();
  const location = useLocation();
  const queryClient = useQueryClient();
  const navState = (location.state as ChartNavState | null) ?? {};

  const [tf, setTf] = useState<Timeframe>(navState.tf ?? 'Daily');
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [settings, setSettings] = useState<ChartSettings>(DEFAULT_CHART_SETTINGS);
  const [settingsReady, setSettingsReady] = useState(false);
  const [drawings, setDrawings] = useState<ChartDrawing[]>([]);
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

  const markInterest = useMutation({
    mutationFn: (status: 'interested' | 'not_interested') => {
      const signal = navState.signal;
      const pine = chart.data?.pine;
      const entry = signal?.entry ?? pine?.close;
      if (entry == null || !Number.isFinite(entry)) {
        throw new Error('No entry price available');
      }
      const tp = signal?.tp ?? pine?.tp ?? undefined;
      const sl = signal?.sl ?? pine?.sl ?? undefined;
      const riskUsd = navState.riskUsd ?? settings.risk_dollars ?? 100;
      const shares =
        signal?.shares ??
        sharesFromRisk(entry, sl ?? null, riskUsd);
      const periodKey = navState.periodKey;
      if (!periodKey) {
        throw new Error('Missing period — open chart from a scan result');
      }

      return api.createTrade({
        symbol: signal?.symbol ?? chart.data?.tvSymbol ?? ticker,
        yahooTicker: signal?.yahooTicker ?? chart.data?.yahooTicker ?? ticker,
        companyName: signal?.companyName ?? chart.data?.companyName,
        tf: navState.tf ?? tf,
        entry,
        tp: tp ?? undefined,
        sl: sl ?? undefined,
        rrAtEntry: signal?.rr ?? pine?.rr ?? undefined,
        shares,
        riskUsd,
        asOf: signal?.asOf,
        runId: navState.runId,
        periodKey,
        status,
        source: 'manual',
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['trades'] });
      queryClient.invalidateQueries({ queryKey: ['signals'] });
    },
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

  useEffect(() => {
    const onWheel = (e: WheelEvent) => {
      if (e.ctrlKey) e.preventDefault();
    };
    document.addEventListener('wheel', onWheel, { passive: false });
    return () => document.removeEventListener('wheel', onWheel);
  }, []);

  const pine = chart.data?.pine;
  const wm = chart.data?.watermark;
  const canMark =
    Boolean(navState.periodKey) &&
    (navState.signal?.entry != null || (pine?.close != null && Number.isFinite(pine.close)));
  const marking = markInterest.isPending;

  return (
    <div className="chart-page">
      <div className="chart-head">
        <button type="button" className="btn-sm ghost" onClick={() => navigate(-1)}>
          Back
        </button>
        <div className="chart-head-title">
          <strong>{chart.data?.tvSymbol ?? ticker}</strong>
          <span className="muted small block ellipsis">{chart.data?.companyName ?? ''}</span>
          {pine ? (
            <div className="chart-pine-row">
              <span className={`badge ${pine.valid ? 'up' : 'down'}`}>
                {pine.valid ? 'VALID' : 'NO SIGNAL'}
              </span>
              {pine.isNew ? <span className="badge up">NEW</span> : null}
              {pine.strong ? <span className="badge">STRONG</span> : null}
              <span className="chart-pine-metric">
                <span>Close</span> {pine.close?.toFixed(2) ?? 'n/a'}
              </span>
              <span className="chart-pine-metric">
                <span>RR</span> {pine.rr != null ? pine.rr.toFixed(2) : 'n/a'}
              </span>
              <span className="chart-pine-metric">
                <span>TP</span> {pine.tp != null ? pine.tp.toFixed(2) : 'n/a'}
              </span>
              <span className="chart-pine-metric">
                <span>SL</span> {pine.sl != null ? pine.sl.toFixed(2) : 'n/a'}
              </span>
            </div>
          ) : null}
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

      <div className="chart-stage">
        <div className="chart-host" ref={containerRef} />
        {crosshair ? (
          <p className="chart-legend small" style={{ color: settings.wm_text_color }}>
            {crosshair}
          </p>
        ) : null}
        {settings.show_watermark && wm?.lines?.length ? (
          <div
            className="chart-watermark"
            style={{ color: settings.wm_text_color, fontSize: settings.wm_font_size }}
          >
            {wm.lines.map((line) => (
              <div key={line}>{line}</div>
            ))}
          </div>
        ) : null}
      </div>

      {chart.isLoading ? <p className="muted small">Loading bars…</p> : null}
      {chart.error ? <p className="error">{(chart.error as Error).message}</p> : null}
      {markInterest.error ? (
        <p className="error">{(markInterest.error as Error).message}</p>
      ) : null}

      <div className="card-actions chart-actions">
        <button
          type="button"
          className="btn-sm"
          disabled={!canMark || marking}
          onClick={() => markInterest.mutate('interested')}
        >
          {marking ? 'Saving…' : 'Interested'}
        </button>
        <button
          type="button"
          className="btn-sm ghost"
          disabled={!canMark || marking}
          onClick={() => markInterest.mutate('not_interested')}
        >
          Not Interested
        </button>
        <a
          className="btn-sm ghost"
          href="https://app.fastgraphs.com/dashboard"
          target="_blank"
          rel="noreferrer"
        >
          FastGraph
        </a>
        <a
          className="btn-sm ghost"
          href={`https://www.tradingview.com/chart/?symbol=${encodeURIComponent(
            chart.data?.tvSymbol ?? ticker,
          )}&interval=${tf === 'Weekly' ? 'W' : tf === 'Monthly' ? 'M' : 'D'}`}
          target="_blank"
          rel="noreferrer"
        >
          TradingView
        </a>
      </div>

      <ChartSettingsPanel
        open={settingsOpen}
        value={settings}
        onChange={setSettings}
        onClose={() => setSettingsOpen(false)}
        onSave={() => savePreset.mutate()}
        onReset={() => setSettings(DEFAULT_CHART_SETTINGS)}
      />

      {wm?.description ? (
        <section className="card">
          <h3>About</h3>
          <p className="muted small">{wm.description}</p>
        </section>
      ) : null}
    </div>
  );
}
