import { useEffect, useMemo, useRef, useState } from 'react';
import { useLocation, useNavigate, useParams } from 'react-router-dom';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import {
  api,
  type ChartDrawing,
  type ChartSettings,
  type Interest,
  type ResultRow,
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
import { investedFromShares, sharesFromRisk } from '../lib/positionSize';

type ChartNavState = { row?: ResultRow };

function money(n: number) {
  return n.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

export function ChartPage() {
  const { ticker = '' } = useParams();
  const navigate = useNavigate();
  const location = useLocation();
  const queryClient = useQueryClient();
  const navState = (location.state as ChartNavState | null) ?? {};

  const [tf, setTf] = useState<Timeframe>(navState.row?.tf ?? 'Daily');
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [settings, setSettings] = useState<ChartSettings>(DEFAULT_CHART_SETTINGS);
  const [settingsReady, setSettingsReady] = useState(false);
  const [drawings, setDrawings] = useState<ChartDrawing[]>([]);
  const [crosshair, setCrosshair] = useState<string>('');

  const containerRef = useRef<HTMLDivElement | null>(null);
  const destroyRef = useRef<(() => void) | null>(null);

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

  // The mark lives on the tracked signal, so a chart opened straight from a URL has to find it.
  const tracked = useQuery({
    queryKey: ['tracked-signal', ticker, tf],
    queryFn: () => api.lookupSignal(ticker, tf),
    enabled: Boolean(ticker),
    initialData: navState.row?.tf === tf ? navState.row : undefined,
  });

  const markInterest = useMutation({
    mutationFn: (next: Interest | null) => {
      const id = tracked.data?.id;
      if (!id) throw new Error('This symbol is not a tracked signal on this timeframe');
      return api.setInterest(id, next);
    },
    onSuccess: (row) => {
      queryClient.setQueryData(['tracked-signal', ticker, tf], row);
      void queryClient.invalidateQueries({ queryKey: ['results'] });
      void queryClient.invalidateQueries({ queryKey: ['history-trades'] });
    },
  });

  const markStatus = tracked.data?.interest ?? null;

  useEffect(() => {
    if (!containerRef.current || !chart.data) return;
    destroyRef.current?.();
    const mounted = mountSequenceChart(containerRef.current, chart.data, settings, drawings);
    destroyRef.current = mounted.destroy;

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
    const onDblClick = () => {
      mounted.fitContent();
    };
    chartApi.subscribeDblClick(onDblClick);

    return () => {
      chartApi.unsubscribeDblClick(onDblClick);
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
  const row = tracked.data ?? null;
  const riskUsd = row?.riskUsd || settings.risk_dollars || 100;

  const tradeMetrics = useMemo(() => {
    const entry = row?.entry ?? pine?.close ?? null;
    const tp = row?.tp ?? pine?.tp ?? null;
    const sl = row?.sl ?? pine?.sl ?? null;
    const rr = row?.currentRr ?? row?.rr ?? pine?.rr ?? null;
    const shares =
      row?.shares != null && row.shares > 0
        ? row.shares
        : entry != null
          ? sharesFromRisk(entry, sl, riskUsd)
          : 0;
    const dollars = entry != null ? investedFromShares(entry, shares) : 0;
    return { tp, sl, rr, shares, dollars };
  }, [row, pine, riskUsd]);

  const showMetrics = Boolean(pine || row);
  const canMark = Boolean(row?.id);
  const marking = markInterest.isPending;

  return (
    <div className="chart-page">
      <div className="chart-head">
        <button
          type="button"
          className="chart-icon-btn ghost"
          aria-label="Back"
          onClick={() => navigate(-1)}
        >
          ←
        </button>
        <div className="chart-head-title">
          <div className="chart-head-name ellipsis">
            <strong>{chart.data?.tvSymbol ?? ticker}</strong>
            {chart.data?.companyName ? (
              <span className="muted small">{chart.data.companyName}</span>
            ) : null}
          </div>
        </div>
        <button
          type="button"
          className="chart-icon-btn"
          aria-label="Settings"
          onClick={() => setSettingsOpen(true)}
        >
          ⚙
        </button>
      </div>

      {showMetrics ? (
        <div className="chart-pine-row">
          {pine ? (
            <>
              {pine.isNew ? (
                <span className="badge up">NEW</span>
              ) : (
                <span className={`badge ${pine.valid ? 'up' : 'down'}`}>
                  {pine.valid ? 'VALID' : 'NO SIGNAL'}
                </span>
              )}
              {pine.strong ? <span className="badge">STRONG</span> : null}
            </>
          ) : null}
          <span className="chart-pine-metric">
            <span>RR</span>{' '}
            {tradeMetrics.rr != null && Number.isFinite(tradeMetrics.rr)
              ? tradeMetrics.rr.toFixed(2)
              : 'n/a'}
          </span>
          <span className="chart-pine-metric">
            <span>TP</span>{' '}
            {tradeMetrics.tp != null ? money(tradeMetrics.tp) : 'n/a'}
          </span>
          <span className="chart-pine-metric">
            <span>SL</span>{' '}
            {tradeMetrics.sl != null ? money(tradeMetrics.sl) : 'n/a'}
          </span>
          <span className="chart-pine-metric">
            <span>Sh</span> {tradeMetrics.shares}
          </span>
          <span className="chart-pine-metric">
            <span>$</span> {money(tradeMetrics.dollars)}
          </span>
        </div>
      ) : null}

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

      {chart.isLoading ? <p className="muted small chart-status-line">Loading bars…</p> : null}
      {chart.error ? (
        <p className="error chart-status-line">{(chart.error as Error).message}</p>
      ) : null}
      {markInterest.error ? (
        <p className="error chart-status-line">{(markInterest.error as Error).message}</p>
      ) : null}

      <div className="card-actions chart-actions">
        <button
          type="button"
          className={`btn-sm${markStatus === 'interested' ? ' selected' : ' ghost'}`}
          disabled={!canMark || marking}
          onClick={() => markInterest.mutate(markStatus === 'interested' ? null : 'interested')}
        >
          {marking ? 'Saving…' : 'Interested'}
        </button>
        <button
          type="button"
          className={`btn-sm${markStatus === 'not_interested' ? ' danger selected' : ' ghost'}`}
          disabled={!canMark || marking}
          onClick={() =>
            markInterest.mutate(markStatus === 'not_interested' ? null : 'not_interested')
          }
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
    </div>
  );
}
