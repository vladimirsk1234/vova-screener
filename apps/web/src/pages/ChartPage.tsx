import { useEffect, useMemo, useRef, useState } from 'react';
import { useLocation, useNavigate, useParams, useSearchParams } from 'react-router-dom';
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
import { mountSequenceChart, type ChartTrade } from '../components/mountSequenceChart';
import { barsLabel, signedMoney } from '../lib/format';
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
  const [search] = useSearchParams();
  const queryClient = useQueryClient();
  const navState = (location.state as ChartNavState | null) ?? {};
  const tradeId = search.get('trade');

  const [tf, setTf] = useState<Timeframe>(navState.row?.tf ?? 'Daily');
  // A trade opens as a snapshot of itself: the series cut at the bar it broke on, so the structure
  // on screen is the structure that closed it. Everything after that is one tap away.
  const [snapshot, setSnapshot] = useState(true);
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

  const appSettings = useQuery({ queryKey: ['settings'], queryFn: api.settings });
  const maxRiskUsd = appSettings.data?.maxRiskUsd;

  // The mark lives on the tracked signal, so a chart opened straight from a URL has to find it.
  // A closed trade cannot be looked up by ticker — nothing is active — so History passes its id.
  const trade = useQuery({
    queryKey: ['tracked-signal-by-id', tradeId],
    queryFn: () => api.signal(tradeId as string),
    enabled: Boolean(tradeId),
    initialData: navState.row?.id === tradeId ? navState.row : undefined,
  });
  const tracked = useQuery({
    queryKey: ['tracked-signal', ticker, tf],
    queryFn: () => api.lookupSignal(ticker, tf),
    enabled: Boolean(ticker) && !tradeId,
    initialData: navState.row?.tf === tf ? navState.row : undefined,
  });
  const row = (tradeId ? trade.data : tracked.data) ?? null;

  // The trade decides the timeframe: a Weekly trade read on the Daily chart is a different chart.
  const tradeTf = trade.data?.tf;
  useEffect(() => {
    if (tradeTf) setTf(tradeTf);
  }, [tradeTf]);

  const asOf = snapshot ? (trade.data?.exitDate ?? null) : null;
  const numeric = useMemo(() => numericChartParams(settings), [settings]);
  const chart = useQuery({
    queryKey: ['chart', ticker, tf, numeric, maxRiskUsd, asOf],
    queryFn: () => api.chart(ticker, tf, numeric, maxRiskUsd, asOf),
    enabled: Boolean(ticker) && settingsReady && maxRiskUsd != null && !(tradeId && !trade.data),
  });

  const savePreset = useMutation({
    mutationFn: () => api.putPreset('chart', settings),
  });

  const markInterest = useMutation({
    mutationFn: (next: Interest | null) => {
      const id = row?.id;
      if (!id) throw new Error('This symbol is not a tracked signal on this timeframe');
      return api.setInterest(id, next);
    },
    onSuccess: (saved) => {
      queryClient.setQueryData(
        tradeId ? ['tracked-signal-by-id', tradeId] : ['tracked-signal', ticker, tf],
        saved,
      );
      void queryClient.invalidateQueries({ queryKey: ['results'] });
      void queryClient.invalidateQueries({ queryKey: ['history-trades'] });
    },
  });

  const markStatus = row?.interest ?? null;

  // Only a chart opened on a trade draws one. The live chart keeps showing what the engine reports
  // for the latest bar, which is what every other screen is reading from.
  const chartTrade = useMemo<ChartTrade | null>(
    () =>
      tradeId && trade.data
        ? {
            entry: trade.data.entry,
            tp: trade.data.tp,
            sl: trade.data.sl,
            openedAsOf: trade.data.openedAsOf,
            exitDate: trade.data.exitDate,
            exitPrice: trade.data.exitPrice,
          }
        : null,
    [tradeId, trade.data],
  );

  useEffect(() => {
    if (!containerRef.current || !chart.data) return;
    destroyRef.current?.();
    const mounted = mountSequenceChart(
      containerRef.current,
      chart.data,
      settings,
      drawings,
      chartTrade,
    );
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
  }, [chart.data, settings, drawings, chartTrade]);

  useEffect(() => {
    const onWheel = (e: WheelEvent) => {
      if (e.ctrlKey) e.preventDefault();
    };
    document.addEventListener('wheel', onWheel, { passive: false });
    return () => document.removeEventListener('wheel', onWheel);
  }, []);

  const pine = chart.data?.pine;
  const wm = chart.data?.watermark;
  // A tracked signal carries the risk it was sized at; anything else uses the current setting.
  const riskUsd = row?.riskUsd || maxRiskUsd || 100;

  const tradeMetrics = useMemo(() => {
    const entry = row?.entry ?? pine?.close ?? null;
    const tp = row?.tp ?? pine?.tp ?? null;
    const sl = row?.sl ?? pine?.sl ?? null;
    // A closed trade reports the RR it was taken at; a running one reports where it stands now.
    const rr = row?.exitDate ? (row.rr ?? null) : (row?.currentRr ?? row?.rr ?? pine?.rr ?? null);
    const shares =
      row?.shares != null && row.shares > 0
        ? row.shares
        : entry != null
          ? sharesFromRisk(entry, sl, riskUsd)
          : 0;
    const dollars = entry != null ? investedFromShares(entry, shares) : 0;
    return { tp, sl, rr, shares, dollars };
  }, [row, pine, riskUsd]);

  // Realized or closing on the bar in progress — either way the trade is over and the header has
  // its exit to report instead of a NEW / VALID badge about the setup running right now.
  const closedTrade = row && (row.status === 'closed' || row.provisionalClose) ? row : null;
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

      {closedTrade ? (
        <div className="chart-trade-row">
          <span className="badge">
            {closedTrade.provisionalClose ? 'CLOSING' : 'SELL TO CLOSE'}
          </span>
          <span className="chart-pine-metric">
            <span>In</span> {closedTrade.openedAsOf ?? '—'} @ {money(closedTrade.entry)}
          </span>
          <span className="chart-pine-metric">
            <span>Out</span> {closedTrade.exitDate ?? '—'} @{' '}
            {closedTrade.exitPrice != null ? money(closedTrade.exitPrice) : '—'}
          </span>
          <span
            className={`chart-pine-metric ${(closedTrade.pnlUsd ?? 0) >= 0 ? 'up-text' : 'down-text'}`}
          >
            <span>P&amp;L</span> {signedMoney(closedTrade.pnlUsd)}
            {closedTrade.pnlR != null ? ` · ${closedTrade.pnlR.toFixed(2)}R` : ''}
          </span>
          <button
            type="button"
            className={`btn-sm${snapshot ? ' selected' : ' ghost'}`}
            title="Bars up to the exit, structure as it was when the trade closed"
            onClick={() => setSnapshot(!snapshot)}
          >
            {snapshot ? 'Snapshot' : 'Live'}
          </button>
        </div>
      ) : null}

      {showMetrics ? (
        <div className="chart-pine-row">
          {pine && !closedTrade ? (
            <>
              {/* Same rule and the same number as the Results tabs: the signal is NEW on the bar it
                  appeared on and VALID on every bar after it. Nothing else — the RR settings below
                  do not move a signal between the two. */}
              {pine.barsSinceValid === 0 ? (
                <span className="badge up" title={`New signal on the current ${tf} bar`}>
                  NEW
                </span>
              ) : (
                <span
                  className={`badge ${pine.barsSinceValid != null ? 'up' : 'down'}`}
                  title={pine.validSinceAsOf ? `Signal bar ${pine.validSinceAsOf}` : undefined}
                >
                  {pine.barsSinceValid != null
                    ? `VALID · ${barsLabel(pine.barsSinceValid)}`
                    : 'NO SIGNAL'}
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
