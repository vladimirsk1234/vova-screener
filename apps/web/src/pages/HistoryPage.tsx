import { useEffect, useMemo, useState } from 'react';
import { keepPreviousData, useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import {
  HISTORY_GROUP_BYS,
  HISTORY_RANGES,
  HISTORY_TFS,
  UNIVERSES,
  api,
  type HistoryGroupBy,
  type HistoryPeriodSort,
  type HistoryRange,
  type HistoryTf,
  type HistoryTimeframe,
  type HistoryTradeSort,
  type SortDir,
  type Universe,
} from '../lib/api';
import {
  holdLabel,
  money,
  num,
  pct,
  periodLabel,
  realizedRrLabel,
  realizedRrRatio,
  signedMoney,
  signedMultiple,
} from '../lib/format';
import { runHistoryEpsEnrichLoop } from '../lib/historyEpsEnrich';
import { loadHistoryFilters, saveHistoryFilters } from '../lib/tabMemory';
import { useCardFundamentals } from '../lib/useCardFundamentals';
import { useRestoreChartScroll } from '../lib/useRestoreChartScroll';
import { Chips, Switch } from '../components/Chips';
import { SignalCard } from '../components/SignalCard';
import { SortChips } from '../components/SortChips';

type HistoryRangeChip = (typeof HISTORY_RANGES)[number];

const RANGE_LABELS: Record<HistoryRangeChip, string> = {
  all: 'All',
  ytd: 'YTD',
  '1m': '1M',
  '3m': '3M',
  '6m': '6M',
  '1y': '1Y',
};

const PERIOD_SORTS: Array<{ value: HistoryPeriodSort; label: string }> = [
  { value: 'period', label: 'Date' },
  { value: 'pnl', label: 'P&L' },
  { value: 'winRate', label: 'Win %' },
  { value: 'trades', label: 'Count' },
  { value: 'rr', label: 'RR' },
];

const TRADE_SORTS: Array<{ value: HistoryTradeSort; label: string; from?: 'asc' | 'desc' }> = [
  { value: 'date', label: 'Date' },
  { value: 'pnl', label: 'P&L' },
  { value: 'r', label: 'R' },
  { value: 'rr', label: 'RR' },
  { value: 'uv', label: 'UV', from: 'asc' },
  { value: 'interest', label: 'Marked' },
];

function Sparkline({
  points,
  className = 'sparkline',
}: {
  points: Array<{ equity: number }>;
  className?: string;
}) {
  if (points.length < 2) return null;
  const values = points.map((p) => p.equity);
  const min = Math.min(...values, 0);
  const max = Math.max(...values, 0);
  const span = max - min || 1;
  const y = (v: number) => 40 - ((v - min) / span) * 40;
  const x = (i: number) => (i / (values.length - 1)) * 100;
  const path = values
    .map((v, i) => `${i === 0 ? 'M' : 'L'}${x(i).toFixed(2)},${y(v).toFixed(2)}`)
    .join(' ');
  const positive = values[values.length - 1] >= 0;
  const stroke = positive ? '#089981' : '#f23645';
  return (
    <svg viewBox="0 0 100 40" preserveAspectRatio="none" className={className}>
      {/* Break-even, so a curve that never recovers its drawdown reads as one. */}
      <line x1="0" x2="100" y1={y(0)} y2={y(0)} stroke="#2a2e39" strokeWidth="1" />
      <path d={path} fill="none" stroke={stroke} strokeWidth="1.5" />
    </svg>
  );
}

/**
 * Weekly and Monthly are two strategies sharing one screener, and their curves are the
 * quickest way to see which one is actually paying. Shown whatever the filter above is set to.
 */
function TimeframeGrowth({ rows }: { rows: HistoryTimeframe[] }) {
  const active = rows.filter((r) => r.closed > 0);
  if (!active.length) return null;
  return (
    <section className="card">
      <h3 style={{ marginTop: 0 }}>Growth by timeframe</h3>
      <div className="tf-growth">
        {active.map((r) => (
          <div key={r.tf} className="tf-growth-cell">
            <div className="stack-row">
              <strong>{r.tf}</strong>
              <span className={r.pnlUsd >= 0 ? 'up-text' : 'down-text'}>
                {signedMoney(r.pnlUsd)}
              </span>
            </div>
            <Sparkline points={r.equity} className="sparkline sparkline-sm" />
            <p className="muted small" style={{ margin: '4px 0 0' }}>
              {r.closed} closed · {r.winRatePct}% won · {money(r.avgTradeSizeUsd)} ·{' '}
              {pct(r.avgWinPct)} / {pct(r.avgLossPct)} · RR{' '}
              {realizedRrLabel(r.avgWinPct, r.avgLossPct)}
            </p>
          </div>
        ))}
      </div>
    </section>
  );
}

export function HistoryPage() {
  const [universe, setUniverse] = useState<Universe>(() => loadHistoryFilters().universe);
  const [tf, setTf] = useState<HistoryTf>(() => loadHistoryFilters().tf);
  const [groupBy, setGroupBy] = useState<HistoryGroupBy>(() => loadHistoryFilters().groupBy);
  const [range, setRange] = useState<HistoryRangeChip>(() => {
    const saved = loadHistoryFilters().range;
    return (HISTORY_RANGES as readonly HistoryRange[]).includes(saved)
      ? (saved as HistoryRangeChip)
      : 'all';
  });
  const [periodSort, setPeriodSort] = useState<HistoryPeriodSort>('period');
  const [periodDir, setPeriodDir] = useState<SortDir>('desc');
  const [tradeSort, setTradeSort] = useState<HistoryTradeSort>('date');
  const [tradeDir, setTradeDir] = useState<SortDir>('desc');
  const [openPeriod, setOpenPeriod] = useState<string | null>(null);
  const [hideUnprofitable, setHideUnprofitable] = useState(false);
  const queryClient = useQueryClient();

  useEffect(() => {
    saveHistoryFilters({ universe, tf, groupBy, range });
  }, [universe, tf, groupBy, range]);

  const report = useQuery({
    queryKey: ['history', universe, tf, groupBy, range, periodSort, periodDir],
    queryFn: () =>
      api.history({ universe, tf, groupBy, range, sort: periodSort, dir: periodDir }),
    placeholderData: keepPreviousData,
    staleTime: 60_000,
  });

  const trades = useQuery({
    queryKey: ['history-trades', universe, tf, groupBy, range, openPeriod, tradeSort, tradeDir, hideUnprofitable],
    queryFn: () =>
      api.historyTrades({
        universe,
        tf,
        groupBy,
        range,
        periodKey: openPeriod ?? undefined,
        sort: tradeSort,
        dir: tradeDir,
        limit: 200,
        hideUnprofitable,
      }),
    placeholderData: keepPreviousData,
    staleTime: 60_000,
  });

  useRestoreChartScroll(!report.isLoading && !trades.isLoading);

  const invalidateHistory = () => {
    void queryClient.invalidateQueries({ queryKey: ['history-trades'] });
    void queryClient.invalidateQueries({ queryKey: ['history'] });
  };
  const enrichEps = useMutation({
    mutationFn: () => runHistoryEpsEnrichLoop((limit) => api.enrichHistoryEps(limit)),
    onSuccess: invalidateHistory,
  });
  const enrichPremium = useMutation({
    mutationFn: () => api.enrichHistoryPremium(80),
    onSuccess: invalidateHistory,
  });
  const data = report.data;
  const unit = holdLabel(tf);
  const realizedRr = data
    ? realizedRrRatio(data.totals.avgWinPct, data.totals.avgLossPct)
    : null;
  const tradeTickers = useMemo(
    () => (trades.data?.rows ?? []).map((r) => r.yahooTicker),
    [trades.data?.rows],
  );
  const cardFund = useCardFundamentals(tradeTickers);

  const onUniverse = (next: Universe) => {
    setUniverse(next);
    setOpenPeriod(null);
  };

  const onTf = (next: HistoryTf) => {
    setTf(next);
    setOpenPeriod(null);
    if (next !== 'All') setGroupBy(next);
  };

  const onGroupBy = (next: HistoryGroupBy) => {
    setGroupBy(next);
    setOpenPeriod(null);
  };

  const onRange = (next: HistoryRangeChip) => {
    setRange(next);
    setOpenPeriod(null);
  };

  return (
    <div>
      <section className="card">
        <h2>History</h2>
        <p className="muted small">
          Trades closed by a sell-to-close break, on bars that have finished. P&amp;L follows the
          current Max risk; Min RR from Settings filters what counts. Pick Stocks or ETF, then a
          timeframe. Range filters by exit date (Weekly/Monthly bars go back ~10y after
          Rebuild history).
        </p>
        <Chips label="Universe" value={universe} options={UNIVERSES} onChange={onUniverse} />
        <Chips label="Timeframe" value={tf} options={HISTORY_TFS} onChange={onTf} />
        <Chips label="Group by" value={groupBy} options={HISTORY_GROUP_BYS} onChange={onGroupBy} />
        <Chips
          label="Range"
          value={range}
          options={HISTORY_RANGES}
          onChange={onRange}
          format={(v) => RANGE_LABELS[v]}
        />
        <Switch
          label="Hide EPS≤0 at entry"
          checked={hideUnprofitable}
          onChange={setHideUnprofitable}
        />
        <div className="stack-row" style={{ marginTop: 8 }}>
          <button
            type="button"
            className="btn-sm ghost"
            disabled={enrichEps.isPending}
            onClick={() => enrichEps.mutate()}
          >
            {enrichEps.isPending ? 'Tagging EPS…' : 'Tag EPS at entry'}
          </button>
          {enrichEps.data ? (
            <span className="muted small">
              tagged {enrichEps.data.updated}, remaining {enrichEps.data.remaining}
            </span>
          ) : null}
          <button
            type="button"
            className="btn-sm ghost"
            disabled={enrichPremium.isPending}
            onClick={() => enrichPremium.mutate()}
          >
            {enrichPremium.isPending ? 'Tagging UV…' : 'Tag UV at entry'}
          </button>
          {enrichPremium.data ? (
            <span className="muted small">
              tagged {enrichPremium.data.updated}, remaining {enrichPremium.data.remaining}
            </span>
          ) : null}
        </div>
        {enrichEps.error ? (
          <p className="error">
            {(enrichEps.error as Error).message.includes('FMP_API_KEY')
              ? 'Set FMP_API_KEY to tag History EPS at entry.'
              : (enrichEps.error as Error).message}
          </p>
        ) : null}
        {enrichPremium.error ? (
          <p className="error">{(enrichPremium.error as Error).message}</p>
        ) : null}
      </section>

      {report.isLoading ? <p className="empty">Loading…</p> : null}

      {data ? (
        <section className="card">
          <div className="history-stats">
            <div>
              <p className="history-stats-label">Outcome</p>
              <div className="history-stats-grid">
                <div>
                  <span>Win rate</span>
                  <strong>{data.totals.winRatePct}%</strong>
                </div>
                <div>
                  <span>Avg hold ({unit})</span>
                  <strong>{num(data.totals.avgHold)}</strong>
                </div>
                <div>
                  <span>Avg trade size</span>
                  <strong>{money(data.totals.avgTradeSizeUsd)}</strong>
                </div>
                <div>
                  <span>Avg winner</span>
                  <strong className="up-text">{pct(data.totals.avgWinPct)}</strong>
                </div>
                <div>
                  <span>Avg loser</span>
                  <strong className="down-text">{pct(data.totals.avgLossPct)}</strong>
                </div>
                <div>
                  <span>Realized RR</span>
                  <strong>
                    {realizedRrLabel(data.totals.avgWinPct, data.totals.avgLossPct)}
                    {realizedRr != null ? (
                      <span className="muted small"> · {num(realizedRr)}</span>
                    ) : null}
                  </strong>
                </div>
              </div>
            </div>
            <div>
              <p className="history-stats-label">Vs Max risk</p>
              <div className="history-stats-grid">
                <div>
                  <span>Net P&amp;L</span>
                  <strong className={data.totals.pnlUsd >= 0 ? 'up-text' : 'down-text'}>
                    {signedMoney(data.totals.pnlUsd)}
                  </strong>
                </div>
                <div>
                  <span>Risk / trade</span>
                  <strong>{money(data.totals.maxRiskUsd)}</strong>
                </div>
                <div>
                  <span>Closed / active</span>
                  <strong>
                    {data.totals.closed} / {data.totals.active}
                  </strong>
                </div>
                <div>
                  <span>Total risked</span>
                  <strong>{money(data.totals.totalRiskUsd)}</strong>
                </div>
                <div>
                  <span>Profit / risk</span>
                  <strong
                    className={
                      data.totals.profitToRisk == null
                        ? undefined
                        : data.totals.profitToRisk >= 0
                          ? 'up-text'
                          : 'down-text'
                    }
                  >
                    {signedMultiple(data.totals.profitToRisk)}
                  </strong>
                </div>
                <div>
                  <span>Closed invested</span>
                  <strong>{money(data.totals.invested)}</strong>
                </div>
              </div>
            </div>
            <div>
              <p className="history-stats-label">Capital pool</p>
              <div className="history-stats-grid">
                <div>
                  <span>Peak capital</span>
                  <strong>{money(data.totals.peakCapitalUsd)}</strong>
                </div>
                <div>
                  <span>Peaked</span>
                  <strong>
                    {data.totals.peakCapitalAsOf
                      ? periodLabel(data.totals.peakCapitalAsOf, 'Day')
                      : '—'}
                    {data.totals.peakConcurrentPositions > 0 ? (
                      <span className="muted small">
                        {' '}
                        · {data.totals.peakConcurrentPositions} open
                      </span>
                    ) : null}
                  </strong>
                </div>
                <div>
                  <span>Open now</span>
                  <strong>{money(data.totals.openCapitalUsd)}</strong>
                </div>
                <div>
                  <span>ROI on peak</span>
                  <strong
                    className={
                      data.totals.roiOnPeakPct == null
                        ? undefined
                        : data.totals.roiOnPeakPct >= 0
                          ? 'up-text'
                          : 'down-text'
                    }
                  >
                    {pct(data.totals.roiOnPeakPct)}
                  </strong>
                </div>
                <div>
                  <span>Avg capital</span>
                  <strong>{money(data.totals.avgCapitalUsd)}</strong>
                </div>
                <div>
                  <span>ROI on avg</span>
                  <strong
                    className={
                      data.totals.roiOnAvgPct == null
                        ? undefined
                        : data.totals.roiOnAvgPct >= 0
                          ? 'up-text'
                          : 'down-text'
                    }
                  >
                    {pct(data.totals.roiOnAvgPct)}
                  </strong>
                </div>
                <div>
                  <span>S&amp;P (period) return</span>
                  <strong
                    className={
                      data.totals.benchmarkReturnPct == null
                        ? undefined
                        : data.totals.benchmarkReturnPct >= 0
                          ? 'up-text'
                          : 'down-text'
                    }
                  >
                    {pct(data.totals.benchmarkReturnPct)}
                    {data.totals.benchmarkSymbol ? (
                      <span className="muted small"> · {data.totals.benchmarkSymbol}</span>
                    ) : null}
                  </strong>
                </div>
                <div>
                  <span>Alpha vs S&amp;P (on peak)</span>
                  <strong
                    className={
                      data.totals.alphaVsBenchmarkPct == null
                        ? undefined
                        : data.totals.alphaVsBenchmarkPct >= 0
                          ? 'up-text'
                          : 'down-text'
                    }
                  >
                    {pct(data.totals.alphaVsBenchmarkPct)}
                  </strong>
                </div>
                <div>
                  <span>Alpha vs S&amp;P (on avg)</span>
                  <strong
                    className={
                      data.totals.alphaOnAvgPct == null
                        ? undefined
                        : data.totals.alphaOnAvgPct >= 0
                          ? 'up-text'
                          : 'down-text'
                    }
                  >
                    {pct(data.totals.alphaOnAvgPct)}
                  </strong>
                </div>
              </div>
              <p className="muted small" style={{ margin: '8px 0 0' }}>
                Min. cash to take every signal at the current Max risk. A close frees size for a
                new trade the same day. Avg is the calendar-day mean of that sweep (idle days
                count). S&amp;P is SPY total return (Yahoo adjusted close) over the selected
                range; alpha is ROI minus that return.
              </p>
            </div>
          </div>
          <p className="muted small" style={{ margin: '10px 0 0' }}>
            Avg R {num(data.totals.avgR)} · RR at entry {num(data.totals.avgRrEntry)}
          </p>
          <Sparkline points={data.equity} />
          {data.exitReasons.length ? (
            <p className="muted small" style={{ marginBottom: 0 }}>
              {data.exitReasons.map((r) => `${r.reason} ${r.count}`).join(' · ')}
            </p>
          ) : null}
        </section>
      ) : null}

      {data ? <TimeframeGrowth rows={data.timeframes} /> : null}

      {data ? (
        <section className="card">
          <div className="stack-row">
            <h3 style={{ margin: 0 }}>Periods</h3>
            <SortChips
              label="Sort periods"
              options={PERIOD_SORTS}
              value={periodSort}
              dir={periodDir}
              onChange={(next, dir) => {
                setPeriodSort(next);
                setPeriodDir(dir);
              }}
            />
          </div>

          {data.periods.length === 0 ? (
            <p className="muted">No closed signals yet.</p>
          ) : (
            data.periods.map((p) => (
              <button
                key={p.periodKey}
                type="button"
                className={`list-row period-row${openPeriod === p.periodKey ? ' active' : ''}`}
                onClick={() => setOpenPeriod(openPeriod === p.periodKey ? null : p.periodKey)}
              >
                <span>{periodLabel(p.periodKey, groupBy)}</span>
                <span>
                  <span className={p.pnlUsd >= 0 ? 'up-text' : 'down-text'}>
                    {signedMoney(p.pnlUsd)}
                  </span>
                  <span className="muted small">
                    {' '}
                    · {p.trades} · {p.winRatePct}% · R {num(p.avgR)} · hold {num(p.avgHold)} {unit}
                  </span>
                </span>
              </button>
            ))
          )}
        </section>
      ) : null}

      <section className="card">
        <div className="stack-row">
          <h3 style={{ margin: 0 }}>
            {openPeriod ? periodLabel(openPeriod, groupBy) : 'All closed'}
            <span className="muted small"> · {trades.data?.total ?? 0}</span>
          </h3>
          {openPeriod ? (
            <button type="button" className="btn-sm ghost" onClick={() => setOpenPeriod(null)}>
              Clear
            </button>
          ) : null}
        </div>
        <SortChips
          label="Sort closed signals"
          options={TRADE_SORTS}
          value={tradeSort}
          dir={tradeDir}
          onChange={(next, dir) => {
            setTradeSort(next);
            setTradeDir(dir);
          }}
        />
        {trades.data && trades.data.rows.length === 0 ? (
          <p className="muted">Nothing closed here yet.</p>
        ) : null}
      </section>

      {trades.data?.rows.map((row) => (
        <SignalCard
          key={row.id}
          row={row}
          bucket="closed"
          fundamentals={cardFund.data?.[row.yahooTicker.toUpperCase()] ?? null}
        />
      ))}
    </div>
  );
}
