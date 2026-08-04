import { useEffect, useState } from 'react';
import { keepPreviousData, useQuery } from '@tanstack/react-query';
import {
  HISTORY_RANGES,
  TIMEFRAMES,
  UNIVERSES,
  api,
  type HistoryPeriodSort,
  type HistoryRange,
  type HistoryTf,
  type HistoryTimeframe,
  type HistoryTradeSort,
  type SortDir,
  type Timeframe,
  type Universe,
} from '../lib/api';
import { holdLabel, money, num, periodLabel, signedMoney } from '../lib/format';
import { loadHistoryFilters, saveHistoryFilters } from '../lib/tabMemory';
import { Chips } from '../components/Chips';
import { SignalCard } from '../components/SignalCard';
import { SortChips } from '../components/SortChips';

const HISTORY_TFS = ['Daily', 'Weekly', 'Monthly', 'All'] as const satisfies readonly HistoryTf[];

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

const TRADE_SORTS: Array<{ value: HistoryTradeSort; label: string }> = [
  { value: 'date', label: 'Date' },
  { value: 'pnl', label: 'P&L' },
  { value: 'r', label: 'R' },
  { value: 'rr', label: 'RR' },
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
 * Daily, Weekly and Monthly are three strategies sharing one screener, and their curves are the
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
              {r.closed} closed · {r.winRatePct}% won · R {num(r.avgR)}
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
  const [groupBy, setGroupBy] = useState<Timeframe>(() => loadHistoryFilters().groupBy);
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
    queryKey: ['history-trades', universe, tf, groupBy, range, openPeriod, tradeSort, tradeDir],
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
      }),
    placeholderData: keepPreviousData,
    staleTime: 60_000,
  });

  const data = report.data;
  const unit = holdLabel(tf);

  const onUniverse = (next: Universe) => {
    setUniverse(next);
    setOpenPeriod(null);
  };

  const onTf = (next: HistoryTf) => {
    setTf(next);
    setOpenPeriod(null);
    if (next !== 'All') setGroupBy(next);
  };

  const onGroupBy = (next: Timeframe) => {
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
          timeframe. Range filters by exit date (Daily bars go back ~2y, Weekly/Monthly ~10y after
          Rebuild history).
        </p>
        <Chips label="Universe" value={universe} options={UNIVERSES} onChange={onUniverse} />
        <Chips label="Timeframe" value={tf} options={HISTORY_TFS} onChange={onTf} />
        <Chips label="Group by" value={groupBy} options={TIMEFRAMES} onChange={onGroupBy} />
        <Chips
          label="Range"
          value={range}
          options={HISTORY_RANGES}
          onChange={onRange}
          format={(v) => RANGE_LABELS[v]}
        />
      </section>

      {report.isLoading ? <p className="empty">Loading…</p> : null}

      {data ? (
        <section className="card">
          <div className="meta-grid">
            <div>
              <span>Win rate</span>
              <strong>{data.totals.winRatePct}%</strong>
            </div>
            <div>
              <span>Net P&amp;L</span>
              <strong className={data.totals.pnlUsd >= 0 ? 'up-text' : 'down-text'}>
                {signedMoney(data.totals.pnlUsd)}
              </strong>
            </div>
            <div>
              <span>Closed / active</span>
              <strong>
                {data.totals.closed} / {data.totals.active}
              </strong>
            </div>
            <div>
              <span>Avg R</span>
              <strong>{num(data.totals.avgR)}</strong>
            </div>
            <div>
              <span>Avg RR at entry</span>
              <strong>{num(data.totals.avgRrEntry)}</strong>
            </div>
            <div>
              <span>Avg hold ({unit})</span>
              <strong>{num(data.totals.avgHold)}</strong>
            </div>
          </div>
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
        <p className="muted small" style={{ marginBottom: 0 }}>
          Invested {money(data?.totals.invested)}
        </p>
      </section>

      {trades.data?.rows.map((row) => <SignalCard key={row.id} row={row} bucket="closed" />)}
    </div>
  );
}
