import { useState } from 'react';
import { keepPreviousData, useQuery } from '@tanstack/react-query';
import {
  TIMEFRAMES,
  api,
  type HistoryPeriodSort,
  type HistoryTf,
  type HistoryTradeSort,
  type SortDir,
  type Timeframe,
} from '../lib/api';
import { holdLabel, money, num, periodLabel, signedMoney } from '../lib/format';
import { Chips } from '../components/Chips';
import { SignalCard } from '../components/SignalCard';

const HISTORY_TFS = ['Daily', 'Weekly', 'Monthly', 'All'] as const satisfies readonly HistoryTf[];

const PERIOD_SORTS: Array<{ value: HistoryPeriodSort; label: string }> = [
  { value: 'period', label: 'Date' },
  { value: 'pnl', label: 'P&L' },
  { value: 'winRate', label: 'Win %' },
  { value: 'trades', label: 'Count' },
];

const TRADE_SORTS: Array<{ value: HistoryTradeSort; label: string }> = [
  { value: 'date', label: 'Date' },
  { value: 'pnl', label: 'P&L' },
  { value: 'r', label: 'R' },
  { value: 'rr', label: 'RR' },
  { value: 'interest', label: 'Marked' },
];

function Sparkline({ points }: { points: Array<{ equity: number }> }) {
  if (points.length < 2) return null;
  const values = points.map((p) => p.equity);
  const min = Math.min(...values, 0);
  const max = Math.max(...values, 0);
  const span = max - min || 1;
  const path = values
    .map((v, i) => {
      const x = (i / (values.length - 1)) * 100;
      const y = 40 - ((v - min) / span) * 40;
      return `${i === 0 ? 'M' : 'L'}${x.toFixed(2)},${y.toFixed(2)}`;
    })
    .join(' ');
  const positive = values[values.length - 1] >= 0;
  return (
    <svg viewBox="0 0 100 40" preserveAspectRatio="none" className="sparkline">
      <path d={path} fill="none" stroke={positive ? '#089981' : '#f23645'} strokeWidth="1.5" />
    </svg>
  );
}

export function HistoryPage() {
  const [tf, setTf] = useState<HistoryTf>('Daily');
  const [groupBy, setGroupBy] = useState<Timeframe>('Daily');
  const [periodSort, setPeriodSort] = useState<HistoryPeriodSort>('period');
  const [periodDir, setPeriodDir] = useState<SortDir>('desc');
  const [tradeSort, setTradeSort] = useState<HistoryTradeSort>('date');
  const [tradeDir, setTradeDir] = useState<SortDir>('desc');
  const [openPeriod, setOpenPeriod] = useState<string | null>(null);

  const report = useQuery({
    queryKey: ['history', tf, groupBy, periodSort, periodDir],
    queryFn: () => api.history({ tf, groupBy, sort: periodSort, dir: periodDir }),
    placeholderData: keepPreviousData,
    staleTime: 60_000,
  });

  const trades = useQuery({
    queryKey: ['history-trades', tf, groupBy, openPeriod, tradeSort, tradeDir],
    queryFn: () =>
      api.historyTrades({
        tf,
        groupBy,
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

  const onTf = (next: HistoryTf) => {
    setTf(next);
    setOpenPeriod(null);
    if (next !== 'All') setGroupBy(next);
  };

  const onGroupBy = (next: Timeframe) => {
    setGroupBy(next);
    setOpenPeriod(null);
  };

  return (
    <div>
      <section className="card">
        <h2>History</h2>
        <p className="muted small">Closed signals only. Statistics follow the timeframe you pick.</p>
        <Chips label="Timeframe" value={tf} options={HISTORY_TFS} onChange={onTf} />
        <Chips label="Group by" value={groupBy} options={TIMEFRAMES} onChange={onGroupBy} />
      </section>

      {report.isLoading ? <p className="empty">Loading…</p> : null}

      {data ? (
        <section className="card">
          <div className="meta-grid">
            <div>
              <span>Win rate</span>
              {data.totals.winRatePct}%
            </div>
            <div>
              <span>Net P&amp;L</span>
              <span className={data.totals.pnlUsd >= 0 ? 'up-text' : 'down-text'}>
                {signedMoney(data.totals.pnlUsd)}
              </span>
            </div>
            <div>
              <span>Closed / active</span>
              {data.totals.closed} / {data.totals.active}
            </div>
            <div>
              <span>Avg R</span>
              {num(data.totals.avgR)}
            </div>
            <div>
              <span>Avg RR at entry</span>
              {num(data.totals.avgRrEntry)}
            </div>
            <div>
              <span>Avg hold ({unit})</span>
              {num(data.totals.avgHold)}
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

      {data ? (
        <section className="card">
          <div className="stack-row">
            <h3 style={{ margin: 0 }}>Periods</h3>
            <div className="sort-row">
              {PERIOD_SORTS.map((option) => (
                <button
                  key={option.value}
                  type="button"
                  className={`sort-chip${periodSort === option.value ? ' active' : ''}`}
                  onClick={() => {
                    setPeriodDir(periodSort === option.value && periodDir === 'desc' ? 'asc' : 'desc');
                    setPeriodSort(option.value);
                  }}
                >
                  {option.label}
                  {periodSort === option.value ? (periodDir === 'desc' ? ' ↓' : ' ↑') : ''}
                </button>
              ))}
            </div>
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
        <div className="sort-row">
          {TRADE_SORTS.map((option) => (
            <button
              key={option.value}
              type="button"
              className={`sort-chip${tradeSort === option.value ? ' active' : ''}`}
              onClick={() => {
                setTradeDir(tradeSort === option.value && tradeDir === 'desc' ? 'asc' : 'desc');
                setTradeSort(option.value);
              }}
            >
              {option.label}
              {tradeSort === option.value ? (tradeDir === 'desc' ? ' ↓' : ' ↑') : ''}
            </button>
          ))}
        </div>
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
