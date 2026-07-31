import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { api, type Timeframe } from '../lib/api';
import { Chips } from '../components/Chips';

function money(n: number | null | undefined) {
  if (n == null) return '—';
  return n.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

function periodLabel(periodKey: string, tf: Timeframe) {
  if (!periodKey || periodKey === 'unknown') return periodKey || '—';
  if (tf === 'Daily') {
    return new Date(`${periodKey}T12:00:00`).toLocaleDateString(undefined, {
      weekday: 'short',
      year: 'numeric',
      month: 'short',
      day: 'numeric',
    });
  }
  if (tf === 'Monthly') {
    const [y, m] = periodKey.split('-');
    if (!y || !m) return periodKey;
    return new Date(Number(y), Number(m) - 1, 1).toLocaleDateString(undefined, {
      year: 'numeric',
      month: 'long',
    });
  }
  return periodKey;
}

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

export function PnlPage() {
  const [tf, setTf] = useState<Timeframe>('Daily');
  const { data, isLoading } = useQuery({
    queryKey: ['performance', tf],
    queryFn: () => api.performance(tf),
  });

  if (isLoading) return <p className="empty">Loading…</p>;
  if (!data) return <p className="empty">No data</p>;

  return (
    <div>
      <section className="card">
        <h2>Performance</h2>
        <Chips value={tf} options={['Daily', 'Weekly', 'Monthly'] as const} onChange={setTf} />
        <div className="meta-grid">
          <div>
            <span>Win rate</span>
            {data.totals.winRatePct}%
          </div>
          <div>
            <span>Net P&amp;L</span>
            {money(data.totals.pnlUsd)}
          </div>
          <div>
            <span>Avg R</span>
            {data.totals.avgR ?? '—'}
          </div>
          <div>
            <span>Closed / open</span>
            {data.totals.closed} / {data.totals.open}
          </div>
        </div>
        <Sparkline points={data.equity} />
      </section>

      <section className="card">
        <h3>{tf} periods</h3>
        {data.periods.length === 0 ? (
          <p className="muted">No closed {tf} trades yet.</p>
        ) : (
          data.periods.map((p) => (
            <div className="stack-row list-row" key={p.periodKey}>
              <span>{periodLabel(p.periodKey, tf)}</span>
              <span>
                <span className={p.pnlUsd >= 0 ? 'up-text' : 'down-text'}>{money(p.pnlUsd)}</span>
                <span className="muted small">
                  {' '}
                  · {p.trades} trades · {p.winRatePct}% · avgR {p.avgR ?? '—'}
                </span>
              </span>
            </div>
          ))
        )}
      </section>
    </div>
  );
}
