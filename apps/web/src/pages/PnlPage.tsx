import { useQuery } from '@tanstack/react-query';
import { api } from '../lib/api';

function money(n: number | null | undefined) {
  if (n == null) return '—';
  return n.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
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
  const { data, isLoading } = useQuery({ queryKey: ['monthly'], queryFn: api.monthly });

  if (isLoading) return <p className="empty">Loading…</p>;
  if (!data) return <p className="empty">No data</p>;

  return (
    <div>
      <section className="card">
        <h2>Performance</h2>
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
        <h3>Monthly</h3>
        {data.months.length === 0 ? (
          <p className="muted">No closed trades yet.</p>
        ) : (
          data.months.map((m) => (
            <div className="stack-row list-row" key={m.month}>
              <span>{m.month}</span>
              <span>
                <span className={m.pnlUsd >= 0 ? 'up-text' : 'down-text'}>{money(m.pnlUsd)}</span>
                <span className="muted small">
                  {' '}
                  · {m.trades} trades · {m.winRatePct}% · avgR {m.avgR ?? '—'}
                </span>
              </span>
            </div>
          ))
        )}
      </section>
    </div>
  );
}
