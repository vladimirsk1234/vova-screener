import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { api, type Timeframe } from '../lib/api';
import { Chips } from '../components/Chips';

function money(n: number | null | undefined) {
  if (n == null) return '—';
  return n.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

function num(n: number | null | undefined) {
  if (n == null) return '—';
  return String(n);
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

function holdLabel(tf: Timeframe) {
  if (tf === 'Daily') return 'Avg days in trade';
  if (tf === 'Weekly') return 'Avg weeks in trade';
  return 'Avg months in trade';
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
  const [downloading, setDownloading] = useState(false);
  const { data, isLoading } = useQuery({
    queryKey: ['performance', tf],
    queryFn: () => api.performance(tf),
  });

  const onDownload = async () => {
    setDownloading(true);
    try {
      await api.downloadPerformanceReport(tf);
    } catch (e) {
      window.alert((e as Error).message);
    } finally {
      setDownloading(false);
    }
  };

  if (isLoading) return <p className="empty">Loading…</p>;
  if (!data) return <p className="empty">No data</p>;

  return (
    <div>
      <section className="card">
        <div className="stack-row">
          <h2 style={{ margin: 0 }}>Performance</h2>
          <button type="button" className="btn-sm" disabled={downloading} onClick={onDownload}>
            {downloading ? 'Downloading…' : 'Download report'}
          </button>
        </div>
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
            <span>Avg RR entry</span>
            {num(data.totals.avgRrEntry)}
          </div>
          <div>
            <span>Avg RR exit</span>
            {num(data.totals.avgRrExit)}
          </div>
          <div>
            <span>{holdLabel(tf)}</span>
            {num(data.totals.avgHold)}
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
                  · {p.trades} trades · {p.winRatePct}% · RR in {num(p.avgRrEntry)} / out{' '}
                  {num(p.avgRrExit)} · hold {num(p.avgHold)} {data.holdUnit}
                </span>
              </span>
            </div>
          ))
        )}
      </section>
    </div>
  );
}
