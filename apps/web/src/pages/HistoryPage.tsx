import { Link } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { api } from '../lib/api';

export function HistoryPage() {
  const { data, isLoading } = useQuery({ queryKey: ['runs'], queryFn: () => api.runs(30) });

  if (isLoading) return <p className="empty">Loading…</p>;
  if (!data?.length) return <p className="empty">No scan runs yet. Start one from the Scan tab.</p>;

  return (
    <div>
      {data.map((run) => (
        <Link key={run._id} to={`/runs/${run._id}`} className="card block-link">
          <div className="stack-row">
            <strong>
              {run.params.source} · {run.params.direction.toUpperCase()} · {run.params.tf}
            </strong>
            <span className={`badge ${run.status === 'completed' ? 'up' : ''}`}>{run.status}</span>
          </div>
          <p className="muted small">
            {new Date(run.createdAt).toLocaleString()}
            {run.asOf ? ` · as of ${run.asOf}` : ''}
          </p>
          <div className="meta-grid">
            <div>
              <span>Signals</span>
              {run.counters.signals}
            </div>
            <div>
              <span>Scanned</span>
              {run.counters.evaluated}/{run.counters.total}
            </div>
            <div>
              <span>Rejected</span>
              {run.counters.rejected}
            </div>
            <div>
              <span>Took</span>
              {run.timings?.totalMs ? `${Math.round(run.timings.totalMs / 1000)}s` : '—'}
            </div>
          </div>
        </Link>
      ))}
    </div>
  );
}
