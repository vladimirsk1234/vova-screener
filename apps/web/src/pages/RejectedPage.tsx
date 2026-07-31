import { Link, useParams } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { api } from '../lib/api';

export function RejectedPage() {
  const { runId = '' } = useParams();
  const { data, isLoading } = useQuery({
    queryKey: ['rejections', runId],
    queryFn: () => api.rejections(runId),
  });

  if (isLoading) return <p className="empty">Loading…</p>;
  if (!data) return <p className="empty">No data</p>;

  const reasons = Object.entries(data.reasonCounts).sort((a, b) => b[1] - a[1]);

  return (
    <div>
      <section className="card">
        <div className="stack-row">
          <h2 style={{ margin: 0 }}>Rejected</h2>
          <span className="badge">{data.total}</span>
        </div>
        <Link className="link-row" to={`/runs/${runId}`}>
          Back to results
        </Link>
      </section>

      <section className="card">
        <h3>Reason breakdown</h3>
        {reasons.length === 0 ? (
          <p className="muted">Nothing rejected.</p>
        ) : (
          reasons.map(([reason, count]) => (
            <div className="stack-row list-row" key={reason}>
              <span className="warn">{reason}</span>
              <strong>{count}</strong>
            </div>
          ))
        )}
      </section>

      <section className="card">
        <h3>Symbols</h3>
        {data.rows.map((row) => (
          <div className="stack-row list-row" key={row._id}>
            <span>{row.symbol}</span>
            <span className="muted small">{row.reason}</span>
          </div>
        ))}
      </section>
    </div>
  );
}
