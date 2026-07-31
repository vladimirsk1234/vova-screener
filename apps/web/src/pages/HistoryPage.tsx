import { useState } from 'react';
import { Link } from 'react-router-dom';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { api, type Timeframe } from '../lib/api';
import { Chips } from '../components/Chips';

function periodLabel(periodKey: string | undefined, tf: Timeframe, createdAt: string) {
  if (!periodKey) return new Date(createdAt).toLocaleString();
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
    return new Date(Number(y), Number(m) - 1, 1).toLocaleDateString(undefined, {
      year: 'numeric',
      month: 'long',
    });
  }
  return periodKey;
}

export function HistoryPage() {
  const queryClient = useQueryClient();
  const [tf, setTf] = useState<Timeframe>('Daily');
  const { data, isLoading } = useQuery({
    queryKey: ['runs', tf],
    queryFn: () => api.runs({ limit: 60, tf }),
  });

  const reset = useMutation({
    mutationFn: api.resetHistory,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['runs'] });
    },
  });

  const onReset = () => {
    if (
      !window.confirm(
        'Reset all scan history? This deletes all scan runs and signals. Trades are kept.',
      )
    ) {
      return;
    }
    reset.mutate();
  };

  return (
    <div>
      <section className="card">
        <h2>History</h2>
        <Chips value={tf} options={['Daily', 'Weekly', 'Monthly'] as const} onChange={setTf} />
        <button
          type="button"
          className="btn-sm ghost"
          disabled={reset.isPending}
          onClick={onReset}
        >
          {reset.isPending ? 'Resetting…' : 'Reset history'}
        </button>
        {reset.data ? (
          <p className="muted small" style={{ marginBottom: 0 }}>
            Deleted {reset.data.deletedRuns} runs.
          </p>
        ) : null}
      </section>

      {isLoading ? <p className="empty">Loading…</p> : null}
      {!isLoading && !data?.length ? (
        <p className="empty">No {tf} history yet. Start a scan or wait for end-of-period.</p>
      ) : null}

      {data?.map((run) => (
        <Link key={run._id} to={`/runs/${run._id}`} className="card block-link">
          <div className="stack-row">
            <strong>
              {periodLabel(run.periodKey, (run.periodTf ?? run.params.tf) as Timeframe, run.createdAt)}
              {' · '}
              {run.params.source} · {run.params.direction.toUpperCase()}
            </strong>
            <span className={`badge ${run.status === 'completed' ? 'up' : ''}`}>{run.status}</span>
          </div>
          <p className="muted small">
            {run.trigger === 'scheduled' ? 'Scheduled' : 'Manual'}
            {run.asOf ? ` · as of ${run.asOf}` : ''}
            {run.createdAt ? ` · ${new Date(run.createdAt).toLocaleString()}` : ''}
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
