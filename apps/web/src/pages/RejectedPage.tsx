import { Link, useParams } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { api, type RejectDetail } from '../lib/api';

const num = (v: number | null | undefined) => (v == null ? null : v.toFixed(2));

const SEQ_LABEL: Record<string, string> = { '1': 'up', '-1': 'down', '0': 'flat' };

/** Same numbers the TradingView dashboard shows, so a disagreement is traceable. */
function detailLine(detail: RejectDetail): string {
  const parts: string[] = [];
  if (detail.barDate) parts.push(`bar ${detail.barDate}`);
  if (detail.close != null) parts.push(`C ${num(detail.close)}`);
  if (detail.criticalLevel != null) parts.push(`crit ${num(detail.criticalLevel)}`);
  if (detail.seqState != null) parts.push(`seq ${SEQ_LABEL[String(detail.seqState)] ?? detail.seqState}`);
  if (detail.sl != null) parts.push(`SL ${num(detail.sl)}`);
  if (detail.tp != null) parts.push(`TP ${num(detail.tp)}`);
  if (detail.rr != null) parts.push(`RR ${num(detail.rr)} / min ${num(detail.minRr)}`);
  return parts.join(' · ');
}

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
        <Link className="link-row" to="/results/manual">
          Back to manual scan
        </Link>
      </section>

      <section className="card">
        <h3>Reason breakdown</h3>
        <p className="muted small">
          Includes skipped reasons (e.g. NOT_NEW). Rejected rows below are rejects only.
        </p>
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
        <p className="muted small">
          Numbers are from the evaluated bar snapshot. TradingView draws the in-progress bar, so a
          close sitting next to the critical level can be VALID there and rejected here.
        </p>
        {data.rows.map((row) => {
          const detail = row.detail ? detailLine(row.detail) : '';
          return (
            <div className="list-row" key={row._id}>
              <div className="stack-row">
                <span>{row.symbol}</span>
                <span className="muted small">{row.reason}</span>
              </div>
              {detail ? <span className="muted small block">{detail}</span> : null}
            </div>
          );
        })}
      </section>
    </div>
  );
}
