import { useNavigate } from 'react-router-dom';
import type { Bucket, ResultRow } from '../lib/api';
import { barsLabel, money, num, pct, signedMoney } from '../lib/format';

const EXIT_LABEL: Record<string, string> = {
  TP: 'TP hit',
  SL: 'SL hit',
  sell_to_close: 'sell to close',
  signal_lost: 'signal gone',
  manual: 'closed by hand',
};

/**
 * One tracked signal. Same card in every bucket — NEW has no P&L yet, VALID carries the
 * unrealized number and CLOSED the realized one.
 *
 * A signal can be older than the record: the scan may meet a symbol that has already been valid
 * for several bars, so VALID reports the age of the signal separately from the price the tracker
 * measures its P&L against.
 */
export function SignalCard({ row, bucket }: { row: ResultRow; bucket: Bucket }) {
  const navigate = useNavigate();
  const openChart = () =>
    navigate(`/chart/${encodeURIComponent(row.yahooTicker)}`, { state: { row } });

  const showPnl = bucket !== 'new' && row.pnlUsd != null;
  const positive = (row.pnlUsd ?? 0) >= 0;

  return (
    <article
      className="card signal-card compact clickable"
      role="button"
      tabIndex={0}
      onClick={openChart}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          openChart();
        }
      }}
    >
      <div className="signal-card-line1">
        <div className="signal-card-title">
          <strong>{row.symbol}</strong>
          <span className="muted ellipsis">{row.companyName}</span>
        </div>
        {showPnl ? (
          <span className={`badge ${positive ? 'up' : 'down'}`}>
            {signedMoney(row.pnlUsd)}
            {row.pnlPct != null ? ` (${pct(row.pnlPct)})` : ''}
          </span>
        ) : null}
      </div>

      <div className="signal-card-badges">
        {row.interest === 'interested' ? <span className="badge up">INTERESTED</span> : null}
        {row.interest === 'not_interested' ? <span className="badge down">NO INTEREST</span> : null}
        {row.isStrong ? <span className="badge">STRONG</span> : null}
        {row.provisional && bucket === 'new' ? (
          <span className="badge warn-badge" title="Seen mid-period, confirmed at the close">
            LIVE
          </span>
        ) : null}
        {bucket === 'closed' && row.exitReason ? (
          <span className="badge">{EXIT_LABEL[row.exitReason] ?? row.exitReason}</span>
        ) : null}
      </div>

      <div className="signal-card-metrics">
        <span>
          <span className="lbl">E</span> {money(row.entry)}
        </span>
        <span className="sep">·</span>
        <span>
          <span className="lbl">TP</span> {money(row.tp)}
        </span>
        <span className="sep">·</span>
        <span>
          <span className="lbl">SL</span> {money(row.sl)}
        </span>
        <span className="sep">·</span>
        <span>
          <span className="lbl">Sh</span> {row.shares}
        </span>
        <span className="sep">·</span>
        <span>
          <span className="lbl">$</span> {money(row.positionValue)}
        </span>
        <span className="sep">·</span>
        <span>
          <span className="lbl">RR</span>{' '}
          {num(bucket === 'closed' ? row.rr : (row.currentRr ?? row.rr))}
        </span>
      </div>

      <p className="muted small signal-card-foot">
        {bucket === 'closed'
          ? `${row.openedAsOf ?? '—'} → ${row.exitDate ?? '—'} · exit ${money(row.exitPrice)} · ${num(row.pnlR)}R`
          : bucket === 'valid'
            ? `valid ${barsLabel(row.barsSinceValid)} · since ${row.validSinceAsOf ?? row.openedAsOf ?? row.openedPeriodKey} · now ${money(row.lastPrice)} · ${num(row.pnlR)}R`
            : `signal bar ${row.validSinceAsOf ?? row.openedAsOf ?? row.openedPeriodKey}`}
      </p>
    </article>
  );
}
