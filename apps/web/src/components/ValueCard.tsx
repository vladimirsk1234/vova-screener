import type { KeyboardEvent, MouseEvent } from 'react';
import { useNavigate } from 'react-router-dom';
import type { SeqStructStatus, ValueScreenerRow } from '../lib/api';

const STAR_TOTAL = 3;

function starsLabel(stars: number): string {
  const n = Math.max(0, Math.min(STAR_TOTAL, stars));
  return `${'★'.repeat(n)}${'☆'.repeat(STAR_TOTAL - n)} ${n}/${STAR_TOTAL}`;
}

function valuationLabel(premiumPct: number | null): { text: string; className: string } {
  if (premiumPct == null || !Number.isFinite(premiumPct)) {
    return { text: '—', className: '' };
  }
  const abs = Math.abs(premiumPct).toFixed(0);
  if (premiumPct < 0) {
    return {
      text: `${abs}% undervalued`,
      className: premiumPct < -10 ? 'fund-pos' : '',
    };
  }
  if (premiumPct > 0) {
    return {
      text: `${abs}% overvalued`,
      className: premiumPct > 10 ? 'fund-neg' : '',
    };
  }
  return { text: 'fair', className: '' };
}

function taLine(label: string, snap: SeqStructStatus | null | undefined): string {
  if (!snap) return `${label} Seq —  Struct —`;
  return `${label} Seq ${snap.seqEmoji}  Struct ${snap.structEmoji}${snap.structLabel}`;
}

export function ValueCard({ row }: { row: ValueScreenerRow }) {
  const navigate = useNavigate();
  const eps = valuationLabel(row.epsPremiumPct);
  const fcf = valuationLabel(row.fcfPremiumPct);
  const dcf = valuationLabel(row.dcfPremiumPct);

  const openChart = () =>
    navigate(`/chart/${encodeURIComponent(row.yahooTicker)}?view=fundamentals`);

  const onCardClick = (e: MouseEvent | KeyboardEvent) => {
    if ((e.target as HTMLElement).closest('button')) return;
    openChart();
  };

  return (
    <article
      className="card signal-card compact clickable value-card"
      role="button"
      tabIndex={0}
      onClick={onCardClick}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          if ((e.target as HTMLElement).closest('button')) return;
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
        <span className="value-stars" title={`${row.stars} of ${STAR_TOTAL} metrics undervalued`}>
          {starsLabel(row.stars)}
        </span>
      </div>

      <div className="signal-card-fundamentals value-card-metrics">
        <span>
          <span className="lbl">EPS</span>{' '}
          <span className={eps.className}>{eps.text}</span>
        </span>
        <span className="sep">·</span>
        <span>
          <span className="lbl">FCF</span>{' '}
          <span className={fcf.className}>{fcf.text}</span>
        </span>
        <span className="sep">·</span>
        <span>
          <span className="lbl">DCF</span>{' '}
          <span className={dcf.className}>{dcf.text}</span>
        </span>
      </div>

      <div className="value-card-ta" aria-label="TA sequence and structure">
        <span>{taLine('D', row.ta.daily)}</span>
        <span>{taLine('W', row.ta.weekly)}</span>
        <span>{taLine('M', row.ta.monthly)}</span>
      </div>
    </article>
  );
}
