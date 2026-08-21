import type { KeyboardEvent, MouseEvent } from 'react';
import { useNavigate } from 'react-router-dom';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { isLowLtDebt, normalizePctPoints, VALUE_STAR_TOTAL } from '@vova/engine';
import type { Interest, SeqStructStatus, ValueScreenerRow } from '../lib/api';
import { api } from '../lib/api';

function starsLabel(stars: number): string {
  const n = Math.max(0, Math.min(VALUE_STAR_TOTAL, stars));
  return `${'★'.repeat(n)}${'☆'.repeat(VALUE_STAR_TOTAL - n)} ${n}/${VALUE_STAR_TOTAL}`;
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
  const queryClient = useQueryClient();
  const markStatus: Interest | null = row.interest ?? null;
  const eps = valuationLabel(row.epsPremiumPct);
  const fcf = valuationLabel(row.fcfPremiumPct);
  const dcf = valuationLabel(row.dcfPremiumPct);
  const debtPct = normalizePctPoints(row.ltDebtToCapitalTTM);
  const debtLow = isLowLtDebt(row.ltDebtToCapitalTTM);

  const markInterest = useMutation({
    mutationFn: (next: Interest | null) => api.setTickerInterest(row.yahooTicker, next),
    onSuccess: (saved) => {
      queryClient.setQueryData(['ticker-interest', row.yahooTicker], saved);
      void queryClient.invalidateQueries({ queryKey: ['value-screener'] });
    },
  });
  const marking = markInterest.isPending;

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
        <span className="value-stars" title={`${row.stars} of ${VALUE_STAR_TOTAL} metrics (EPS/FCF/DCF undervalued, LT D/C < 50%)`}>
          {starsLabel(row.stars)}
        </span>
      </div>

      {row.interest === 'not_interested' ? (
        <div className="signal-card-badges">
          <span className="badge down">NO INTEREST</span>
        </div>
      ) : null}

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
        <span className="sep">·</span>
        <span>
          <span className="lbl">LT D/C</span>{' '}
          <span className={debtLow ? 'fund-pos' : ''}>
            {debtPct == null ? '—' : `${debtPct.toFixed(0)}%`}
          </span>
        </span>
      </div>

      <div className="value-card-ta" aria-label="TA sequence and structure">
        <span>{taLine('D', row.ta.daily)}</span>
        <span>{taLine('W', row.ta.weekly)}</span>
        <span>{taLine('M', row.ta.monthly)}</span>
      </div>

      <div className="card-actions">
        <button
          type="button"
          className={`btn-sm${markStatus === 'interested' ? ' selected' : ' ghost'}`}
          disabled={marking}
          onClick={(e) => {
            e.stopPropagation();
            markInterest.mutate(markStatus === 'interested' ? null : 'interested');
          }}
        >
          {marking && markInterest.variables === 'interested' ? 'Saving…' : 'Interested'}
        </button>
        <button
          type="button"
          className={`btn-sm${markStatus === 'not_interested' ? ' danger selected' : ' ghost'}`}
          disabled={marking}
          onClick={(e) => {
            e.stopPropagation();
            markInterest.mutate(markStatus === 'not_interested' ? null : 'not_interested');
          }}
        >
          {marking && markInterest.variables === 'not_interested' ? 'Saving…' : 'Not Interested'}
        </button>
      </div>
      {markInterest.error ? (
        <p className="error small signal-card-foot">{(markInterest.error as Error).message}</p>
      ) : null}
    </article>
  );
}
