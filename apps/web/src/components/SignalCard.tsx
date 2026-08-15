import { useMutation, useQueryClient } from '@tanstack/react-query';
import type { KeyboardEvent, MouseEvent } from 'react';
import { useNavigate } from 'react-router-dom';
import { api, type Bucket, type CardFundamentals, type Interest, type ResultRow } from '../lib/api';
import { barsLabel, money, num, pct, signedMoney } from '../lib/format';

/**
 * A trade this app takes ends on the sell-to-close break and on nothing else. The other codes are
 * only ever read, never written: imported journal trades and records left by older builds.
 */
const EXIT_LABEL: Record<string, string> = {
  sell_to_close: 'SELL TO CLOSE',
  TP: 'TP hit',
  SL: 'SL hit',
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
function signalBar(row: ResultRow): string {
  return row.validSinceAsOf ?? row.openedAsOf ?? row.openedPeriodKey;
}

/** An imported journal trade can have no age at all, and "valid —" would read like a bug. */
function validFoot(row: ResultRow): string {
  return [
    row.barsSinceValid != null ? `valid ${barsLabel(row.barsSinceValid)}` : null,
    `since ${signalBar(row)}`,
    `now ${money(row.lastPrice)}`,
    `${num(row.pnlR)}R`,
  ]
    .filter(Boolean)
    .join(' · ');
}

/** FMP mixes decimals (0.18) and whole percents (18); normalize to percent points. */
function asPctPoints(n: number | null | undefined): number | null {
  if (n == null || !Number.isFinite(n)) return null;
  return Math.abs(n) <= 1.5 ? n * 100 : n;
}

function ratio(n: number | null | undefined, digits = 1): string {
  if (n == null || !Number.isFinite(n)) return '—';
  return n.toFixed(digits);
}

/** Current premium vs fair value — same number Settings uses to filter lists. */
function premiumVsFair(row: ResultRow, fund: CardFundamentals | undefined): number | null {
  if (fund?.premiumPct != null && Number.isFinite(fund.premiumPct)) return fund.premiumPct;
  if (!fund?.fairValue || fund.fairValue <= 0) return null;
  const price = row.lastPrice ?? row.entry;
  if (price == null || !Number.isFinite(price)) return null;
  return ((price - fund.fairValue) / fund.fairValue) * 100;
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

export function SignalCard({
  row,
  bucket,
  fundamentals,
}: {
  row: ResultRow;
  bucket: Bucket;
  fundamentals?: CardFundamentals | null;
}) {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const markStatus: Interest | null = row.interest ?? null;

  const markInterest = useMutation({
    mutationFn: (next: Interest | null) => api.setInterest(row.id, next),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ['results'] });
      void queryClient.invalidateQueries({ queryKey: ['history-trades'] });
      void queryClient.invalidateQueries({ queryKey: ['tracked-signal'] });
      void queryClient.invalidateQueries({ queryKey: ['tracked-signal-by-id'] });
    },
  });

  const marking = markInterest.isPending;

  // A closed trade opens as a snapshot of itself — the chart cut at the bar it broke on, with the
  // entry and the exit marked. Anything still running opens on the live chart.
  const openChart = () =>
    navigate(
      `/chart/${encodeURIComponent(row.yahooTicker)}${bucket === 'closed' ? `?trade=${row.id}` : ''}`,
      { state: { row } },
    );

  const onCardClick = (e: MouseEvent | KeyboardEvent) => {
    if ((e.target as HTMLElement).closest('button')) return;
    openChart();
  };

  const showPnl = bucket !== 'new' && row.pnlUsd != null;
  const positive = (row.pnlUsd ?? 0) >= 0;

  const premiumPct = premiumVsFair(row, fundamentals ?? undefined);
  const valuation = valuationLabel(premiumPct);
  const debtPct = asPctPoints(fundamentals?.ltDebtToCapitalTTM ?? null);

  return (
    <article
      className="card signal-card compact clickable"
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
        {showPnl ? (
          <span className={`badge ${positive ? 'up' : 'down'}`}>
            {signedMoney(row.pnlUsd)}
            {row.pnlPct != null ? ` (${pct(row.pnlPct)})` : ''}
          </span>
        ) : null}
      </div>

      <div className="signal-card-badges">
        {row.epsPositiveAtEntry === false ? (
          <span className="badge down" title="FMP EPS was ≤ 0 on the last report before entry">
            EPS≤0 AT ENTRY
          </span>
        ) : null}
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
        {row.provisionalClose ? (
          <span
            className="badge warn-badge"
            title="Break on the bar still running — goes to History if it holds to the close"
          >
            CLOSING
          </span>
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

      <div className="signal-card-fundamentals">
        <span className={valuation.className}>{valuation.text}</span>
        <span className="sep">·</span>
        <span>
          <span className="lbl">Growth</span> {pct(fundamentals?.growthRatePct)}
        </span>
        <span className="sep">·</span>
        <span>
          <span className="lbl">P/E B</span> {ratio(fundamentals?.blendedPe)}
        </span>
        <span className="sep">·</span>
        <span>
          <span className="lbl">LT D/C</span>{' '}
          {debtPct == null ? '—' : `${debtPct.toFixed(0)}%`}
        </span>
      </div>

      <p className="muted small signal-card-foot">
        {bucket === 'closed'
          ? `${row.openedAsOf ?? '—'} → ${row.exitDate ?? '—'} · exit ${money(row.exitPrice)} · ${num(row.pnlR)}R`
          : bucket === 'valid'
            ? validFoot(row)
            : `signal bar ${signalBar(row)}`}
      </p>

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
