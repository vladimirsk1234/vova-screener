import { useMemo, useState, type KeyboardEvent, type MouseEvent } from 'react';
import { useNavigate } from 'react-router-dom';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { api, type BuySignal, type Timeframe, type Trade, type TradeStatus } from '../lib/api';
import { Chips } from '../components/Chips';
import { investedFromShares } from '../lib/positionSize';

function money(n: number | null | undefined) {
  if (n == null) return '—';
  return n.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

function periodLabel(periodKey: string | undefined, tf: Timeframe, fallback: string) {
  if (!periodKey) return fallback;
  if (tf === 'Daily') {
    return new Date(`${periodKey}T12:00:00`).toLocaleDateString(undefined, {
      weekday: 'short',
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

type PeriodSummary = {
  periodKey: string;
  count: number;
  investedUsd: number;
  pnlUsd: number;
};

function summarizeByPeriod(trades: Trade[], tf: Timeframe): PeriodSummary[] {
  const map = new Map<string, PeriodSummary>();
  for (const t of trades) {
    const key =
      t.periodKey ||
      (t.status === 'closed' && t.exitDate
        ? tf === 'Monthly'
          ? t.exitDate.slice(0, 7)
          : t.exitDate.slice(0, 10)
        : 'unknown');
    const row = map.get(key) ?? { periodKey: key, count: 0, investedUsd: 0, pnlUsd: 0 };
    row.count += 1;
    row.investedUsd += t.investedUsd ?? (t.entry ?? 0) * (t.shares || 0);
    if (t.status === 'open') row.pnlUsd += t.unrealizedUsd ?? 0;
    else if (t.status === 'closed') row.pnlUsd += t.pnlUsd ?? 0;
    map.set(key, row);
  }
  return [...map.values()].sort((a, b) => b.periodKey.localeCompare(a.periodKey));
}

const TAB_OPTIONS = ['interested', 'open', 'closed', 'not_interested', 'dismissed'] as const;
type Tab = (typeof TAB_OPTIONS)[number];

function tabLabel(v: Tab) {
  if (v === 'not_interested') return 'NO INTEREST';
  return v.toUpperCase();
}

function tradeToSignal(trade: Trade): BuySignal {
  const shares = trade.shares || 0;
  return {
    kind: 'buy',
    symbol: trade.symbol,
    tvSymbol: trade.symbol,
    yahooTicker: trade.yahooTicker,
    companyName: trade.companyName ?? trade.yahooTicker,
    tvUrl: '',
    entry: trade.entry,
    tp: trade.tp ?? trade.entry,
    sl: trade.sl ?? trade.entry,
    rr: trade.rrAtEntry ?? null,
    shares,
    positionValue: trade.investedUsd ?? investedFromShares(trade.entry, shares),
    isNew: false,
    isStrong: false,
    atr: 0,
    asOf: trade.asOf ?? '',
  };
}

function RiskInput({
  tradeId,
  value,
  disabled,
  onCommit,
}: {
  tradeId: string;
  value: number;
  disabled?: boolean;
  onCommit: (id: string, riskUsd: number) => void;
}) {
  const [draft, setDraft] = useState(String(value));
  const [prev, setPrev] = useState(value);
  if (value !== prev) {
    setPrev(value);
    setDraft(String(value));
  }

  const commit = () => {
    const next = Number(draft);
    if (!Number.isFinite(next) || next <= 0) {
      setDraft(String(value));
      return;
    }
    if (next === value) return;
    onCommit(tradeId, next);
  };

  const onKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      (e.target as HTMLInputElement).blur();
    }
  };

  return (
    <input
      className="trade-risk-input"
      type="number"
      inputMode="decimal"
      min={1}
      step="1"
      disabled={disabled}
      value={draft}
      onClick={(e) => e.stopPropagation()}
      onChange={(e) => setDraft(e.target.value)}
      onBlur={commit}
      onKeyDown={onKeyDown}
      aria-label="Risk dollars"
    />
  );
}

export function TradesPage() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [tf, setTf] = useState<Timeframe>('Daily');
  const [tab, setTab] = useState<Tab>('interested');
  const trades = useQuery({
    queryKey: ['trades', tab, tf],
    queryFn: () => api.trades({ status: tab as TradeStatus, tf }),
  });

  const invalidate = () => {
    queryClient.invalidateQueries({ queryKey: ['trades'] });
    queryClient.invalidateQueries({ queryKey: ['performance'] });
    queryClient.invalidateQueries({ queryKey: ['monthly'] });
  };

  const closeTrade = useMutation({
    mutationFn: ({ id, exitPrice }: { id: string; exitPrice: number }) =>
      api.closeTrade(id, { exitPrice, exitReason: 'manual' }),
    onSuccess: invalidate,
  });
  const dismissTrade = useMutation({
    mutationFn: api.dismissTrade,
    onSuccess: invalidate,
  });
  const removeTrade = useMutation({ mutationFn: api.deleteTrade, onSuccess: invalidate });
  const refresh = useMutation({
    mutationFn: () => api.refreshTrades(),
    onSuccess: invalidate,
  });
  const updateRisk = useMutation({
    mutationFn: ({ id, riskUsd }: { id: string; riskUsd: number }) =>
      api.updateTradeRisk(id, riskUsd),
    onSuccess: invalidate,
  });

  const rows = trades.data ?? [];
  const periods = useMemo(() => summarizeByPeriod(rows, tf), [rows, tf]);
  const showPnlSummary = tab === 'open' || tab === 'closed';

  const totals = useMemo(() => {
    let investedUsd = 0;
    let pnlUsd = 0;
    let pnlKnown = 0;
    for (const t of rows) {
      investedUsd += t.investedUsd ?? (t.entry ?? 0) * (t.shares || 0);
      if (tab === 'open') {
        if (t.unrealizedUsd != null) {
          pnlUsd += t.unrealizedUsd;
          pnlKnown += 1;
        }
      } else if (tab === 'closed') {
        pnlUsd += t.pnlUsd ?? 0;
        pnlKnown += 1;
      }
    }
    return {
      count: rows.length,
      investedUsd,
      pnlUsd,
      pnlKnown,
    };
  }, [rows, tab]);

  const onClose = (trade: Trade) => {
    const suggested = trade.currentPrice ?? trade.entry;
    const input = window.prompt(`Exit price for ${trade.symbol}`, String(suggested));
    if (!input) return;
    const exitPrice = Number(input);
    if (!Number.isFinite(exitPrice)) return;
    closeTrade.mutate({ id: trade._id, exitPrice });
  };

  const openChart = (trade: Trade) => {
    const isInterest = trade.status === 'interested' || trade.status === 'not_interested';
    navigate(`/chart/${encodeURIComponent(trade.yahooTicker)}`, {
      state: {
        signal: tradeToSignal(trade),
        runId: undefined,
        riskUsd: trade.riskUsd || 100,
        tf: trade.tf,
        periodKey: trade.periodKey,
        markStatus: isInterest ? trade.status : null,
      },
    });
  };

  return (
    <div>
      <section className="card">
        <h2>Trade journal</h2>
        <Chips value={tf} options={['Daily', 'Weekly', 'Monthly'] as const} onChange={setTf} />
        <Chips value={tab} options={TAB_OPTIONS} onChange={setTab} format={tabLabel} />
        {tab === 'open' ? (
          <button
            type="button"
            className="btn btn-accent"
            disabled={refresh.isPending}
            onClick={() => refresh.mutate()}
          >
            {refresh.isPending ? 'Checking…' : 'Check TP/SL / sell-to-close'}
          </button>
        ) : null}
        {refresh.data && tab === 'open' ? (
          <p className="muted small" style={{ marginBottom: 0 }}>
            Checked {refresh.data.checked}, auto-closed {refresh.data.closed}.
          </p>
        ) : null}
      </section>

      {showPnlSummary && !trades.isLoading ? (
        <section className="card">
          <h3>
            {tf} · {tab} summary
          </h3>
          <div className="meta-grid">
            <div>
              <span>Trades</span>
              {totals.count}
            </div>
            <div>
              <span>Invested</span>
              {money(totals.investedUsd)}
            </div>
            <div>
              <span>{tab === 'open' ? 'Open P&L' : 'Realized P&L'}</span>
              <span className={(totals.pnlUsd ?? 0) >= 0 ? 'up-text' : 'down-text'}>
                {tab === 'open' && totals.pnlKnown === 0 ? '—' : money(totals.pnlUsd)}
              </span>
            </div>
            <div>
              <span>{tab === 'open' ? 'Open P&L %' : 'Return on invested'}</span>
              {totals.investedUsd > 0 && (tab !== 'open' || totals.pnlKnown > 0)
                ? `${((totals.pnlUsd / totals.investedUsd) * 100).toFixed(2)}%`
                : '—'}
            </div>
          </div>

          {periods.length > 0 ? (
            <>
              <h3 style={{ marginTop: 16 }}>By period</h3>
              {periods.map((p) => (
                <div className="stack-row list-row" key={p.periodKey}>
                  <span>{periodLabel(p.periodKey, tf, p.periodKey)}</span>
                  <span className="muted small">
                    {p.count} · invested {money(p.investedUsd)} ·{' '}
                    <span className={p.pnlUsd >= 0 ? 'up-text' : 'down-text'}>{money(p.pnlUsd)}</span>
                  </span>
                </div>
              ))}
            </>
          ) : null}
        </section>
      ) : null}

      {trades.isLoading ? <p className="empty">Loading…</p> : null}
      {trades.data?.length === 0 ? (
        <p className="empty">
          No {tabLabel(tab).toLowerCase()} {tf} items.
          {tab === 'interested'
            ? ' Mark symbols Interested from the chart view.'
            : tab === 'not_interested'
              ? ' Mark symbols Not Interested from the chart view.'
              : tab === 'open'
                ? ' Interested symbols become open trades after end-of-period scans.'
                : ''}
        </p>
      ) : null}

      {trades.data?.map((trade) => {
        const invested = trade.investedUsd ?? trade.entry * (trade.shares || 0);
        const isInterest = trade.status === 'interested' || trade.status === 'not_interested';
        const canOpenChart = isInterest || trade.status === 'open';
        const pnl = trade.status === 'open' ? trade.unrealizedUsd : trade.pnlUsd;
        const positive = (pnl ?? 0) >= 0;
        const stopBubble = (e: MouseEvent) => e.stopPropagation();
        return (
          <article
            className={`card signal-card${canOpenChart ? ' clickable' : ''}`}
            key={trade._id}
            role={canOpenChart ? 'button' : undefined}
            tabIndex={canOpenChart ? 0 : undefined}
            onClick={canOpenChart ? () => openChart(trade) : undefined}
            onKeyDown={
              canOpenChart
                ? (e) => {
                    if (e.key === 'Enter') openChart(trade);
                  }
                : undefined
            }
          >
            <div className="stack-row">
              <strong>{trade.symbol}</strong>
              <span
                className={`badge ${
                  trade.status === 'interested'
                    ? 'up'
                    : trade.status === 'not_interested'
                      ? 'down'
                      : positive
                        ? 'up'
                        : 'down'
                }`}
              >
                {isInterest
                  ? trade.status === 'interested'
                    ? 'interested'
                    : 'no interest'
                  : trade.status === 'dismissed'
                    ? 'dismissed'
                    : pnl == null
                      ? '—'
                      : `${positive ? '+' : ''}${money(pnl)}`}
              </span>
            </div>
            <p className="muted small ellipsis">
              {trade.companyName ?? trade.yahooTicker} · {trade.tf}
              {trade.source ? ` · ${trade.source}` : ''}
              {trade.periodKey ? ` · ${trade.periodKey}` : ''}
            </p>

            <div className="meta-grid">
              <div>
                <span>Entry</span>
                {money(trade.entry)}
              </div>
              <div>
                <span>{trade.status === 'open' ? 'Now' : isInterest ? 'RR' : 'Exit'}</span>
                {trade.status === 'open'
                  ? money(trade.currentPrice)
                  : isInterest
                    ? (trade.rrAtEntry ?? '—')
                    : money(trade.exitPrice)}
              </div>
              <div>
                <span>Invested</span>
                {money(invested)}
              </div>
              <div>
                <span>{trade.status === 'open' ? 'Open P&L' : isInterest ? 'Risk $' : 'P&L'}</span>
                {isInterest ? (
                  <RiskInput
                    tradeId={trade._id}
                    value={trade.riskUsd || 0}
                    disabled={updateRisk.isPending}
                    onCommit={(id, riskUsd) => updateRisk.mutate({ id, riskUsd })}
                  />
                ) : pnl == null ? (
                  '—'
                ) : (
                  money(pnl)
                )}
              </div>
              <div>
                <span>TP / SL</span>
                {money(trade.tp)} / {money(trade.sl)}
              </div>
              <div>
                <span>Shares</span>
                {trade.shares}
              </div>
              <div>
                <span>
                  {trade.status === 'open'
                    ? 'Unrealized R'
                    : trade.status === 'closed'
                      ? 'Realized R'
                      : 'Period'}
                </span>
                {trade.status === 'open'
                  ? (trade.unrealizedR ?? '—')
                  : trade.status === 'closed'
                    ? (trade.pnlR ?? '—')
                    : (trade.periodKey ?? '—')}
              </div>
              <div>
                <span>
                  {trade.status === 'open'
                    ? 'Open %'
                    : trade.status === 'closed'
                      ? 'Closed'
                      : isInterest
                        ? 'As of'
                        : 'Dismissed'}
                </span>
                {trade.status === 'open'
                  ? trade.unrealizedPct != null
                    ? `${trade.unrealizedPct}%`
                    : '—'
                  : trade.status === 'closed'
                    ? `${trade.exitDate ?? '—'} (${trade.exitReason ?? '—'})`
                    : isInterest
                      ? (trade.asOf ?? new Date(trade.openedAt).toLocaleDateString())
                      : new Date(trade.openedAt).toLocaleDateString()}
              </div>
            </div>

            <div className="card-actions" onClick={stopBubble}>
              {trade.status === 'open' ? (
                <>
                  <button type="button" className="btn-sm" onClick={() => onClose(trade)}>
                    Close
                  </button>
                  <button
                    type="button"
                    className="btn-sm ghost"
                    onClick={() => dismissTrade.mutate(trade._id)}
                  >
                    Dismiss
                  </button>
                </>
              ) : null}
              <button
                type="button"
                className="btn-sm ghost"
                onClick={() => removeTrade.mutate(trade._id)}
              >
                Delete
              </button>
            </div>
          </article>
        );
      })}
    </div>
  );
}
