import { useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { api, type Trade } from '../lib/api';
import { Chips } from '../components/Chips';

function money(n: number | null | undefined) {
  if (n == null) return '—';
  return n.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

export function TradesPage() {
  const queryClient = useQueryClient();
  const [tab, setTab] = useState<'open' | 'closed' | 'dismissed'>('open');
  const trades = useQuery({ queryKey: ['trades', tab], queryFn: () => api.trades(tab) });

  const invalidate = () => {
    queryClient.invalidateQueries({ queryKey: ['trades'] });
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
  const refresh = useMutation({ mutationFn: api.refreshTrades, onSuccess: invalidate });

  const onClose = (trade: Trade) => {
    const suggested = trade.currentPrice ?? trade.entry;
    const input = window.prompt(`Exit price for ${trade.symbol}`, String(suggested));
    if (!input) return;
    const exitPrice = Number(input);
    if (!Number.isFinite(exitPrice)) return;
    closeTrade.mutate({ id: trade._id, exitPrice });
  };

  return (
    <div>
      <section className="card">
        <h2>Trade journal</h2>
        <Chips
          value={tab}
          options={['open', 'closed', 'dismissed'] as const}
          onChange={setTab}
          format={(v) => v.toUpperCase()}
        />
        <button
          type="button"
          className="btn btn-accent"
          disabled={refresh.isPending}
          onClick={() => refresh.mutate()}
        >
          {refresh.isPending ? 'Checking…' : 'Check TP/SL / sell-to-close'}
        </button>
        {refresh.data ? (
          <p className="muted small" style={{ marginBottom: 0 }}>
            Checked {refresh.data.checked}, auto-closed {refresh.data.closed}.
          </p>
        ) : null}
      </section>

      {trades.isLoading ? <p className="empty">Loading…</p> : null}
      {trades.data?.length === 0 ? (
        <p className="empty">
          No {tab} trades.
          {tab === 'open' ? ' Auto trades open after end-of-period scans, or add from a result card.' : ''}
        </p>
      ) : null}

      {trades.data?.map((trade) => {
        const pnl = trade.status === 'open' ? trade.unrealizedUsd : trade.pnlUsd;
        const positive = (pnl ?? 0) >= 0;
        return (
          <article className="card signal-card" key={trade._id}>
            <div className="stack-row">
              <strong>{trade.symbol}</strong>
              <span className={`badge ${positive ? 'up' : 'down'}`}>
                {trade.status === 'dismissed'
                  ? 'dismissed'
                  : pnl == null
                    ? '—'
                    : `${positive ? '+' : ''}${money(pnl)}`}
              </span>
            </div>
            <p className="muted small ellipsis">
              {trade.companyName ?? trade.yahooTicker} · {trade.tf}
              {trade.source ? ` · ${trade.source}` : ''}
            </p>

            <div className="meta-grid">
              <div>
                <span>Entry</span>
                {money(trade.entry)}
              </div>
              <div>
                <span>{trade.status === 'open' ? 'Now' : 'Exit'}</span>
                {money(trade.status === 'open' ? trade.currentPrice : trade.exitPrice)}
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
                <span>{trade.status === 'open' ? 'Opened' : trade.status === 'closed' ? 'Closed' : 'Dismissed'}</span>
                {trade.status === 'closed'
                  ? `${trade.exitDate ?? '—'} (${trade.exitReason ?? '—'})`
                  : new Date(trade.openedAt).toLocaleDateString()}
              </div>
            </div>

            <div className="card-actions">
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
