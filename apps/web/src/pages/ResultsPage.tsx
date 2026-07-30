import { useState } from 'react';
import { Link, useNavigate, useParams } from 'react-router-dom';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { api, type BuySignal, type SellSignal, type Signal } from '../lib/api';
import { Switch } from '../components/Chips';

function money(n: number) {
  return n.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

function BuyCard({
  signal,
  isNewSinceLast,
  onAdd,
  adding,
}: {
  signal: BuySignal;
  isNewSinceLast: boolean;
  onAdd: () => void;
  adding: boolean;
}) {
  const navigate = useNavigate();
  return (
    <article className="card signal-card">
      <div
        role="button"
        tabIndex={0}
        onClick={() => navigate(`/chart/${encodeURIComponent(signal.yahooTicker)}`)}
        onKeyDown={(e) => {
          if (e.key === 'Enter') navigate(`/chart/${encodeURIComponent(signal.yahooTicker)}`);
        }}
      >
        <div className="stack-row">
          <strong>{signal.symbol}</strong>
          <span className="badge up">RR {signal.rr ?? 'n/a'}</span>
        </div>
        <p className="muted ellipsis">{signal.companyName}</p>

        <div className="meta-grid">
          <div>
            <span>Entry</span>
            {money(signal.entry)}
          </div>
          <div>
            <span>TP</span>
            {money(signal.tp)}
          </div>
          <div>
            <span>SL</span>
            {money(signal.sl)}
          </div>
          <div>
            <span>Shares</span>
            {signal.shares}
          </div>
        </div>

        <div className="chip-row" style={{ marginTop: 10 }}>
          {signal.isStrong ? <span className="badge">STRONG</span> : null}
          {signal.isNew ? <span className="badge up">NEW</span> : null}
          {isNewSinceLast ? <span className="badge">NEW SINCE LAST</span> : null}
        </div>
      </div>

      <div className="card-actions">
        <button type="button" className="btn-sm" disabled={adding} onClick={onAdd}>
          {adding ? 'Adding…' : 'Add to journal'}
        </button>
        <a className="btn-sm ghost" href={signal.tvUrl} target="_blank" rel="noreferrer">
          TradingView
        </a>
      </div>
    </article>
  );
}

function SellCard({ signal }: { signal: SellSignal }) {
  const navigate = useNavigate();
  const positive = signal.pnlUsd >= 0;
  return (
    <article
      className="card signal-card"
      role="button"
      tabIndex={0}
      onClick={() => navigate(`/chart/${encodeURIComponent(signal.yahooTicker)}`)}
      onKeyDown={(e) => {
        if (e.key === 'Enter') navigate(`/chart/${encodeURIComponent(signal.yahooTicker)}`);
      }}
    >
      <div className="stack-row">
        <strong>{signal.symbol}</strong>
        <span className={`badge ${positive ? 'up' : 'down'}`}>
          {positive ? '+' : ''}
          {money(signal.pnlUsd)} ({signal.pnlPct}%)
        </span>
      </div>
      <p className="muted ellipsis">{signal.companyName}</p>
      <div className="meta-grid">
        <div>
          <span>Entry</span>
          {money(signal.entry)}
        </div>
        <div>
          <span>Exit</span>
          {money(signal.exit)}
        </div>
        <div>
          <span>RR entry</span>
          {signal.rrAtEntry ?? 'n/a'}
        </div>
        <div>
          <span>RR close</span>
          {signal.rrAtClose ?? 'n/a'}
        </div>
      </div>
    </article>
  );
}

export function ResultsPage() {
  const { runId = '' } = useParams();
  const queryClient = useQueryClient();
  const [onlyNew, setOnlyNew] = useState(false);
  const [onlyStrong, setOnlyStrong] = useState(false);

  const { data, isLoading, error } = useQuery({
    queryKey: ['signals', runId, onlyNew, onlyStrong],
    queryFn: () => api.signals(runId, { onlyNew, onlyStrong }),
  });

  const addTrade = useMutation({
    mutationFn: (signal: BuySignal) =>
      api.createTrade({
        symbol: signal.symbol,
        yahooTicker: signal.yahooTicker,
        companyName: signal.companyName,
        tf: data?.run.params.tf ?? 'Daily',
        entry: signal.entry,
        tp: signal.tp,
        sl: signal.sl,
        rrAtEntry: signal.rr,
        shares: signal.shares,
        riskUsd: data?.run.params.riskPerTrade ?? 0,
        asOf: signal.asOf,
        runId,
      }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['trades'] }),
  });

  if (isLoading) return <p className="empty">Loading results…</p>;
  if (error) return <p className="error">{(error as Error).message}</p>;
  if (!data) return <p className="empty">No data</p>;

  const { run, rows, count, newSymbols } = data;
  const newSet = new Set(newSymbols);

  return (
    <div>
      <section className="card">
        <div className="stack-row">
          <h2 style={{ margin: 0 }}>Results</h2>
          <span className="badge">{count}</span>
        </div>
        <p className="muted">
          {run.params.source} · {run.params.direction.toUpperCase()} · {run.params.tf}
          {run.asOf ? ` · as of ${run.asOf}` : ''}
        </p>
        <Switch label="New only" checked={onlyNew} onChange={setOnlyNew} />
        {run.params.direction === 'buy' ? (
          <Switch label="Strong only" checked={onlyStrong} onChange={setOnlyStrong} />
        ) : null}
        <Link className="link-row" to={`/runs/${runId}/rejected`}>
          Rejected / skipped ({run.counters.rejected})
        </Link>
      </section>

      {run.summary ? (
        <section className="card accent-border">
          <h3>Sell summary</h3>
          <div className="meta-grid">
            <div>
              <span>Win rate</span>
              {run.summary.winRatePct}%
            </div>
            <div>
              <span>Net P&amp;L</span>
              {money(run.summary.pnlUsd)}
            </div>
            <div>
              <span>Invested</span>
              {money(run.summary.invested)}
            </div>
            <div>
              <span>Avg RR entry</span>
              {run.summary.avgEntryRr}
            </div>
          </div>
        </section>
      ) : null}

      {rows.length === 0 ? (
        <p className="empty">No signals matched. Check rejected reasons for why.</p>
      ) : (
        rows.map((signal: Signal) =>
          signal.kind === 'buy' ? (
            <BuyCard
              key={signal.yahooTicker}
              signal={signal}
              isNewSinceLast={newSet.has(signal.symbol)}
              adding={addTrade.isPending}
              onAdd={() => addTrade.mutate(signal)}
            />
          ) : (
            <SellCard key={signal.yahooTicker} signal={signal} />
          ),
        )
      )}
    </div>
  );
}
