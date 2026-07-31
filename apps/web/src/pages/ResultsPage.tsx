import { useState } from 'react';
import { Link, useNavigate, useParams } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { api, type BuySignal, type SellSignal, type Signal } from '../lib/api';
import { formatDataAge } from '../lib/freshness';
import { Switch } from '../components/Chips';

function money(n: number) {
  return n.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

function BuyCard({
  signal,
  runId,
  riskUsd,
  tf,
  periodKey,
}: {
  signal: BuySignal;
  runId: string;
  riskUsd: number;
  tf: string;
  periodKey?: string;
}) {
  const navigate = useNavigate();
  const openChart = () =>
    navigate(`/chart/${encodeURIComponent(signal.yahooTicker)}`, {
      state: { signal, runId, riskUsd, tf, periodKey },
    });

  return (
    <article
      className="card signal-card compact"
      role="button"
      tabIndex={0}
      onClick={openChart}
      onKeyDown={(e) => {
        if (e.key === 'Enter') openChart();
      }}
    >
      <div className="signal-card-line1">
        <div className="signal-card-title">
          <strong>{signal.symbol}</strong>
          <span className="muted ellipsis">({signal.companyName})</span>
          {signal.interestMark === 'interested' ? (
            <span className="badge up">INTERESTED</span>
          ) : null}
        </div>
      </div>

      <div className="signal-card-metrics">
        <span>
          <span className="lbl">E</span> {money(signal.entry)}
        </span>
        <span className="sep">·</span>
        <span>
          <span className="lbl">TP</span> {money(signal.tp)}
        </span>
        <span className="sep">·</span>
        <span>
          <span className="lbl">SL</span> {money(signal.sl)}
        </span>
        <span className="sep">·</span>
        <span>
          <span className="lbl">Sh</span> {signal.shares}
        </span>
        <span className="sep">·</span>
        <span>
          <span className="lbl">$</span> {money(signal.positionValue)}
        </span>
        <span className="sep">·</span>
        <span>
          <span className="lbl">RR</span> {signal.rr ?? 'n/a'}
        </span>
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
  const [onlyNew, setOnlyNew] = useState(false);
  const [onlyStrong, setOnlyStrong] = useState(false);

  const { data, isLoading, error } = useQuery({
    queryKey: ['signals', runId, onlyNew, onlyStrong],
    queryFn: () => api.signals(runId, { onlyNew, onlyStrong }),
  });

  if (isLoading) return <p className="empty">Loading results…</p>;
  if (error) {
    const msg = (error as Error).message;
    const gone = /404|not found/i.test(msg);
    if (gone) {
      try {
        localStorage.removeItem('vova.activeRunId');
      } catch {
        /* ignore */
      }
    }
    return (
      <div className="card">
        <p className="error">{gone ? 'This scan was deleted or never existed (history was reset).' : msg}</p>
        <p className="muted">Start a new scan from the Scan tab, or open a run from History.</p>
        <div className="card-actions">
          <Link className="btn-sm" to="/">
            Scan
          </Link>
          <Link className="btn-sm ghost" to="/history">
            History
          </Link>
        </div>
      </div>
    );
  }
  if (!data) return <p className="empty">No data</p>;

  const { run, rows, count } = data;
  const dataAge = formatDataAge(run.barsOldestAt);

  return (
    <div>
      <section className="card">
        <div className="stack-row">
          <h2 style={{ margin: 0 }}>Results</h2>
          <span className="badge">{count}</span>
        </div>
        <p className="muted">
          {run.params.source} · {run.params.direction.toUpperCase()} · {run.params.tf}
          {run.asOf ? ` · bar ${run.asOf}` : ''}
        </p>
        {dataAge ? (
          <p className="muted small">
            Bars pulled {dataAge}. TradingView shows the live in-progress bar, this run scored the
            snapshot above.
          </p>
        ) : null}
        <Switch label="New only" checked={onlyNew} onChange={setOnlyNew} />
        {run.params.direction === 'buy' ? (
          <Switch label="Strong only" checked={onlyStrong} onChange={setOnlyStrong} />
        ) : null}
        <Link className="link-row" to={`/runs/${runId}/rejected`}>
          Rejected ({run.counters.rejected}) · Skipped ({run.counters.skipped ?? 0})
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
              runId={runId}
              riskUsd={run.params.riskPerTrade ?? 0}
              tf={run.params.tf}
              periodKey={run.periodKey}
            />
          ) : (
            <SellCard key={signal.yahooTicker} signal={signal} />
          ),
        )
      )}
    </div>
  );
}
