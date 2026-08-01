import { useEffect, useMemo, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { TIMEFRAMES, UNIVERSES, api, type BuySignal, type Timeframe } from '../lib/api';
import { barsLabel, money, num } from '../lib/format';
import { Chips } from '../components/Chips';
import { SegmentedTabs } from '../components/SegmentedTabs';
import { SortChips, type SortDir, type SortOption } from '../components/SortChips';
import { useScanProgress } from '../lib/useScanProgress';

const ACTIVE_RUN_KEY = 'vova.manualRunId';
const TICKERS_KEY = 'vova.manualTickers';
const TERMINAL = ['completed', 'cancelled', 'failed'];

type ManualSort = 'rr' | 'entry' | 'symbol';

const SORTS: SortOption<ManualSort>[] = [
  { value: 'rr', label: 'RR' },
  { value: 'entry', label: 'Entry' },
  { value: 'symbol', label: 'A-Z', from: 'asc' },
];

/**
 * The only screen that starts a scan. Manual runs are ad-hoc checks against a live chart, so
 * they always pull fresh bars and never feed the tracked-signal history.
 */
export function ManualPage() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [tickers, setTickers] = useState(
    () => localStorage.getItem(TICKERS_KEY) ?? 'AAPL, TSLA, NVDA, MSFT, AMD',
  );
  const [tf, setTf] = useState<Timeframe>('Daily');
  const [runId, setRunId] = useState<string | null>(() => localStorage.getItem(ACTIVE_RUN_KEY));
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sort, setSort] = useState<ManualSort>('rr');
  const [dir, setDir] = useState<SortDir>('desc');
  const progress = useScanProgress(runId);

  const settings = useQuery({ queryKey: ['settings'], queryFn: api.settings });

  const run = useQuery({
    queryKey: ['run', runId],
    queryFn: () => api.run(runId as string),
    enabled: Boolean(runId),
    retry: false,
    refetchInterval: (q) => {
      const status = q.state.data?.status;
      return status && TERMINAL.includes(status) ? false : 2_000;
    },
  });

  const done = Boolean(run.data && TERMINAL.includes(run.data.status));
  const running = Boolean(runId && !done);

  const signals = useQuery({
    queryKey: ['manual-signals', runId],
    queryFn: () => api.signals(runId as string),
    enabled: Boolean(runId) && done,
  });

  useEffect(() => {
    if (!runId || !run.isError) return;
    localStorage.removeItem(ACTIVE_RUN_KEY);
    setRunId(null);
  }, [runId, run.isError]);

  useEffect(() => {
    if (!runId || !progress || !TERMINAL.includes(progress.phase)) return;
    void queryClient.invalidateQueries({ queryKey: ['run', runId] });
  }, [progress?.phase, runId, queryClient]);

  const onStart = async () => {
    setStarting(true);
    setError(null);
    try {
      localStorage.setItem(TICKERS_KEY, tickers);
      const res = await api.startScan({
        source: 'MANUAL SCAN',
        manualTickers: tickers,
        tf,
        direction: 'buy',
        minRr: 0,
        noRrReq: true,
        useLastHlSl: true,
        newOnly: false,
        riskPerTrade: settings.data?.maxRiskUsd ?? 100,
        forceRefresh: true,
      });
      localStorage.setItem(ACTIVE_RUN_KEY, res.runId);
      setRunId(res.runId);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setStarting(false);
    }
  };

  const rows = useMemo(() => {
    const order = dir === 'asc' ? 1 : -1;
    // MIN RR is "any", so a signal can have no computable RR. Ranking those lowest keeps them at
    // one end of the list instead of scattered through it, matching how Results sorts them.
    const rank = (s: BuySignal) => (sort === 'rr' ? (s.rr ?? -Infinity) : s.entry);
    return [...(signals.data?.rows ?? [])].sort((a, b) => {
      if (sort === 'symbol') return a.symbol.localeCompare(b.symbol) * order;
      const x = rank(a);
      const y = rank(b);
      return (x === y ? 0 : x < y ? -1 : 1) * order || a.symbol.localeCompare(b.symbol);
    });
  }, [signals.data, sort, dir]);

  return (
    <div>
      <section className="results-head">
        <SegmentedTabs
          label="Universe"
          segments={[
            ...UNIVERSES.map((u) => ({
              value: u,
              to: `/results/${u}/Daily/new`,
              label: u,
            })),
            { value: 'manual' as const, to: '/results/manual', label: 'Manual' },
          ]}
        />
      </section>

      <section className="card">
        <h2>Manual scan</h2>
        <p className="muted small">
          Stocks and ETF are scanned in the background — this is for checking a handful of
          symbols right now.
        </p>

        <div className="field">
          <label htmlFor="manual-tickers">Tickers</label>
          <textarea
            id="manual-tickers"
            rows={3}
            value={tickers}
            disabled={running}
            onChange={(e) => setTickers(e.target.value)}
          />
        </div>

        <Chips label="Timeframe" value={tf} options={TIMEFRAMES} onChange={setTf} disabled={running} />

        {running ? (
          <button
            type="button"
            className="btn btn-danger"
            onClick={() => void api.cancelScan(runId as string).catch(() => undefined)}
          >
            STOP
          </button>
        ) : (
          <button type="button" className="btn btn-primary" disabled={starting} onClick={onStart}>
            {starting ? 'STARTING…' : 'START SCAN'}
          </button>
        )}

        {error ? <p className="error">{error}</p> : null}
      </section>

      {runId ? (
        <section className="card">
          <div className="stack-row">
            <h3 style={{ margin: 0 }}>{progress?.phase ?? run.data?.status ?? 'queued'}</h3>
            <span className="muted">{progress?.percent ?? 0}%</span>
          </div>
          <div className="bar">
            <div className="bar-fill" style={{ width: `${progress?.percent ?? 0}%` }} />
          </div>
          <p className="muted small" style={{ margin: '8px 0 0' }}>
            {progress?.message ?? 'Waiting for worker…'}
          </p>
          {done && run.data ? (
            <Link className="link-row" to={`/results/manual/rejected/${runId}`}>
              Rejected ({run.data.counters.rejected}) · Skipped ({run.data.counters.skipped ?? 0})
            </Link>
          ) : null}
        </section>
      ) : null}

      {done && rows.length === 0 ? (
        <p className="empty">No valid signals. Check the rejected reasons for why.</p>
      ) : null}

      {rows.length > 1 ? (
        <div className="results-meta manual-sort">
          <span className="muted small">{rows.length} signals</span>
          <SortChips
            options={SORTS}
            value={sort}
            dir={dir}
            onChange={(next, nextDir) => {
              setSort(next);
              setDir(nextDir);
            }}
          />
        </div>
      ) : null}

      {rows.map((signal: BuySignal) => (
        <article
          key={signal.yahooTicker}
          className="card signal-card compact clickable"
          role="button"
          tabIndex={0}
          onClick={() => navigate(`/chart/${encodeURIComponent(signal.yahooTicker)}`)}
          onKeyDown={(e) => {
            if (e.key === 'Enter') navigate(`/chart/${encodeURIComponent(signal.yahooTicker)}`);
          }}
        >
          <div className="signal-card-line1">
            <div className="signal-card-title">
              <strong>{signal.symbol}</strong>
              <span className="muted ellipsis">{signal.companyName}</span>
            </div>
            {/* Same rule as the Results tabs: bar zero of the valid run is NEW. */}
            {signal.barsSinceValid === 0 ? (
              <span className="badge up">NEW</span>
            ) : (
              <span className="badge" title={`Valid since ${signal.validSinceAsOf ?? '—'}`}>
                {barsLabel(signal.barsSinceValid)}
              </span>
            )}
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
              <span className="lbl">RR</span> {num(signal.rr)}
            </span>
          </div>
        </article>
      ))}
    </div>
  );
}
