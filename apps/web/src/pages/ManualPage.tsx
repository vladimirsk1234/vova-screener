import { useEffect, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { parseListEntry } from '@vova/engine';
import { TIMEFRAMES, UNIVERSES, api, type BuySignal, type Rejection, type Timeframe } from '../lib/api';
import { barsLabel, money, num } from '../lib/format';
import { readManualSearchHistory, rememberManualSearch, rememberResolvedManualSearch } from '../lib/manualSearchHistory';
import { resultsPathForUniverse } from '../lib/tabMemory';
import { Chips } from '../components/Chips';
import { SegmentedTabs } from '../components/SegmentedTabs';
import { useScanProgress } from '../lib/useScanProgress';

const ACTIVE_RUN_KEY = 'vova.manualRunId';
const TICKER_KEY = 'vova.manualTickers';
const TERMINAL = ['completed', 'cancelled', 'failed'];

function parseOneManualTicker(
  text: string,
): { ok: true; ticker: string } | { ok: false; error: string } {
  const parts = text
    .split(/[,\n]/)
    .map((s) => s.trim())
    .filter(Boolean);
  if (parts.length === 0) return { ok: false, error: 'Enter one ticker.' };
  if (parts.length > 1) return { ok: false, error: 'One ticker at a time.' };
  if (parts[0].split(/\s+/).filter(Boolean).length > 1) {
    return { ok: false, error: 'One ticker at a time.' };
  }
  const parsed = parseListEntry(parts[0]);
  if (!parsed) return { ok: false, error: 'Enter one ticker.' };
  return { ok: true, ticker: parsed.yahoo };
}

function initialTicker(): string {
  const history = readManualSearchHistory();
  if (history[0]) return history[0];
  const saved = localStorage.getItem(TICKER_KEY) ?? '';
  const parsed = parseOneManualTicker(saved);
  return parsed.ok ? parsed.ticker : '';
}

/**
 * The only screen that starts a scan. Manual runs are ad-hoc checks against a live chart, so
 * they always pull fresh bars and never feed the tracked-signal history.
 */
export function ManualPage() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [ticker, setTicker] = useState(initialTicker);
  const [history, setHistory] = useState(readManualSearchHistory);
  const [lastScanned, setLastScanned] = useState(() => readManualSearchHistory()[0] ?? '');
  const [tf, setTf] = useState<Timeframe>('Daily');
  const [runId, setRunId] = useState<string | null>(() => localStorage.getItem(ACTIVE_RUN_KEY));
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);
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

  const rejections = useQuery({
    queryKey: ['manual-rejections', runId],
    queryFn: () => api.rejections(runId as string, 20),
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

  const signal = signals.data?.rows[0] ?? null;
  const rejection = rejections.data?.rows[0] ?? null;
  const fromRun = parseOneManualTicker(run.data?.params.manualTickers ?? '');
  const chartTicker =
    signal?.yahooTicker ||
    rejection?.yahooTicker ||
    lastScanned ||
    (fromRun.ok ? fromRun.ticker : '');

  useEffect(() => {
    if (!done) return;
    const resolved = signal?.yahooTicker || rejection?.yahooTicker;
    if (!resolved) return;
    const typed = lastScanned.trim().toUpperCase();
    if (resolved.toUpperCase() === typed) return;
    setHistory(rememberResolvedManualSearch(typed, resolved));
    setLastScanned(resolved);
    localStorage.setItem(TICKER_KEY, resolved);
    setTicker((current) => {
      const cur = current.trim().toUpperCase();
      if (!cur || cur === typed) return resolved;
      return current;
    });
  }, [done, lastScanned, rejection?.yahooTicker, signal?.yahooTicker]);

  const onStart = async () => {
    const parsed = parseOneManualTicker(ticker);
    if (!parsed.ok) {
      setError(parsed.error);
      return;
    }
    setStarting(true);
    setError(null);
    try {
      const res = await api.startScan({
        source: 'MANUAL SCAN',
        manualTickers: parsed.ticker,
        tf,
        direction: 'buy',
        minRr: 0,
        noRrReq: true,
        useLastHlSl: true,
        newOnly: false,
        riskPerTrade: settings.data?.maxRiskUsd ?? 100,
        forceRefresh: true,
      });
      localStorage.setItem(TICKER_KEY, parsed.ticker);
      setHistory(rememberManualSearch(parsed.ticker));
      setLastScanned(parsed.ticker);
      localStorage.setItem(ACTIVE_RUN_KEY, res.runId);
      setRunId(res.runId);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setStarting(false);
    }
  };

  return (
    <div>
      <section className="results-head">
        <SegmentedTabs
          label="Universe"
          segments={[
            ...UNIVERSES.map((u) => ({
              value: u,
              to: resultsPathForUniverse(u),
              label: u,
            })),
            { value: 'manual' as const, to: '/results/manual', label: 'Manual' },
          ]}
        />
      </section>

      <section className="card">
        <h2>Manual scan</h2>
        <p className="muted small">
          Stocks and ETF are scanned in the background — this is for checking one symbol right
          now.
        </p>

        <div className="field">
          <label htmlFor="manual-tickers">Ticker</label>
          <input
            id="manual-tickers"
            type="text"
            autoCapitalize="characters"
            autoComplete="off"
            spellCheck={false}
            value={ticker}
            disabled={running}
            onChange={(e) => setTicker(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !running && !starting) void onStart();
            }}
          />
        </div>

        {history.length ? (
          <div className="manual-history" role="group" aria-label="Recent tickers">
            {history.map((item) => (
              <button
                key={item}
                type="button"
                className={`sort-chip${item === ticker.trim().toUpperCase() ? ' active' : ''}`}
                disabled={running}
                onClick={() => setTicker(item)}
              >
                {item}
              </button>
            ))}
          </div>
        ) : null}

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
          <button type="button" className="btn btn-primary" disabled={starting} onClick={() => void onStart()}>
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

      {done && chartTicker ? (
        <ManualResultCard
          ticker={chartTicker}
          signal={signal}
          rejection={rejection}
          onOpen={(path) => navigate(path)}
        />
      ) : null}
    </div>
  );
}

function ManualResultCard({
  ticker,
  signal,
  rejection,
  onOpen,
}: {
  ticker: string;
  signal: BuySignal | null;
  rejection: Rejection | null;
  onOpen: (path: string) => void;
}) {
  const yahoo = signal?.yahooTicker ?? rejection?.yahooTicker ?? ticker;
  const symbol = signal?.symbol ?? rejection?.symbol ?? ticker;
  const name = signal?.companyName ?? '';
  const taPath = `/chart/${encodeURIComponent(yahoo)}`;
  const fundPath = `${taPath}?view=fundamentals`;

  return (
    <article className="card signal-card compact">
      <div className="signal-card-line1">
        <div className="signal-card-title">
          <strong>{symbol}</strong>
          {name ? <span className="muted ellipsis">{name}</span> : null}
        </div>
        {signal ? (
          signal.barsSinceValid === 0 ? (
            <span className="badge up">NEW</span>
          ) : (
            <span className="badge" title={`Valid since ${signal.validSinceAsOf ?? '—'}`}>
              {barsLabel(signal.barsSinceValid)}
            </span>
          )
        ) : rejection ? (
          <span className="badge warn-badge">{rejection.reason}</span>
        ) : (
          <span className="badge">No signal</span>
        )}
      </div>
      {signal ? (
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
      ) : (
        <p className="muted small" style={{ margin: '6px 0 0' }}>
          {rejection
            ? 'No valid signal — TA and Fundamentals are still available.'
            : 'Scan finished without a signal. TA and Fundamentals are still available.'}
        </p>
      )}
      <div className="card-actions">
        <button type="button" className="btn-sm ghost" onClick={() => onOpen(taPath)}>
          TA
        </button>
        <button type="button" className="btn-sm ghost" onClick={() => onOpen(fundPath)}>
          Fundamentals
        </button>
      </div>
    </article>
  );
}
