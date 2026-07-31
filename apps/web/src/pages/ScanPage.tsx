import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { api, type Direction, type ScanParams, type SourceLabel, type Timeframe } from '../lib/api';
import { useScanProgress } from '../lib/useScanProgress';
import { Chips, Switch } from '../components/Chips';

const DEFAULTS: ScanParams = {
  source: 'MANUAL SCAN',
  manualTickers: 'AAPL, TSLA, NVDA, MSFT, AMD',
  tf: 'Weekly',
  direction: 'buy',
  minRr: 1.5,
  riskPerTrade: 100,
  noRrReq: false,
  useLastHlSl: true,
  newOnly: true,
  forceRefresh: false,
};

const ACTIVE_RUN_KEY = 'vova.activeRunId';
const TERMINAL = ['completed', 'cancelled', 'failed'];

export function ScanPage() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [params, setParams] = useState<ScanParams>(DEFAULTS);
  const [runId, setRunId] = useState<string | null>(() => localStorage.getItem(ACTIVE_RUN_KEY));
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const progress = useScanProgress(runId);

  const universe = useQuery({ queryKey: ['universe'], queryFn: api.universeSummary });

  useEffect(() => {
    api
      .getPreset<Partial<ScanParams>>('scan')
      .then((saved) => {
        if (saved && Object.keys(saved).length) setParams((p) => ({ ...p, ...saved }));
      })
      .catch(() => undefined);
  }, []);

  const run = useQuery({
    queryKey: ['run', runId],
    queryFn: () => api.run(runId as string),
    enabled: Boolean(runId),
    refetchInterval: (q) => {
      const status = q.state.data?.status;
      return status && TERMINAL.includes(status) ? false : 2_000;
    },
  });

  useEffect(() => {
    if (!runId || !progress || !TERMINAL.includes(progress.phase)) return;
    void queryClient.invalidateQueries({ queryKey: ['run', runId] });
  }, [progress?.phase, runId, queryClient]);

  const progressDone = Boolean(progress && TERMINAL.includes(progress.phase));
  const status = progressDone && progress ? progress.phase : (run.data?.status ?? 'queued');
  const isRunning = Boolean(
    runId && !progressDone && ['queued', 'running'].includes(run.data?.status ?? 'queued'),
  );
  const patch = (next: Partial<ScanParams>) => setParams((p) => ({ ...p, ...next }));

  const onStart = async () => {
    setStarting(true);
    setError(null);
    try {
      await api.putPreset('scan', params).catch(() => undefined);
      const res = await api.startScan(params);
      localStorage.setItem(ACTIVE_RUN_KEY, res.runId);
      setRunId(res.runId);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setStarting(false);
    }
  };

  const onStop = async () => {
    if (!runId) return;
    await api.cancelScan(runId).catch(() => undefined);
  };

  const counters = progress?.counters ?? run.data?.counters;
  const done = progressDone || Boolean(run.data && TERMINAL.includes(run.data.status));
  const universeCount =
    params.source === 'Stocks'
      ? universe.data?.stocks
      : params.source === 'ETF'
        ? universe.data?.etf
        : undefined;

  return (
    <div>
      <section className="card">
        <h2>Scan</h2>
        <p className="muted">
          {params.source} · {params.direction.toUpperCase()} · {params.tf}
          {universeCount ? ` · ${universeCount} symbols` : ''}
        </p>

        <Chips
          label="Source"
          value={params.source}
          options={['Stocks', 'ETF', 'MANUAL SCAN'] as const satisfies readonly SourceLabel[]}
          onChange={(source) => patch({ source })}
          disabled={isRunning}
        />
        <Chips
          label="Direction"
          value={params.direction}
          options={['buy', 'sell'] as const satisfies readonly Direction[]}
          onChange={(direction) => patch({ direction })}
          disabled={isRunning}
          format={(v) => v.toUpperCase()}
        />
        <Chips
          label="Timeframe"
          value={params.tf}
          options={['Daily', 'Weekly', 'Monthly'] as const satisfies readonly Timeframe[]}
          onChange={(tf) => patch({ tf })}
          disabled={isRunning}
        />

        {params.source === 'MANUAL SCAN' && (
          <div className="field">
            <label htmlFor="manual">Tickers</label>
            <textarea
              id="manual"
              rows={3}
              value={params.manualTickers}
              disabled={isRunning}
              onChange={(e) => patch({ manualTickers: e.target.value })}
            />
          </div>
        )}

        <div className="field-grid">
          <div className="field">
            <label htmlFor="minRr">Min RR</label>
            <input
              id="minRr"
              type="number"
              inputMode="decimal"
              min={0.1}
              step={0.1}
              value={params.minRr}
              disabled={isRunning || params.noRrReq}
              onChange={(e) => patch({ minRr: Number(e.target.value) })}
            />
          </div>
          <div className="field">
            <label htmlFor="risk">Risk $</label>
            <input
              id="risk"
              type="number"
              inputMode="decimal"
              min={1}
              step={1}
              value={params.riskPerTrade}
              disabled={isRunning}
              onChange={(e) => patch({ riskPerTrade: Number(e.target.value) })}
            />
          </div>
        </div>

        <Switch
          label="Any valid signal (no RR req)"
          checked={params.noRrReq}
          disabled={isRunning}
          onChange={(noRrReq) => patch({ noRrReq })}
        />
        <Switch
          label="New signals only"
          checked={params.newOnly}
          disabled={isRunning}
          onChange={(newOnly) => patch({ newOnly })}
        />
        <Switch
          label="Use last HL for SL"
          checked={params.useLastHlSl}
          disabled={isRunning}
          onChange={(useLastHlSl) => patch({ useLastHlSl })}
        />
        <Switch
          label="Force fresh download (ignore cache)"
          checked={Boolean(params.forceRefresh)}
          disabled={isRunning}
          onChange={(forceRefresh) => patch({ forceRefresh })}
        />

        {isRunning ? (
          <button type="button" className="btn btn-danger" onClick={onStop}>
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
            <h3>{status}</h3>
            <span className="muted">{progress?.percent ?? 0}%</span>
          </div>
          <div className="bar">
            <div className="bar-fill" style={{ width: `${progress?.percent ?? 0}%` }} />
          </div>
          <p className="muted" style={{ margin: '8px 0 0' }}>
            {progress?.message ?? 'Waiting for worker…'}
          </p>
          {counters ? (
            <div className="meta-grid">
              <div>
                <span>Scanned</span>
                {counters.evaluated ?? 0}/{counters.total ?? 0}
              </div>
              <div>
                <span>Signals</span>
                {counters.signals ?? 0}
              </div>
              <div>
                <span>Rejected</span>
                {counters.rejected ?? 0}
              </div>
              <div>
                <span>From cache</span>
                {counters.fromCache ?? 0}
              </div>
            </div>
          ) : null}

          {done ? (
            <button
              type="button"
              className="btn btn-accent"
              style={{ marginTop: 12 }}
              onClick={() => navigate(`/runs/${runId}`)}
            >
              View results ({counters?.signals ?? run.data?.counters.signals ?? 0})
            </button>
          ) : null}
        </section>
      ) : null}
    </div>
  );
}
