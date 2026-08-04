import { useEffect, useMemo, useRef, useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import {
  TIMEFRAMES,
  UNIVERSES,
  api,
  type AppSettings,
  type ResultsSummary,
  type Timeframe,
  type Universe,
} from '../lib/api';
import { formatAge, TF_SHORT } from '../lib/format';
import { Chips } from './Chips';

/** Every list the tracked universes feed, invalidated together when a scan changes them. */
const SCAN_FED = ['results', 'results-summary', 'history', 'history-trades', 'tracked-signal'];

type ScanTf = Timeframe | 'all';
const SCAN_TFS: readonly ScanTf[] = ['all', ...TIMEFRAMES];

/** The universe and timeframe a scan is on right now, or null when nothing is running. */
function scanning(data: ResultsSummary | undefined): { universe: Universe; tf: Timeframe } | null {
  if (!data) return null;
  for (const universe of UNIVERSES) {
    for (const tf of TIMEFRAMES) {
      if (data[universe]?.[tf]?.scan.running) return { universe, tf };
    }
  }
  return null;
}

/** Newest finish across every universe and timeframe — how fresh the screens are at best. */
function lastFinished(data: ResultsSummary | undefined): string | null {
  let newest: string | null = null;
  for (const universe of UNIVERSES) {
    for (const tf of TIMEFRAMES) {
      const at = data?.[universe]?.[tf]?.scan.finishedAt;
      if (at && (!newest || at > newest)) newest = at;
    }
  }
  return newest;
}

function invalidateLists(queryClient: ReturnType<typeof useQueryClient>) {
  for (const key of ['results', 'results-summary', 'history', 'history-trades', 'tracked-signal', 'chart']) {
    void queryClient.invalidateQueries({ queryKey: [key] });
  }
}

/**
 * Global knobs: risk per signal, RR floor for every list/stat, rescanning, and maintenance.
 * Scan shape stays fixed in the backend (buy signals); Min RR only filters what is shown.
 */
export function SettingsSheet({ open, onClose }: { open: boolean; onClose: () => void }) {
  const queryClient = useQueryClient();
  const settings = useQuery({ queryKey: ['settings'], queryFn: api.settings, enabled: open });
  const [riskDraft, setRiskDraft] = useState('');
  const [rrDraft, setRrDraft] = useState('');
  const [scanTf, setScanTf] = useState<ScanTf>('all');
  // Queued, but the first run document may not exist yet — without this the button would flick
  // back to idle in the seconds between the request returning and the scan showing up.
  const [awaiting, setAwaiting] = useState(false);

  useEffect(() => {
    if (settings.data) {
      setRiskDraft(String(settings.data.maxRiskUsd));
      setRrDraft(String(settings.data.minRr));
    }
  }, [settings.data]);

  // Shares its key with the Results header, so opening this sheet is also what speeds that up
  // while a scan is on: react-query polls at the shortest interval any observer asks for.
  const summary = useQuery({
    queryKey: ['results-summary'],
    queryFn: api.resultsSummary,
    enabled: open,
    refetchInterval: (query) => (awaiting || scanning(query.state.data) ? 5_000 : false),
  });
  const busy = scanning(summary.data);
  const wasBusy = useRef(false);

  // A finished pass has rewritten every list, and this poll is the only thing that knows.
  useEffect(() => {
    if (busy) {
      wasBusy.current = true;
      return;
    }
    if (!wasBusy.current) return;
    wasBusy.current = false;
    setAwaiting(false);
    for (const key of SCAN_FED) void queryClient.invalidateQueries({ queryKey: [key] });
  }, [busy, queryClient]);

  // Nothing ever started: background scanning is off, or the pass failed before writing a run.
  useEffect(() => {
    if (!awaiting) return;
    const timer = setTimeout(() => setAwaiting(false), 120_000);
    return () => clearTimeout(timer);
  }, [awaiting]);

  const rescan = useMutation({
    mutationFn: () => api.runScanNow(scanTf),
    onSuccess: (result) => {
      if (result.started) setAwaiting(true);
      void queryClient.invalidateQueries({ queryKey: ['results-summary'] });
    },
  });

  const running = awaiting || Boolean(busy);
  const scanAge = useMemo(() => formatAge(lastFinished(summary.data)), [summary.data]);

  const save = useMutation({
    mutationFn: (patch: Partial<AppSettings>) => api.saveSettings(patch),
    onSuccess: (next) => {
      queryClient.setQueryData(['settings'], next);
      // Max risk re-sizes open positions before the response; Min RR only changes what lists show.
      // History always re-reads under the current risk, so it needs a refresh either way.
      invalidateLists(queryClient);
    },
  });

  const reset = useMutation({
    mutationFn: api.resetHistory,
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ['results'] });
      void queryClient.invalidateQueries({ queryKey: ['results-summary'] });
      void queryClient.invalidateQueries({ queryKey: ['history'] });
      void queryClient.invalidateQueries({ queryKey: ['history-trades'] });
    },
  });

  const [rebuildAwaiting, setRebuildAwaiting] = useState(false);
  const rebuild = useMutation({
    mutationFn: api.rebuildHistory,
    onSuccess: (result) => {
      if (result.started) setRebuildAwaiting(true);
      void queryClient.invalidateQueries({ queryKey: ['history-rebuild'] });
    },
  });
  const rebuildStatus = useQuery({
    queryKey: ['history-rebuild'],
    queryFn: api.historyRebuildStatus,
    enabled: open,
    refetchInterval: (query) =>
      rebuildAwaiting || query.state.data?.status === 'running' ? 2_000 : false,
  });
  const rebuildRunning =
    rebuildAwaiting || rebuild.isPending || rebuildStatus.data?.status === 'running';

  useEffect(() => {
    const status = rebuildStatus.data?.status;
    if (!rebuildAwaiting) return;
    if (status === 'running') return;
    if (status === 'done' || status === 'failed') {
      setRebuildAwaiting(false);
      void queryClient.invalidateQueries({ queryKey: ['history'] });
      void queryClient.invalidateQueries({ queryKey: ['history-trades'] });
    }
  }, [rebuildAwaiting, rebuildStatus.data?.status, queryClient]);

  if (!open) return null;

  const commitRisk = () => {
    const next = Number(riskDraft);
    if (!Number.isFinite(next) || next <= 0) {
      setRiskDraft(String(settings.data?.maxRiskUsd ?? ''));
      return;
    }
    if (next === settings.data?.maxRiskUsd) return;
    save.mutate({ maxRiskUsd: next });
  };

  const commitRr = () => {
    const next = Number(rrDraft);
    if (!Number.isFinite(next) || next < 0) {
      setRrDraft(String(settings.data?.minRr ?? ''));
      return;
    }
    if (next === settings.data?.minRr) return;
    save.mutate({ minRr: next });
  };

  return (
    <>
      <div className="sheet-backdrop" onClick={onClose} aria-hidden />
      <div className="chart-settings-sheet" role="dialog" aria-label="Settings">
      <div className="chart-settings-head">
        <strong>Settings</strong>
        <button type="button" className="btn-sm ghost" onClick={onClose}>
          Close
        </button>
      </div>

      <div className="field">
        <label htmlFor="max-risk">Max risk per signal ($)</label>
        <input
          id="max-risk"
          type="number"
          inputMode="decimal"
          min={1}
          step={1}
          value={riskDraft}
          onChange={(e) => setRiskDraft(e.target.value)}
          onBlur={commitRisk}
          onKeyDown={(e) => {
            if (e.key === 'Enter') (e.target as HTMLInputElement).blur();
          }}
        />
      </div>
      <p className="muted small">
        One risk for every signal: position size is this divided by the distance to SL. Changing it
        re-sizes all open signals right away. History statistics recompute under this risk; closed
        rows in Results keep the size they were closed at.
      </p>

      <div className="field">
        <label htmlFor="min-rr">Min RR</label>
        <input
          id="min-rr"
          type="number"
          inputMode="decimal"
          min={0}
          step={0.1}
          value={rrDraft}
          onChange={(e) => setRrDraft(e.target.value)}
          onBlur={commitRr}
          onKeyDown={(e) => {
            if (e.key === 'Enter') (e.target as HTMLInputElement).blur();
          }}
        />
      </div>
      <p className="muted small">
        Floor on entry RR for NEW, VALID, CLOSED and all History statistics. 0 shows everything.
        Scans still track every signal; this only filters what the lists and stats include.
      </p>

      <div className="field">
        <Chips
          label="Rescan"
          value={scanTf}
          options={SCAN_TFS}
          disabled={running || rescan.isPending}
          onChange={setScanTf}
          format={(v) => (v === 'all' ? 'All' : TF_SHORT[v])}
        />
      </div>
      <div className="chart-settings-actions">
        <button
          type="button"
          className="btn-sm"
          disabled={running || rescan.isPending}
          onClick={() => rescan.mutate()}
        >
          {running || rescan.isPending ? 'Scanning…' : 'Run scan now'}
        </button>
      </div>
      <p className="muted small">
        {busy
          ? `Scanning ${busy.universe} ${TF_SHORT[busy.tf]}. Stocks and ETF go one timeframe at a time, and the lists fill in as each finishes.`
          : running
            ? 'Starting. Stocks and ETF go one timeframe at a time, and the lists fill in as each finishes.'
            : 'Re-downloads every symbol in Stocks and ETF and rebuilds the lists from it. Scans otherwise run on their own — one hourly pass over Daily, Weekly and Monthly together.'}
        {scanAge ? ` Last finished ${scanAge}.` : ' No scan has finished yet.'}
      </p>
      {rescan.data && !rescan.data.started ? (
        <p className="muted small">{rescan.data.reason ?? 'A scan is already running.'}</p>
      ) : null}
      {rescan.error ? <p className="error">{(rescan.error as Error).message}</p> : null}

      <div className="chart-settings-actions">
        <button
          type="button"
          className="btn-sm"
          disabled={rebuildRunning || running}
          onClick={() => {
            if (
              !window.confirm(
                'Replay the close-scan ledger over every cached symbol and add missing closed trades to History? Existing rows are kept.',
              )
            ) {
              return;
            }
            rebuild.mutate();
          }}
        >
          {rebuildRunning ? 'Rebuilding…' : 'Rebuild history'}
        </button>
      </div>
      <p className="muted small">
        Fills History from the close-scan replay on the bar cache (run a scan first if bars are
        cold). Does not delete anything. Yahoo windows: Daily ~2y, Weekly and Monthly ~10y — that is
        as far back as rebuild can go.
        {rebuildStatus.data?.status === 'running' && rebuildStatus.data.progress.tf
          ? ` Now ${rebuildStatus.data.progress.universe} ${TF_SHORT[rebuildStatus.data.progress.tf]} (${rebuildStatus.data.progress.symbolsDone}/${rebuildStatus.data.progress.symbolsTotal}).`
          : ''}
        {rebuildStatus.data?.status === 'done'
          ? ` Last run inserted ${rebuildStatus.data.counts.inserted}, skipped ${rebuildStatus.data.counts.skipped}, no bars ${rebuildStatus.data.counts.noBars}.`
          : ''}
      </p>
      {rebuild.data && !rebuild.data.started ? (
        <p className="muted small">{rebuild.data.reason ?? 'A history rebuild is already running.'}</p>
      ) : null}
      {rebuildStatus.data?.status === 'failed' ? (
        <p className="error">{rebuildStatus.data.error ?? 'History rebuild failed.'}</p>
      ) : null}
      {rebuild.error ? <p className="error">{(rebuild.error as Error).message}</p> : null}

      <div className="chart-settings-actions">
        <button
          type="button"
          className="btn-sm danger"
          disabled={reset.isPending || rebuildRunning}
          onClick={() => {
            if (!window.confirm('Delete every scan run and tracked signal? This cannot be undone.')) {
              return;
            }
            reset.mutate();
          }}
        >
          {reset.isPending ? 'Resetting…' : 'Reset all history'}
        </button>
      </div>
      {reset.data ? (
        <p className="muted small">
          Deleted {reset.data.deletedRuns} runs and {reset.data.deletedSignals} tracked signals.
        </p>
      ) : null}
        {save.error ? <p className="error">{(save.error as Error).message}</p> : null}
      </div>
    </>
  );
}
