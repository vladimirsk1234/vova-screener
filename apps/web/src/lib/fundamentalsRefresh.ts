/** Shared EOD / catch-up line for Value and Fundamentals. */

export type RefreshRunStatus = {
  status: string;
  done: number;
  total: number;
};

export type RefreshCoverage = {
  complete: number;
  universe: number;
};

export function fundamentalsUpdateBanner(input: {
  lastRun?: RefreshRunStatus | null;
  coverage?: RefreshCoverage | null;
}): { text: string; pct: number } | null {
  const run = input.lastRun;
  const cov = input.coverage;
  const running = Boolean(run && run.status === 'running' && run.total > 0);
  const incomplete = cov != null && (cov.universe <= 0 || cov.complete < cov.universe);
  if (!running && !incomplete) return null;
  const scored = cov && cov.universe > 0 ? ` · ${cov.complete}/${cov.universe} scored` : '';
  const text =
    running && run
      ? `Updating ${run.done}/${run.total}${scored}`
      : `Starting fundamentals update…${scored}`;
  const pct =
    running && run
      ? Math.round((run.done / run.total) * 100)
      : cov && cov.universe > 0
        ? Math.round((cov.complete / cov.universe) * 100)
        : 0;
  return { text, pct: Math.max(0, Math.min(100, pct)) };
}

export function refreshPollMs(
  input:
    | {
        lastRun?: RefreshRunStatus | null;
        coverage?: RefreshCoverage | null;
      }
    | undefined,
): number | false {
  const run = input?.lastRun;
  const cov = input?.coverage;
  if (run?.status === 'running') return 3_000;
  if (cov == null || cov.universe <= 0 || cov.complete < cov.universe) return 5_000;
  return false;
}
