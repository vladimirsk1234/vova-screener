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
  if (!running || !run) return null;
  const scored = cov && cov.universe > 0 ? ` · ${cov.complete}/${cov.universe} scored` : '';
  const pct =
    cov && cov.universe > 0
      ? Math.round((cov.complete / cov.universe) * 100)
      : Math.round((run.done / run.total) * 100);
  return {
    text: `Updating ${run.done}/${run.total}${scored}`,
    pct: Math.max(0, Math.min(100, pct)),
  };
}

export function refreshPollMs(
  input:
    | {
        lastRun?: RefreshRunStatus | null;
        coverage?: RefreshCoverage | null;
      }
    | undefined,
): number | false {
  if (input?.lastRun?.status === 'running') return 3_000;
  return false;
}
