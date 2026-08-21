/** Decide whether boot / poll should start a fundamentals FMP walk. */

export type FundamentalsCatchUpKind = 'missing' | 'full';

export function fundamentalsCatchUpKind(input: {
  fmpConfigured: boolean;
  busy: boolean;
  universe: number;
  complete: number;
  pastEodSlot: boolean;
  todayFullDone: boolean;
  /** A missing/full walk already finished on this NY calendar day. */
  completedPassToday: boolean;
}): FundamentalsCatchUpKind | null {
  if (!input.fmpConfigured || input.busy) return null;
  // Universe import is async — do not record a 0-ticker "full" run or skip forever.
  if (input.universe <= 0) return null;
  // One gap-fill per day. Failed names wait for the weekday EOD full pull.
  if (input.complete < input.universe && !input.completedPassToday) return 'missing';
  if (!input.pastEodSlot) return null;
  if (input.todayFullDone) return null;
  return 'full';
}

export function refreshProgressPct(input: {
  run?: { status: string; done: number; total: number } | null;
  coverage?: { complete: number; universe: number } | null;
}): number {
  const run = input.run;
  const cov = input.coverage;
  if (cov && cov.universe > 0) {
    return Math.max(0, Math.min(100, Math.round((cov.complete / cov.universe) * 100)));
  }
  if (run?.status === 'running' && run.total > 0) {
    return Math.max(0, Math.min(100, Math.round((run.done / run.total) * 100)));
  }
  return 0;
}
