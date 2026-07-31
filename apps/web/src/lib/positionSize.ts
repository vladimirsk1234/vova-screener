/** Position sizing helpers — match engine evaluate (Math.round). */

export function round2(n: number) {
  return Math.round(n * 100) / 100;
}

export function sharesFromRisk(
  entry: number,
  sl: number | null | undefined,
  riskUsd: number,
): number {
  if (sl == null || !Number.isFinite(sl) || !Number.isFinite(entry)) return 0;
  const risk = entry - sl;
  if (risk <= 0 || riskUsd <= 0) return 0;
  return Math.max(0, Math.round(riskUsd / risk));
}

export function investedFromShares(entry: number, shares: number): number {
  if (!Number.isFinite(entry) || !Number.isFinite(shares) || shares <= 0) return 0;
  return round2(entry * shares);
}
