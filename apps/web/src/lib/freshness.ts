/**
 * Data-age labels. A scan evaluates a stored bar snapshot, TradingView draws the
 * live in-progress bar, so the age of the snapshot explains most disagreements.
 */
export function formatDataAge(iso: string | null | undefined, now = Date.now()): string | null {
  if (!iso) return null;
  const ts = new Date(iso).getTime();
  if (!Number.isFinite(ts)) return null;
  const minutes = Math.max(0, Math.round((now - ts) / 60_000));
  if (minutes < 1) return 'just now';
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 48) return `${hours}h ago`;
  return `${Math.round(hours / 24)}d ago`;
}
