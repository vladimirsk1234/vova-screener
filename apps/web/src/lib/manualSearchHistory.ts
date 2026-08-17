const HISTORY_KEY = 'vova.manualSearchHistory';
const MAX_HISTORY = 10;

/** Last 10 tickers that actually went through a manual START SCAN. Newest first. */
export function readManualSearchHistory(): string[] {
  try {
    const raw = localStorage.getItem(HISTORY_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    const out: string[] = [];
    const seen = new Set<string>();
    for (const item of parsed) {
      const ticker = String(item ?? '')
        .trim()
        .toUpperCase();
      if (!ticker || seen.has(ticker)) continue;
      seen.add(ticker);
      out.push(ticker);
      if (out.length >= MAX_HISTORY) break;
    }
    return out;
  } catch {
    return [];
  }
}

/** Move ticker to the front; the 11th oldest entry is dropped. */
export function rememberManualSearch(ticker: string): string[] {
  const next = ticker.trim().toUpperCase();
  if (!next) return readManualSearchHistory();
  const history = [next, ...readManualSearchHistory().filter((t) => t !== next)].slice(
    0,
    MAX_HISTORY,
  );
  localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
  return history;
}
