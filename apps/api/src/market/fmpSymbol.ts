/** Yahoo ↔ FMP symbol mapping (no HTTP / Nest dependencies). */

/** Dotted/alternate FMP form (BRK-B → BRK.B) — fallback only. */
export function fmpMappedSymbol(yahooTicker: string): string {
  const s = String(yahooTicker || '')
    .trim()
    .toUpperCase();
  if (!s) return s;
  if (/\.(TO|V|NE|CN)$/.test(s)) return s;
  const classShare = s.match(/^([A-Z0-9]+)-([A-Z])$/);
  if (classShare) return `${classShare[1]}.${classShare[2]}`;
  return s.replace(/-/g, '.');
}

/**
 * Symbol forms to try, in order. FMP serves US class shares under the dash form (BRK-B);
 * the dotted form answers HTTP 402, so keep it only as fallback.
 */
export function fmpSymbolCandidates(yahooTicker: string): string[] {
  const raw = String(yahooTicker || '')
    .trim()
    .toUpperCase();
  if (!raw) return [];
  const forms = [raw];
  const mapped = fmpMappedSymbol(raw);
  if (mapped && mapped !== raw) forms.push(mapped);
  return forms;
}

/** Preferred FMP symbol (first candidate). */
export function yahooToFmpSymbol(yahooTicker: string): string {
  return fmpSymbolCandidates(yahooTicker)[0] ?? '';
}
