/** Ticker list parsing — port of ticker_data.py list helpers (pure, no I/O). */

const TV_TO_YAHOO_SUFFIX: Record<string, string> = {
  TSX: '.TO',
  TSXV: '.V',
  NEO: '.NE',
  CSE: '.CN',
};
const YAHOO_CANADIAN_SUFFIXES = Object.values(TV_TO_YAHOO_SUFFIX);

export function normalizeYahooTicker(sym: string): string {
  const s = String(sym || '')
    .trim()
    .toUpperCase()
    .replace(/\//g, '-');
  if (!s) return s;
  const suffixes = [...YAHOO_CANADIAN_SUFFIXES].sort((a, b) => b.length - a.length);
  for (const suffix of suffixes) {
    const dashSuf = suffix.replace('.', '-').toUpperCase();
    const sufU = suffix.toUpperCase();
    if (s.endsWith(dashSuf)) {
      const base = s.slice(0, -dashSuf.length).replace(/\./g, '-');
      return `${base}${sufU}`;
    }
    if (s.endsWith(sufU)) {
      const base = s.slice(0, -sufU.length).replace(/\./g, '-');
      return `${base}${sufU}`;
    }
  }
  return s.replace(/\./g, '-');
}

function tvToYahooSymbol(ex: string, rawSym: string): string {
  const exU = ex.trim().toUpperCase();
  const upper = rawSym.trim().toUpperCase();
  if (exU in TV_TO_YAHOO_SUFFIX) {
    const suffix = TV_TO_YAHOO_SUFFIX[exU];
    // Only treat as already-suffixed when the exchange dot-suffix is present
    // (SHOP.TO). Bare endings like BTO / SSV must still get .TO / .V.
    if (upper.endsWith(suffix.toUpperCase())) return normalizeYahooTicker(upper);
    if (YAHOO_CANADIAN_SUFFIXES.some((s) => s !== suffix && upper.endsWith(s.toUpperCase()))) {
      return normalizeYahooTicker(upper);
    }
    const base = upper.includes('.') ? upper.replace(/\./g, '-') : upper;
    return normalizeYahooTicker(`${base}${suffix}`);
  }
  return normalizeYahooTicker(rawSym);
}

export type ParsedEntry = {
  yahoo: string;
  tv: string;
  name: string | null;
};

export function parseListEntry(part: string): ParsedEntry | null {
  const p = part.trim();
  if (!p || p.startsWith('#')) return null;
  let companyName: string | null = null;
  let tvPart = p;
  if (p.includes('|')) {
    const [left, name] = p.split('|', 2);
    tvPart = left.trim();
    if (name.trim()) companyName = name.trim();
  }
  if (!tvPart) return null;
  if (tvPart.includes(':')) {
    const [ex, raw] = tvPart.split(':', 2);
    return {
      yahoo: tvToYahooSymbol(ex, raw),
      tv: `${ex.trim().toUpperCase()}:${raw.trim().toUpperCase()}`,
      name: companyName,
    };
  }
  const yahoo = normalizeYahooTicker(tvPart);
  return { yahoo, tv: yahoo, name: companyName };
}

export type ParsedList = {
  tickers: string[];
  entries: ParsedEntry[];
  tvByYahoo: Record<string, string>;
  nameByYahoo: Record<string, string>;
};

export function parseListText(raw: string): ParsedList {
  const parts = raw.includes('\n') ? raw.split(/\r?\n/) : raw.split(',');
  const tickers: string[] = [];
  const entries: ParsedEntry[] = [];
  const tvByYahoo: Record<string, string> = {};
  const nameByYahoo: Record<string, string> = {};
  const seen = new Set<string>();
  for (const part of parts) {
    const parsed = parseListEntry(part);
    if (!parsed) continue;
    if (seen.has(parsed.yahoo)) continue;
    seen.add(parsed.yahoo);
    tickers.push(parsed.yahoo);
    entries.push(parsed);
    tvByYahoo[parsed.yahoo] = parsed.tv;
    if (parsed.name) nameByYahoo[parsed.yahoo] = parsed.name;
  }
  return { tickers, entries, tvByYahoo, nameByYahoo };
}

export function parseManualTickers(text: string): ParsedList {
  const parts = text
    .split(/[,\n]/)
    .map((s) => s.trim())
    .filter(Boolean);
  return parseListText(parts.join('\n'));
}
