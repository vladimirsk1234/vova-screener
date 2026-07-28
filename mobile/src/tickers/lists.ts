/** Parse TV list files and manual tickers — port of ticker_data list helpers. */
import { Asset } from 'expo-asset';
import * as FileSystem from 'expo-file-system/legacy';
import type { SourceLabel } from '../types';

const TV_TO_YAHOO_SUFFIX: Record<string, string> = {
  TSX: '.TO',
  TSXV: '.V',
  NEO: '.NE',
  CSE: '.CN',
};
const YAHOO_CANADIAN_SUFFIXES = Object.values(TV_TO_YAHOO_SUFFIX);

function normalizeYahooTicker(sym: string): string {
  let s = String(sym || '')
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
    if (upper.endsWith(suffix.toUpperCase())) return normalizeYahooTicker(upper);
    if (
      YAHOO_CANADIAN_SUFFIXES.some(
        (s) => s !== suffix && upper.endsWith(s.toUpperCase()),
      )
    ) {
      return normalizeYahooTicker(upper);
    }
    let base =
      upper.includes('.') && !upper.endsWith(suffix.toUpperCase())
        ? upper.replace(/\./g, '-')
        : upper;
    const bare = suffix.toUpperCase().replace('.', '');
    if (base.endsWith(bare)) return normalizeYahooTicker(base);
    return normalizeYahooTicker(`${base.split('.')[0]}${suffix}`);
  }
  return normalizeYahooTicker(rawSym);
}

export type ParsedEntry = {
  yahoo: string;
  tv: string;
  name: string | null;
};

export function parseListEntry(part: string): ParsedEntry | null {
  let p = part.trim();
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
    const yahoo = tvToYahooSymbol(ex, raw);
    return { yahoo, tv: `${ex.trim().toUpperCase()}:${raw.trim().toUpperCase()}`, name: companyName };
  }
  const yahoo = normalizeYahooTicker(tvPart);
  return { yahoo, tv: yahoo, name: companyName };
}

export function parseListText(raw: string): {
  tickers: string[];
  tvByYahoo: Record<string, string>;
  nameByYahoo: Record<string, string>;
} {
  const parts = raw.includes('\n') ? raw.split(/\r?\n/) : raw.split(',');
  const tickers: string[] = [];
  const tvByYahoo: Record<string, string> = {};
  const nameByYahoo: Record<string, string> = {};
  const seen = new Set<string>();
  for (const part of parts) {
    const parsed = parseListEntry(part);
    if (!parsed) continue;
    if (seen.has(parsed.yahoo)) continue;
    seen.add(parsed.yahoo);
    tickers.push(parsed.yahoo);
    tvByYahoo[parsed.yahoo] = parsed.tv;
    if (parsed.name) nameByYahoo[parsed.yahoo] = parsed.name;
  }
  return { tickers, tvByYahoo, nameByYahoo };
}

export function parseManualTickers(text: string): {
  tickers: string[];
  tvByYahoo: Record<string, string>;
  nameByYahoo: Record<string, string>;
} {
  const parts = text
    .split(/[,\n]/)
    .map((s) => s.trim())
    .filter(Boolean);
  return parseListText(parts.join('\n'));
}

const LIST_MODULES: Record<string, number> = {
  'TV-LIST-ETF.txt': require('../../assets/lists/TV-LIST-ETF.txt'),
  'TV-LIST-BIG_CAP_10B.txt': require('../../assets/lists/TV-LIST-BIG_CAP_10B.txt'),
  'TV-LIST-SMALL_CAP_2B-10B.txt': require('../../assets/lists/TV-LIST-SMALL_CAP_2B-10B.txt'),
  'TV-LIST-US-CANADA-FULL.txt': require('../../assets/lists/TV-LIST-US-CANADA-FULL.txt'),
  'STOCK-TICKERS.txt': require('../../assets/lists/STOCK-TICKERS.txt'),
};

async function loadAssetText(moduleId: number): Promise<string> {
  const asset = Asset.fromModule(moduleId);
  await asset.downloadAsync();
  const uri = asset.localUri || asset.uri;
  return FileSystem.readAsStringAsync(uri);
}

export async function loadMergedStocks(): Promise<{
  tickers: string[];
  tvByYahoo: Record<string, string>;
  nameByYahoo: Record<string, string>;
}> {
  const files = [
    'TV-LIST-BIG_CAP_10B.txt',
    'TV-LIST-SMALL_CAP_2B-10B.txt',
    'TV-LIST-US-CANADA-FULL.txt',
    'STOCK-TICKERS.txt',
  ];
  const tickers: string[] = [];
  const tvByYahoo: Record<string, string> = {};
  const nameByYahoo: Record<string, string> = {};
  const seen = new Set<string>();
  for (const f of files) {
    const raw = await loadAssetText(LIST_MODULES[f]);
    const parsed = parseListText(raw);
    for (const t of parsed.tickers) {
      if (seen.has(t)) continue;
      seen.add(t);
      tickers.push(t);
      tvByYahoo[t] = parsed.tvByYahoo[t];
      if (parsed.nameByYahoo[t]) nameByYahoo[t] = parsed.nameByYahoo[t];
    }
  }
  return { tickers, tvByYahoo, nameByYahoo };
}

export async function loadEtfList() {
  const raw = await loadAssetText(LIST_MODULES['TV-LIST-ETF.txt']);
  return parseListText(raw);
}

export async function resolveSourceTickers(
  source: SourceLabel,
  manualText: string,
): Promise<{
  tickers: string[];
  tvByYahoo: Record<string, string>;
  nameByYahoo: Record<string, string>;
}> {
  if (source === 'MANUAL SCAN') return parseManualTickers(manualText);
  if (source === 'ETF') return loadEtfList();
  return loadMergedStocks();
}
