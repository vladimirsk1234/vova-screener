/** Financial Modeling Prep (stable API) client for fundamentals + prices. */
import { Injectable, Logger, ServiceUnavailableException } from '@nestjs/common';
import { fmpSymbolCandidates } from './fmpSymbol';
import {
  encodeAssumptionsForFmp,
  mapCustomDcf,
  sanitizeCustomDcfAssumptions,
  type CustomDcfAssumptions,
  type CustomDcfPayload,
} from './fmpCustomDcf';

export { fmpMappedSymbol, fmpSymbolCandidates, yahooToFmpSymbol } from './fmpSymbol';
export {
  CUSTOM_DCF_ASSUMPTION_KEYS,
  FMP_PERCENT_QUERY_KEYS,
  customDcfCacheKey,
  emptyCustomDcf,
  encodeAssumptionsForFmp,
  fmpYearNum,
  mapCustomDcf,
  sanitizeCustomDcfAssumptions,
  toFmpDecimal,
  type CustomDcfAssumptionKey,
  type CustomDcfAssumptions,
  type CustomDcfPayload,
  type CustomDcfYear,
} from './fmpCustomDcf';

const BASE = 'https://financialmodelingprep.com/stable';

type Json = Record<string, unknown>;

function asArr(v: unknown): Json[] {
  return Array.isArray(v) ? (v as Json[]) : [];
}

function num(v: unknown): number | null {
  if (v == null || v === '') return null;
  if (typeof v === 'string') {
    const trimmed = v.trim();
    if (!trimmed) return null;
    const n = Number(trimmed);
    return Number.isFinite(n) ? n : null;
  }
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}

function str(v: unknown): string | null {
  if (v == null) return null;
  const s = String(v).trim();
  return s || null;
}

function sleep(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

type ForexCacheEntry = { at: number; rates: Map<number, number> };

@Injectable()
export class FmpClient {
  private readonly log = new Logger(FmpClient.name);
  private inflight = 0;
  private readonly waiters: Array<() => void> = [];
  /** Keep the shared key under FMP's burst limit — cards + a Fundamentals page share this. */
  private readonly maxConcurrent = 2;
  private readonly forexCache = new Map<string, ForexCacheEntry>();
  private readonly forexTtlMs = 24 * 60 * 60 * 1000;

  private apiKey(): string {
    const key = process.env.FMP_API_KEY?.trim();
    if (!key) {
      throw new ServiceUnavailableException(
        'FMP_API_KEY is not set. Add your Financial Modeling Prep key to use fundamentals analysis.',
      );
    }
    return key;
  }

  configured(): boolean {
    return Boolean(process.env.FMP_API_KEY?.trim());
  }

  /** Pick the first candidate FMP accepts (cheap profile probe before a full pull). */
  async resolveFmpSymbol(yahooTicker: string): Promise<string> {
    const candidates = fmpSymbolCandidates(yahooTicker);
    if (!candidates.length) return '';
    if (candidates.length === 1) return candidates[0];
    for (const symbol of candidates) {
      try {
        const profile = await this.profile(symbol);
        if (profile.companyName) return symbol;
      } catch {
        /* try next candidate */
      }
      try {
        const income = await this.incomeAnnual(symbol, 2);
        if (income.length) return symbol;
      } catch {
        /* try next candidate */
      }
    }
    return candidates[0];
  }

  private async limit<T>(fn: () => Promise<T>): Promise<T> {
    if (this.inflight >= this.maxConcurrent) {
      await new Promise<void>((resolve) => this.waiters.push(resolve));
    }
    this.inflight += 1;
    try {
      return await fn();
    } finally {
      this.inflight -= 1;
      this.waiters.shift()?.();
    }
  }

  private async get<T = unknown>(path: string, params: Record<string, string | number> = {}): Promise<T> {
    return this.limit(() => this.getOnce<T>(path, params, true));
  }

  private async getOnce<T>(
    path: string,
    params: Record<string, string | number>,
    retry: boolean,
  ): Promise<T> {
    const q = new URLSearchParams({ apikey: this.apiKey() });
    for (const [k, v] of Object.entries(params)) q.set(k, String(v));
    const url = `${BASE}${path}?${q.toString()}`;
    const res = await fetch(url, {
      headers: { Accept: 'application/json' },
    });
    if (!res.ok) {
      const body = await res.text().catch(() => '');
      this.log.warn(`FMP ${path} → ${res.status} ${body.slice(0, 200)}`);
      if (retry && (res.status === 429 || res.status === 503)) {
        await sleep(800);
        return this.getOnce<T>(path, params, false);
      }
      throw new ServiceUnavailableException(`FMP request failed (${res.status}) for ${path}`);
    }
    return (await res.json()) as T;
  }

  async profile(symbol: string) {
    const rows = asArr(await this.get('/profile', { symbol }));
    const p = rows[0] ?? {};
    return {
      symbol: str(p.symbol) ?? symbol,
      companyName: str(p.companyName) ?? str(p.companyname),
      currency: str(p.currency),
      exchange: str(p.exchange) ?? str(p.exchangeFullName),
      industry: str(p.industry),
      sector: str(p.sector),
      description: str(p.description),
      mktCap: num(p.mktCap) ?? num(p.marketCap),
      price: num(p.price),
      beta: num(p.beta),
      lastDiv: num(p.lastDiv),
      image: str(p.image),
      country: str(p.country),
      website: str(p.website),
      isEtf: Boolean(p.isEtf),
      isFund: Boolean(p.isFund),
      isActivelyTrading: p.isActivelyTrading !== false,
    };
  }

  async incomeAnnual(symbol: string, limit = 30) {
    return asArr(await this.get('/income-statement', { symbol, period: 'annual', limit }));
  }

  async incomeQuarterly(symbol: string, limit = 40) {
    try {
      return asArr(await this.get('/income-statement', { symbol, period: 'quarter', limit }));
    } catch {
      return [];
    }
  }

  async cashFlowAnnual(symbol: string, limit = 30) {
    try {
      return asArr(await this.get('/cash-flow-statement', { symbol, period: 'annual', limit }));
    } catch {
      return [];
    }
  }

  async cashFlowQuarterly(symbol: string, limit = 40) {
    try {
      return asArr(await this.get('/cash-flow-statement', { symbol, period: 'quarter', limit }));
    } catch {
      return [];
    }
  }

  async keyMetricsAnnual(symbol: string, limit = 30) {
    try {
      return asArr(await this.get('/key-metrics', { symbol, period: 'annual', limit }));
    } catch {
      return [];
    }
  }

  async ratiosAnnual(symbol: string, limit = 30) {
    try {
      return asArr(await this.get('/ratios', { symbol, period: 'annual', limit }));
    } catch {
      return [];
    }
  }

  async keyMetricsTtm(symbol: string) {
    try {
      const rows = asArr(await this.get('/key-metrics-ttm', { symbol }));
      return rows[0] ?? null;
    } catch {
      return null;
    }
  }

  async ratiosTtm(symbol: string) {
    try {
      const rows = asArr(await this.get('/ratios-ttm', { symbol }));
      return rows[0] ?? null;
    } catch {
      return null;
    }
  }

  async ownerEarnings(symbol: string) {
    try {
      return asArr(await this.get('/owner-earnings', { symbol }));
    } catch {
      return [];
    }
  }

  async dcf(symbol: string) {
    try {
      const rows = asArr(await this.get('/discounted-cash-flow', { symbol }));
      const r = rows[0] ?? {};
      return {
        dcf: num(r.dcf) ?? num(r.equityValuePerShare),
        price: num(r['Stock Price'] ?? r.price),
        date: str(r.date),
      };
    } catch {
      return { dcf: null, price: null, date: null };
    }
  }

  /**
   * Unlevered Custom DCF. Query overrides are optional — omit them and FMP fills
   * historical averages. Not the simple `/discounted-cash-flow` single number.
   */
  async customDcf(symbol: string, assumptions: CustomDcfAssumptions = {}): Promise<CustomDcfPayload> {
    const params: Record<string, string | number> = { symbol };
    const clean = sanitizeCustomDcfAssumptions(assumptions);
    for (const [k, v] of Object.entries(encodeAssumptionsForFmp(clean))) {
      if (v != null && Number.isFinite(v)) params[k] = v;
    }
    const raw = await this.get('/custom-discounted-cash-flow', params);
    return mapCustomDcf(symbol, symbol, raw, clean);
  }

  async financialScores(symbol: string) {
    try {
      const rows = asArr(await this.get('/financial-scores', { symbol }));
      const r = rows[0] ?? {};
      return {
        altmanZScore: num(r.altmanZScore),
        piotroskiScore: num(r.piotroskiScore),
        workingCapital: num(r.workingCapital),
      };
    } catch {
      return { altmanZScore: null, piotroskiScore: null, workingCapital: null };
    }
  }

  async analystEstimates(symbol: string, limit = 10) {
    try {
      return asArr(await this.get('/analyst-estimates', { symbol, period: 'annual', limit }));
    } catch {
      return [];
    }
  }

  async dividends(symbol: string) {
    try {
      return asArr(await this.get('/dividends', { symbol }));
    } catch {
      return [];
    }
  }

  async enterpriseValues(symbol: string, limit = 8) {
    try {
      return asArr(await this.get('/enterprise-values', { symbol, limit }));
    } catch {
      return [];
    }
  }

  /**
   * Per-symbol earnings reports (past + upcoming). Empty on error / missing key.
   * https://financialmodelingprep.com/stable/earnings?symbol=AAPL
   */
  async earnings(symbol: string): Promise<FmpEarningsRow[]> {
    try {
      return asArr(await this.get('/earnings', { symbol })).map((r) => ({
        date: str(r.date),
        epsActual: num(r.epsActual) ?? num(r.eps),
        epsEstimated: num(r.epsEstimated) ?? num(r.estimatedEps),
      }));
    } catch {
      return [];
    }
  }

  /**
   * Last reported diluted EPS on or before `asOf` (annual preferred, else quarterly).
   * Used for History "profitable at entry".
   */
  async epsAsOf(symbol: string, asOf: string): Promise<{ eps: number | null; date: string | null }> {
    const [annual, quarterly] = await Promise.all([
      this.incomeAnnual(symbol, 30),
      this.incomeQuarterly(symbol, 40),
    ]);
    const pick = (rows: Json[]) => {
      let best: { date: string; eps: number } | null = null;
      for (const r of rows) {
        const date = str(r.date);
        const eps = num(r.epsdiluted) ?? num(r.epsDiluted) ?? num(r.eps);
        if (!date || date > asOf || eps == null) continue;
        if (!best || date > best.date) best = { date, eps };
      }
      return best;
    };
    const a = pick(annual);
    if (a) return { eps: a.eps, date: a.date };
    const q = pick(quarterly);
    return { eps: q?.eps ?? null, date: q?.date ?? null };
  }

  /**
   * Year → close price near fiscal year end (from daily EOD history).
   * Uses light endpoint when available; falls back gracefully.
   */
  async yearEndCloses(symbol: string, years: number[]): Promise<Map<number, number>> {
    const out = new Map<number, number>();
    if (!years.length) return out;
    const minY = Math.min(...years);
    const from = `${minY - 1}-01-01`;
    let rows: Json[] = [];
    try {
      rows = asArr(
        await this.get('/historical-price-eod/light', {
          symbol,
          from,
        }),
      );
    } catch {
      try {
        rows = asArr(await this.get('/historical-price-eod/full', { symbol, from }));
      } catch {
        return out;
      }
    }
    // Prefer the last trading day of each calendar year.
    const byYear = new Map<number, { date: string; close: number }>();
    for (const r of rows) {
      const date = str(r.date);
      const close = num(r.close) ?? num(r.price);
      if (!date || close == null) continue;
      const y = Number(date.slice(0, 4));
      if (!Number.isFinite(y)) continue;
      const prev = byYear.get(y);
      if (!prev || date > prev.date) byYear.set(y, { date, close });
    }
    for (const y of years) {
      const hit = byYear.get(y);
      if (hit) out.set(y, hit.close);
    }
    return out;
  }

  /**
   * Year-end "foreign units per 1 USD" (USDCNY ≈ 7). Used to turn filing-currency
   * amounts into USD (or another listing currency via two legs).
   */
  async forexForeignPerUsd(currency: string, years: number[]): Promise<Map<number, number>> {
    const out = new Map<number, number>();
    const cur = String(currency || '')
      .trim()
      .toUpperCase();
    if (cur === 'USD') {
      for (const y of years) out.set(y, 1);
      return out;
    }
    if (!cur || !years.length) return out;

    const cacheKey = cur;
    const hit = this.forexCache.get(cacheKey);
    if (hit && Date.now() - hit.at < this.forexTtlMs) {
      for (const y of years) {
        const v = hit.rates.get(y);
        if (v != null) out.set(y, v);
      }
      if (out.size === years.length) return out;
    }

    const pair = forexPairFor(cur);
    if (pair) {
      try {
        const closes = await this.yearEndCloses(pair.symbol, years);
        const merged = hit?.rates ?? new Map<number, number>();
        for (const y of years) {
          const raw = closes.get(y);
          if (raw == null || !(raw > 0)) continue;
          const oriented = orientForexClose(raw, pair.fallback);
          const value = pair.multiplier * oriented;
          merged.set(y, value);
          out.set(y, value);
        }
        this.forexCache.set(cacheKey, { at: Date.now(), rates: merged });
      } catch (err) {
        this.log.warn(
          `Forex ${pair.symbol} failed: ${err instanceof Error ? err.message : String(err)}`,
        );
      }
    }
    return out;
  }
}

function forexPairFor(currency: string): { symbol: string; fallback: number; multiplier: number } | null {
  if (currency === 'GBX' || currency === 'GBP') {
    return { symbol: 'USDGBP', fallback: 0.79, multiplier: currency === 'GBX' ? 100 : 1 };
  }
  if (currency === 'ZAC' || currency === 'ZAR') {
    return { symbol: 'USDZAR', fallback: 18, multiplier: currency === 'ZAC' ? 100 : 1 };
  }
  const aliases: Record<string, string> = { RMB: 'CNY', CNH: 'CNY' };
  const iso = aliases[currency] ?? currency;
  if (iso === 'USD') return null;
  const fallback = (
    {
      CNY: 7.2,
      TWD: 32,
      HKD: 7.8,
      JPY: 150,
      KRW: 1350,
      INR: 84,
      EUR: 0.92,
      CAD: 1.37,
      AUD: 1.55,
      BRL: 5.5,
      ARS: 1100,
      MXN: 18,
      SGD: 1.34,
      CHF: 0.88,
    } as Record<string, number>
  )[iso];
  return { symbol: `USD${iso}`, fallback: fallback ?? 1, multiplier: 1 };
}

/** FMP sometimes stores the inverse pair. Keep the value nearer the known USD cross. */
function orientForexClose(close: number, fallback: number): number {
  if (!(close > 0) || !(fallback > 0)) return close;
  const inv = 1 / close;
  const dFwd = Math.abs(Math.log(close / fallback));
  const dInv = Math.abs(Math.log(inv / fallback));
  return dInv + 0.15 < dFwd ? inv : close;
}

export type FmpEarningsRow = {
  date: string | null;
  epsActual: number | null;
  epsEstimated: number | null;
};

export const fmpNum = num;
export const fmpStr = str;
