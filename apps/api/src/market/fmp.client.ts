/** Financial Modeling Prep (stable API) client for fundamentals + prices. */
import { Injectable, Logger, ServiceUnavailableException } from '@nestjs/common';

const BASE = 'https://financialmodelingprep.com/stable';

type Json = Record<string, unknown>;

function asArr(v: unknown): Json[] {
  return Array.isArray(v) ? (v as Json[]) : [];
}

function num(v: unknown): number | null {
  if (v == null || v === '') return null;
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}

function str(v: unknown): string | null {
  if (v == null) return null;
  const s = String(v).trim();
  return s || null;
}

/** Yahoo → FMP symbol (class shares BRK-B → BRK.B; Canadian suffixes kept). */
export function yahooToFmpSymbol(yahooTicker: string): string {
  const s = String(yahooTicker || '')
    .trim()
    .toUpperCase();
  if (!s) return s;
  if (/\.(TO|V|NE|CN)$/.test(s)) return s;
  const classShare = s.match(/^([A-Z0-9]+)-([A-Z])$/);
  if (classShare) return `${classShare[1]}.${classShare[2]}`;
  return s.replace(/-/g, '.');
}

function sleep(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

@Injectable()
export class FmpClient {
  private readonly log = new Logger(FmpClient.name);
  private inflight = 0;
  private readonly waiters: Array<() => void> = [];
  /** Keep the shared key under FMP's burst limit — cards + a Fundamentals page share this. */
  private readonly maxConcurrent = 2;

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
}

export const fmpNum = num;
export const fmpStr = str;
