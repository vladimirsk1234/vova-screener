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

@Injectable()
export class FmpClient {
  private readonly log = new Logger(FmpClient.name);

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

  private async get<T = unknown>(path: string, params: Record<string, string | number> = {}): Promise<T> {
    const q = new URLSearchParams({ apikey: this.apiKey() });
    for (const [k, v] of Object.entries(params)) q.set(k, String(v));
    const url = `${BASE}${path}?${q.toString()}`;
    const res = await fetch(url, {
      headers: { Accept: 'application/json' },
    });
    if (!res.ok) {
      const body = await res.text().catch(() => '');
      this.log.warn(`FMP ${path} → ${res.status} ${body.slice(0, 200)}`);
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
      isEtf: Boolean(p.isEtf),
      isActivelyTrading: p.isActivelyTrading !== false,
    };
  }

  async incomeAnnual(symbol: string, limit = 30) {
    return asArr(await this.get('/income-statement', { symbol, period: 'annual', limit }));
  }

  async cashFlowAnnual(symbol: string, limit = 30) {
    return asArr(await this.get('/cash-flow-statement', { symbol, period: 'annual', limit }));
  }

  async keyMetricsAnnual(symbol: string, limit = 30) {
    return asArr(await this.get('/key-metrics', { symbol, period: 'annual', limit }));
  }

  async ratiosAnnual(symbol: string, limit = 30) {
    return asArr(await this.get('/ratios', { symbol, period: 'annual', limit }));
  }

  async keyMetricsTtm(symbol: string) {
    const rows = asArr(await this.get('/key-metrics-ttm', { symbol }));
    return rows[0] ?? null;
  }

  async ratiosTtm(symbol: string) {
    const rows = asArr(await this.get('/ratios-ttm', { symbol }));
    return rows[0] ?? null;
  }

  async ownerEarnings(symbol: string) {
    return asArr(await this.get('/owner-earnings', { symbol }));
  }

  async dcf(symbol: string) {
    const rows = asArr(await this.get('/discounted-cash-flow', { symbol }));
    const r = rows[0] ?? {};
    return {
      dcf: num(r.dcf) ?? num(r.equityValuePerShare),
      price: num(r['Stock Price'] ?? r.price),
      date: str(r.date),
    };
  }

  async financialScores(symbol: string) {
    const rows = asArr(await this.get('/financial-scores', { symbol }));
    const r = rows[0] ?? {};
    return {
      altmanZScore: num(r.altmanZScore),
      piotroskiScore: num(r.piotroskiScore),
      workingCapital: num(r.workingCapital),
    };
  }

  async analystEstimates(symbol: string, limit = 10) {
    try {
      return asArr(await this.get('/analyst-estimates', { symbol, period: 'annual', limit }));
    } catch {
      return [];
    }
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
