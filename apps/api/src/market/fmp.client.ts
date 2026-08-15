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

  /**
   * Unlevered Custom DCF. Query overrides are optional — omit them and FMP fills
   * historical averages. Not the simple `/discounted-cash-flow` single number.
   */
  async customDcf(symbol: string, assumptions: CustomDcfAssumptions = {}): Promise<CustomDcfPayload> {
    const params: Record<string, string | number> = { symbol };
    const clean = sanitizeCustomDcfAssumptions(assumptions);
    for (const [k, v] of Object.entries(clean)) {
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

export const CUSTOM_DCF_ASSUMPTION_KEYS = [
  'revenueGrowthPct',
  'ebitdaPct',
  'operatingCashFlowPct',
  'capitalExpenditurePct',
  'longTermGrowthRate',
  'taxRate',
  'riskFreeRate',
  'marketRiskPremium',
  'costOfEquity',
  'costOfDebt',
] as const;

export type CustomDcfAssumptionKey = (typeof CUSTOM_DCF_ASSUMPTION_KEYS)[number];
export type CustomDcfAssumptions = Partial<Record<CustomDcfAssumptionKey, number>>;

export type CustomDcfYear = {
  year: number;
  revenue: number | null;
  ebitda: number | null;
  ebit: number | null;
  depreciation: number | null;
  capitalExpenditure: number | null;
  ufcf: number | null;
  pvUfcf: number | null;
};

export type CustomDcfPayload = {
  yahooTicker: string;
  fmpSymbol: string;
  model: 'unlevered';
  price: number | null;
  equityValuePerShare: number | null;
  premiumPct: number | null;
  enterpriseValue: number | null;
  equityValue: number | null;
  netDebt: number | null;
  terminalValue: number | null;
  presentTerminalValue: number | null;
  sumPvUfcf: number | null;
  dilutedShares: number | null;
  wacc: number | null;
  beta: number | null;
  costOfEquity: number | null;
  costOfDebt: number | null;
  afterTaxCostOfDebt: number | null;
  taxRate: number | null;
  riskFreeRate: number | null;
  marketRiskPremium: number | null;
  debtWeighting: number | null;
  equityWeighting: number | null;
  longTermGrowthRate: number | null;
  revenueGrowthPct: number | null;
  ebitdaPct: number | null;
  capitalExpenditurePct: number | null;
  operatingCashFlowPct: number | null;
  years: CustomDcfYear[];
  /** True when WACC − g is under 1pp — Gordon growth is unstable. */
  fragile: boolean;
  /** presentTerminalValue / enterpriseValue × 100. */
  terminalSharePct: number | null;
  asOf: string;
  cached: boolean;
};

/** FMP wants 0.08 for 8%. Accept 8 or 0.08 from query/UI. */
export function toFmpDecimal(n: number): number {
  return Math.abs(n) > 1.5 ? n / 100 : n;
}

export function sanitizeCustomDcfAssumptions(raw: Record<string, unknown>): CustomDcfAssumptions {
  const out: CustomDcfAssumptions = {};
  for (const key of CUSTOM_DCF_ASSUMPTION_KEYS) {
    const n = num(raw[key]);
    if (n == null) continue;
    out[key] = toFmpDecimal(n);
  }
  return out;
}

export function customDcfCacheKey(symbol: string, assumptions: CustomDcfAssumptions): string {
  const parts = CUSTOM_DCF_ASSUMPTION_KEYS.map((k) => {
    const v = assumptions[k];
    return v == null || !Number.isFinite(v) ? '' : `${k}=${v}`;
  });
  return `${symbol.toUpperCase()}|${parts.join('&')}`;
}

export function emptyCustomDcf(yahooTicker: string, fmpSymbol: string): CustomDcfPayload {
  return {
    yahooTicker,
    fmpSymbol,
    model: 'unlevered',
    price: null,
    equityValuePerShare: null,
    premiumPct: null,
    enterpriseValue: null,
    equityValue: null,
    netDebt: null,
    terminalValue: null,
    presentTerminalValue: null,
    sumPvUfcf: null,
    dilutedShares: null,
    wacc: null,
    beta: null,
    costOfEquity: null,
    costOfDebt: null,
    afterTaxCostOfDebt: null,
    taxRate: null,
    riskFreeRate: null,
    marketRiskPremium: null,
    debtWeighting: null,
    equityWeighting: null,
    longTermGrowthRate: null,
    revenueGrowthPct: null,
    ebitdaPct: null,
    capitalExpenditurePct: null,
    operatingCashFlowPct: null,
    years: [],
    fragile: false,
    terminalSharePct: null,
    asOf: new Date().toISOString(),
    cached: false,
  };
}

function asRows(v: unknown): Json[] {
  if (Array.isArray(v)) return v as Json[];
  if (v && typeof v === 'object') {
    const obj = v as Json;
    if (Array.isArray(obj.data)) return obj.data as Json[];
    if (obj.year != null || obj.symbol != null || obj.equityValuePerShare != null || obj.dcf != null) {
      return [obj];
    }
  }
  return [];
}

function pickNum(r: Json, ...keys: string[]): number | null {
  for (const k of keys) {
    const n = num(r[k]);
    if (n != null) return n;
  }
  return null;
}

/** Rates in the FMP payload mix 0.08 and 8. Store as decimals. */
function asDecimal(n: number | null): number | null {
  if (n == null) return null;
  return toFmpDecimal(n);
}

function mapCustomDcf(
  yahooTicker: string,
  fmpSymbol: string,
  raw: unknown,
  requested: CustomDcfAssumptions,
): CustomDcfPayload {
  const rows = asRows(raw)
    .slice()
    .sort((a, b) => (num(a.year) ?? 0) - (num(b.year) ?? 0));
  const empty = emptyCustomDcf(yahooTicker, fmpSymbol);
  if (!rows.length) return empty;

  const last = rows[rows.length - 1] ?? {};
  const first = rows[0] ?? {};
  const wacc = asDecimal(pickNum(last, 'wacc', 'WACC'));
  const years: CustomDcfYear[] = rows.map((r, i) => {
    const year = num(r.year) ?? i + 1;
    const ufcf =
      pickNum(r, 'ufcf', 'unleveredFreeCashFlow', 'freeCashFlow', 'fcf') ?? null;
    const pvGiven = pickNum(r, 'pvUfcf', 'presentValueUfcf', 'pvFreeCashFlow');
    const pvUfcf =
      pvGiven ??
      (ufcf != null && wacc != null && wacc > -0.99
        ? ufcf / Math.pow(1 + wacc, i + 1)
        : null);
    return {
      year,
      revenue: pickNum(r, 'revenue'),
      ebitda: pickNum(r, 'ebitda'),
      ebit: pickNum(r, 'ebit'),
      depreciation: pickNum(r, 'depreciation'),
      capitalExpenditure: pickNum(r, 'capitalExpenditure', 'capex'),
      ufcf,
      pvUfcf,
    };
  });

  const price = pickNum(last, 'price', 'Stock Price') ?? pickNum(first, 'price', 'Stock Price');
  const equityValuePerShare =
    pickNum(last, 'equityValuePerShare', 'dcf', 'equityValuePerShareToday') ??
    pickNum(first, 'equityValuePerShare', 'dcf');
  const enterpriseValue = pickNum(last, 'enterpriseValue') ?? pickNum(first, 'enterpriseValue');
  const presentTerminalValue =
    pickNum(last, 'presentTerminalValue', 'pvTerminalValue') ??
    pickNum(first, 'presentTerminalValue', 'pvTerminalValue');
  const g = asDecimal(
    requested.longTermGrowthRate ?? pickNum(last, 'longTermGrowthRate', 'terminalGrowthRate'),
  );
  const premiumPct =
    price != null && equityValuePerShare != null && equityValuePerShare > 0
      ? ((price - equityValuePerShare) / equityValuePerShare) * 100
      : null;
  const terminalSharePct =
    presentTerminalValue != null && enterpriseValue != null && enterpriseValue !== 0
      ? (presentTerminalValue / enterpriseValue) * 100
      : null;
  const fragile = wacc != null && g != null && wacc - g < 0.01;

  return {
    ...empty,
    price,
    equityValuePerShare,
    premiumPct,
    enterpriseValue,
    equityValue: pickNum(last, 'equityValue') ?? pickNum(first, 'equityValue'),
    netDebt: pickNum(last, 'netDebt') ?? pickNum(first, 'netDebt'),
    terminalValue: pickNum(last, 'terminalValue') ?? pickNum(first, 'terminalValue'),
    presentTerminalValue,
    sumPvUfcf: pickNum(last, 'sumPvUfcf') ?? pickNum(first, 'sumPvUfcf'),
    dilutedShares:
      pickNum(last, 'dilutedSharesOutstanding', 'dilutedShares', 'sharesOutstanding') ??
      pickNum(first, 'dilutedSharesOutstanding', 'dilutedShares', 'sharesOutstanding'),
    wacc,
    beta: pickNum(last, 'beta') ?? pickNum(first, 'beta'),
    costOfEquity: asDecimal(
      requested.costOfEquity ?? pickNum(last, 'costOfEquity') ?? pickNum(first, 'costOfEquity'),
    ),
    costOfDebt: asDecimal(
      requested.costOfDebt ?? pickNum(last, 'costOfDebt') ?? pickNum(first, 'costOfDebt'),
    ),
    afterTaxCostOfDebt: asDecimal(
      pickNum(last, 'afterTaxCostOfDebt') ?? pickNum(first, 'afterTaxCostOfDebt'),
    ),
    taxRate: asDecimal(requested.taxRate ?? pickNum(last, 'taxRate') ?? pickNum(first, 'taxRate')),
    riskFreeRate: asDecimal(
      requested.riskFreeRate ?? pickNum(last, 'riskFreeRate') ?? pickNum(first, 'riskFreeRate'),
    ),
    marketRiskPremium: asDecimal(
      requested.marketRiskPremium ??
        pickNum(last, 'marketRiskPremium') ??
        pickNum(first, 'marketRiskPremium'),
    ),
    debtWeighting: asDecimal(pickNum(last, 'debtWeighting') ?? pickNum(first, 'debtWeighting')),
    equityWeighting: asDecimal(pickNum(last, 'equityWeighting') ?? pickNum(first, 'equityWeighting')),
    longTermGrowthRate: g,
    revenueGrowthPct: asDecimal(
      requested.revenueGrowthPct ??
        pickNum(first, 'revenuePercentage', 'revenueGrowthPct', 'revenueGrowth'),
    ),
    ebitdaPct: asDecimal(
      requested.ebitdaPct ?? pickNum(first, 'ebitdaPercentage', 'ebitdaPct', 'ebitdaMargin'),
    ),
    capitalExpenditurePct: asDecimal(
      requested.capitalExpenditurePct ??
        pickNum(first, 'capitalExpenditurePercentage', 'capitalExpenditurePct'),
    ),
    operatingCashFlowPct: asDecimal(
      requested.operatingCashFlowPct ??
        pickNum(first, 'operatingCashFlowPercentage', 'operatingCashFlowPct'),
    ),
    years,
    fragile,
    terminalSharePct,
  };
}

export const fmpNum = num;
export const fmpStr = str;
