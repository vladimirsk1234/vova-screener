/** Assemble Fast Graphs–style fundamentals payload from FMP (prices for Performance via Yahoo). */
import { Injectable, Logger, NotFoundException, ServiceUnavailableException } from '@nestjs/common';
import {
  annualizedPriceReturnPct,
  buildValuationSeries,
  completeFiscalYears,
  yoyChgPct,
  type AnnualFundamentalPoint,
  type ForwardMetricPoint,
  type ValuationMetric,
  type ValuationSeriesPoint,
} from '@vova/engine';
import { InjectModel } from '@nestjs/mongoose';
import type { Model } from 'mongoose';
import { INSTRUMENT_FUNDAMENTALS } from '../db/schemas';
import { BarsService } from '../market/bars.service';
import {
  FmpClient,
  customDcfCacheKey,
  emptyCustomDcf,
  fmpNum,
  fmpStr,
  sanitizeCustomDcfAssumptions,
  yahooToFmpSymbol,
  type CustomDcfPayload,
} from '../market/fmp.client';
import type { FundamentalsFilter } from '../settings/settings.module';
import { UniverseService } from '../universe/universe.service';

const CACHE_TTL_MS = 24 * 60 * 60 * 1000;
const DCF_TTL_MS = 60 * 60 * 1000;
const FAIL_TTL_MS = 30 * 60 * 1000;
const WARM_GAP_MS = 250;
const WARM_QUEUE_CAP = 300;
const METRICS: ValuationMetric[] = ['eps', 'revenue', 'fcf', 'ownerEarnings'];
const CARD_BATCH_LIMIT = 150;
const CARD_CONCURRENCY = 5;

function uniqueTickers(tickers: string[]): string[] {
  const unique: string[] = [];
  const seen = new Set<string>();
  for (const raw of tickers) {
    const t = String(raw || '')
      .trim()
      .toUpperCase();
    if (!t || seen.has(t)) continue;
    seen.add(t);
    unique.push(t);
  }
  return unique;
}

function emptyProfile(symbol: string) {
  return {
    symbol,
    companyName: null as string | null,
    currency: null as string | null,
    exchange: null as string | null,
    industry: null as string | null,
    sector: null as string | null,
    description: null as string | null,
    mktCap: null as number | null,
    price: null as number | null,
    beta: null as number | null,
    lastDiv: null as number | null,
    image: null as string | null,
    country: null as string | null,
    website: null as string | null,
    isEtf: false,
    isFund: false,
    isActivelyTrading: true,
  };
}

type CacheEntry = { at: number; payload: FundamentalsPayload };
type DcfCacheEntry = { at: number; payload: CustomDcfPayload };

/** Slim valuation fields for Results / History signal cards. */
export type CardFundamentals = {
  fairValue: number | null;
  /** (price − fairValue) / fairValue × 100. Null when either side is missing. */
  premiumPct: number | null;
  growthRatePct: number | null;
  blendedPe: number | null;
  ltDebtToCapitalTTM: number | null;
};

export type ValuationSets = {
  undervalued: string[];
  overvalued: string[];
};

type CardCacheEntry = { at: number; metrics: CardFundamentals };

export type EstimateRow = {
  year: number;
  date: string;
  eps: number | null;
  epsChgPct: number | null;
  dividend: number | null;
  analysts: number | null;
  estimated: boolean;
};

export type HorizonReturns = {
  y1: number | null;
  y3: number | null;
  y5: number | null;
  y10: number | null;
};

export type PerformanceYear = {
  year: number;
  tickerClose: number | null;
  spyClose: number | null;
  tickerRetPct: number | null;
  spyRetPct: number | null;
  eps: number | null;
  epsChgPct: number | null;
};

export type FundamentalsPayload = {
  provider: 'fmp';
  yahooTicker: string;
  fmpSymbol: string;
  tvSymbol: string;
  profile: {
    companyName: string;
    currency: string | null;
    exchange: string | null;
    sector: string | null;
    industry: string | null;
    description: string | null;
    mktCap: number | null;
    price: number | null;
    beta: number | null;
    country: string | null;
    website: string | null;
    image: string | null;
  };
  snapshot: {
    peTTM: number | null;
    /** Trailing-twelve-month diluted EPS (anchor fallback when no forward estimate). */
    ttmEps: number | null;
    pbTTM: number | null;
    psTTM: number | null;
    pegTTM: number | null;
    roeTTM: number | null;
    roicTTM: number | null;
    dividendYieldTTM: number | null;
    earningsYieldTTM: number | null;
    blendedPe: number | null;
    fwdPe: number | null;
    fwdEps: number | null;
    tev: number | null;
    ltDebtToCapitalTTM: number | null;
    spCreditRating: null;
    estAnnualRorPct: number | null;
    futurePrice: number | null;
    debtToEquityTTM: number | null;
    currentRatioTTM: number | null;
    profitMarginTTM: number | null;
    operatingMarginTTM: number | null;
    fcfYieldTTM: number | null;
    dcf: number | null;
    dcfPremiumPct: number | null;
    altmanZScore: number | null;
    piotroskiScore: number | null;
  };
  valuation: ReturnType<typeof buildValuationSeries>;
  forecastSeries: ValuationSeriesPoint[];
  estimates: EstimateRow[];
  performance: {
    price: HorizonReturns;
    spy: HorizonReturns;
    eps: HorizonReturns;
    years: PerformanceYear[];
  };
  annual: AnnualFundamentalPoint[];
  incomeTrend: Array<{
    year: number;
    date: string;
    revenue: number | null;
    netIncome: number | null;
    eps: number | null;
    epsChgPct: number | null;
    dividend: number | null;
    operatingCashFlow: number | null;
    freeCashFlow: number | null;
  }>;
  asOf: string;
  cached: boolean;
};

function yearOf(date: string | null): number | null {
  if (!date || date.length < 4) return null;
  const y = Number(date.slice(0, 4));
  return Number.isFinite(y) ? y : null;
}

function byDateKey(rows: Record<string, unknown>[]): Map<string, Record<string, unknown>> {
  const m = new Map<string, Record<string, unknown>>();
  for (const r of rows) {
    const d = fmpStr(r.date) ?? fmpStr(r.calendarYear);
    if (d) m.set(d.slice(0, 10), r);
  }
  return m;
}

function horizonsFrom(
  compute: (years: number) => number | null,
): HorizonReturns {
  return {
    y1: compute(1),
    y3: compute(3),
    y5: compute(5),
    y10: compute(10),
  };
}

/** FMP mixes decimals (0.18) and whole percents (18); normalize to percent points. */
function asPctPoints(n: number | null | undefined): number | null {
  if (n == null || !Number.isFinite(n)) return null;
  return Math.abs(n) <= 1.5 ? n * 100 : n;
}

/**
 * FMP only estimates EPS, so revenue / FCF / owner-earnings views keep the trailing growth rate
 * and the last reported year as their fair value anchor.
 */
function forwardFor(metric: ValuationMetric, estimates: EstimateRow[]): ForwardMetricPoint[] {
  if (metric !== 'eps') return [];
  return estimates.map((e) => ({ year: e.year, metric: e.eps }));
}

/**
 * TTM diluted EPS for the fair-value anchor when analyst estimates are missing.
 * Prefer an explicit per-share TTM field; otherwise invert price / PE_TTM.
 */
function ttmEpsFrom(
  keyTtm: Record<string, unknown> | null | undefined,
  ratiosTtm: Record<string, unknown> | null | undefined,
  price: number | null,
  peTTM: number | null,
): number | null {
  const fromFields =
    fmpNum(keyTtm?.netIncomePerShareTTM) ??
    fmpNum(keyTtm?.epsTTM) ??
    fmpNum(keyTtm?.netIncomePerShare) ??
    fmpNum(ratiosTtm?.netIncomePerShareTTM);
  if (fromFields != null && fromFields > 0) return fromFields;
  if (price != null && peTTM != null && peTTM > 0 && price > 0) {
    const implied = price / peTTM;
    return Number.isFinite(implied) && implied > 0 ? implied : null;
  }
  return null;
}

@Injectable()
export class FundamentalsService {
  private readonly log = new Logger(FundamentalsService.name);
  private readonly cache = new Map<string, CacheEntry>();
  private readonly dcfCache = new Map<string, DcfCacheEntry>();
  private readonly cardCache = new Map<string, CardCacheEntry>();
  private readonly failedAt = new Map<string, number>();
  private readonly warmQueued = new Set<string>();
  private readonly warmQueue: string[] = [];
  private warming = false;

  constructor(
    @InjectModel(INSTRUMENT_FUNDAMENTALS) private readonly store: Model<any>,
    private readonly fmp: FmpClient,
    private readonly universe: UniverseService,
    private readonly bars: BarsService,
  ) {}

  async get(yahooTicker: string, metric: ValuationMetric = 'eps'): Promise<FundamentalsPayload> {
    if (!METRICS.includes(metric)) metric = 'eps';
    const ticker = yahooTicker.toUpperCase();
    const cacheKey = `${ticker}|${metric}`;
    const hit = this.cache.get(cacheKey);
    if (hit && Date.now() - hit.at < CACHE_TTL_MS) {
      return { ...hit.payload, cached: true };
    }

    const stored = await this.loadStored(ticker);
    if (stored) {
      const payload = this.payloadForMetric(stored, metric);
      this.remember(ticker, payload);
      return { ...payload, cached: true };
    }

    const payload = await this.fetchFresh(ticker, metric);
    await this.persist(ticker, payload, 'full');
    return payload;
  }

  /**
   * Unlevered Custom DCF from FMP. In-memory only — never written to the Lynch Mongo snapshot
   * and never called from the scheduled fundamentals refresh.
   */
  async getCustomDcf(
    yahooTicker: string,
    rawAssumptions: Record<string, unknown> = {},
  ): Promise<CustomDcfPayload> {
    const ticker = yahooTicker.toUpperCase();
    const fmpSymbol = yahooToFmpSymbol(ticker);
    const assumptions = sanitizeCustomDcfAssumptions(rawAssumptions);
    const cacheKey = customDcfCacheKey(fmpSymbol, assumptions);
    const hit = this.dcfCache.get(cacheKey);
    if (hit && Date.now() - hit.at < DCF_TTL_MS) {
      return { ...hit.payload, yahooTicker: ticker, cached: true };
    }

    if (!this.fmp.configured()) {
      throw new ServiceUnavailableException(
        'FMP_API_KEY is not set. Add your Financial Modeling Prep key to use fundamentals analysis.',
      );
    }

    try {
      const payload = await this.fmp.customDcf(fmpSymbol, assumptions);
      const next: CustomDcfPayload = { ...payload, yahooTicker: ticker, fmpSymbol, cached: false };
      this.dcfCache.set(cacheKey, { at: Date.now(), payload: next });
      return next;
    } catch (err) {
      if (err instanceof ServiceUnavailableException && String(err.message).includes('FMP_API_KEY')) {
        throw err;
      }
      this.log.warn(
        `Custom DCF failed for ${ticker}: ${err instanceof Error ? err.message : String(err)}`,
      );
      return emptyCustomDcf(ticker, fmpSymbol);
    }
  }

  async epsAsOf(
    yahooTicker: string,
    asOf: string,
  ): Promise<{ eps: number | null; positive: boolean | null; asOf: string; reportDate: string | null }> {
    const fmpSymbol = yahooToFmpSymbol(yahooTicker);
    const { eps, date } = await this.fmp.epsAsOf(fmpSymbol, asOf);
    return {
      eps,
      positive: eps == null ? null : eps > 0,
      asOf,
      reportDate: date,
    };
  }

  /**
   * Batch card metrics. Mongo first; FMP only for names that have never been stored.
   */
  async getCardMetrics(tickers: string[]): Promise<Record<string, CardFundamentals>> {
    const unique = uniqueTickers(tickers).slice(0, CARD_BATCH_LIMIT);
    if (!unique.length) return {};

    const out: Record<string, CardFundamentals> = {};
    const now = Date.now();
    const needStore: string[] = [];

    for (const ticker of unique) {
      const fromFull = this.cardFromFullCache(ticker, now);
      if (fromFull) {
        out[ticker] = fromFull;
        continue;
      }
      const cardHit = this.cardCache.get(ticker);
      if (cardHit && now - cardHit.at < CACHE_TTL_MS) {
        out[ticker] = cardHit.metrics;
        continue;
      }
      needStore.push(ticker);
    }

    if (needStore.length) {
      const docs = await this.store
        .find({ yahooTicker: { $in: needStore } })
        .lean<any[]>()
        .exec();
      for (const doc of docs) {
        const metrics = this.cardFromDoc(doc);
        if (!metrics) continue;
        out[doc.yahooTicker] = metrics;
        this.cardCache.set(doc.yahooTicker, { at: now, metrics });
        if (doc.payload) this.remember(doc.yahooTicker, doc.payload as FundamentalsPayload);
      }
    }

    const needFetch = unique.filter((t) => !out[t]);
    if (!needFetch.length || !this.fmp.configured()) return out;

    for (let i = 0; i < needFetch.length; i += CARD_CONCURRENCY) {
      const chunk = needFetch.slice(i, i + CARD_CONCURRENCY);
      await Promise.all(
        chunk.map(async (ticker) => {
          try {
            const metrics = await this.fetchCardSlim(ticker);
            this.cardCache.set(ticker, { at: Date.now(), metrics });
            await this.persistCard(ticker, metrics);
            out[ticker] = metrics;
          } catch (err) {
            this.failedAt.set(ticker, Date.now());
            this.log.warn(
              `Card metrics failed for ${ticker}: ${err instanceof Error ? err.message : String(err)}`,
            );
          }
        }),
      );
    }

    return out;
  }

  /** Tickers that pass the Settings valuation filter, or null when the filter is off. */
  async tickersForFilter(
    filter: FundamentalsFilter,
    tickers: string[],
    _opts: { warm?: boolean } = {},
  ): Promise<string[] | null> {
    if (filter !== 'undervalued' && filter !== 'overvalued') return null;
    const unique = uniqueTickers(tickers);
    if (!unique.length) return [];
    const premium = filter === 'undervalued' ? { $lt: 0 } : { $gt: 0 };
    const rows = await this.store
      .find({ yahooTicker: { $in: unique }, premiumPct: premium })
      .select('yahooTicker')
      .lean<Array<{ yahooTicker: string }>>()
      .exec();
    return rows.map((r) => r.yahooTicker);
  }

  /** Daily job: /profile + recompute premium from stored annual. No-op without a full payload. */
  async refreshPrice(yahooTicker: string): Promise<boolean> {
    const ticker = yahooTicker.toUpperCase();
    const stored = await this.loadStored(ticker);
    if (!stored) return false;
    const profile = await this.fmp.profile(yahooToFmpSymbol(ticker)).catch(() => null);
    const price = profile?.price;
    if (price == null || !Number.isFinite(price) || price <= 0) return false;
    const next = this.repricePayload(stored, price, profile);
    await this.persist(ticker, next, 'price');
    return true;
  }

  async storedCount(): Promise<number> {
    return this.store.countDocuments().exec();
  }

  /** Weekly / first-fill job: full 13-endpoint fetch. Keeps the old doc on failure. */
  async refreshFull(yahooTicker: string): Promise<boolean> {
    const ticker = yahooTicker.toUpperCase();
    try {
      const payload = await this.fetchFresh(ticker, 'eps');
      await this.persist(ticker, payload, 'full');
      return true;
    } catch (err) {
      this.log.warn(
        `Full fundamentals refresh failed for ${ticker}: ${err instanceof Error ? err.message : String(err)}`,
      );
      return false;
    }
  }

  private async loadStored(ticker: string): Promise<FundamentalsPayload | null> {
    const doc = await this.store.findOne({ yahooTicker: ticker }).lean<any>().exec();
    if (!doc?.payload || typeof doc.payload !== 'object') return null;
    return doc.payload as FundamentalsPayload;
  }

  private remember(ticker: string, payload: FundamentalsPayload) {
    const t = ticker.toUpperCase();
    this.cache.set(`${t}|eps`, { at: Date.now(), payload: { ...payload, cached: false } });
    const metrics = this.metricsFromPayload(payload);
    this.cardCache.set(t, { at: Date.now(), metrics });
  }

  private async persist(
    ticker: string,
    payload: FundamentalsPayload,
    kind: 'full' | 'price',
  ): Promise<void> {
    const t = ticker.toUpperCase();
    const card = this.metricsFromPayload(payload);
    const now = new Date();
    const set: Record<string, unknown> = {
      yahooTicker: t,
      payload: { ...payload, cached: false },
      fairValue: card.fairValue,
      premiumPct: card.premiumPct,
      growthRatePct: card.growthRatePct,
      blendedPe: card.blendedPe,
      ltDebtToCapitalTTM: card.ltDebtToCapitalTTM,
      updatedAt: now,
    };
    if (kind === 'full') set.fetchedAt = now;
    await this.store.updateOne({ yahooTicker: t }, { $set: set }, { upsert: true }).exec();
    this.remember(t, payload);
  }

  private async persistCard(ticker: string, metrics: CardFundamentals): Promise<void> {
    const t = ticker.toUpperCase();
    await this.store
      .updateOne(
        { yahooTicker: t },
        {
          $set: {
            yahooTicker: t,
            fairValue: metrics.fairValue,
            premiumPct: metrics.premiumPct,
            growthRatePct: metrics.growthRatePct,
            blendedPe: metrics.blendedPe,
            ltDebtToCapitalTTM: metrics.ltDebtToCapitalTTM,
            updatedAt: new Date(),
          },
        },
        { upsert: true },
      )
      .exec();
  }

  private cardFromDoc(doc: any): CardFundamentals | null {
    if (doc?.payload && typeof doc.payload === 'object') {
      return this.metricsFromPayload(doc.payload as FundamentalsPayload);
    }
    if (doc?.premiumPct == null && doc?.fairValue == null) return null;
    const num = (v: unknown) => (typeof v === 'number' && Number.isFinite(v) ? v : null);
    return {
      fairValue: num(doc.fairValue),
      premiumPct: num(doc.premiumPct),
      growthRatePct: num(doc.growthRatePct),
      blendedPe: num(doc.blendedPe),
      ltDebtToCapitalTTM: num(doc.ltDebtToCapitalTTM),
    };
  }

  private payloadForMetric(stored: FundamentalsPayload, metric: ValuationMetric): FundamentalsPayload {
    if (metric === 'eps') return stored;
    const valuation = buildValuationSeries(stored.annual, metric, {
      currentPrice: stored.profile.price,
      windowYears: 5,
      forward: forwardFor(metric, stored.estimates),
      ttmMetric: null,
    });
    return {
      ...stored,
      valuation,
      forecastSeries: this.extendForecast(
        valuation.series,
        stored.estimates,
        valuation.summary.fairValueRatio,
      ),
    };
  }

  private repricePayload(
    stored: FundamentalsPayload,
    price: number,
    profile: { companyName?: string | null; mktCap?: number | null; beta?: number | null } | null,
  ): FundamentalsPayload {
    const valuation = buildValuationSeries(stored.annual, 'eps', {
      currentPrice: price,
      windowYears: 5,
      forward: forwardFor('eps', stored.estimates),
      ttmMetric: stored.snapshot.ttmEps,
    });
    const fwdEps = stored.snapshot.fwdEps;
    const fwdPe = fwdEps != null && fwdEps > 0 ? price / fwdEps : null;
    const peForBlend = stored.snapshot.peTTM ?? valuation.summary.currentPe;
    const blendedPe =
      peForBlend != null && fwdPe != null ? (peForBlend + fwdPe) / 2 : peForBlend ?? fwdPe;
    const dcfVal = stored.snapshot.dcf;
    const dcfPremiumPct =
      dcfVal != null && dcfVal > 0 ? ((price - dcfVal) / dcfVal) * 100 : stored.snapshot.dcfPremiumPct;
    const fv = valuation.summary.fairValue;
    const growth = valuation.summary.growthRatePct;
    const divYldPts = asPctPoints(stored.snapshot.dividendYieldTTM) ?? 0;
    const reversion =
      fv != null && fv > 0 && price > 0 ? (Math.pow(fv / price, 1 / 5) - 1) * 100 : 0;
    const estAnnualRorPct =
      growth == null && fv == null ? null : (growth ?? 0) + divYldPts + reversion;
    return {
      ...stored,
      profile: {
        ...stored.profile,
        price,
        companyName: profile?.companyName ?? stored.profile.companyName,
        mktCap: profile?.mktCap ?? stored.profile.mktCap,
        beta: profile?.beta ?? stored.profile.beta,
      },
      snapshot: {
        ...stored.snapshot,
        blendedPe,
        fwdPe,
        dcfPremiumPct,
        estAnnualRorPct,
      },
      valuation,
      forecastSeries: this.extendForecast(
        valuation.series,
        stored.estimates,
        valuation.summary.fairValueRatio,
      ),
      asOf: new Date().toISOString(),
      cached: false,
    };
  }

  /** Cached card metrics only. Missing key → omitted (caller treats as unknown). */
  peekCardMetrics(tickers: string[]): Record<string, CardFundamentals> {
    const out: Record<string, CardFundamentals> = {};
    const now = Date.now();
    for (const ticker of uniqueTickers(tickers)) {
      const fromFull = this.cardFromFullCache(ticker, now);
      if (fromFull) {
        out[ticker] = fromFull;
        continue;
      }
      const cardHit = this.cardCache.get(ticker);
      if (cardHit && now - cardHit.at < CACHE_TTL_MS) out[ticker] = cardHit.metrics;
    }
    return out;
  }

  /** One-at-a-time FMP fill so a filter click cannot stampede /profile. */
  private enqueueWarm(tickers: string[]) {
    const now = Date.now();
    for (const ticker of tickers) {
      if (this.warmQueue.length >= WARM_QUEUE_CAP) break;
      const failed = this.failedAt.get(ticker);
      if (failed && now - failed < FAIL_TTL_MS) continue;
      if (this.warmQueued.has(ticker)) continue;
      this.warmQueued.add(ticker);
      this.warmQueue.push(ticker);
    }
    void this.drainWarm();
  }

  private async drainWarm() {
    if (this.warming) return;
    this.warming = true;
    try {
      while (this.warmQueue.length) {
        const ticker = this.warmQueue.shift();
        if (!ticker) break;
        this.warmQueued.delete(ticker);
        try {
          const got = await this.getCardMetrics([ticker]);
          if (!got[ticker]) this.failedAt.set(ticker, Date.now());
        } catch {
          this.failedAt.set(ticker, Date.now());
        }
        await new Promise((resolve) => setTimeout(resolve, WARM_GAP_MS));
      }
    } finally {
      this.warming = false;
      if (this.warmQueue.length) void this.drainWarm();
    }
  }

  private cardFromFullCache(ticker: string, now: number): CardFundamentals | null {
    const hit = this.cache.get(`${ticker}|eps`);
    if (!hit || now - hit.at >= CACHE_TTL_MS) return null;
    return this.metricsFromPayload(hit.payload);
  }

  private metricsFromPayload(payload: FundamentalsPayload): CardFundamentals {
    return {
      fairValue: payload.valuation.summary.fairValue,
      premiumPct: payload.valuation.summary.premiumPct,
      growthRatePct: payload.valuation.summary.growthRatePct,
      blendedPe: payload.snapshot.blendedPe,
      ltDebtToCapitalTTM: payload.snapshot.ltDebtToCapitalTTM,
    };
  }

  /** Income + TTM ratios + estimates + profile — enough for the four card fields. */
  private async fetchCardSlim(yahooTicker: string): Promise<CardFundamentals> {
    const fmpSymbol = yahooToFmpSymbol(yahooTicker);
    const [profile, income, keyTtm, ratiosTtm, estimatesRaw] = await Promise.all([
      this.fmp.profile(fmpSymbol).catch(() => emptyProfile(fmpSymbol)),
      this.fmp.incomeAnnual(fmpSymbol).catch(() => []),
      this.fmp.keyMetricsTtm(fmpSymbol),
      this.fmp.ratiosTtm(fmpSymbol),
      this.fmp.analystEstimates(fmpSymbol),
    ]);

    const annual: AnnualFundamentalPoint[] = [];
    for (const inc of income) {
      const date = fmpStr(inc.date) ?? fmpStr(inc.calendarYear);
      const y = yearOf(date) ?? Number(fmpStr(inc.calendarYear));
      if (!date || !Number.isFinite(y)) continue;
      const eps =
        fmpNum(inc.epsdiluted) ??
        fmpNum(inc.epsDiluted) ??
        fmpNum(inc.eps);
      annual.push({
        date: date.slice(0, 10),
        year: y,
        price: null,
        eps,
        revenuePerShare: null,
        fcfPerShare: null,
        ownerEarningsPerShare: null,
        pe: null,
        revenue: fmpNum(inc.revenue),
        netIncome: fmpNum(inc.netIncome),
        operatingCashFlow: null,
        freeCashFlow: null,
      });
    }
    annual.sort((a, b) => a.year - b.year);
    const completed = completeFiscalYears(annual);

    const price = profile.price ?? null;

    // Estimates first: they drive the FG-style growth rate and the fair value anchor.
    const lastHistYear = completed[completed.length - 1]?.year ?? 0;
    const estimateParsed = estimatesRaw
      .map((row) => {
        const date = fmpStr(row.date) ?? '';
        const y = yearOf(date) ?? Number(fmpStr(row.calendarYear));
        return {
          year: y,
          eps:
            fmpNum(row.estimatedEpsAvg) ??
            fmpNum(row.estimatedEps) ??
            fmpNum(row.epsAvg),
        };
      })
      .filter((r) => Number.isFinite(r.year) && r.year > lastHistYear)
      .sort((a, b) => a.year - b.year);

    const peTTM =
      fmpNum(ratiosTtm?.priceToEarningsRatioTTM) ??
      fmpNum(ratiosTtm?.peRatioTTM) ??
      fmpNum(keyTtm?.peRatioTTM);
    const ttmEps = ttmEpsFrom(keyTtm, ratiosTtm, price, peTTM);

    const valuation = buildValuationSeries(completed, 'eps', {
      currentPrice: price,
      windowYears: 5,
      forward: estimateParsed.map((e) => ({ year: e.year, metric: e.eps })),
      ttmMetric: ttmEps,
    });

    const ltDebtToCapitalTTM =
      fmpNum(ratiosTtm?.longTermDebtToCapitalRatioTTM) ??
      fmpNum(keyTtm?.longTermDebtToCapitalRatioTTM);

    const fwdEps = estimateParsed[0]?.eps ?? null;
    const fwdPe = price != null && fwdEps != null && fwdEps > 0 ? price / fwdEps : null;
    const peForBlend = peTTM ?? valuation.summary.currentPe;
    const blendedPe =
      peForBlend != null && fwdPe != null ? (peForBlend + fwdPe) / 2 : peForBlend ?? fwdPe;

    return {
      fairValue: valuation.summary.fairValue,
      premiumPct: valuation.summary.premiumPct,
      growthRatePct: valuation.summary.growthRatePct,
      blendedPe,
      ltDebtToCapitalTTM,
    };
  }

  private extendForecast(
    series: ValuationSeriesPoint[],
    estimates: EstimateRow[],
    fairValueRatio: number | null,
  ): ValuationSeriesPoint[] {
    const out: ValuationSeriesPoint[] = series.map((p) => ({ ...p, estimated: false }));
    const lastHistYear = out[out.length - 1]?.year ?? 0;
    for (const est of estimates) {
      if (!est.estimated || est.year <= lastHistYear) continue;
      const eps = est.eps;
      const positive = eps != null && Number.isFinite(eps) && eps > 0;
      const fv = positive && fairValueRatio != null ? eps * fairValueRatio : null;
      out.push({
        date: est.date,
        year: est.year,
        price: null,
        metric: eps,
        earningsPower: fv,
        fairValue: fv,
        normalValue: null,
        pe: null,
        estimated: true,
      });
    }
    return out;
  }

  private async fetchFresh(
    yahooTicker: string,
    metric: ValuationMetric,
  ): Promise<FundamentalsPayload> {
    const fmpSymbol = yahooToFmpSymbol(yahooTicker);
    const instrument = await this.universe.findOne(yahooTicker);

    const [
      profile,
      income,
      cashFlow,
      keyMetrics,
      ratios,
      keyTtm,
      ratiosTtm,
      ownerEarn,
      dcf,
      scores,
      estimatesRaw,
      dividendsRaw,
      evRows,
      tickerBarsRes,
      spyBarsRes,
    ] = await Promise.all([
      this.fmp.profile(fmpSymbol).catch(() => emptyProfile(fmpSymbol)),
      this.fmp.incomeAnnual(fmpSymbol).catch(() => []),
      this.fmp.cashFlowAnnual(fmpSymbol),
      this.fmp.keyMetricsAnnual(fmpSymbol),
      this.fmp.ratiosAnnual(fmpSymbol),
      this.fmp.keyMetricsTtm(fmpSymbol),
      this.fmp.ratiosTtm(fmpSymbol),
      this.fmp.ownerEarnings(fmpSymbol).catch(() => []),
      this.fmp.dcf(fmpSymbol).catch(() => ({ dcf: null, price: null, date: null })),
      this.fmp.financialScores(fmpSymbol).catch(() => ({
        altmanZScore: null,
        piotroskiScore: null,
        workingCapital: null,
      })),
      this.fmp.analystEstimates(fmpSymbol),
      this.fmp.dividends(fmpSymbol),
      this.fmp.enterpriseValues(fmpSymbol),
      this.bars.getBars(yahooTicker, 'Daily', { maxAgeHours: 24 }).catch(() => ({
        bars: null,
        fromCache: false,
        fetchedAt: null,
      })),
      this.bars.getBars('SPY', 'Daily', { maxAgeHours: 24 }).catch(() => ({
        bars: null,
        fromCache: false,
        fetchedAt: null,
      })),
    ]);

    if (!profile.companyName && !income.length) {
      throw new NotFoundException(`No FMP fundamentals for ${fmpSymbol}`);
    }

    const incomeByDate = byDateKey(income);
    const cfByDate = byDateKey(cashFlow);
    const kmByDate = byDateKey(keyMetrics);
    const ratioByDate = byDateKey(ratios);

    const ownerByYear = new Map<number, number>();
    for (const row of ownerEarn) {
      const y = yearOf(fmpStr(row.date));
      const oe = fmpNum(row.ownerEarnings);
      const shares =
        fmpNum(row.averageSharesOutstanding) ??
        fmpNum(row.weightedAverageShsOut) ??
        fmpNum(row.shares);
      if (y == null || oe == null || !shares || shares <= 0) continue;
      ownerByYear.set(y, oe / shares);
    }

    const divByYear = new Map<number, number>();
    for (const row of dividendsRaw) {
      const y = yearOf(fmpStr(row.date));
      const d = fmpNum(row.adjDividend) ?? fmpNum(row.dividend);
      if (y == null || d == null) continue;
      divByYear.set(y, (divByYear.get(y) ?? 0) + d);
    }

    const dates = new Set<string>();
    for (const d of incomeByDate.keys()) dates.add(d);
    for (const d of kmByDate.keys()) dates.add(d);
    const dateList = [...dates].sort();
    const tickerCloses = (tickerBarsRes.bars ?? []).map((b) => ({ date: b.date, close: b.close }));
    const spyCloses = (spyBarsRes.bars ?? []).map((b) => ({ date: b.date, close: b.close }));
    const yearEnd = yearEndMap(tickerCloses);

    let annual: AnnualFundamentalPoint[] = [];
    for (const date of dateList) {
      const y = yearOf(date);
      if (y == null) continue;
      const inc = incomeByDate.get(date) ?? {};
      const cf = cfByDate.get(date) ?? {};
      const km = kmByDate.get(date) ?? {};
      const rt = ratioByDate.get(date) ?? {};

      const eps =
        fmpNum(inc.epsdiluted) ??
        fmpNum(inc.epsDiluted) ??
        fmpNum(inc.eps) ??
        fmpNum(km.netIncomePerShare);
      const revenuePerShare = fmpNum(km.revenuePerShare);
      const fcfPerShare =
        fmpNum(km.freeCashFlowPerShare) ??
        (() => {
          const fcf = fmpNum(cf.freeCashFlow);
          const shares =
            fmpNum(inc.weightedAverageShsOutDil) ??
            fmpNum(inc.weightedAverageShsOut) ??
            fmpNum(km.weightedAverageShsOut);
          if (fcf == null || !shares || shares <= 0) return null;
          return fcf / shares;
        })();
      const price = yearEnd.get(y) ?? null;
      const pe =
        fmpNum(rt.priceToEarningsRatio) ??
        fmpNum(km.peRatio) ??
        (price != null && eps != null && eps > 0 ? price / eps : null);

      annual.push({
        date,
        year: y,
        price,
        eps,
        revenuePerShare,
        fcfPerShare,
        ownerEarningsPerShare: ownerByYear.get(y) ?? null,
        pe,
        revenue: fmpNum(inc.revenue),
        netIncome: fmpNum(inc.netIncome),
        operatingCashFlow: fmpNum(cf.operatingCashFlow),
        freeCashFlow: fmpNum(cf.freeCashFlow),
        dividend: divByYear.get(y) ?? null,
      });
    }

    annual = completeFiscalYears(annual);

    if (!annual.length) {
      this.log.warn(`Empty annual series for ${fmpSymbol}`);
    }

    const price = profile.price ?? annual[annual.length - 1]?.price ?? null;

    // Estimates are parsed before the valuation because they drive the FG-style growth rate and
    // the fair value anchor. Only EPS has forward data — revenue/FCF per share fall back to
    // trailing growth.
    const lastHistYear = annual[annual.length - 1]?.year ?? 0;
    const estimateParsed: EstimateRow[] = estimatesRaw
      .map((row) => {
        const date = fmpStr(row.date) ?? '';
        const y = yearOf(date) ?? Number(fmpStr(row.calendarYear));
        return {
          year: y,
          date: date || (Number.isFinite(y) ? `${y}-12-31` : ''),
          eps:
            fmpNum(row.estimatedEpsAvg) ??
            fmpNum(row.estimatedEps) ??
            fmpNum(row.epsAvg),
          epsChgPct: null as number | null,
          dividend: null as number | null,
          analysts:
            fmpNum(row.numberAnalystEstimatedEps) ??
            fmpNum(row.numberAnalysts) ??
            fmpNum(row.analysts),
          estimated: true,
        };
      })
      .filter((r) => Number.isFinite(r.year) && r.year > lastHistYear)
      .sort((a, b) => a.year - b.year);

    const histEps = annual.map((a) => a.eps);
    const lastHistEps = histEps[histEps.length - 1] ?? null;
    for (let i = 0; i < estimateParsed.length; i++) {
      const prev = i === 0 ? lastHistEps : estimateParsed[i - 1]!.eps;
      estimateParsed[i]!.epsChgPct = yoyChgPct(estimateParsed[i]!.eps, prev);
    }

    const peTTM =
      fmpNum(ratiosTtm?.priceToEarningsRatioTTM) ??
      fmpNum(ratiosTtm?.peRatioTTM) ??
      fmpNum(keyTtm?.peRatioTTM);
    const ttmEps = metric === 'eps' ? ttmEpsFrom(keyTtm, ratiosTtm, price, peTTM) : null;

    const valuation = buildValuationSeries(annual, metric, {
      currentPrice: price,
      windowYears: 5,
      forward: forwardFor(metric, estimateParsed),
      ttmMetric: ttmEps,
    });

    const pbTTM = fmpNum(ratiosTtm?.priceToBookRatioTTM) ?? fmpNum(keyTtm?.pbRatioTTM);
    const psTTM = fmpNum(ratiosTtm?.priceToSalesRatioTTM) ?? fmpNum(keyTtm?.ptbRatioTTM);
    const pegTTM = fmpNum(ratiosTtm?.priceToEarningsGrowthRatioTTM) ?? fmpNum(ratiosTtm?.pegRatioTTM);
    const roeTTM = fmpNum(keyTtm?.returnOnEquityTTM) ?? fmpNum(ratiosTtm?.returnOnEquityTTM);
    const roicTTM = fmpNum(keyTtm?.returnOnInvestedCapitalTTM) ?? fmpNum(keyTtm?.roicTTM);
    const dividendYieldTTM =
      fmpNum(ratiosTtm?.dividendYieldTTM) ?? fmpNum(keyTtm?.dividendYieldTTM);
    const earningsYieldTTM =
      fmpNum(ratiosTtm?.earningsYieldTTM) ??
      fmpNum(keyTtm?.earningsYieldTTM) ??
      (price != null &&
      valuation.summary.latestMetric != null &&
      valuation.summary.latestMetric > 0
        ? valuation.summary.latestMetric / price
        : null);
    const debtToEquityTTM =
      fmpNum(ratiosTtm?.debtEquityRatioTTM) ?? fmpNum(ratiosTtm?.debtToEquityTTM);
    const currentRatioTTM = fmpNum(ratiosTtm?.currentRatioTTM);
    const profitMarginTTM =
      fmpNum(ratiosTtm?.netProfitMarginTTM) ?? fmpNum(keyTtm?.netProfitMarginTTM);
    const operatingMarginTTM =
      fmpNum(ratiosTtm?.operatingProfitMarginTTM) ?? fmpNum(ratiosTtm?.operatingMarginTTM);
    const fcfYieldTTM = fmpNum(keyTtm?.freeCashFlowYieldTTM);
    const ltDebtToCapitalTTM =
      fmpNum(ratiosTtm?.longTermDebtToCapitalRatioTTM) ??
      fmpNum(keyTtm?.longTermDebtToCapitalRatioTTM);
    const tev =
      fmpNum(keyTtm?.enterpriseValueTTM) ??
      fmpNum(evRows[0]?.enterpriseValue) ??
      fmpNum(evRows[0]?.enterpriseValueTTM);

    const fwdEps = estimateParsed[0]?.eps ?? null;
    const fwdPe = price != null && fwdEps != null && fwdEps > 0 ? price / fwdEps : null;
    const peForBlend = peTTM ?? valuation.summary.currentPe;
    const blendedPe =
      peForBlend != null && fwdPe != null ? (peForBlend + fwdPe) / 2 : peForBlend ?? fwdPe;

    const fvRatio = valuation.summary.fairValueRatio;
    // Fair value already sits on the first estimate, so the future price has to read the far end of
    // the forecast — otherwise the two numbers are the same.
    const horizonEps = [...estimateParsed]
      .filter((e) => e.eps != null && Number.isFinite(e.eps) && e.eps > 0)
      .pop()?.eps ?? null;
    const futurePrice =
      horizonEps != null && fvRatio != null ? horizonEps * fvRatio : null;

    const divYldPts = asPctPoints(dividendYieldTTM) ?? 0;
    const growth = valuation.summary.growthRatePct;
    const fv = valuation.summary.fairValue;
    const reversion =
      price != null && fv != null && price > 0 && fv > 0
        ? (Math.pow(fv / price, 1 / 5) - 1) * 100
        : 0;
    const estAnnualRorPct =
      growth == null && fv == null
        ? null
        : (growth ?? 0) + divYldPts + reversion;

    const dcfVal = dcf.dcf;
    const dcfPremiumPct =
      dcfVal != null && price != null && dcfVal > 0 ? ((price - dcfVal) / dcfVal) * 100 : null;

    const incomeTrend = annual
      .slice()
      .reverse()
      .slice(0, 12)
      .map((a, idx, arr) => {
        const older = arr[idx + 1];
        return {
          year: a.year,
          date: a.date,
          revenue: a.revenue,
          netIncome: a.netIncome,
          eps: a.eps,
          epsChgPct: yoyChgPct(a.eps, older?.eps ?? null),
          dividend: a.dividend ?? null,
          operatingCashFlow: a.operatingCashFlow,
          freeCashFlow: a.freeCashFlow,
        };
      });

    const forecastSeries = this.extendForecast(valuation.series, estimateParsed, fvRatio);
    const performance = this.buildPerformance(annual, tickerCloses, spyCloses);

    return {
      provider: 'fmp',
      yahooTicker,
      fmpSymbol,
      tvSymbol: instrument?.tvSymbol ?? yahooTicker,
      profile: {
        companyName: profile.companyName ?? instrument?.companyName ?? yahooTicker,
        currency: profile.currency,
        exchange: profile.exchange,
        sector: profile.sector,
        industry: profile.industry,
        description: profile.description,
        mktCap: profile.mktCap,
        price,
        beta: profile.beta,
        country: profile.country ?? null,
        website: profile.website ?? null,
        image: profile.image ?? null,
      },
      snapshot: {
        peTTM,
        ttmEps,
        pbTTM,
        psTTM,
        pegTTM,
        roeTTM,
        roicTTM,
        dividendYieldTTM,
        earningsYieldTTM,
        blendedPe,
        fwdPe,
        fwdEps,
        tev,
        ltDebtToCapitalTTM,
        spCreditRating: null,
        estAnnualRorPct,
        futurePrice,
        debtToEquityTTM,
        currentRatioTTM,
        profitMarginTTM,
        operatingMarginTTM,
        fcfYieldTTM,
        dcf: dcfVal,
        dcfPremiumPct,
        altmanZScore: scores.altmanZScore,
        piotroskiScore: scores.piotroskiScore,
      },
      valuation,
      forecastSeries,
      estimates: estimateParsed,
      performance,
      annual,
      incomeTrend,
      asOf: new Date().toISOString(),
      cached: false,
    };
  }

  private buildPerformance(
    annual: AnnualFundamentalPoint[],
    tickerBars: Array<{ date: string; close: number }>,
    spyBars: Array<{ date: string; close: number }>,
  ): FundamentalsPayload['performance'] {
    const price = horizonsFrom((y) => annualizedPriceReturnPct(tickerBars, y));
    const spy = horizonsFrom((y) => annualizedPriceReturnPct(spyBars, y));
    const eps = horizonsFrom((y) => this.epsHorizon(annual, y));

    const tickerYear = yearEndMap(tickerBars);
    const spyYear = yearEndMap(spyBars);
    const years = [...new Set([...tickerYear.keys(), ...spyYear.keys(), ...annual.map((a) => a.year)])]
      .sort((a, b) => a - b)
      .slice(-12);
    const epsByYear = new Map(annual.map((a) => [a.year, a.eps]));
    const rows: PerformanceYear[] = years.map((year, i) => {
      const tickerClose = tickerYear.get(year) ?? null;
      const spyClose = spyYear.get(year) ?? null;
      const prevT = i > 0 ? tickerYear.get(years[i - 1]!) : undefined;
      const prevS = i > 0 ? spyYear.get(years[i - 1]!) : undefined;
      const epsVal = epsByYear.get(year) ?? null;
      const prevEps = i > 0 ? (epsByYear.get(years[i - 1]!) ?? null) : null;
      return {
        year,
        tickerClose,
        spyClose,
        tickerRetPct: yoyChgPct(tickerClose, prevT ?? null),
        spyRetPct: yoyChgPct(spyClose, prevS ?? null),
        eps: epsVal,
        epsChgPct: yoyChgPct(epsVal, prevEps),
      };
    });

    return { price, spy, eps, years: rows.reverse() };
  }

  private epsHorizon(annual: AnnualFundamentalPoint[], years: number): number | null {
    const withEps = annual.filter((a) => a.eps != null && a.eps > 0);
    if (withEps.length < 2) return null;
    const last = withEps[withEps.length - 1]!;
    const target = last.year - years;
    let first = withEps[0]!;
    for (const p of withEps) {
      if (p.year <= target) first = p;
    }
    const span = last.year - first.year;
    if (span < Math.max(1, years * 0.75) || first.eps == null || last.eps == null) return null;
    return yoyChgPct(last.eps, first.eps) == null
      ? null
      : ((Math.pow(last.eps / first.eps, 1 / span) - 1) * 100);
  }
}

function yearEndMap(bars: Array<{ date: string; close: number }>): Map<number, number> {
  const m = new Map<number, { date: string; close: number }>();
  for (const b of bars) {
    const y = Number(b.date.slice(0, 4));
    if (!Number.isFinite(y)) continue;
    const prev = m.get(y);
    if (!prev || b.date > prev.date) m.set(y, b);
  }
  return new Map([...m.entries()].map(([y, v]) => [y, v.close]));
}
