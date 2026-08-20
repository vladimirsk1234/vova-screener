/** Assemble Fast Graphs–style fundamentals payload from FMP (prices for Performance via Yahoo). */
import { Injectable, Logger, NotFoundException, ServiceUnavailableException } from '@nestjs/common';
import {
  FUNDAMENTALS_SCALE_VERSION,
  annualizedPriceReturnPct,
  bestValuePremium,
  buildValuationSeries,
  compareValueRows,
  completeFiscalYears,
  impliedPe,
  interestRankOf,
  peInBand,
  rowMatchesStarsFilter,
  scaleDcf,
  scalePerShare,
  scaleTev,
  scoreValueStars,
  seqStructFromBars,
  ttmFromQuarterly,
  yoyChgPct,
  type AnnualFundamentalPoint,
  type ChartFundamentals,
  type ForwardMetricPoint,
  type FundamentalsScale,
  type OhlcSeries,
  type SeqStructStatus,
  type Timeframe,
  type ValuationMetric,
  type ValuationSeriesPoint,
  type ValueInterest,
  type ValueScreenerSort,
  type ValueSortDir,
  type ValueStarsFilter,
} from '@vova/engine';
import { InjectModel } from '@nestjs/mongoose';
import type { Model } from 'mongoose';
import { INSTRUMENT_FUNDAMENTALS, FUNDAMENTALS_REFRESH_RUN } from '../db/schemas';
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
  type FmpEarningsRow,
} from '../market/fmp.client';
import type { FundamentalsFilter } from '../settings/settings.module';
import { UniverseService } from '../universe/universe.service';
import {
  buildScaleForTicker,
  fxToListingByYear,
  hasCurrentScale,
  incomeAnchor,
  mostCommonReportedCurrency,
  scaleAnnualPoint,
} from './listing-scale';

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

export type TaSnapshotMap = {
  daily?: SeqStructStatus | null;
  weekly?: SeqStructStatus | null;
  monthly?: SeqStructStatus | null;
};

export type TickerInterest = {
  yahooTicker: string;
  interest: ValueInterest | null;
  interestRank: number;
};

export type ValueScreenerRow = {
  yahooTicker: string;
  symbol: string;
  tvSymbol: string;
  companyName: string;
  stars: number;
  epsPremiumPct: number | null;
  fcfPremiumPct: number | null;
  dcfPremiumPct: number | null;
  epsFairValue: number | null;
  fcfFairValue: number | null;
  dcfFairValue: number | null;
  bestPremiumPct: number | null;
  growthRatePct: number | null;
  blendedPe: number | null;
  ltDebtToCapitalTTM: number | null;
  interest: ValueInterest | null;
  interestRank: number;
  ta: TaSnapshotMap;
};

export type ValueScreenerCoverage = {
  universe: number;
  stored: number;
  reliable: number;
  complete: number;
};

export type ValueScreenerLastRun = {
  kind: string;
  trigger: string;
  status: string;
  startedAt: string | null;
  finishedAt: string | null;
  total: number;
  done: number;
  ok: number;
  skip: number;
  fail: number;
} | null;

export type ValueScreenerPage = {
  rows: ValueScreenerRow[];
  total: number;
  counts: { all: number; undervalued: number; 0: number; 1: number; 2: number; 3: number };
  coverage: ValueScreenerCoverage;
  lastFullAt: string | null;
  lastRun: ValueScreenerLastRun;
};

type StarFields = {
  epsFairValue: number | null;
  fcfFairValue: number | null;
  dcfFairValue: number | null;
  epsPremiumPct: number | null;
  fcfPremiumPct: number | null;
  dcfPremiumPct: number | null;
  stars: number;
  bestPremiumPct: number | null;
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
  /** Listing-currency / ADR alignment applied before PE15. Missing on pre-fix cache. */
  scale?: FundamentalsScale | null;
  snapshot: {
    peTTM: number | null;
    /** Trailing-twelve-month diluted EPS (anchor fallback when no forward estimate). */
    ttmEps: number | null;
    /** Period-end of the last quarter included in a quarterly-built TTM. */
    ttmAsOf?: string | null;
    /** Trailing-twelve-month FCF per share from the last four cash-flow quarters. */
    ttmFcf?: number | null;
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
    /** YYYY-MM-DD of the next (or still-pending) earnings report. */
    nextEarningsDate: string | null;
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
  /** Reported quarters (listing-scaled). Used to step EPS / FCF charts mid-FY. */
  quarters?: Array<{ date: string; eps: number | null; fcfPerShare?: number | null }>;
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

function quarterlyEpsRows(
  rows: Record<string, unknown>[],
): Array<{ date: string; eps: number | null }> {
  return rows
    .map((row) => ({
      date: (fmpStr(row.date) ?? '').slice(0, 10),
      eps: fmpNum(row.epsdiluted) ?? fmpNum(row.epsDiluted) ?? fmpNum(row.eps),
    }))
    .filter((q) => q.date.length === 10);
}

function quarterlyFundamentalRows(
  incomeQ: Record<string, unknown>[],
  cashQ: Record<string, unknown>[],
): Array<{ date: string; eps: number | null; fcfPerShare: number | null }> {
  const incByDate = byDateKey(incomeQ);
  const cfByDate = byDateKey(cashQ);
  const dates = new Set<string>([...incByDate.keys(), ...cfByDate.keys()]);
  const out: Array<{ date: string; eps: number | null; fcfPerShare: number | null }> = [];
  for (const date of [...dates].sort()) {
    if (date.length !== 10) continue;
    const inc = incByDate.get(date) ?? {};
    const cf = cfByDate.get(date) ?? {};
    const eps =
      fmpNum(inc.epsdiluted) ?? fmpNum(inc.epsDiluted) ?? fmpNum(inc.eps);
    const fcf = fmpNum(cf.freeCashFlow);
    const shares =
      fmpNum(inc.weightedAverageShsOutDil) ??
      fmpNum(inc.weightedAverageShsOut) ??
      fmpNum(cf.weightedAverageShsOutDil) ??
      fmpNum(cf.weightedAverageShsOut);
    const fcfPerShare =
      fcf != null && shares != null && shares > 0 ? fcf / shares : null;
    out.push({ date, eps, fcfPerShare });
  }
  return out;
}

function pickTtmEps(
  quarterly: Array<{ date: string; eps: number | null }>,
  keyTtm: Record<string, unknown> | null | undefined,
  ratiosTtm: Record<string, unknown> | null | undefined,
  price: number | null,
  peTTM: number | null,
): { ttm: number | null; asOf: string | null } {
  const fromQ = ttmFromQuarterly(quarterly);
  if (fromQ.ttm != null && fromQ.ttm > 0) return fromQ;
  return { ttm: ttmEpsFrom(keyTtm, ratiosTtm, price, peTTM), asOf: fromQ.asOf };
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

function todayIso(now = new Date()): string {
  return now.toISOString().slice(0, 10);
}

function formatPeStr(pe: number | null | undefined): string {
  if (pe == null || !Number.isFinite(pe)) return 'N/A';
  return pe.toFixed(2);
}

function formatMcapStr(mcap: number | null | undefined): string {
  if (mcap == null || !Number.isFinite(mcap) || mcap <= 0) return 'N/A';
  if (mcap >= 1e12) return `${Number((mcap / 1e12).toFixed(2))}T`;
  if (mcap >= 1e9) return `${Number((mcap / 1e9).toFixed(2))}B`;
  return `${Number((mcap / 1e6).toFixed(2))}M`;
}

/** Daily % change for the chart watermark. Empty string when bars are missing (not N/A). */
export function formatDailyChgStr(
  close: number | null | undefined,
  prev: number | null | undefined,
): string {
  if (close == null || prev == null || !Number.isFinite(close) || !Number.isFinite(prev) || prev === 0) {
    return '';
  }
  const chg = ((close - prev) / prev) * 100;
  if (!Number.isFinite(chg)) return '';
  const sign = chg >= 0 ? '+' : '';
  return `${sign}${chg.toFixed(2)}%`;
}

/** Nearest report on/after today; else the latest past date still missing an actual. */
export function pickNextEarningsDate(
  rows: Array<Pick<FmpEarningsRow, 'date' | 'epsActual'>>,
  today = todayIso(),
): string | null {
  const future: string[] = [];
  const pendingPast: string[] = [];
  for (const r of rows) {
    const d = r.date;
    if (!d) continue;
    if (d >= today) future.push(d);
    else if (r.epsActual == null) pendingPast.push(d);
  }
  future.sort();
  if (future[0]) return future[0];
  pendingPast.sort();
  return pendingPast.length ? pendingPast[pendingPast.length - 1]! : null;
}

export function formatEarnStr(dateIso: string | null | undefined, today = todayIso()): string {
  if (!dateIso) return 'N/A';
  const t0 = Date.parse(`${today}T00:00:00Z`);
  const t1 = Date.parse(`${dateIso.slice(0, 10)}T00:00:00Z`);
  if (!Number.isFinite(t0) || !Number.isFinite(t1)) return 'N/A';
  const days = Math.round((t1 - t0) / 86_400_000);
  if (days <= 0) return 'Today';
  return `${days}d`;
}

function emptyChartFundamentals(ticker: string): ChartFundamentals {
  return {
    company_name: ticker,
    pe_str: 'N/A',
    earn_str: 'N/A',
    mcap_str: 'N/A',
    daily_chg_str: '',
    description: null,
  };
}

function finiteOrNull(n: number | null | undefined): number | null {
  return n != null && Number.isFinite(n) ? n : null;
}

/** Keep a stored report date only while it is still today or in the future. */
function usableEarnDate(dateIso: string | null | undefined, today = todayIso()): string | null {
  if (!dateIso) return null;
  const d = dateIso.slice(0, 10);
  return d >= today ? d : null;
}

type ChartFundCacheEntry = {
  at: number;
  company_name: string;
  pe: number | null;
  mcap: number | null;
  nextEarningsDate: string | null;
  description: string | null;
};

function finiteNum(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

function parseInterest(value: unknown): ValueInterest | null {
  return value === 'interested' || value === 'not_interested' ? value : null;
}

function starFieldsFromPremia(
  premia: {
    epsFairValue: number | null;
    fcfFairValue: number | null;
    dcfFairValue: number | null;
    epsPremiumPct: number | null;
    fcfPremiumPct: number | null;
    dcfPremiumPct: number | null;
  },
): StarFields {
  const score = scoreValueStars(premia);
  return {
    ...premia,
    stars: score.stars,
    bestPremiumPct: bestValuePremium(premia),
  };
}

function starFieldsFromPayload(payload: FundamentalsPayload): StarFields {
  const reliable = payload.scale?.reliable !== false;
  const epsFairValue = reliable ? finiteNum(payload.valuation?.summary?.fairValue) : null;
  const epsPremiumPct = reliable ? finiteNum(payload.valuation?.summary?.premiumPct) : null;
  let fcfFairValue: number | null = null;
  let fcfPremiumPct: number | null = null;
  if (reliable && Array.isArray(payload.annual) && payload.annual.length) {
    const fcfVal = buildValuationSeries(payload.annual, 'fcf', {
      currentPrice: payload.profile?.price ?? null,
      windowYears: 5,
      forward: [],
      ttmMetric: finiteNum(payload.snapshot?.ttmFcf),
    });
    fcfFairValue = finiteNum(fcfVal.summary.fairValue);
    fcfPremiumPct = finiteNum(fcfVal.summary.premiumPct);
  }
  const dcfFairValue = reliable ? finiteNum(payload.snapshot?.dcf) : null;
  const dcfPremiumPct = reliable ? finiteNum(payload.snapshot?.dcfPremiumPct) : null;
  return starFieldsFromPremia({
    epsFairValue,
    fcfFairValue,
    dcfFairValue,
    epsPremiumPct,
    fcfPremiumPct,
    dcfPremiumPct,
  });
}

function tfKey(tf: Timeframe): 'daily' | 'weekly' | 'monthly' {
  if (tf === 'Weekly') return 'weekly';
  if (tf === 'Monthly') return 'monthly';
  return 'daily';
}

@Injectable()
export class FundamentalsService {
  private readonly log = new Logger(FundamentalsService.name);
  private readonly cache = new Map<string, CacheEntry>();
  private readonly dcfCache = new Map<string, DcfCacheEntry>();
  private readonly cardCache = new Map<string, CardCacheEntry>();
  private readonly chartFundCache = new Map<string, ChartFundCacheEntry>();
  private readonly failedAt = new Map<string, number>();
  private readonly warmQueued = new Set<string>();
  private readonly warmQueue: string[] = [];
  private warming = false;

  constructor(
    @InjectModel(INSTRUMENT_FUNDAMENTALS) private readonly store: Model<any>,
    @InjectModel(FUNDAMENTALS_REFRESH_RUN) private readonly refreshRuns: Model<any>,
    private readonly fmp: FmpClient,
    private readonly universe: UniverseService,
    private readonly bars: BarsService,
  ) {}

  /**
   * Mongo first. Listed tickers never hit FMP on miss (EOD job fills them).
   * Unknown Manual tickers still pull live so a one-off chart works.
   */
  async get(yahooTicker: string, metric: ValuationMetric = 'eps'): Promise<FundamentalsPayload> {
    if (!METRICS.includes(metric)) metric = 'eps';
    const ticker = yahooTicker.toUpperCase();
    const cacheKey = `${ticker}|${metric}`;
    const hit = this.cache.get(cacheKey);
    if (hit && Date.now() - hit.at < CACHE_TTL_MS) {
      return { ...hit.payload, cached: true };
    }

    const stored = await this.loadStored(ticker);
    if (
      stored &&
      hasCurrentScale(stored, FUNDAMENTALS_SCALE_VERSION) &&
      Array.isArray(stored.quarters) &&
      Object.prototype.hasOwnProperty.call(stored.snapshot ?? {}, 'ttmFcf')
    ) {
      const payload = this.payloadForMetric(stored, metric);
      this.remember(ticker, payload);
      return { ...payload, cached: true };
    }

    const listed = await this.universe.isInTrackedUniverse(ticker);
    if (listed) {
      throw new NotFoundException(
        `No fundamentals in Mongo for ${ticker} yet — wait for the EOD refresh`,
      );
    }

    const payload = await this.fetchFresh(ticker, metric);
    await this.persist(ticker, payload, 'full');
    return payload;
  }

  /**
   * Slim PE / mcap / days-to-earnings for the chart watermark.
   * Mongo / memory only — never FMP on the read path.
   */
  async getChartFundamentals(
    yahooTicker: string,
    opts: { close?: number | null; prevDailyClose?: number | null } = {},
  ): Promise<ChartFundamentals> {
    const ticker = String(yahooTicker || '')
      .trim()
      .toUpperCase();
    const dailyChg = formatDailyChgStr(opts.close, opts.prevDailyClose);
    if (!ticker) return { ...emptyChartFundamentals(''), daily_chg_str: dailyChg };

    try {
      const hit = this.chartFundCache.get(ticker);
      if (hit && Date.now() - hit.at < CACHE_TTL_MS) {
        return {
          company_name: hit.company_name,
          pe_str: formatPeStr(hit.pe),
          earn_str: formatEarnStr(hit.nextEarningsDate),
          mcap_str: formatMcapStr(hit.mcap),
          daily_chg_str: dailyChg,
          description: hit.description,
        };
      }

      const stored = await this.loadStored(ticker);
      let pe = finiteOrNull(stored?.snapshot.peTTM);
      const ttmEps = finiteOrNull(stored?.snapshot.ttmEps);
      let mcap = finiteOrNull(stored?.profile.mktCap);
      let nextEarn = usableEarnDate(stored?.snapshot.nextEarningsDate);
      const companyName = stored?.profile.companyName || ticker;
      const description = stored?.profile.description ?? null;

      if (pe == null && opts.close != null && ttmEps != null && ttmEps !== 0) {
        const implied = opts.close / ttmEps;
        if (Number.isFinite(implied)) pe = implied;
      }

      this.chartFundCache.set(ticker, {
        at: Date.now(),
        company_name: companyName,
        pe,
        mcap,
        nextEarningsDate: nextEarn,
        description,
      });

      return {
        company_name: companyName,
        pe_str: formatPeStr(pe),
        earn_str: formatEarnStr(nextEarn),
        mcap_str: formatMcapStr(mcap),
        daily_chg_str: dailyChg,
        description,
      };
    } catch (err) {
      this.log.warn(
        `chart fundamentals ${ticker}: ${err instanceof Error ? err.message : String(err)}`,
      );
      return { ...emptyChartFundamentals(ticker), daily_chg_str: dailyChg };
    }
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
   * Batch card metrics from Mongo / memory only. Never FMP — EOD job fills gaps.
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
      .find({
        yahooTicker: { $in: unique },
        premiumPct: premium,
        scaleVersion: FUNDAMENTALS_SCALE_VERSION,
        valuationReliable: { $ne: false },
      })
      .select('yahooTicker')
      .lean<Array<{ yahooTicker: string }>>()
      .exec();
    return rows.map((r) => r.yahooTicker);
  }

  /** Drop pre-normalization Mongo snapshots so XYF-style unit errors cannot linger. */
  async invalidateUnscaledStore(): Promise<number> {
    const res = await this.store
      .updateMany(
        {
          $or: [
            { scaleVersion: { $exists: false } },
            { scaleVersion: { $ne: FUNDAMENTALS_SCALE_VERSION } },
          ],
        },
        {
          $unset: {
            payload: 1,
            fairValue: 1,
            premiumPct: 1,
            growthRatePct: 1,
            blendedPe: 1,
            ltDebtToCapitalTTM: 1,
            epsFairValue: 1,
            fcfFairValue: 1,
            dcfFairValue: 1,
            epsPremiumPct: 1,
            fcfPremiumPct: 1,
            dcfPremiumPct: 1,
            stars: 1,
            bestPremiumPct: 1,
          },
          $set: {
            updatedAt: new Date(),
            // Stamp current scale so the next boot does not wipe the same docs again.
            scaleVersion: FUNDAMENTALS_SCALE_VERSION,
          },
        },
      )
      .exec();
    this.cache.clear();
    this.cardCache.clear();
    this.chartFundCache.clear();
    const n = res.modifiedCount ?? 0;
    if (n) this.log.log(`Cleared ${n} unscaled instrumentFundamentals snapshots`);
    return n;
  }

  /** Daily job: /profile + recompute premium from stored annual. No-op without a full payload. */
  async refreshPrice(yahooTicker: string): Promise<boolean> {
    const ticker = yahooTicker.toUpperCase();
    const stored = await this.loadStored(ticker);
    if (!stored || !hasCurrentScale(stored, FUNDAMENTALS_SCALE_VERSION)) return false;
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

  /** STOCK-TICKERS coverage for Value: stored / reliable / complete (full payload). */
  async coverageStats(): Promise<ValueScreenerCoverage> {
    const stocks = await this.universe.listStockEntries();
    const tickers = stocks.map((s) => s.yahooTicker);
    const universe = tickers.length;
    if (!universe) return { universe: 0, stored: 0, reliable: 0, complete: 0 };

    const docs = await this.store
      .find({ yahooTicker: { $in: tickers } })
      .select('yahooTicker scaleVersion valuationReliable payload')
      .lean<
        Array<{
          yahooTicker: string;
          scaleVersion?: number;
          valuationReliable?: boolean;
          payload?: unknown;
        }>
      >()
      .exec();

    let stored = 0;
    let reliable = 0;
    let complete = 0;
    for (const doc of docs) {
      stored += 1;
      const scaled = doc.scaleVersion === FUNDAMENTALS_SCALE_VERSION;
      const ok = scaled && doc.valuationReliable !== false;
      if (ok) reliable += 1;
      if (ok && doc.payload && typeof doc.payload === 'object') complete += 1;
    }
    return { universe, stored, reliable, complete };
  }

  /** Tickers among `tickers` that already have a full scaled payload. */
  async completeTickerSet(tickers: string[]): Promise<Set<string>> {
    const out = new Set<string>();
    if (!tickers.length) return out;
    const docs = await this.store
      .find({
        yahooTicker: { $in: tickers },
        scaleVersion: FUNDAMENTALS_SCALE_VERSION,
        valuationReliable: { $ne: false },
        payload: { $exists: true, $ne: null },
      })
      .select('yahooTicker')
      .lean<Array<{ yahooTicker: string }>>()
      .exec();
    for (const d of docs) out.add(String(d.yahooTicker).toUpperCase());
    return out;
  }

  async latestRefreshRun(): Promise<ValueScreenerLastRun> {
    const doc = await this.refreshRuns
      .findOne()
      .sort({ startedAt: -1 })
      .lean<any>()
      .exec();
    if (!doc) return null;
    return {
      kind: String(doc.kind ?? 'full'),
      trigger: String(doc.trigger ?? 'cron'),
      status: String(doc.status ?? 'completed'),
      startedAt: doc.startedAt ? new Date(doc.startedAt).toISOString() : null,
      finishedAt: doc.finishedAt ? new Date(doc.finishedAt).toISOString() : null,
      total: Number(doc.total) || 0,
      done: Number(doc.done) || 0,
      ok: Number(doc.ok) || 0,
      skip: Number(doc.skip) || 0,
      fail: Number(doc.fail) || 0,
    };
  }

  async lastFullAtIso(): Promise<string | null> {
    const doc = await this.refreshRuns
      .findOne({ status: 'completed' })
      .sort({ finishedAt: -1 })
      .select('finishedAt')
      .lean<{ finishedAt?: Date }>()
      .exec();
    return doc?.finishedAt ? new Date(doc.finishedAt).toISOString() : null;
  }

  /**
   * Recompute EPS/FCF/DCF premiums and N/3 stars from stored payloads.
   * Needed once for docs written before those fields existed.
   */
  async backfillStarFields(): Promise<number> {
    const docs = await this.store
      .find({ payload: { $exists: true, $ne: null }, stars: { $exists: false } })
      .select('yahooTicker payload')
      .lean<Array<{ yahooTicker: string; payload: FundamentalsPayload }>>()
      .exec();
    if (!docs.length) return 0;
    const ops = [];
    for (const doc of docs) {
      if (!doc.payload || typeof doc.payload !== 'object') continue;
      if (!hasCurrentScale(doc.payload, FUNDAMENTALS_SCALE_VERSION)) continue;
      const stars = starFieldsFromPayload(doc.payload);
      ops.push({
        updateOne: {
          filter: { yahooTicker: doc.yahooTicker },
          update: { $set: { ...stars, updatedAt: new Date() } },
        },
      });
    }
    if (!ops.length) return 0;
    const res = await this.store.bulkWrite(ops, { ordered: false });
    this.log.log(`Backfilled Value stars on ${res.modifiedCount ?? ops.length} fundamentals docs`);
    return res.modifiedCount ?? ops.length;
  }

  async listScreener(opts: {
    stars?: ValueStarsFilter;
    sort?: ValueScreenerSort;
    dir?: ValueSortDir;
    limit?: number;
    offset?: number;
  }): Promise<ValueScreenerPage> {
    const filter: ValueStarsFilter = opts.stars ?? 'undervalued';
    const sort: ValueScreenerSort = opts.sort ?? 'stars';
    const dir: ValueSortDir = opts.dir ?? 'desc';
    const limit = Math.min(Math.max(opts.limit ?? 100, 1), 200);
    const offset = Math.max(opts.offset ?? 0, 0);

    await this.backfillStarFields().catch((err) => {
      this.log.warn(
        `Value stars backfill skipped: ${err instanceof Error ? err.message : String(err)}`,
      );
    });

    const stocks = await this.universe.listStockEntries();
    const byYahoo = new Map(stocks.map((s) => [s.yahooTicker, s]));
    const tickers = stocks.map((s) => s.yahooTicker);
    if (!tickers.length) {
      return {
        rows: [],
        total: 0,
        counts: { all: 0, undervalued: 0, 0: 0, 1: 0, 2: 0, 3: 0 },
        coverage: { universe: 0, stored: 0, reliable: 0, complete: 0 },
        lastFullAt: null,
        lastRun: null,
      };
    }

    const docs = await this.store
      .find({
        yahooTicker: { $in: tickers },
        scaleVersion: FUNDAMENTALS_SCALE_VERSION,
        valuationReliable: { $ne: false },
      })
      .select(
        'yahooTicker fairValue premiumPct growthRatePct blendedPe ltDebtToCapitalTTM epsFairValue fcfFairValue dcfFairValue epsPremiumPct fcfPremiumPct dcfPremiumPct stars bestPremiumPct interest interestRank taSnapshot',
      )
      .lean<any[]>()
      .exec();

    const scored: ValueScreenerRow[] = [];
    const counts = { all: 0, undervalued: 0, 0: 0, 1: 0, 2: 0, 3: 0 };

    for (const doc of docs) {
      const listing = byYahoo.get(String(doc.yahooTicker || '').toUpperCase());
      if (!listing) continue;
      const epsPremiumPct = finiteNum(doc.epsPremiumPct) ?? finiteNum(doc.premiumPct);
      const fcfPremiumPct = finiteNum(doc.fcfPremiumPct);
      const dcfPremiumPct = finiteNum(doc.dcfPremiumPct);
      const epsFairValue = finiteNum(doc.epsFairValue) ?? finiteNum(doc.fairValue);
      const fcfFairValue = finiteNum(doc.fcfFairValue);
      const dcfFairValue = finiteNum(doc.dcfFairValue);
      const computed = starFieldsFromPremia({
        epsFairValue,
        fcfFairValue,
        dcfFairValue,
        epsPremiumPct,
        fcfPremiumPct,
        dcfPremiumPct,
      });
      const starCount = finiteNum(doc.stars) ?? computed.stars;
      const bestPremiumPct = finiteNum(doc.bestPremiumPct) ?? computed.bestPremiumPct;
      counts.all += 1;
      if (starCount >= 1) counts.undervalued += 1;
      if (starCount === 0) counts[0] += 1;
      if (starCount === 1) counts[1] += 1;
      if (starCount === 2) counts[2] += 1;
      if (starCount === 3) counts[3] += 1;
      if (!rowMatchesStarsFilter(starCount, filter)) continue;

      const interest = parseInterest(doc.interest);
      const ta = (doc.taSnapshot ?? {}) as TaSnapshotMap;
      scored.push({
        yahooTicker: listing.yahooTicker,
        symbol: listing.symbol,
        tvSymbol: listing.tvSymbol,
        companyName: listing.companyName,
        stars: starCount,
        epsPremiumPct,
        fcfPremiumPct,
        dcfPremiumPct,
        epsFairValue,
        fcfFairValue,
        dcfFairValue,
        bestPremiumPct,
        growthRatePct: finiteNum(doc.growthRatePct),
        blendedPe: finiteNum(doc.blendedPe),
        ltDebtToCapitalTTM: finiteNum(doc.ltDebtToCapitalTTM),
        interest,
        interestRank: finiteNum(doc.interestRank) ?? interestRankOf(interest),
        ta: {
          daily: ta.daily ?? null,
          weekly: ta.weekly ?? null,
          monthly: ta.monthly ?? null,
        },
      });
    }

    scored.sort((a, b) => compareValueRows(a, b, sort, dir));
    const rows = scored.slice(offset, offset + limit);
    await this.fillTaForRows(rows);
    const [coverage, lastRun, lastFullAt] = await Promise.all([
      this.coverageStats(),
      this.latestRefreshRun(),
      this.lastFullAtIso(),
    ]);
    return { rows, total: scored.length, counts, coverage, lastFullAt, lastRun };
  }

  async getTickerInterest(yahooTicker: string): Promise<TickerInterest> {
    const t = yahooTicker.trim().toUpperCase();
    if (!t) throw new NotFoundException('ticker required');
    const doc = await this.store
      .findOne({ yahooTicker: t })
      .select('yahooTicker interest interestRank')
      .lean<any>()
      .exec();
    const interest = parseInterest(doc?.interest);
    return {
      yahooTicker: t,
      interest,
      interestRank: finiteNum(doc?.interestRank) ?? interestRankOf(interest),
    };
  }

  async setTickerInterest(yahooTicker: string, interest: ValueInterest | null): Promise<TickerInterest> {
    const t = yahooTicker.trim().toUpperCase();
    if (!t) throw new NotFoundException('ticker required');
    const rank = interestRankOf(interest);
    await this.store
      .updateOne(
        { yahooTicker: t },
        {
          $set: {
            yahooTicker: t,
            interest,
            interestRank: rank,
            interestAt: interest ? new Date() : null,
            updatedAt: new Date(),
          },
        },
        { upsert: true },
      )
      .exec();
    return { yahooTicker: t, interest, interestRank: rank };
  }

  /** Merge one timeframe's Seq/Struct into instrumentFundamentals.taSnapshot. */
  async mergeTaSnapshot(
    yahooTicker: string,
    tf: Timeframe,
    status: SeqStructStatus,
  ): Promise<void> {
    const t = yahooTicker.toUpperCase();
    const key = tfKey(tf);
    await this.store
      .updateOne(
        { yahooTicker: t },
        {
          $set: {
            yahooTicker: t,
            [`taSnapshot.${key}`]: status,
            'taSnapshot.updatedAt': new Date(),
            updatedAt: new Date(),
          },
        },
        { upsert: true },
      )
      .exec();
  }

  async mergeTaSnapshots(
    items: Array<{ yahooTicker: string; tf: Timeframe; status: SeqStructStatus }>,
  ): Promise<void> {
    if (!items.length) return;
    const ops = items.map((item) => ({
      updateOne: {
        filter: { yahooTicker: item.yahooTicker.toUpperCase() },
        update: {
          $set: {
            yahooTicker: item.yahooTicker.toUpperCase(),
            [`taSnapshot.${tfKey(item.tf)}`]: item.status,
            'taSnapshot.updatedAt': new Date(),
            updatedAt: new Date(),
          },
        },
        upsert: true,
      },
    }));
    await this.store.bulkWrite(ops, { ordered: false });
  }

  async persistTaFromBars(yahooTicker: string, tf: Timeframe, bars: OhlcSeries | null): Promise<void> {
    if (!bars?.length) return;
    const status = seqStructFromBars(bars, tf);
    if (!status) return;
    await this.mergeTaSnapshot(yahooTicker, tf, status);
  }

  private async fillTaForRows(rows: ValueScreenerRow[]): Promise<void> {
    const tfs: Timeframe[] = ['Daily', 'Weekly', 'Monthly'];
    const need = rows.filter(
      (row) => !row.ta.daily || !row.ta.weekly || !row.ta.monthly,
    );
    if (!need.length) return;
    const cached = await this.bars.getCachedMany(
      need.map((r) => r.yahooTicker),
      tfs,
    );
    const writes: Array<{ yahooTicker: string; tf: Timeframe; status: SeqStructStatus }> = [];
    for (const row of need) {
      for (const tf of tfs) {
        const key = tfKey(tf);
        if (row.ta[key]) continue;
        const bars = cached.get(`${row.yahooTicker}|${tf}`);
        if (!bars) continue;
        const status = seqStructFromBars(bars, tf);
        if (!status) continue;
        row.ta[key] = status;
        writes.push({ yahooTicker: row.yahooTicker, tf, status });
      }
    }
    if (writes.length) {
      await this.mergeTaSnapshots(writes).catch((err) => {
        this.log.warn(
          `TA snapshot persist failed: ${err instanceof Error ? err.message : String(err)}`,
        );
      });
    }
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
    const payload = doc.payload as FundamentalsPayload;
    if (!hasCurrentScale(payload, FUNDAMENTALS_SCALE_VERSION)) return null;
    return payload;
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
    const stars = starFieldsFromPayload(payload);
    const now = new Date();
    const set: Record<string, unknown> = {
      yahooTicker: t,
      payload: { ...payload, cached: false },
      fairValue: card.fairValue,
      premiumPct: card.premiumPct,
      growthRatePct: card.growthRatePct,
      blendedPe: card.blendedPe,
      ltDebtToCapitalTTM: card.ltDebtToCapitalTTM,
      ...stars,
      updatedAt: now,
    };
    if (kind === 'full') set.fetchedAt = now;
    set.scaleVersion = payload.scale?.version ?? FUNDAMENTALS_SCALE_VERSION;
    set.valuationReliable = payload.scale?.reliable !== false;
    await this.store.updateOne({ yahooTicker: t }, { $set: set }, { upsert: true }).exec();
    this.remember(t, payload);
  }

  private async persistCard(
    ticker: string,
    metrics: CardFundamentals,
    scale?: FundamentalsScale | null,
  ): Promise<void> {
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
            epsFairValue: metrics.fairValue,
            epsPremiumPct: metrics.premiumPct,
            scaleVersion: scale?.version ?? FUNDAMENTALS_SCALE_VERSION,
            valuationReliable: scale?.reliable !== false,
            updatedAt: new Date(),
          },
        },
        { upsert: true },
      )
      .exec();
  }

  private cardFromDoc(doc: any): CardFundamentals | null {
    if (doc?.scaleVersion != null && doc.scaleVersion !== FUNDAMENTALS_SCALE_VERSION) return null;
    const payload = doc?.payload && typeof doc.payload === 'object' ? (doc.payload as FundamentalsPayload) : null;
    if (payload && hasCurrentScale(payload, FUNDAMENTALS_SCALE_VERSION)) {
      return this.metricsFromPayload(payload);
    }
    if (doc?.scaleVersion === FUNDAMENTALS_SCALE_VERSION) {
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
    return null;
  }

  private payloadForMetric(stored: FundamentalsPayload, metric: ValuationMetric): FundamentalsPayload {
    if (metric === 'eps') return stored;
    const valuation = buildValuationSeries(stored.annual, metric, {
      currentPrice: stored.profile.price,
      windowYears: 5,
      forward: forwardFor(metric, stored.estimates),
      ttmMetric: metric === 'fcf' ? stored.snapshot.ttmFcf ?? null : null,
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
        nextEarningsDate: stored.snapshot.nextEarningsDate ?? null,
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
    const reliable = payload.scale?.reliable !== false;
    return {
      fairValue: reliable ? payload.valuation.summary.fairValue : null,
      premiumPct: reliable ? payload.valuation.summary.premiumPct : null,
      growthRatePct: payload.valuation.summary.growthRatePct,
      blendedPe: payload.snapshot.blendedPe,
      ltDebtToCapitalTTM: payload.snapshot.ltDebtToCapitalTTM,
    };
  }

  /** Income + TTM ratios + estimates + profile — enough for the four card fields. */
  private async fetchCardSlim(
    yahooTicker: string,
  ): Promise<{ metrics: CardFundamentals; scale: FundamentalsScale }> {
    const fmpSymbol = yahooToFmpSymbol(yahooTicker);
    const [profile, income, incomeQuarterly, keyTtm, ratiosTtm, estimatesRaw] = await Promise.all([
      this.fmp.profile(fmpSymbol).catch(() => emptyProfile(fmpSymbol)),
      this.fmp.incomeAnnual(fmpSymbol).catch(() => []),
      this.fmp.incomeQuarterly(fmpSymbol),
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
        dilutedShares:
          fmpNum(inc.weightedAverageShsOutDil) ?? fmpNum(inc.weightedAverageShsOut),
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

    const peTTMRaw =
      fmpNum(ratiosTtm?.priceToEarningsRatioTTM) ??
      fmpNum(ratiosTtm?.peRatioTTM) ??
      fmpNum(keyTtm?.peRatioTTM);
    const ttmPick = pickTtmEps(quarterlyEpsRows(incomeQuarterly), keyTtm, ratiosTtm, price, peTTMRaw);
    const ttmEpsRaw = ttmPick.ttm;
    const aligned = await this.alignToListing({
      yahooTicker,
      listingCurrency: profile.currency,
      price,
      income,
      annual: completed,
      estimateEps: estimateParsed,
      ttmEps: ttmEpsRaw,
      peTTM: peTTMRaw,
      tev: null,
      dcf: null,
      mktCap: profile.mktCap,
    });

    const valuation = buildValuationSeries(aligned.annual, 'eps', {
      currentPrice: price,
      windowYears: 5,
      forward: aligned.estimateEps.map((e) => ({ year: e.year, metric: e.eps })),
      ttmMetric: aligned.ttmEps,
    });

    const ltDebtToCapitalTTM =
      fmpNum(ratiosTtm?.longTermDebtToCapitalRatioTTM) ??
      fmpNum(keyTtm?.longTermDebtToCapitalRatioTTM);

    const fwdEps = aligned.estimateEps[0]?.eps ?? null;
    const fwdPe = price != null && fwdEps != null && fwdEps > 0 ? price / fwdEps : null;
    const peForBlend = aligned.peTTM ?? valuation.summary.currentPe;
    const blendedPe =
      peForBlend != null && fwdPe != null ? (peForBlend + fwdPe) / 2 : peForBlend ?? fwdPe;

    const reliable = aligned.scale.reliable !== false;
    return {
      metrics: {
        fairValue: reliable ? valuation.summary.fairValue : null,
        premiumPct: reliable ? valuation.summary.premiumPct : null,
        growthRatePct: valuation.summary.growthRatePct,
        blendedPe,
        ltDebtToCapitalTTM,
      },
      scale: aligned.scale,
    };
  }

  private async alignToListing(opts: {
    yahooTicker: string;
    listingCurrency: string | null;
    price: number | null;
    income: Record<string, unknown>[];
    annual: AnnualFundamentalPoint[];
    estimateEps: Array<{ year: number; eps: number | null }>;
    ttmEps: number | null;
    peTTM: number | null;
    tev: number | null;
    dcf: number | null;
    mktCap: number | null;
  }): Promise<{
    annual: AnnualFundamentalPoint[];
    estimateEps: Array<{ year: number; eps: number | null }>;
    ttmEps: number | null;
    peTTM: number | null;
    tev: number | null;
    dcf: number | null;
    mktCap: number | null;
    scale: FundamentalsScale;
  }> {
    const reported =
      mostCommonReportedCurrency(opts.income) ?? incomeAnchor(opts.income)?.reportedCurrency ?? null;
    const years = [...new Set(opts.annual.map((p) => p.year))].sort((a, b) => a - b);
    const fxByYear = await fxToListingByYear(
      this.fmp,
      reported,
      opts.listingCurrency,
      years.length ? years : [new Date().getUTCFullYear()],
    );
    const lastYear = years[years.length - 1];
    const latestFx = (lastYear != null ? fxByYear.get(lastYear) : undefined) ?? 1;
    const last = opts.annual[opts.annual.length - 1];
    const anchor = incomeAnchor(opts.income);
    const scale = buildScaleForTicker({
      ticker: opts.yahooTicker,
      listingCurrency: opts.listingCurrency,
      reportedCurrency: reported ?? anchor?.reportedCurrency ?? null,
      fxToListing: latestFx,
      netIncome: last?.netIncome ?? anchor?.netIncome ?? null,
      fmpEps: last?.eps ?? anchor?.fmpEps ?? opts.ttmEps,
      dilutedShares: last?.dilutedShares ?? anchor?.dilutedShares ?? null,
      price: opts.price,
      peTtm: opts.peTTM,
    });

    const annual = opts.annual.map((p) =>
      scaleAnnualPoint(p, scale, fxByYear.get(p.year), opts.peTTM),
    );
    const estimateEps = opts.estimateEps.map((e) => ({
      ...e,
      eps: scalePerShare(e.eps, scale),
    }));
    let ttmEps = scalePerShare(opts.ttmEps, scale);
    const lastEps = annual[annual.length - 1]?.eps ?? null;
    if (!peInBand(impliedPe(opts.price, ttmEps)) && peInBand(impliedPe(opts.price, lastEps))) {
      ttmEps = lastEps;
    }
    const scaledUnits = scale.fxToListing !== 1 || scale.perShareFactor !== 1;
    let peTTM = impliedPe(opts.price, ttmEps);
    if (!scaledUnits && peInBand(opts.peTTM)) peTTM = opts.peTTM;
    if (!peInBand(peTTM)) peTTM = impliedPe(opts.price, lastEps);

    const latestShares = last?.dilutedShares ?? anchor?.dilutedShares ?? null;
    let mktCap = opts.mktCap;
    if (opts.price != null && opts.price > 0 && latestShares != null && latestShares > 0) {
      const ads = scale.adrRatio > 1 ? latestShares / scale.adrRatio : latestShares;
      const implied = opts.price * ads;
      if (
        implied > 0 &&
        (mktCap == null || mktCap <= 0 || mktCap / implied < 0.4 || mktCap / implied > 2.5)
      ) {
        mktCap = implied;
      }
    }

    const tev = scaleTev(opts.tev, mktCap, scale);
    const dcf = scaleDcf(opts.dcf, scale, opts.price);
    if (!peInBand(impliedPe(opts.price, ttmEps ?? lastEps))) {
      scale.reliable = false;
    }

    return { annual, estimateEps, ttmEps, peTTM, tev, dcf, mktCap, scale };
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
      incomeQuarterly,
      cashFlowQuarterly,
      dividendsRaw,
      evRows,
      tickerBarsRes,
      spyBarsRes,
      earningsRows,
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
      this.fmp.incomeQuarterly(fmpSymbol),
      this.fmp.cashFlowQuarterly(fmpSymbol),
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
      this.fmp.earnings(fmpSymbol),
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
        dilutedShares:
          fmpNum(inc.weightedAverageShsOutDil) ??
          fmpNum(inc.weightedAverageShsOut) ??
          fmpNum(km.weightedAverageShsOut),
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

    const peTTMRaw =
      fmpNum(ratiosTtm?.priceToEarningsRatioTTM) ??
      fmpNum(ratiosTtm?.peRatioTTM) ??
      fmpNum(keyTtm?.peRatioTTM);
    const rawQuarters = quarterlyFundamentalRows(incomeQuarterly, cashFlowQuarterly);
    const ttmPick = pickTtmEps(
      rawQuarters.map((q) => ({ date: q.date, eps: q.eps })),
      keyTtm,
      ratiosTtm,
      price,
      peTTMRaw,
    );
    const ttmEpsRaw = ttmPick.ttm;
    const ttmFcfRaw = ttmFromQuarterly(
      rawQuarters.map((q) => ({ date: q.date, metric: q.fcfPerShare })),
    );
    const aligned = await this.alignToListing({
      yahooTicker,
      listingCurrency: profile.currency,
      price,
      income,
      annual,
      estimateEps: estimateParsed.map((e) => ({ year: e.year, eps: e.eps })),
      ttmEps: ttmEpsRaw,
      peTTM: peTTMRaw,
      tev:
        fmpNum(keyTtm?.enterpriseValueTTM) ??
        fmpNum(evRows[0]?.enterpriseValue) ??
        fmpNum(evRows[0]?.enterpriseValueTTM),
      dcf: dcf.dcf,
      mktCap: profile.mktCap,
    });
    annual = aligned.annual;
    for (let i = 0; i < estimateParsed.length; i++) {
      const scaled = aligned.estimateEps[i]?.eps ?? null;
      estimateParsed[i]!.eps = scaled;
    }
    const lastHistEpsScaled = annual[annual.length - 1]?.eps ?? null;
    for (let i = 0; i < estimateParsed.length; i++) {
      const prev = i === 0 ? lastHistEpsScaled : estimateParsed[i - 1]!.eps;
      estimateParsed[i]!.epsChgPct = yoyChgPct(estimateParsed[i]!.eps, prev);
    }
    const peTTM = aligned.peTTM;
    const ttmEps = aligned.ttmEps;
    const quarters = rawQuarters
      .map((q) => ({
        date: q.date,
        eps: scalePerShare(q.eps, aligned.scale),
        fcfPerShare: scalePerShare(q.fcfPerShare, aligned.scale),
      }))
      .sort((a, b) => a.date.localeCompare(b.date));
    const ttmFcf = scalePerShare(ttmFcfRaw.ttm, aligned.scale);

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
    const earningsYieldFromPrice =
      price != null &&
      ttmEps != null &&
      ttmEps > 0 &&
      price > 0
        ? ttmEps / price
        : price != null &&
            valuation.summary.latestMetric != null &&
            valuation.summary.latestMetric > 0
          ? valuation.summary.latestMetric / price
          : null;
    const scaledUnits = aligned.scale.fxToListing !== 1 || aligned.scale.perShareFactor !== 1;
    const earningsYieldTTM = scaledUnits
      ? earningsYieldFromPrice
      : (fmpNum(ratiosTtm?.earningsYieldTTM) ??
        fmpNum(keyTtm?.earningsYieldTTM) ??
        earningsYieldFromPrice);
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
    const tev = aligned.tev;

    const fwdEps = estimateParsed[0]?.eps ?? null;
    const fwdPe = price != null && fwdEps != null && fwdEps > 0 ? price / fwdEps : null;
    const peForBlend = peTTM ?? valuation.summary.currentPe;
    const blendedPe =
      peForBlend != null && fwdPe != null ? (peForBlend + fwdPe) / 2 : peForBlend ?? fwdPe;

    const fvRatio = valuation.summary.fairValueRatio;
    // Today's FV is TTM / last FY × the trailing-window ratio. Future price uses the far estimate.
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

    const dcfVal = aligned.dcf;
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
        mktCap: aligned.mktCap,
        price,
        beta: profile.beta,
        country: profile.country ?? null,
        website: profile.website ?? null,
        image: profile.image ?? null,
      },
      scale: aligned.scale,
      snapshot: {
        peTTM,
        ttmEps,
        ttmAsOf: ttmPick.asOf,
        ttmFcf,
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
        nextEarningsDate: pickNextEarningsDate(earningsRows),
      },
      valuation,
      forecastSeries,
      estimates: estimateParsed,
      performance,
      annual,
      quarters,
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
