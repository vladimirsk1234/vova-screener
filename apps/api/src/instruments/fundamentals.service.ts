/** Assemble Fast Graphs–style fundamentals payload from FMP (prices for Performance via Yahoo). */
import { Injectable, Logger, NotFoundException } from '@nestjs/common';
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
import { BarsService } from '../market/bars.service';
import { FmpClient, fmpNum, fmpStr, yahooToFmpSymbol } from '../market/fmp.client';
import { UniverseService } from '../universe/universe.service';

const CACHE_TTL_MS = 24 * 60 * 60 * 1000;
const METRICS: ValuationMetric[] = ['eps', 'revenue', 'fcf', 'ownerEarnings'];
const CARD_BATCH_LIMIT = 150;
const CARD_CONCURRENCY = 5;

type CacheEntry = { at: number; payload: FundamentalsPayload };

/** Slim valuation fields for Results / History signal cards. */
export type CardFundamentals = {
  fairValue: number | null;
  growthRatePct: number | null;
  blendedPe: number | null;
  ltDebtToCapitalTTM: number | null;
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
  private readonly cardCache = new Map<string, CardCacheEntry>();

  constructor(
    private readonly fmp: FmpClient,
    private readonly universe: UniverseService,
    private readonly bars: BarsService,
  ) {}

  async get(yahooTicker: string, metric: ValuationMetric = 'eps'): Promise<FundamentalsPayload> {
    if (!METRICS.includes(metric)) metric = 'eps';
    const cacheKey = `${yahooTicker.toUpperCase()}|${metric}`;
    const hit = this.cache.get(cacheKey);
    if (hit && Date.now() - hit.at < CACHE_TTL_MS) {
      return { ...hit.payload, cached: true };
    }

    for (const m of METRICS) {
      if (m === metric) continue;
      const other = this.cache.get(`${yahooTicker.toUpperCase()}|${m}`);
      if (other && Date.now() - other.at < CACHE_TTL_MS) {
        const valuation = buildValuationSeries(other.payload.annual, metric, {
          currentPrice: other.payload.profile.price,
          windowYears: 5,
          forward: forwardFor(metric, other.payload.estimates),
          ttmMetric: metric === 'eps' ? other.payload.snapshot.ttmEps : null,
        });
        const payload: FundamentalsPayload = {
          ...other.payload,
          valuation,
          forecastSeries: this.extendForecast(
            valuation.series,
            other.payload.estimates,
            valuation.summary.fairValueRatio,
          ),
          cached: true,
        };
        this.cache.set(cacheKey, { at: other.at, payload: { ...payload, cached: false } });
        return payload;
      }
    }

    const payload = await this.fetchFresh(yahooTicker, metric);
    this.cache.set(cacheKey, { at: Date.now(), payload: { ...payload, cached: false } });
    return payload;
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
   * Batch card metrics for Results / History. Reuses the full fundamentals cache when warm;
   * otherwise a slim FMP fetch (income + TTM + estimates + profile). Missing key → {}.
   */
  async getCardMetrics(tickers: string[]): Promise<Record<string, CardFundamentals>> {
    if (!this.fmp.configured()) return {};

    const unique: string[] = [];
    const seen = new Set<string>();
    for (const raw of tickers) {
      const t = String(raw || '')
        .trim()
        .toUpperCase();
      if (!t || seen.has(t)) continue;
      seen.add(t);
      unique.push(t);
      if (unique.length >= CARD_BATCH_LIMIT) break;
    }
    if (!unique.length) return {};

    const out: Record<string, CardFundamentals> = {};
    const needFetch: string[] = [];
    const now = Date.now();

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
      needFetch.push(ticker);
    }

    for (let i = 0; i < needFetch.length; i += CARD_CONCURRENCY) {
      const chunk = needFetch.slice(i, i + CARD_CONCURRENCY);
      await Promise.all(
        chunk.map(async (ticker) => {
          try {
            const metrics = await this.fetchCardSlim(ticker);
            this.cardCache.set(ticker, { at: Date.now(), metrics });
            out[ticker] = metrics;
          } catch (err) {
            this.log.warn(
              `Card metrics failed for ${ticker}: ${err instanceof Error ? err.message : String(err)}`,
            );
          }
        }),
      );
    }

    return out;
  }

  private cardFromFullCache(ticker: string, now: number): CardFundamentals | null {
    const hit = this.cache.get(`${ticker}|eps`);
    if (!hit || now - hit.at >= CACHE_TTL_MS) return null;
    return this.metricsFromPayload(hit.payload);
  }

  private metricsFromPayload(payload: FundamentalsPayload): CardFundamentals {
    return {
      fairValue: payload.valuation.summary.fairValue,
      growthRatePct: payload.valuation.summary.growthRatePct,
      blendedPe: payload.snapshot.blendedPe,
      ltDebtToCapitalTTM: payload.snapshot.ltDebtToCapitalTTM,
    };
  }

  /** Income + TTM ratios + estimates + profile — enough for the four card fields. */
  private async fetchCardSlim(yahooTicker: string): Promise<CardFundamentals> {
    const fmpSymbol = yahooToFmpSymbol(yahooTicker);
    const [profile, income, keyTtm, ratiosTtm, estimatesRaw] = await Promise.all([
      this.fmp.profile(fmpSymbol),
      this.fmp.incomeAnnual(fmpSymbol),
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
      this.fmp.profile(fmpSymbol),
      this.fmp.incomeAnnual(fmpSymbol),
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
