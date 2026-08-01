/** Assemble Fast Graphs–style fundamentals payload from FMP. */
import { Injectable, Logger, NotFoundException } from '@nestjs/common';
import {
  buildValuationSeries,
  type AnnualFundamentalPoint,
  type ValuationMetric,
} from '@vova/engine';
import { FmpClient, fmpNum, fmpStr, yahooToFmpSymbol } from '../market/fmp.client';
import { UniverseService } from '../universe/universe.service';

const CACHE_TTL_MS = 24 * 60 * 60 * 1000;
const METRICS: ValuationMetric[] = ['eps', 'revenue', 'fcf', 'ownerEarnings'];

type CacheEntry = { at: number; payload: FundamentalsPayload };

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
  };
  snapshot: {
    peTTM: number | null;
    pbTTM: number | null;
    psTTM: number | null;
    pegTTM: number | null;
    roeTTM: number | null;
    roicTTM: number | null;
    dividendYieldTTM: number | null;
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
  /** All metric valuations for quick switching without refetch (series rebuilt client-side too). */
  annual: AnnualFundamentalPoint[];
  incomeTrend: Array<{
    year: number;
    date: string;
    revenue: number | null;
    netIncome: number | null;
    eps: number | null;
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

@Injectable()
export class FundamentalsService {
  private readonly log = new Logger(FundamentalsService.name);
  private readonly cache = new Map<string, CacheEntry>();

  constructor(
    private readonly fmp: FmpClient,
    private readonly universe: UniverseService,
  ) {}

  async get(yahooTicker: string, metric: ValuationMetric = 'eps'): Promise<FundamentalsPayload> {
    if (!METRICS.includes(metric)) metric = 'eps';
    const cacheKey = `${yahooTicker.toUpperCase()}|${metric}`;
    const hit = this.cache.get(cacheKey);
    if (hit && Date.now() - hit.at < CACHE_TTL_MS) {
      return { ...hit.payload, cached: true };
    }

    // Warm from any metric cache for the same ticker (rebuild valuation only).
    for (const m of METRICS) {
      if (m === metric) continue;
      const other = this.cache.get(`${yahooTicker.toUpperCase()}|${m}`);
      if (other && Date.now() - other.at < CACHE_TTL_MS) {
        const valuation = buildValuationSeries(other.payload.annual, metric, {
          currentPrice: other.payload.profile.price,
        });
        const payload: FundamentalsPayload = {
          ...other.payload,
          valuation,
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
    ] = await Promise.all([
      this.fmp.profile(fmpSymbol),
      this.fmp.incomeAnnual(fmpSymbol),
      this.fmp.cashFlowAnnual(fmpSymbol),
      this.fmp.keyMetricsAnnual(fmpSymbol),
      this.fmp.ratiosAnnual(fmpSymbol),
      this.fmp.keyMetricsTtm(fmpSymbol),
      this.fmp.ratiosTtm(fmpSymbol),
      this.fmp.ownerEarnings(fmpSymbol),
      this.fmp.dcf(fmpSymbol),
      this.fmp.financialScores(fmpSymbol),
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

    const dates = new Set<string>();
    for (const d of incomeByDate.keys()) dates.add(d);
    for (const d of kmByDate.keys()) dates.add(d);
    const dateList = [...dates].sort();

    const years = dateList
      .map((d) => yearOf(d))
      .filter((y): y is number => y != null);
    const yearEnd = await this.fmp.yearEndCloses(fmpSymbol, [...new Set(years)]);

    const annual: AnnualFundamentalPoint[] = [];
    for (const date of dateList) {
      const y = yearOf(date);
      if (y == null) continue;
      const inc = incomeByDate.get(date) ?? {};
      const cf = cfByDate.get(date) ?? {};
      const km = kmByDate.get(date) ?? {};
      const rt = ratioByDate.get(date) ?? {};

      const eps =
        fmpNum(km.netIncomePerShare) ??
        fmpNum(inc.epsdiluted) ??
        fmpNum(inc.epsDiluted) ??
        fmpNum(inc.eps);
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
      });
    }

    annual.sort((a, b) => a.year - b.year);

    if (!annual.length) {
      this.log.warn(`Empty annual series for ${fmpSymbol}`);
    }

    const price = profile.price ?? annual[annual.length - 1]?.price ?? null;
    const valuation = buildValuationSeries(annual, metric, { currentPrice: price });

    const peTTM =
      fmpNum(ratiosTtm?.priceToEarningsRatioTTM) ??
      fmpNum(ratiosTtm?.peRatioTTM) ??
      fmpNum(keyTtm?.peRatioTTM);
    const pbTTM =
      fmpNum(ratiosTtm?.priceToBookRatioTTM) ?? fmpNum(keyTtm?.pbRatioTTM);
    const psTTM =
      fmpNum(ratiosTtm?.priceToSalesRatioTTM) ?? fmpNum(keyTtm?.ptbRatioTTM);
    const pegTTM = fmpNum(ratiosTtm?.priceToEarningsGrowthRatioTTM) ?? fmpNum(ratiosTtm?.pegRatioTTM);
    const roeTTM = fmpNum(keyTtm?.returnOnEquityTTM) ?? fmpNum(ratiosTtm?.returnOnEquityTTM);
    const roicTTM = fmpNum(keyTtm?.returnOnInvestedCapitalTTM) ?? fmpNum(keyTtm?.roicTTM);
    const dividendYieldTTM =
      fmpNum(ratiosTtm?.dividendYieldTTM) ?? fmpNum(keyTtm?.dividendYieldTTM);
    const debtToEquityTTM =
      fmpNum(ratiosTtm?.debtEquityRatioTTM) ?? fmpNum(ratiosTtm?.debtToEquityTTM);
    const currentRatioTTM = fmpNum(ratiosTtm?.currentRatioTTM);
    const profitMarginTTM =
      fmpNum(ratiosTtm?.netProfitMarginTTM) ?? fmpNum(keyTtm?.netProfitMarginTTM);
    const operatingMarginTTM =
      fmpNum(ratiosTtm?.operatingProfitMarginTTM) ?? fmpNum(ratiosTtm?.operatingMarginTTM);
    const fcfYieldTTM = fmpNum(keyTtm?.freeCashFlowYieldTTM);

    const dcfVal = dcf.dcf;
    const dcfPremiumPct =
      dcfVal != null && price != null && dcfVal > 0 ? ((price - dcfVal) / dcfVal) * 100 : null;

    const incomeTrend = annual
      .slice()
      .reverse()
      .slice(0, 12)
      .map((a) => ({
        year: a.year,
        date: a.date,
        revenue: a.revenue,
        netIncome: a.netIncome,
        eps: a.eps,
        operatingCashFlow: a.operatingCashFlow,
        freeCashFlow: a.freeCashFlow,
      }));

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
      },
      snapshot: {
        peTTM,
        pbTTM,
        psTTM,
        pegTTM,
        roeTTM,
        roicTTM,
        dividendYieldTTM,
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
      annual,
      incomeTrend,
      asOf: new Date().toISOString(),
      cached: false,
    };
  }
}
