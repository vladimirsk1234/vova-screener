import { useEffect, useMemo, useRef, useState } from 'react';
import { useQuery, type UseQueryResult } from '@tanstack/react-query';
import {
  expectedDcfFairValueByYear,
  fairValueFromEstimate,
  formatScaleCaption,
  nextIsoDate,
  sliceToWindow,
  type ValuationMetric,
  type ValuationSeriesPoint,
  type ValuationSummary,
  type ValuationWindowYears,
} from '@vova/engine';
import {
  api,
  type CustomDcfAssumptions,
  type CustomDcfPayload,
  type FundamentalsPayload,
  type HorizonReturns,
} from '../lib/api';
import { Chips } from '../components/Chips';

const METRICS = [
  { id: 'eps' as const, label: 'EPS' },
  { id: 'revenue' as const, label: 'Sales/sh' },
  { id: 'fcf' as const, label: 'FCF/sh' },
  { id: 'ownerEarnings' as const, label: 'Owner earn.' },
];

export const FUND_TABS = ['summary', 'forecasting', 'dcf', 'performance', 'profile'] as const;
export type FundTab = (typeof FUND_TABS)[number];
const TAB_LABEL: Record<FundTab, string> = {
  summary: 'Summary',
  forecasting: 'Forecasting',
  dcf: 'DCF',
  performance: 'Performance',
  profile: 'Profile',
};

function money(n: number | null | undefined, digits = 2) {
  if (n == null || !Number.isFinite(n)) return '—';
  return n.toLocaleString(undefined, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

function pct(n: number | null | undefined) {
  if (n == null || !Number.isFinite(n)) return '—';
  const sign = n > 0 ? '+' : '';
  return `${sign}${n.toFixed(1)}%`;
}

/** FMP mixes decimals (0.18) and whole percents (18); normalize to percent points. */
function asPctPoints(n: number | null | undefined): number | null {
  if (n == null || !Number.isFinite(n)) return null;
  return Math.abs(n) <= 1.5 ? n * 100 : n;
}

function ratio(n: number | null | undefined, digits = 2) {
  if (n == null || !Number.isFinite(n)) return '—';
  return n.toFixed(digits);
}

function multiple(n: number | null | undefined) {
  if (n == null || !Number.isFinite(n)) return '—';
  const digits = n >= 1 ? 1 : n >= 0.1 ? 2 : 3;
  return n.toFixed(digits);
}

function compact(n: number | null | undefined) {
  if (n == null || !Number.isFinite(n)) return '—';
  return new Intl.NumberFormat(undefined, {
    notation: 'compact',
    maximumFractionDigits: 2,
  }).format(n);
}

function fvRuleLabel(rule: string | undefined) {
  if (rule === 'pe15') return 'PE 15';
  if (rule === 'lynch_peg') return 'Lynch PEG=1';
  return 'N/A';
}

/** Forward growth spans the window through the last estimate, so a fixed "5y" would be wrong. */
function growthLabel(source: string | undefined, windowYears: ValuationWindowYears) {
  if (source === 'forward') return 'Growth (fwd)';
  if (windowYears == null) return 'Growth (max)';
  return `Growth (${windowYears}y)`;
}

function growthRateLabel(source: string | undefined, windowYears: ValuationWindowYears) {
  if (source === 'forward') return 'Growth Rate (fwd)';
  if (windowYears == null) return 'Growth Rate (max)';
  return `Growth Rate (${windowYears}y)`;
}

export function FundamentalsPanel({
  ticker,
  metric,
  setMetric,
  windowYears,
  fundQ,
  valuation,
  tab,
  onTabChange,
  onDcfChartSeries,
}: {
  ticker: string;
  metric: ValuationMetric;
  setMetric: (metric: ValuationMetric) => void;
  windowYears: ValuationWindowYears;
  fundQ: UseQueryResult<FundamentalsPayload>;
  valuation: { summary: ValuationSummary } | null;
  tab: FundTab;
  onTabChange: (tab: FundTab) => void;
  onDcfChartSeries?: (series: ValuationSeriesPoint[]) => void;
}) {

  const fyRows = useMemo(() => {
    if (!fundQ.data) return [];
    const windowed = sliceToWindow(fundQ.data.annual, windowYears);
    const minYear = windowed[0]?.year;
    if (minYear == null) return [];
    return fundQ.data.incomeTrend.filter((row) => row.year >= minYear);
  }, [fundQ.data, windowYears]);

  const snap = fundQ.data?.snapshot;
  const profile = fundQ.data?.profile;
  const summary = valuation?.summary;
  const premiumClass =
    summary?.premiumPct == null
      ? ''
      : summary.premiumPct > 10
        ? 'fund-neg'
        : summary.premiumPct < -10
          ? 'fund-pos'
          : '';

  return (
    <div className="fund-panel">
      <Chips
        value={tab}
        options={FUND_TABS}
        format={(id) => TAB_LABEL[id]}
        onChange={onTabChange}
      />

      {fundQ.isLoading ? <p className="muted small">Loading FMP fundamentals…</p> : null}
      {fundQ.error ? (
        <p className="error">
          {(fundQ.error as Error).message.includes('FMP_API_KEY')
            ? 'Set FMP_API_KEY on the API server to load fundamentals.'
            : (fundQ.error as Error).message}
        </p>
      ) : null}

      {tab === 'forecasting' ? (
        <>
          <section className="fund-hero">
            <div className="fund-hero-main">
              <p className="fund-kicker">Forecast</p>
              <h2 className="fund-headline">
                {money(snap?.futurePrice)}
                <span className="fund-headline-unit"> future price</span>
              </h2>
              <p className="fund-sub">
                Price {money(summary?.currentPrice)} ·{' '}
                <span className={premiumClass}>{pct(summary?.premiumPct)} vs fair</span>
                {snap?.estAnnualRorPct != null ? <> · Est. ROR {pct(snap.estAnnualRorPct)}</> : null}
                {formatScaleCaption(fundQ.data?.scale ?? null) ? (
                  <> · {formatScaleCaption(fundQ.data?.scale ?? null)}</>
                ) : null}
              </p>
            </div>
            <dl className="fund-hero-stats">
              <div>
                <dt>{growthLabel(summary?.growthSource, windowYears)}</dt>
                <dd>{pct(summary?.growthRatePct)}</dd>
              </div>
              <div>
                <dt>FV ratio</dt>
                <dd>
                  {summary?.fairValueRatio != null ? `${money(summary.fairValueRatio, 2)}×` : '—'}
                  <span className="fund-hero-hint">{fvRuleLabel(summary?.fairValueRule)}</span>
                </dd>
              </div>
              <div>
                <dt>Normal P/E</dt>
                <dd>{summary ? `${multiple(summary.normalMultiple)}×` : '—'}</dd>
              </div>
            </dl>
          </section>
        </>
      ) : null}

      {tab === 'summary' ? (
        <>
          <Chips
            value={metric}
            options={METRICS.map((m) => m.id)}
            format={(id) => METRICS.find((m) => m.id === id)?.label ?? id}
            onChange={setMetric}
          />

          {snap ? (
            <div className="fund-layout">
              <aside className="fund-sidebar">
                <Metric
                  label={growthRateLabel(summary?.growthSource, windowYears)}
                  value={pct(summary?.growthRatePct)}
                />
                <Metric
                  label="Fair Value Ratio"
                  value={
                    summary?.fairValueRatio != null
                      ? `${ratio(summary.fairValueRatio)}×`
                      : '—'
                  }
                />
                <Metric label="Normal P/E" value={`${multiple(summary?.normalMultiple)}×`} />
                <Metric label="Blended P/E" value={ratio(snap.blendedPe)} />
                <Metric label="EPS Yld" value={pct(asPctPoints(snap.earningsYieldTTM))} />
                <Metric label="Div Yld" value={pct(asPctPoints(snap.dividendYieldTTM))} />
                <Metric label="S&P Credit Rating" value="—" />
                <Metric label="Market Cap" value={compact(profile?.mktCap)} />
                <Metric label="TEV" value={compact(snap.tev)} />
                <Metric
                  label="LT Debt/Capital"
                  value={pct(asPctPoints(snap.ltDebtToCapitalTTM))}
                />
                <Metric label="Country" value={profile?.country ?? '—'} />
                <Metric label="Industry" value={profile?.industry ?? '—'} />
                <Metric
                  label="Units"
                  value={formatScaleCaption(fundQ.data?.scale ?? null) ?? (profile?.currency ?? '—')}
                />
              </aside>
            </div>
          ) : null}
        </>
      ) : null}

      {tab === 'forecasting' && snap ? (
        <div className="fund-layout">
          <aside className="fund-sidebar">
            <Metric label="Est. Annual ROR" value={pct(snap.estAnnualRorPct)} />
            <Metric label="Fair Value $" value={money(summary?.fairValue)} />
            <Metric label="Future price" value={money(snap.futurePrice)} />
            <Metric label="Fwd EPS" value={money(snap.fwdEps)} />
            <Metric label="Fwd P/E" value={ratio(snap.fwdPe)} />
            <Metric label="Blended P/E" value={ratio(snap.blendedPe)} />
            <Metric label="Div Yld" value={pct(asPctPoints(snap.dividendYieldTTM))} />
          </aside>
        </div>
      ) : null}

      {tab === 'summary' && fyRows.length ? (
        <section className="fund-section">
          <h3 className="fund-section-title">FY EPS / Chg / Div</h3>
          <div className="fund-table-wrap">
            <table className="fund-table">
              <thead>
                <tr>
                  <th>Year</th>
                  <th>EPS</th>
                  <th>% Chg</th>
                  <th>Div</th>
                </tr>
              </thead>
              <tbody>
                {fyRows.map((row) => (
                  <tr key={row.date}>
                    <td>{row.year}</td>
                    <td>{money(row.eps)}</td>
                    <td>{pct(row.epsChgPct)}</td>
                    <td>{money(row.dividend)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      ) : null}

      {tab === 'forecasting' ? (
        <section className="fund-section">
          <h3 className="fund-section-title">Analyst estimates</h3>
          {fundQ.data?.estimates?.length ? (
            <div className="fund-table-wrap">
              <table className="fund-table">
                <thead>
                  <tr>
                    <th>Year</th>
                    <th>EPS est.</th>
                    <th>% Chg</th>
                    <th>Fair Value $</th>
                    <th># Analysts</th>
                  </tr>
                </thead>
                <tbody>
                  {fundQ.data.estimates.map((row) => (
                    <tr key={row.date || row.year}>
                      <td>{row.year}</td>
                      <td>{money(row.eps)}</td>
                      <td>{pct(row.epsChgPct)}</td>
                      <td>{money(fairValueFromEstimate(row.eps, summary?.fairValueRatio))}</td>
                      <td>{row.analysts != null ? String(row.analysts) : '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p className="muted small">No analyst estimates from FMP for this symbol.</p>
          )}
        </section>
      ) : null}

      {tab === 'dcf' ? (
        <DcfTab
          ticker={ticker}
          lynchFairValue={summary?.fairValue ?? null}
          price={summary?.currentPrice ?? profile?.price ?? null}
          onDcfChartSeries={onDcfChartSeries}
        />
      ) : null}

      {tab === 'performance' && fundQ.data ? (
        <PerformanceTab
          ticker={ticker}
          price={fundQ.data.performance.price}
          spy={fundQ.data.performance.spy}
          eps={fundQ.data.performance.eps}
          years={fundQ.data.performance.years}
        />
      ) : null}

      {tab === 'profile' && profile ? (
        <section className="fund-section">
          <h3 className="fund-section-title">Company profile</h3>
          {profile.description ? <p className="fund-about fund-about--full">{profile.description}</p> : null}
          <div className="fund-metric-grid">
            <Metric label="Sector" value={profile.sector ?? '—'} />
            <Metric label="Industry" value={profile.industry ?? '—'} />
            <Metric label="Country" value={profile.country ?? '—'} />
            <Metric label="Exchange" value={profile.exchange ?? '—'} />
            <Metric
              label="Units"
              value={formatScaleCaption(fundQ.data?.scale ?? null) ?? (profile.currency ?? '—')}
            />
            <Metric label="Market Cap" value={compact(profile.mktCap)} />
            <Metric label="Beta" value={ratio(profile.beta)} />
          </div>
          {profile.website ? (
            <p className="muted small">
              <a href={profile.website} target="_blank" rel="noreferrer">
                {profile.website}
              </a>
            </p>
          ) : null}
        </section>
      ) : null}

      {tab === 'summary' && snap ? (
        <section className="fund-section">
          <h3 className="fund-section-title">More ratios</h3>
          <div className="fund-metric-grid">
            <Metric label="P/E TTM" value={ratio(snap.peTTM)} />
            <Metric label="PEG" value={ratio(snap.pegTTM)} />
            <Metric label="P/B" value={ratio(snap.pbTTM)} />
            <Metric label="ROE" value={pct(asPctPoints(snap.roeTTM))} />
            <Metric label="ROIC" value={pct(asPctPoints(snap.roicTTM))} />
            <Metric label="Op. margin" value={pct(asPctPoints(snap.operatingMarginTTM))} />
            <Metric label="DCF" value={money(snap.dcf)} />
            <Metric label="Piotroski" value={snap.piotroskiScore != null ? String(snap.piotroskiScore) : '—'} />
          </div>
        </section>
      ) : null}

      {tab !== 'dcf' ? (
        <p className="muted small fund-footnote">
          Fair value = GAAP diluted EPS × 15× when 5y EPS CAGR &lt; 15%, else PEG=1 (ratio = growth %).
          Normal P/E is the median price/EPS on the selected 5Y / 10Y / MAX window. Per-share
          figures are converted to the listing currency (and per ADS when the ADR ratio is known).
          Source: Financial Modeling Prep GAAP diluted, not FAST Graphs adjusted operating EPS.
          S&amp;P credit rating is not in FMP.
          {snap?.ttmAsOf ? ` TTM uses the last four reported quarters through ${snap.ttmAsOf}.` : ''}
          {fundQ.data?.cached ? ' · cached' : ''}
        </p>
      ) : null}
    </div>
  );
}

const DCF_PRESETS = ['conservative', 'base', 'optimistic'] as const;
type DcfPreset = (typeof DCF_PRESETS)[number];
const DCF_PRESET_LABEL: Record<DcfPreset, string> = {
  conservative: 'Conservative',
  base: 'Base',
  optimistic: 'Optimistic',
};

const DCF_FIELDS = [
  { key: 'revenueGrowthPct', label: 'Revenue growth %' },
  { key: 'ebitdaPct', label: 'EBITDA %' },
  { key: 'capitalExpenditurePct', label: 'Capex %' },
  { key: 'longTermGrowthRate', label: 'Long-term g %' },
  { key: 'riskFreeRate', label: 'Risk-free %' },
  { key: 'marketRiskPremium', label: 'ERP %' },
] as const;

type DcfFieldKey = (typeof DCF_FIELDS)[number]['key'];
type DcfDraft = Record<DcfFieldKey, string>;

function emptyDcfDraft(): DcfDraft {
  return {
    revenueGrowthPct: '',
    ebitdaPct: '',
    capitalExpenditurePct: '',
    longTermGrowthRate: '',
    riskFreeRate: '',
    marketRiskPremium: '',
  };
}

function rateToPctInput(n: number | null | undefined): string {
  if (n == null || !Number.isFinite(n)) return '';
  const pts = Math.abs(n) <= 1.5 ? n * 100 : n;
  return String(Math.round(pts * 1000) / 1000);
}

function draftFromAssumptions(a: CustomDcfAssumptions): DcfDraft {
  return {
    revenueGrowthPct: rateToPctInput(a.revenueGrowthPct),
    ebitdaPct: rateToPctInput(a.ebitdaPct),
    capitalExpenditurePct: rateToPctInput(a.capitalExpenditurePct),
    longTermGrowthRate: rateToPctInput(a.longTermGrowthRate),
    riskFreeRate: rateToPctInput(a.riskFreeRate),
    marketRiskPremium: rateToPctInput(a.marketRiskPremium),
  };
}

function draftFromPayload(data: CustomDcfPayload): DcfDraft {
  return draftFromAssumptions(assumptionsFromPayload(data));
}

function assumptionsFromPayload(data: CustomDcfPayload): CustomDcfAssumptions {
  const out: CustomDcfAssumptions = {};
  const put = (key: DcfFieldKey, n: number | null) => {
    if (n != null && Number.isFinite(n)) out[key] = n;
  };
  put('revenueGrowthPct', data.revenueGrowthPct);
  put('ebitdaPct', data.ebitdaPct);
  put('capitalExpenditurePct', data.capitalExpenditurePct);
  put('longTermGrowthRate', data.longTermGrowthRate);
  put('riskFreeRate', data.riskFreeRate);
  put('marketRiskPremium', data.marketRiskPremium);
  return out;
}

function draftToAssumptions(draft: DcfDraft): CustomDcfAssumptions {
  const out: CustomDcfAssumptions = {};
  for (const { key } of DCF_FIELDS) {
    const raw = draft[key].trim();
    if (!raw) continue;
    const n = Number(raw);
    if (!Number.isFinite(n)) continue;
    out[key] = n / 100;
  }
  return out;
}

function presetOverrides(
  base: CustomDcfAssumptions,
  kind: Exclude<DcfPreset, 'base'>,
): CustomDcfAssumptions {
  const growth = base.revenueGrowthPct ?? 0.08;
  const g = base.longTermGrowthRate ?? 0.025;
  const erp = base.marketRiskPremium ?? 0.05;
  if (kind === 'conservative') {
    return {
      revenueGrowthPct: growth * 0.5,
      longTermGrowthRate: Math.min(g, 0.02),
      marketRiskPremium: erp + 0.01,
    };
  }
  return {
    revenueGrowthPct: growth * 1.25,
    longTermGrowthRate: Math.min(g + 0.005, 0.04),
    marketRiskPremium: Math.max(erp - 0.005, 0.03),
  };
}

function dcfChartSeriesFromPayload(data: CustomDcfPayload): ValuationSeriesPoint[] {
  const yearly = expectedDcfFairValueByYear({
    years: data.years,
    wacc: data.wacc,
    terminalValue: data.terminalValue,
    netDebt: data.netDebt,
    dilutedShares: data.dilutedShares,
  });
  const points: ValuationSeriesPoint[] = [];
  const asOf = (data.asOf || new Date().toISOString()).slice(0, 10);
  if (
    data.equityValuePerShare != null &&
    Number.isFinite(data.equityValuePerShare) &&
    data.equityValuePerShare > 0
  ) {
    points.push({
      date: asOf,
      year: Number(asOf.slice(0, 4)),
      price: data.price,
      metric: null,
      earningsPower: data.equityValuePerShare,
      fairValue: data.equityValuePerShare,
      normalValue: null,
      pe: null,
      forecast: true,
    });
  }
  let prev = points[points.length - 1]?.date ?? '';
  for (const row of yearly) {
    if (
      row.fairValuePerShare == null ||
      !Number.isFinite(row.fairValuePerShare) ||
      row.fairValuePerShare <= 0
    ) {
      continue;
    }
    let date = `${row.year}-12-31`;
    if (prev && date <= prev) date = nextIsoDate(prev);
    points.push({
      date,
      year: row.year,
      price: null,
      metric: null,
      earningsPower: row.fairValuePerShare,
      fairValue: row.fairValuePerShare,
      normalValue: null,
      pe: null,
      forecast: true,
    });
    prev = date;
  }
  return points;
}

function DcfTab({
  ticker,
  lynchFairValue,
  price: lynchPrice,
  onDcfChartSeries,
}: {
  ticker: string;
  lynchFairValue: number | null;
  price: number | null;
  onDcfChartSeries?: (series: ValuationSeriesPoint[]) => void;
}) {
  const seeded = useRef(false);
  const [draft, setDraft] = useState<DcfDraft>(emptyDcfDraft);
  const [applied, setApplied] = useState<CustomDcfAssumptions>({});
  const [base, setBase] = useState<CustomDcfAssumptions | null>(null);
  const [preset, setPreset] = useState<DcfPreset>('base');

  useEffect(() => {
    seeded.current = false;
    setApplied({});
    setBase(null);
    setDraft(emptyDcfDraft());
    setPreset('base');
  }, [ticker]);

  const dcfQ = useQuery({
    queryKey: ['custom-dcf', ticker, applied],
    queryFn: () => api.customDcf(ticker, applied),
    enabled: Boolean(ticker),
    staleTime: 60_000,
  });

  useEffect(() => {
    if (!dcfQ.data || seeded.current) return;
    seeded.current = true;
    setBase(assumptionsFromPayload(dcfQ.data));
    setDraft(draftFromPayload(dcfQ.data));
  }, [dcfQ.data]);

  const data = dcfQ.data;
  const price = data?.price ?? lynchPrice;
  const dcfPrice = data?.equityValuePerShare ?? null;
  const premiumClass =
    data?.premiumPct == null
      ? ''
      : data.premiumPct > 10
        ? 'fund-neg'
        : data.premiumPct < -10
          ? 'fund-pos'
          : '';

  const applyPreset = (next: DcfPreset) => {
    setPreset(next);
    const snapshot = base ?? (data ? assumptionsFromPayload(data) : null);
    if (next === 'base') {
      if (snapshot) setDraft(draftFromAssumptions(snapshot));
      setApplied({});
      return;
    }
    if (!snapshot) return;
    const overrides = presetOverrides(snapshot, next);
    setDraft(draftFromAssumptions({ ...snapshot, ...overrides }));
    setApplied(overrides);
  };

  const recalculate = () => {
    setApplied(draftToAssumptions(draft));
  };

  const yearlyFairValue = useMemo(
    () =>
      data
        ? expectedDcfFairValueByYear({
            years: data.years,
            wacc: data.wacc,
            terminalValue: data.terminalValue,
            netDebt: data.netDebt,
            dilutedShares: data.dilutedShares,
          })
        : [],
    [data],
  );

  useEffect(() => {
    if (!onDcfChartSeries) return;
    onDcfChartSeries(data ? dcfChartSeriesFromPayload(data) : []);
    return () => onDcfChartSeries([]);
  }, [data, onDcfChartSeries]);

  const latestUfcf = data?.years?.length ? data.years[data.years.length - 1]?.ufcf ?? null : null;

  return (
    <>
      <section className="fund-hero">
        <div className="fund-hero-main">
          <p className="fund-kicker">Custom DCF</p>
          <h2 className="fund-headline">
            {money(dcfPrice)}
            <span className="fund-headline-unit"> DCF / share</span>
          </h2>
          <p className="fund-sub">
            Price {money(price)} ·{' '}
            <span className={premiumClass}>{pct(data?.premiumPct)} vs DCF</span>
            {lynchFairValue != null ? <> · Lynch FV {money(lynchFairValue)}</> : null}
          </p>
        </div>
        <dl className="fund-hero-stats">
          <div>
            <dt>WACC</dt>
            <dd>{pct(data?.wacc != null ? data.wacc * 100 : null)}</dd>
          </div>
          <div>
            <dt>Long-term g</dt>
            <dd>{pct(data?.longTermGrowthRate != null ? data.longTermGrowthRate * 100 : null)}</dd>
          </div>
          <div>
            <dt>Terminal of EV</dt>
            <dd>{data?.terminalSharePct != null ? `${data.terminalSharePct.toFixed(0)}%` : '—'}</dd>
          </div>
        </dl>
      </section>

      {dcfQ.isLoading ? <p className="muted small">Loading FMP Custom DCF…</p> : null}
      {dcfQ.error ? (
        <p className="error">
          {(dcfQ.error as Error).message.includes('FMP_API_KEY')
            ? 'Set FMP_API_KEY on the API server to load DCF.'
            : (dcfQ.error as Error).message}
        </p>
      ) : null}

      <section className="fund-section">
        <h3 className="fund-section-title">Health</h3>
        <div className="fund-metric-grid">
          <Metric
            label="Rev. growth"
            value={pct(data?.revenueGrowthPct != null ? data.revenueGrowthPct * 100 : null)}
          />
          <Metric label="UFCF (last yr)" value={compact(latestUfcf)} />
          <Metric label="Net debt" value={compact(data?.netDebt)} />
          <Metric label="Beta" value={ratio(data?.beta)} />
          <Metric
            label="Cost of equity"
            value={pct(data?.costOfEquity != null ? data.costOfEquity * 100 : null)}
          />
          <Metric
            label="Cost of debt"
            value={pct(data?.costOfDebt != null ? data.costOfDebt * 100 : null)}
          />
        </div>
        {data?.fragile ? (
          <p className="error small">
            WACC and long-term g are within 1pp — terminal value is unstable.
          </p>
        ) : null}
        {data?.terminalSharePct != null && data.terminalSharePct > 75 ? (
          <p className="muted small">
            Terminal value is {data.terminalSharePct.toFixed(0)}% of enterprise value. Most of the
            fair price is the perpetuity assumption, not the forecast years.
          </p>
        ) : null}
      </section>

      <section className="fund-section">
        <h3 className="fund-section-title">Assumptions</h3>
        <Chips
          value={preset}
          options={DCF_PRESETS}
          format={(id) => DCF_PRESET_LABEL[id]}
          onChange={applyPreset}
        />
        <div className="fund-dcf-form">
          {DCF_FIELDS.map((field) => (
            <label key={field.key} className="field">
              <span>{field.label}</span>
              <input
                type="number"
                inputMode="decimal"
                step="0.1"
                value={draft[field.key]}
                onChange={(e) => setDraft((prev) => ({ ...prev, [field.key]: e.target.value }))}
              />
            </label>
          ))}
        </div>
        <div className="fund-dcf-actions">
          <button type="button" className="btn-sm" onClick={recalculate} disabled={dcfQ.isFetching}>
            {dcfQ.isFetching ? 'Calculating…' : 'Recalculate'}
          </button>
        </div>
      </section>

      {data?.years?.length ? (
        <section className="fund-section">
          <h3 className="fund-section-title">Forecast</h3>
          <div className="fund-table-wrap">
            <table className="fund-table">
              <thead>
                <tr>
                  <th>Year</th>
                  <th>Revenue</th>
                  <th>EBITDA</th>
                  <th>UFCF</th>
                  <th>PV</th>
                  <th>FV / share</th>
                </tr>
              </thead>
              <tbody>
                {data.years.map((row, i) => (
                  <tr key={`${row.year}-${i}`}>
                    <td>{row.year}</td>
                    <td>{compact(row.revenue)}</td>
                    <td>{compact(row.ebitda)}</td>
                    <td>{compact(row.ufcf)}</td>
                    <td>{compact(row.pvUfcf)}</td>
                    <td>{money(yearlyFairValue[i]?.fairValuePerShare)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      ) : dcfQ.isSuccess && !dcfQ.isLoading ? (
        <p className="muted small">No Custom DCF projection from FMP for this symbol.</p>
      ) : null}

      <section className="fund-section">
        <h3 className="fund-section-title">Bridge to equity</h3>
        <div className="fund-metric-grid">
          <Metric label="PV of UFCF" value={compact(data?.sumPvUfcf)} />
          <Metric label="Terminal value" value={compact(data?.terminalValue)} />
          <Metric label="PV of terminal" value={compact(data?.presentTerminalValue)} />
          <Metric label="Enterprise value" value={compact(data?.enterpriseValue)} />
          <Metric label="Net debt" value={compact(data?.netDebt)} />
          <Metric label="Equity value" value={compact(data?.equityValue)} />
          <Metric label="DCF / share" value={money(dcfPrice)} />
          <Metric label="Lynch fair value" value={money(lynchFairValue)} />
        </div>
      </section>

      <p className="muted small fund-footnote">
        Unlevered Custom DCF from Financial Modeling Prep: projected free cash flow discounted at
        WACC, then net debt subtracted. FV / share is the roll-forward intrinsic value at each
        year-end (remaining UFCF + terminal, discounted to that date). Not a bank/insurance model.
        Long-term g must stay below WACC. This is not the PE15 / Lynch fair value used by the
        Settings filter.
        {data?.cached ? ' · cached' : ''}
      </p>
    </>
  );
}

function PerformanceTab({
  ticker,
  price,
  spy,
  eps,
  years,
}: {
  ticker: string;
  price: HorizonReturns;
  spy: HorizonReturns;
  eps: HorizonReturns;
  years: Array<{
    year: number;
    tickerClose: number | null;
    spyClose: number | null;
    tickerRetPct: number | null;
    spyRetPct: number | null;
    eps: number | null;
    epsChgPct: number | null;
  }>;
}) {
  return (
    <>
      <section className="fund-section">
        <h3 className="fund-section-title">Annualized returns</h3>
        <div className="fund-table-wrap">
          <table className="fund-table">
            <thead>
              <tr>
                <th></th>
                <th>1Y</th>
                <th>3Y</th>
                <th>5Y</th>
                <th>10Y</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>{ticker} price</td>
                <td>{pct(price.y1)}</td>
                <td>{pct(price.y3)}</td>
                <td>{pct(price.y5)}</td>
                <td>{pct(price.y10)}</td>
              </tr>
              <tr>
                <td>SPY</td>
                <td>{pct(spy.y1)}</td>
                <td>{pct(spy.y3)}</td>
                <td>{pct(spy.y5)}</td>
                <td>{pct(spy.y10)}</td>
              </tr>
              <tr>
                <td>EPS CAGR</td>
                <td>{pct(eps.y1)}</td>
                <td>{pct(eps.y3)}</td>
                <td>{pct(eps.y5)}</td>
                <td>{pct(eps.y10)}</td>
              </tr>
            </tbody>
          </table>
        </div>
        <p className="muted small">Price vs SPY from Yahoo bars. EPS from FMP. No SPY EPS line.</p>
      </section>
      {years.length ? (
        <section className="fund-section">
          <h3 className="fund-section-title">Calendar years</h3>
          <div className="fund-table-wrap">
            <table className="fund-table">
              <thead>
                <tr>
                  <th>Year</th>
                  <th>{ticker}</th>
                  <th>SPY</th>
                  <th>EPS %chg</th>
                </tr>
              </thead>
              <tbody>
                {years.map((row) => (
                  <tr key={row.year}>
                    <td>{row.year}</td>
                    <td>{pct(row.tickerRetPct)}</td>
                    <td>{pct(row.spyRetPct)}</td>
                    <td>{pct(row.epsChgPct)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      ) : null}
    </>
  );
}

function Metric({
  label,
  value,
  tone,
}: {
  label: string;
  value: string;
  tone?: number | null;
}) {
  const cls =
    tone == null || !Number.isFinite(tone) ? '' : tone > 10 ? 'fund-neg' : tone < -10 ? 'fund-pos' : '';
  return (
    <div className="fund-metric">
      <span className="fund-metric-label">{label}</span>
      <span className={`fund-metric-value ${cls}`}>{value}</span>
    </div>
  );
}
