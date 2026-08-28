import { useEffect, useMemo, useRef, useState } from 'react';
import { useQuery, type UseQueryResult } from '@tanstack/react-query';
import {
  buildForecastScenarios,
  dividendCoverage,
  dividendStreak,
  forecastGrowthFromEstimates,
  expectedDcfFairValueByYear,
  fairValueFromEstimate,
  formatScaleCaption,
  pickMetric,
  sliceToWindow,
  yoyChgPct,
  type ChartValuationMetric,
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
import { isFundamentalsPendingError } from '../lib/apiError';
import { fundamentalsUpdateBanner, refreshPollMs } from '../lib/fundamentalsRefresh';
import {
  dcfChartSeriesFromPayload,
  dcfFairValueToday,
  EMPTY_DCF_SCENARIO_SERIES,
  type DcfScenarioSeries,
} from '../lib/dcfChart';
import { Chips } from '../components/Chips';

const METRICS: { id: ChartValuationMetric; label: string }[] = [
  { id: 'operatingEps', label: 'Op. EPS' },
  { id: 'fcf', label: 'FCF/sh' },
];

const METRIC_TABLE_LABEL: Record<ChartValuationMetric, string> = {
  operatingEps: 'Op. EPS',
  fcf: 'FCF/sh',
};

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

function selectedTtm(
  metric: ChartValuationMetric,
  snap:
    | { ttmEps: number | null; ttmOperatingEps?: number | null; ttmFcf?: number | null }
    | undefined,
): number | null {
  if (!snap) return null;
  if (metric === 'fcf') return snap.ttmFcf ?? null;
  return snap.ttmOperatingEps ?? null;
}

function compact(n: number | null | undefined) {
  if (n == null || !Number.isFinite(n)) return '—';
  return new Intl.NumberFormat(undefined, {
    notation: 'compact',
    maximumFractionDigits: 2,
  }).format(n);
}

function scorecardLine(bucket?: {
  beat: number;
  meet: number;
  miss: number;
  total: number;
  beatPct: number | null;
}) {
  if (!bucket || bucket.total <= 0) return '—';
  const beat = bucket.beatPct != null ? `${bucket.beatPct.toFixed(0)}% beat` : '';
  return `${bucket.beat}/${bucket.meet}/${bucket.miss} (${bucket.total})${beat ? ` · ${beat}` : ''}`;
}

function fvRuleLabel(rule: string | undefined) {
  if (rule === 'gdf') return 'GDF';
  if (rule === 'gdf_pe_g') return 'GDF…P/E=G';
  if (rule === 'pe_g') return 'P/E=G';
  // Cached Mongo / older payloads
  if (rule === 'pe15') return 'GDF…P/E=G';
  if (rule === 'lynch_peg') return 'P/E=G';
  return 'N/A';
}

/** Prefer the actual CAGR span when history is shorter than the selected window. */
function growthSpanLabelYears(
  windowYears: ValuationWindowYears,
  growthSpanYears: number | null | undefined,
): ValuationWindowYears | number | null {
  if (growthSpanYears != null && Number.isFinite(growthSpanYears) && growthSpanYears >= 1) {
    if (windowYears == null || growthSpanYears < windowYears) return growthSpanYears;
  }
  return windowYears;
}

function growthRateLabel(
  source: string | undefined,
  windowYears: ValuationWindowYears,
  growthSpanYears?: number | null,
) {
  if (source === 'forward') return 'Growth Rate (fwd)';
  const years = growthSpanLabelYears(windowYears, growthSpanYears);
  if (years == null) return 'Growth Rate (max)';
  return `Growth Rate (${years}y)`;
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
  metric: ChartValuationMetric;
  setMetric: (metric: ChartValuationMetric) => void;
  windowYears: ValuationWindowYears;
  fundQ: UseQueryResult<FundamentalsPayload>;
  valuation: { summary: ValuationSummary } | null;
  tab: FundTab;
  onTabChange: (tab: FundTab) => void;
  onDcfChartSeries?: (series: DcfScenarioSeries) => void;
}) {

  const refreshQ = useQuery({
    queryKey: ['fundamentals-refresh'],
    queryFn: () => api.fundamentalsRefresh(),
    enabled: Boolean(ticker),
    refetchInterval: (q) => refreshPollMs(q.state.data),
  });
  const pending = fundQ.isLoading || (fundQ.isError && isFundamentalsPendingError(fundQ.error));
  const fatal =
    fundQ.isError && !isFundamentalsPendingError(fundQ.error)
      ? (fundQ.error as Error)
      : null;
  const banner = fundamentalsUpdateBanner({
    lastRun: refreshQ.data?.lastRun,
    coverage: refreshQ.data?.coverage,
  });

  const fyRows = useMemo(() => {
    if (!fundQ.data) return [];
    const windowed = sliceToWindow(fundQ.data.annual, windowYears);
    return windowed
      .slice()
      .reverse()
      .map((row, idx, arr) => {
        const metricVal = pickMetric(row, metric);
        const older = arr[idx + 1];
        return {
          year: row.year,
          date: row.date,
          metric: metricVal,
          metricChgPct: yoyChgPct(metricVal, older ? pickMetric(older, metric) : null),
          dividend: row.dividend ?? null,
          operatingCashFlow: row.operatingCashFlow,
          freeCashFlow: row.freeCashFlow,
          dilutedShares: row.dilutedShares ?? null,
        };
      });
  }, [fundQ.data, windowYears, metric]);

  const [customPe, setCustomPe] = useState('');
  const customMultiple = (() => {
    const n = Number(customPe);
    return Number.isFinite(n) && n > 0 ? n : null;
  })();

  const snap = fundQ.data?.snapshot;
  const profile = fundQ.data?.profile;
  const summary = valuation?.summary;
  const forecastTtm = selectedTtm(metric, snap);
  const divStreak = useMemo(
    () => dividendStreak(fundQ.data?.annual ?? []),
    [fundQ.data?.annual],
  );
  const forecast = useMemo(() => {
    if (!summary) return null;
    const estimates = fundQ.data?.estimates ?? [];
    const box = forecastGrowthFromEstimates(estimates);
    const ttm = fundQ.data?.snapshot.ttmEps;
    const streetFv =
      ttm != null && Number.isFinite(ttm) && ttm > 0 && box.fairValueRatio != null
        ? ttm * box.fairValueRatio
        : null;
    return buildForecastScenarios({
      price: summary.currentPrice,
      fairValue: streetFv,
      fairValueRatio: undefined,
      normalMultiple: summary.normalMultiple,
      customMultiple,
      dividendYieldPct: asPctPoints(fundQ.data?.snapshot.dividendYieldTTM),
      estimates,
    });
  }, [summary, customMultiple, fundQ.data?.snapshot.dividendYieldTTM, fundQ.data?.snapshot.ttmEps, fundQ.data?.estimates]);
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

      {banner ? (
        <div className="fund-refresh" role="status">
          <p className="fund-refresh-text">{banner.text}</p>
          <div className="bar" aria-hidden="true">
            <div className="bar-fill" style={{ width: `${banner.pct}%` }} />
          </div>
        </div>
      ) : null}

      {pending ? <p className="muted small">Loading FMP fundamentals…</p> : null}
      {fatal ? (
        <p className="error">
          {fatal.message.includes('FMP_API_KEY')
            ? 'Set FMP_API_KEY on the API server to load fundamentals.'
            : fatal.message}
        </p>
      ) : null}

      {tab === 'summary' || tab === 'forecasting' ? (
        <Chips
          value={metric}
          options={METRICS.map((m) => m.id)}
          format={(id) => METRICS.find((m) => m.id === id)?.label ?? id}
          onChange={setMetric}
        />
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
                <span className={premiumClass}>
                  {forecast?.marginOfSafetyPct != null
                    ? `${pct(forecast.marginOfSafetyPct)} margin of safety`
                    : `${pct(summary?.premiumPct)} vs fair`}
                </span>
                {forecast?.rorPegPct != null ? <> · Est. ROR {pct(forecast.rorPegPct)}</> : null}
                {forecast?.horizonYears != null ? (
                  <> · {forecast.horizonYears.toFixed(2)}y horizon</>
                ) : null}
                {formatScaleCaption(fundQ.data?.scale ?? null) ? (
                  <> · {formatScaleCaption(fundQ.data?.scale ?? null)}</>
                ) : null}
              </p>
            </div>
            <dl className="fund-hero-stats">
              <div>
                <dt>Growth (fwd)</dt>
                <dd>{pct(forecast?.growthRatePct)}</dd>
              </div>
              <div>
                <dt>FV ratio</dt>
                <dd>
                  {forecast?.fairValueRatio != null ? `${money(forecast.fairValueRatio, 2)}×` : '—'}
                  <span className="fund-hero-hint">{fvRuleLabel(forecast?.fairValueRule ?? undefined)}</span>
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
          {snap ? (
            <div className="fund-layout">
              <aside className="fund-sidebar">
                <Metric
                  label={growthRateLabel(summary?.growthSource, windowYears, summary?.growthSpanYears)}
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
                {divStreak.consecPaid > 0 ? (
                  <>
                    <Metric label="Consec. Div Paid" value={String(divStreak.consecPaid)} />
                    <Metric
                      label="Consec. Div Increases"
                      value={String(divStreak.consecIncreases)}
                    />
                    <Metric label="Div CAGR" value={pct(divStreak.avgGrowthPct)} />
                  </>
                ) : null}
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
            <Metric label="Est. ROR (P=E=G)" value={pct(forecast?.rorPegPct)} />
            <Metric label="Est. ROR (Normal P/E)" value={pct(forecast?.rorNormalPct)} />
            <Metric
              label="Future price (P=E=G)"
              value={money(forecast?.futurePricePeg ?? snap.futurePrice)}
            />
            <Metric label="Future price (Normal)" value={money(forecast?.futurePriceNormal)} />
            <Metric
              label="Horizon"
              value={
                forecast?.horizonYears != null ? `${forecast.horizonYears.toFixed(2)}y` : '—'
              }
            />
            <Metric label="Margin of safety" value={pct(forecast?.marginOfSafetyPct)} />
            <Metric
              label="Fair Value $"
              value={money(
                forecastTtm != null && forecast?.fairValueRatio != null
                  ? forecastTtm * forecast.fairValueRatio
                  : summary?.fairValue,
              )}
            />
            <Metric label="Fwd EPS" value={money(snap.fwdEps)} />
            <Metric label="Fwd P/E" value={ratio(snap.fwdPe)} />
            <Metric label="Blended P/E" value={ratio(snap.blendedPe)} />
            <Metric label="Div Yld" value={pct(asPctPoints(snap.dividendYieldTTM))} />
            <label className="fund-custom-pe">
              <span>Custom P/E</span>
              <input
                type="number"
                min={1}
                step={0.1}
                value={customPe}
                onChange={(e) => setCustomPe(e.target.value)}
                placeholder="21"
              />
            </label>
            {customMultiple != null ? (
              <Metric label={`Est. ROR (${customMultiple}×)`} value={pct(forecast?.rorCustomPct)} />
            ) : null}
          </aside>
        </div>
      ) : null}

      {tab === 'summary' && fyRows.length ? (
        <section className="fund-section">
          <h3 className="fund-section-title">FY {METRIC_TABLE_LABEL[metric]} / Chg / Div</h3>
          <div className="fund-table-wrap">
            <table className="fund-table">
              <thead>
                <tr>
                  <th>Year</th>
                  <th>{METRIC_TABLE_LABEL[metric]}</th>
                  <th>% Chg</th>
                  <th>Div</th>
                  <th>OCF cov</th>
                  <th>FCF cov</th>
                </tr>
              </thead>
              <tbody>
                {fyRows.map((row) => {
                  const cover = dividendCoverage({
                    dividend: row.dividend,
                    dilutedShares: row.dilutedShares,
                    operatingCashFlow: row.operatingCashFlow,
                    freeCashFlow: row.freeCashFlow,
                  });
                  return (
                    <tr key={row.date}>
                      <td>{row.year}</td>
                      <td>{money(row.metric)}</td>
                      <td>{pct(row.metricChgPct)}</td>
                      <td>{money(row.dividend)}</td>
                      <td>{cover.ocfCover != null ? `${cover.ocfCover.toFixed(1)}×` : '—'}</td>
                      <td>
                        {cover.fcfCover != null ? `${cover.fcfCover.toFixed(1)}×` : '—'}
                        {cover.status !== 'none' ? ` ${cover.status}` : ''}
                      </td>
                    </tr>
                  );
                })}
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
                      <td>{money(fairValueFromEstimate(row.eps, forecast?.fairValueRatio ?? summary?.fairValueRatio))}</td>
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

      {tab === 'forecasting' && snap?.analystScorecard ? (
        <section className="fund-section">
          <h3 className="fund-section-title">Analyst scorecard</h3>
          <div className="fund-metric-grid">
            <Metric
              label="1Y beat / meet / miss"
              value={scorecardLine(snap.analystScorecard.y1)}
            />
            <Metric
              label="2Y beat / meet / miss"
              value={scorecardLine(snap.analystScorecard.y2)}
            />
          </div>
        </section>
      ) : null}

      {tab === 'dcf' ? (
        <DcfTab
          ticker={ticker}
          lynchFairValue={summary?.fairValue ?? null}
          price={summary?.currentPrice ?? profile?.price ?? null}
          lastHistDate={fundQ.data?.annual[fundQ.data.annual.length - 1]?.date ?? null}
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
          Historical Graph Key uses trailing metric CAGR on the selected 1Y / 3Y / 5Y / 8Y / 10Y /
          15Y / MAX window:
          8.5+2g when 0 ≤ growth &lt; 5%, 15× when growth is negative or 5–15% (or a short CAGR
          span), else P/E = growth %. Long windows do not take P/E=G from a stub early base
          when the 5Y path is still below 15% (CRM 10Y). Default chart metric is Op. EPS (NOPAT /
          diluted shares).
          FCF/sh is the cash-flow companion; both plot FV, last value, and a 3y dashed overlay.
          GAAP diluted EPS stays internal (Street estimates, ADR scale, FCF/sh growth borrow)
          and is not a Summary chip. For ADRs / foreign books (filing currency ≠ listing, e.g.
          NOK EUR vs NYSE USD) that internal EPS line uses FMP historical consensus (epsAvg) in
          listing units; FX-scaled GAAP stays on gaapEps. Street estimates are already
          listing-currency and are not FX-converted again. 1 NOK ADR = 1 Helsinki share.
          Forecasting uses a separate Street-to-Street CAGR and can flip the rule (AAPL
          Historical 25.67× vs Forecasting 15×). Est. ROR = (future price / today)^(1/horizon)
          − 1 + dividend yield. First estimate % Chg is blank so history and Street are not mixed.
          FCF/sh keeps trailing growth (borrowed from the internal EPS orange box). Dividends are
          summed on the fiscal year (FG DPS), not the calendar year; streak / Div CAGR come from
          that series. Normal P/E uses the last close on or before each FY-end. Results / Value
          cards persist the same 5Y Op. EPS trailing valuation as this Summary default. Snapshot
          DCF is FMP&apos;s simple
          headline; the DCF tab is Custom DCF. FMP has no FactSet-adjusted operating series, FG
          score, or S&amp;P credit rating.
          {snap?.ttmAsOf ? ` TTM through ${snap.ttmAsOf}.` : ''}
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

function DcfTab({
  ticker,
  lynchFairValue,
  price: lynchPrice,
  lastHistDate,
  onDcfChartSeries,
}: {
  ticker: string;
  lynchFairValue: number | null;
  price: number | null;
  lastHistDate?: string | null;
  onDcfChartSeries?: (series: DcfScenarioSeries) => void;
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

  const baseQ = useQuery({
    queryKey: ['custom-dcf', ticker, {}],
    queryFn: () => api.customDcf(ticker, {}),
    enabled: Boolean(ticker),
    staleTime: 60_000,
  });

  useEffect(() => {
    if (!baseQ.data || seeded.current) return;
    seeded.current = true;
    setBase(assumptionsFromPayload(baseQ.data));
    setDraft(draftFromPayload(baseQ.data));
  }, [baseQ.data]);

  const consOverrides = useMemo(
    () => (base ? presetOverrides(base, 'conservative') : null),
    [base],
  );
  const optOverrides = useMemo(() => (base ? presetOverrides(base, 'optimistic') : null), [base]);

  const consQ = useQuery({
    queryKey: ['custom-dcf', ticker, consOverrides],
    queryFn: () => api.customDcf(ticker, consOverrides!),
    enabled: Boolean(ticker && consOverrides),
    staleTime: 60_000,
  });
  const optQ = useQuery({
    queryKey: ['custom-dcf', ticker, optOverrides],
    queryFn: () => api.customDcf(ticker, optOverrides!),
    enabled: Boolean(ticker && optOverrides),
    staleTime: 60_000,
  });

  const data = dcfQ.data;
  const price = data?.price ?? lynchPrice;

  const payloadFor = (id: DcfPreset): CustomDcfPayload | undefined => {
    if (preset === id && data) return data;
    if (id === 'conservative') return consQ.data;
    if (id === 'optimistic') return optQ.data;
    return baseQ.data;
  };
  const todayOf = (id: DcfPreset) => {
    const payload = payloadFor(id);
    return payload ? dcfFairValueToday(payload, lastHistDate) : null;
  };
  const fvToday = {
    conservative: todayOf('conservative'),
    base: todayOf('base'),
    optimistic: todayOf('optimistic'),
  };
  const dcfPrice = fvToday[preset];
  const premiumPct =
    price != null && dcfPrice != null && dcfPrice > 0
      ? ((price - dcfPrice) / dcfPrice) * 100
      : (data?.premiumPct ?? null);
  const premiumClass =
    premiumPct == null
      ? ''
      : premiumPct > 10
        ? 'fund-neg'
        : premiumPct < -10
          ? 'fund-pos'
          : '';

  const applyPreset = (next: DcfPreset) => {
    setPreset(next);
    const snapshot = base ?? (baseQ.data ? assumptionsFromPayload(baseQ.data) : null);
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

  const yearlyFairValue = useMemo(() => {
    if (!data) return new Map<number, number | null>();
    const rows = expectedDcfFairValueByYear({
      years: data.years,
      wacc: data.wacc,
      terminalValue: data.terminalValue,
      netDebt: data.netDebt,
      dilutedShares: data.dilutedShares,
      lastHistDate,
    });
    return new Map(rows.map((r) => [r.year, r.fairValuePerShare]));
  }, [data, lastHistDate]);

  const chartSeries = useMemo<DcfScenarioSeries>(() => {
    const out: DcfScenarioSeries = {
      conservative: consQ.data ? dcfChartSeriesFromPayload(consQ.data, lastHistDate) : [],
      base: baseQ.data ? dcfChartSeriesFromPayload(baseQ.data, lastHistDate) : [],
      optimistic: optQ.data ? dcfChartSeriesFromPayload(optQ.data, lastHistDate) : [],
    };
    if (data) out[preset] = dcfChartSeriesFromPayload(data, lastHistDate);
    return out;
  }, [baseQ.data, consQ.data, optQ.data, data, preset, lastHistDate]);

  useEffect(() => {
    if (!onDcfChartSeries) return;
    onDcfChartSeries(chartSeries);
    return () => onDcfChartSeries(EMPTY_DCF_SCENARIO_SERIES);
  }, [chartSeries, onDcfChartSeries]);

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
            <span className={premiumClass}>{pct(premiumPct)} vs DCF</span>
            {lynchFairValue != null ? <> · Lynch FV {money(lynchFairValue)}</> : null}
          </p>
          <ul className="fund-dcf-today" aria-label="DCF fair value today">
            {DCF_PRESETS.map((id) => (
              <li key={id} className={preset === id ? 'is-active' : undefined}>
                <span>{DCF_PRESET_LABEL[id]}</span>
                <strong>{money(fvToday[id])}</strong>
              </li>
            ))}
          </ul>
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
                    <td>{money(yearlyFairValue.get(row.year) ?? null)}</td>
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
        <p className="muted small">
          Price vs SPY from Yahoo bars. EPS CAGR uses the internal GAAP/Street EPS series (FMP
          GAAP diluted for same-currency US names; FMP Street consensus for ADR / foreign
          books). No SPY EPS line.
        </p>
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
