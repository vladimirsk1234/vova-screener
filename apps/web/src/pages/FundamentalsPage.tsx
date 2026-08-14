import { useEffect, useMemo, useRef, useState } from 'react';
import { Link, useNavigate, useParams } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { buildValuationSeries, sliceToWindow, type ValuationMetric, type ValuationWindowYears } from '@vova/engine';
import { api, type HorizonReturns } from '../lib/api';
import { Chips } from '../components/Chips';
import { mountValuationChart } from '../components/mountValuationChart';

const METRICS = [
  { id: 'eps' as const, label: 'EPS' },
  { id: 'revenue' as const, label: 'Sales/sh' },
  { id: 'fcf' as const, label: 'FCF/sh' },
  { id: 'ownerEarnings' as const, label: 'Owner earn.' },
];

const TABS = ['summary', 'forecasting', 'performance', 'profile'] as const;
type FundTab = (typeof TABS)[number];
const TAB_LABEL: Record<FundTab, string> = {
  summary: 'Summary',
  forecasting: 'Forecasting',
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

/** Forward growth spans the window through the last estimate, so "5y" would be wrong. */
function growthLabel(source: string | undefined) {
  return source === 'forward' ? 'Growth (fwd)' : 'Growth (5y)';
}

export function FundamentalsPage() {
  const { ticker = '' } = useParams();
  const navigate = useNavigate();
  const [metric, setMetric] = useState<ValuationMetric>('eps');
  const [windowYears, setWindowYears] = useState<ValuationWindowYears>(5);
  const [tab, setTab] = useState<FundTab>('summary');
  const hostRef = useRef<HTMLDivElement | null>(null);
  const destroyRef = useRef<(() => void) | null>(null);

  const fundQ = useQuery({
    queryKey: ['fundamentals', ticker],
    queryFn: () => api.fundamentals(ticker, 'eps'),
    enabled: Boolean(ticker),
    staleTime: 60_000,
  });

  const valuation = useMemo(() => {
    if (!fundQ.data) return null;
    return buildValuationSeries(fundQ.data.annual, metric, {
      currentPrice: fundQ.data.profile.price,
      windowYears,
      // FMP only estimates EPS; the other metrics stay on trailing growth.
      forward:
        metric === 'eps'
          ? fundQ.data.estimates.map((e) => ({ year: e.year, metric: e.eps }))
          : [],
    });
  }, [fundQ.data, metric, windowYears]);

  const chartSeries = useMemo(() => {
    if (!fundQ.data || !valuation) return [];
    if (tab !== 'forecasting') return valuation.series;
    const lastYear = valuation.series[valuation.series.length - 1]?.year ?? 0;
    const extra = fundQ.data.forecastSeries.filter(
      (p) => p.estimated && p.year > lastYear,
    );
    return [...valuation.series, ...extra];
  }, [fundQ.data, valuation, tab]);

  const fyRows = useMemo(() => {
    if (!fundQ.data) return [];
    const windowed = sliceToWindow(fundQ.data.annual, windowYears);
    const minYear = windowed[0]?.year;
    if (minYear == null) return [];
    return fundQ.data.incomeTrend.filter((row) => row.year >= minYear);
  }, [fundQ.data, windowYears]);

  useEffect(() => {
    if (!hostRef.current || !chartSeries.length || (tab !== 'summary' && tab !== 'forecasting')) {
      return;
    }
    destroyRef.current?.();
    const mounted = mountValuationChart(hostRef.current, chartSeries);
    destroyRef.current = mounted.destroy;
    return () => {
      destroyRef.current?.();
      destroyRef.current = null;
    };
  }, [chartSeries, tab]);

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
    <div className="fund-page">
      <div className="chart-head">
        <button
          type="button"
          className="chart-icon-btn ghost"
          aria-label="Back"
          onClick={() => navigate(-1)}
        >
          ←
        </button>
        <div className="chart-head-title">
          <div className="chart-head-name ellipsis">
            <strong>{ticker}</strong>
            {profile?.companyName ? (
              <span className="muted small">{profile.companyName}</span>
            ) : null}
          </div>
        </div>
        <Link className="btn-sm ghost" to={`/chart/${encodeURIComponent(ticker)}`}>
          Price chart
        </Link>
      </div>

      <Chips
        value={tab}
        options={TABS}
        format={(id) => TAB_LABEL[id]}
        onChange={setTab}
      />

      {fundQ.isLoading ? <p className="muted small">Loading FMP fundamentals…</p> : null}
      {fundQ.error ? (
        <p className="error">
          {(fundQ.error as Error).message.includes('FMP_API_KEY')
            ? 'Set FMP_API_KEY on the API server to load fundamentals.'
            : (fundQ.error as Error).message}
        </p>
      ) : null}

      {tab === 'summary' || tab === 'forecasting' ? (
        <>
          <section className="fund-hero">
            <div className="fund-hero-main">
              <p className="fund-kicker">
                {tab === 'forecasting' ? 'Forecast' : 'Valuation'}
              </p>
              <h2 className="fund-headline">
                {tab === 'forecasting' ? money(snap?.futurePrice) : money(summary?.fairValue)}
                <span className="fund-headline-unit">
                  {tab === 'forecasting' ? ' future price' : ' fair value'}
                </span>
              </h2>
              <p className="fund-sub">
                Price {money(summary?.currentPrice)} ·{' '}
                <span className={premiumClass}>{pct(summary?.premiumPct)} vs fair</span>
                {tab === 'forecasting' && snap?.estAnnualRorPct != null ? (
                  <> · Est. ROR {pct(snap.estAnnualRorPct)}</>
                ) : null}
              </p>
            </div>
            <dl className="fund-hero-stats">
              <div>
                <dt>{growthLabel(summary?.growthSource)}</dt>
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
                <dd>{summary ? money(summary.normalMultiple, 1) : '—'}×</dd>
              </div>
            </dl>
          </section>

          {tab === 'summary' || tab === 'forecasting' ? (
            <>
              {tab === 'summary' ? (
                <Chips
                  value={metric}
                  options={METRICS.map((m) => m.id)}
                  format={(id) => METRICS.find((m) => m.id === id)?.label ?? id}
                  onChange={setMetric}
                />
              ) : null}
              <Chips
                value={windowYears == null ? 'max' : String(windowYears)}
                options={['5', '10', 'max']}
                format={(id) => (id === 'max' ? 'MAX' : `${id}Y`)}
                onChange={(id) => setWindowYears(id === 'max' ? null : (Number(id) as 5 | 10))}
              />
            </>
          ) : null}

          <div className="fund-layout">
            <div className="fund-chart-stage">
              <div className="fund-chart-host" ref={hostRef} />
              <ul className="fund-legend" aria-label="Chart legend">
                <li>
                  <span className="fund-swatch fund-swatch--power" /> EPS
                </li>
                <li>
                  <span className="fund-swatch fund-swatch--price" /> Price
                </li>
                <li>
                  <span className="fund-swatch fund-swatch--fair" /> Fair value
                </li>
                <li>
                  <span className="fund-swatch fund-swatch--normal" /> Normal P/E
                </li>
              </ul>
            </div>

            {tab === 'summary' && snap ? (
              <aside className="fund-sidebar">
                <Metric
                  label={
                    summary?.growthSource === 'forward' ? 'Growth Rate (fwd)' : 'Growth Rate'
                  }
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
                <Metric label="Normal P/E" value={`${ratio(summary?.normalMultiple, 1)}×`} />
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
              </aside>
            ) : null}

            {tab === 'forecasting' && snap ? (
              <aside className="fund-sidebar">
                <Metric label="Est. Annual ROR" value={pct(snap.estAnnualRorPct)} />
                <Metric label="Fair Value $" value={money(summary?.fairValue)} />
                <Metric label="Future price" value={money(snap.futurePrice)} />
                <Metric label="Fwd EPS" value={money(snap.fwdEps)} />
                <Metric label="Fwd P/E" value={ratio(snap.fwdPe)} />
                <Metric label="Blended P/E" value={ratio(snap.blendedPe)} />
                <Metric label="Div Yld" value={pct(asPctPoints(snap.dividendYieldTTM))} />
              </aside>
            ) : null}
          </div>
        </>
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
                    <th># Analysts</th>
                  </tr>
                </thead>
                <tbody>
                  {fundQ.data.estimates.map((row) => (
                    <tr key={row.date || row.year}>
                      <td>{row.year}</td>
                      <td>{money(row.eps)}</td>
                      <td>{pct(row.epsChgPct)}</td>
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

      <p className="muted small fund-footnote">
        Fair value = GAAP diluted EPS × 15× when 5y EPS CAGR &lt; 15%, else PEG=1 (ratio = growth %).
        Normal P/E is the median price/EPS on the selected 5Y / 10Y / MAX window. Figures from
        Financial Modeling Prep — GAAP diluted, not FAST Graphs adjusted operating EPS. S&amp;P
        credit rating is not in FMP.
        {fundQ.data?.cached ? ' · cached' : ''}
      </p>
    </div>
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
