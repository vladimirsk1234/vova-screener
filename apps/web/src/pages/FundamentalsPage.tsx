import { useEffect, useMemo, useRef, useState } from 'react';
import { Link, useNavigate, useParams } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { buildValuationSeries, type ValuationMetric } from '@vova/engine';
import { api } from '../lib/api';
import { Chips } from '../components/Chips';
import { mountValuationChart } from '../components/mountValuationChart';

const METRICS = [
  { id: 'eps' as const, label: 'EPS' },
  { id: 'revenue' as const, label: 'Sales/sh' },
  { id: 'fcf' as const, label: 'FCF/sh' },
  { id: 'ownerEarnings' as const, label: 'Owner earn.' },
];

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

function ratio(n: number | null | undefined) {
  if (n == null || !Number.isFinite(n)) return '—';
  return n.toFixed(2);
}

function compact(n: number | null | undefined) {
  if (n == null || !Number.isFinite(n)) return '—';
  return new Intl.NumberFormat(undefined, {
    notation: 'compact',
    maximumFractionDigits: 2,
  }).format(n);
}

export function FundamentalsPage() {
  const { ticker = '' } = useParams();
  const navigate = useNavigate();
  const [metric, setMetric] = useState<ValuationMetric>('eps');
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
    if (metric === fundQ.data.valuation.summary.metric) return fundQ.data.valuation;
    return buildValuationSeries(fundQ.data.annual, metric, {
      currentPrice: fundQ.data.profile.price,
    });
  }, [fundQ.data, metric]);

  useEffect(() => {
    if (!hostRef.current || !valuation) return;
    destroyRef.current?.();
    const mounted = mountValuationChart(hostRef.current, valuation.series);
    destroyRef.current = mounted.destroy;
    return () => {
      destroyRef.current?.();
      destroyRef.current = null;
    };
  }, [valuation]);

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
      <header className="chart-topbar">
        <button type="button" className="chart-icon-btn" aria-label="Back" onClick={() => navigate(-1)}>
          ←
        </button>
        <div className="chart-title-block">
          <h1 className="chart-title">{ticker}</h1>
          <p className="muted small">{profile?.companyName ?? 'Fundamentals'}</p>
        </div>
        <Link className="btn-sm ghost" to={`/chart/${encodeURIComponent(ticker)}`}>
          Price chart
        </Link>
      </header>

      <section className="fund-hero">
        <div className="fund-hero-main">
          <p className="fund-kicker">Valuation</p>
          <h2 className="fund-headline">
            {summary ? money(summary.fairValue) : '—'}
            <span className="fund-headline-unit"> fair value</span>
          </h2>
          <p className="fund-sub">
            Price {money(summary?.currentPrice)} ·{' '}
            <span className={premiumClass}>{pct(summary?.premiumPct)} vs fair</span>
          </p>
        </div>
        <dl className="fund-hero-stats">
          <div>
            <dt>Normal mult.</dt>
            <dd>{summary ? money(summary.normalMultiple, 1) : '—'}×</dd>
          </div>
          <div>
            <dt>CAGR</dt>
            <dd>{pct(summary?.metricCagrPct)}</dd>
          </div>
          <div>
            <dt>Years</dt>
            <dd>{summary?.years ?? '—'}</dd>
          </div>
        </dl>
      </section>

      <Chips
        value={metric}
        options={METRICS.map((m) => m.id)}
        format={(id) => METRICS.find((m) => m.id === id)?.label ?? id}
        onChange={setMetric}
      />

      <div className="fund-chart-stage">
        <div className="fund-chart-host" ref={hostRef} />
        <ul className="fund-legend" aria-label="Chart legend">
          <li>
            <span className="fund-swatch fund-swatch--power" /> Earnings power
          </li>
          <li>
            <span className="fund-swatch fund-swatch--price" /> Price
          </li>
          <li>
            <span className="fund-swatch fund-swatch--fair" /> Fair value
          </li>
        </ul>
      </div>

      {fundQ.isLoading ? <p className="muted small">Loading FMP fundamentals…</p> : null}
      {fundQ.error ? (
        <p className="error">
          {(fundQ.error as Error).message.includes('FMP_API_KEY')
            ? 'Set FMP_API_KEY on the API server to load fundamentals.'
            : (fundQ.error as Error).message}
        </p>
      ) : null}

      {snap ? (
        <section className="fund-section">
          <h3 className="fund-section-title">Snapshot</h3>
          <div className="fund-metric-grid">
            <Metric label="P/E TTM" value={ratio(snap.peTTM)} />
            <Metric label="PEG" value={ratio(snap.pegTTM)} />
            <Metric label="P/B" value={ratio(snap.pbTTM)} />
            <Metric label="P/S" value={ratio(snap.psTTM)} />
            <Metric label="ROE" value={pct(asPctPoints(snap.roeTTM))} />
            <Metric label="ROIC" value={pct(asPctPoints(snap.roicTTM))} />
            <Metric label="Op. margin" value={pct(asPctPoints(snap.operatingMarginTTM))} />
            <Metric label="Profit margin" value={pct(asPctPoints(snap.profitMarginTTM))} />
            <Metric label="FCF yield" value={pct(asPctPoints(snap.fcfYieldTTM))} />
            <Metric label="Div yield" value={pct(asPctPoints(snap.dividendYieldTTM))} />
            <Metric label="D/E" value={ratio(snap.debtToEquityTTM)} />
            <Metric label="Current ratio" value={ratio(snap.currentRatioTTM)} />
            <Metric label="DCF" value={money(snap.dcf)} />
            <Metric label="vs DCF" value={pct(snap.dcfPremiumPct)} tone={snap.dcfPremiumPct} />
            <Metric label="Piotroski" value={snap.piotroskiScore != null ? String(snap.piotroskiScore) : '—'} />
            <Metric label="Altman Z" value={ratio(snap.altmanZScore)} />
          </div>
        </section>
      ) : null}

      {fundQ.data?.incomeTrend?.length ? (
        <section className="fund-section">
          <h3 className="fund-section-title">Annual trend</h3>
          <div className="fund-table-wrap">
            <table className="fund-table">
              <thead>
                <tr>
                  <th>Year</th>
                  <th>Revenue</th>
                  <th>Net income</th>
                  <th>EPS</th>
                  <th>FCF</th>
                </tr>
              </thead>
              <tbody>
                {fundQ.data.incomeTrend.map((row) => (
                  <tr key={row.date}>
                    <td>{row.year}</td>
                    <td>{compact(row.revenue)}</td>
                    <td>{compact(row.netIncome)}</td>
                    <td>{money(row.eps)}</td>
                    <td>{compact(row.freeCashFlow)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      ) : null}

      {profile?.description ? (
        <section className="fund-section">
          <h3 className="fund-section-title">About</h3>
          <p className="fund-about">{profile.description}</p>
          <p className="muted small">
            {[profile.sector, profile.industry, profile.exchange].filter(Boolean).join(' · ')}
            {profile.mktCap != null ? ` · Cap ${compact(profile.mktCap)}` : ''}
          </p>
        </section>
      ) : null}

      <p className="muted small fund-footnote">
        Fair value ≈ selected metric × normal multiple (median historical price/metric). Inspired by
        FAST Graphs; figures from Financial Modeling Prep — not identical to FAST Graphs operating
        earnings methodology.
        {fundQ.data?.cached ? ' · cached' : ''}
      </p>
    </div>
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
