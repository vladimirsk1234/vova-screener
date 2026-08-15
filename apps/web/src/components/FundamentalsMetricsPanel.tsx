import { useMemo, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  buildValuationSeries,
  formatScaleCaption,
  type ValuationMetric,
  type ValuationWindowYears,
} from '@vova/engine';
import { api } from '../lib/api';
import { Chips } from './Chips';

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

function multiple(n: number | null | undefined) {
  if (n == null || !Number.isFinite(n)) return '—';
  const digits = n >= 1 ? 1 : n >= 0.1 ? 2 : 3;
  return n.toFixed(digits);
}

function fvRuleLabel(rule: string | undefined) {
  if (rule === 'pe15') return 'PE 15';
  if (rule === 'lynch_peg') return 'Lynch PEG=1';
  return 'N/A';
}

function growthLabel(source: string | undefined) {
  return source === 'forward' ? 'Growth (fwd)' : 'Growth (5y)';
}

export function FundamentalsMetricsPanel({ ticker }: { ticker: string }) {
  const [metric, setMetric] = useState<ValuationMetric>('eps');
  const [windowYears, setWindowYears] = useState<ValuationWindowYears>(5);

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
      ttmMetric: metric === 'eps' ? fundQ.data.snapshot.ttmEps : null,
    });
  }, [fundQ.data, metric, windowYears]);

  const summary = valuation?.summary;
  const snap = fundQ.data?.snapshot;
  const metricLabel = METRICS.find((m) => m.id === metric)?.label ?? 'EPS';
  const earningsValue =
    metric === 'eps' && snap?.ttmEps != null ? snap.ttmEps : (summary?.latestMetric ?? null);
  const earningsLabel = metric === 'eps' ? 'EPS (TTM)' : metricLabel;
  const scaleCaption = formatScaleCaption(fundQ.data?.scale ?? null);
  const premiumClass =
    summary?.premiumPct == null
      ? ''
      : summary.premiumPct > 10
        ? 'fund-neg'
        : summary.premiumPct < -10
          ? 'fund-pos'
          : '';

  return (
    <div className="fund-panel fund-metrics-panel">
      {fundQ.isLoading ? <p className="muted small">Loading FMP fundamentals…</p> : null}
      {fundQ.error ? (
        <p className="error">
          {(fundQ.error as Error).message.includes('FMP_API_KEY')
            ? 'Set FMP_API_KEY on the API server to load fundamentals.'
            : (fundQ.error as Error).message}
        </p>
      ) : null}

      <section className="fund-hero">
        <div className="fund-hero-main">
          <p className="fund-kicker">Valuation</p>
          <h2 className="fund-headline">
            {money(summary?.fairValue)}
            <span className="fund-headline-unit"> fair value</span>
          </h2>
          <p className="fund-sub">
            Price {money(summary?.currentPrice)} ·{' '}
            <span className={premiumClass}>{pct(summary?.premiumPct)} vs fair</span>
          </p>
          <p className="fund-sub">
            {earningsLabel} {money(earningsValue)}
            {scaleCaption ? ` · ${scaleCaption}` : ''}
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
            <dd>{summary ? `${multiple(summary.normalMultiple)}×` : '—'}</dd>
          </div>
        </dl>
      </section>

      <Chips
        value={metric}
        options={METRICS.map((m) => m.id)}
        format={(id) => METRICS.find((m) => m.id === id)?.label ?? id}
        onChange={setMetric}
      />
      <Chips
        value={windowYears == null ? 'max' : String(windowYears)}
        options={['5', '10', 'max']}
        format={(id) => (id === 'max' ? 'MAX' : `${id}Y`)}
        onChange={(id) => setWindowYears(id === 'max' ? null : (Number(id) as 5 | 10))}
      />
    </div>
  );
}
