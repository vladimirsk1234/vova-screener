import { buildDcfChartSeries, expectedDcfFairValueToday } from '@vova/engine';
import type { CustomDcfPayload, ValuationSeriesPoint } from './api';

export type DcfScenario = 'conservative' | 'base' | 'optimistic';
export type DcfScenarioSeries = Record<DcfScenario, ValuationSeriesPoint[]>;

export const EMPTY_DCF_SCENARIO_SERIES: DcfScenarioSeries = {
  conservative: [],
  base: [],
  optimistic: [],
};

export function dcfModelInput(data: CustomDcfPayload) {
  return {
    years: data.years,
    wacc: data.wacc,
    terminalValue: data.terminalValue,
    netDebt: data.netDebt,
    dilutedShares: data.dilutedShares,
  };
}

export function dcfFairValueToday(data: CustomDcfPayload): number | null {
  const local = expectedDcfFairValueToday(dcfModelInput(data));
  if (local != null && Number.isFinite(local) && local > 0) return local;
  if (
    data.equityValuePerShare != null &&
    Number.isFinite(data.equityValuePerShare) &&
    data.equityValuePerShare > 0
  ) {
    return data.equityValuePerShare;
  }
  return null;
}

export function dcfChartSeriesFromPayload(data: CustomDcfPayload): ValuationSeriesPoint[] {
  return buildDcfChartSeries({
    ...dcfModelInput(data),
    asOf: data.asOf || new Date().toISOString(),
    fmpEquityValuePerShare: data.equityValuePerShare,
  }).map((p) => ({
    date: p.date,
    year: p.year ?? Number(p.date.slice(0, 4)),
    price: p.year == null ? data.price : null,
    metric: null,
    earningsPower: p.fairValue,
    fairValue: p.fairValue,
    normalValue: null,
    pe: null,
    forecast: true,
    estimated: p.year != null,
  }));
}

export function dcfScenarioHasPoints(series: DcfScenarioSeries): boolean {
  return series.conservative.length > 0 || series.base.length > 0 || series.optimistic.length > 0;
}

export function flattenDcfScenarioSeries(series: DcfScenarioSeries): ValuationSeriesPoint[] {
  return [...series.conservative, ...series.base, ...series.optimistic];
}
