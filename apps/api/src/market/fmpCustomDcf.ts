/**
 * FMP Custom DCF mapping, assumption sanitization, and query encoding.
 * Kept out of the Nest client so unit tests can run without decorators.
 */

type Json = Record<string, unknown>;

function num(v: unknown): number | null {
  if (v == null || v === '') return null;
  if (typeof v === 'string') {
    const trimmed = v.trim();
    if (!trimmed) return null;
    const n = Number(trimmed);
    return Number.isFinite(n) ? n : null;
  }
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}

/** FMP Custom DCF years arrive as strings ("2026") and sometimes as ISO dates. */
export function fmpYearNum(v: unknown): number | null {
  if (typeof v === 'number' && Number.isFinite(v)) return Math.trunc(v);
  if (typeof v === 'string') {
    const trimmed = v.trim();
    if (!trimmed) return null;
    const direct = Number(trimmed);
    if (Number.isFinite(direct) && Math.abs(direct) >= 1000) return Math.trunc(direct);
    const m = /(\d{4})/.exec(trimmed);
    return m ? Number(m[1]) : null;
  }
  return num(v);
}

export const CUSTOM_DCF_ASSUMPTION_KEYS = [
  'revenueGrowthPct',
  'ebitdaPct',
  'operatingCashFlowPct',
  'capitalExpenditurePct',
  'longTermGrowthRate',
  'taxRate',
  'riskFreeRate',
  'marketRiskPremium',
  'costOfEquity',
  'costOfDebt',
] as const;

/**
 * FMP `/custom-discounted-cash-flow` query units are inconsistent.
 * These fields are stored as percents (4.72, 4, 9.37) — not 0.0472.
 * Growth / margin / capex fields stay decimals (0.0409 = 4.09%).
 */
export const FMP_PERCENT_QUERY_KEYS = [
  'longTermGrowthRate',
  'taxRate',
  'riskFreeRate',
  'marketRiskPremium',
  'costOfEquity',
  'costOfDebt',
  'wacc',
  'afterTaxCostOfDebt',
  'debtWeighting',
  'equityWeighting',
] as const;

export type CustomDcfAssumptionKey = (typeof CUSTOM_DCF_ASSUMPTION_KEYS)[number];
export type CustomDcfAssumptions = Partial<Record<CustomDcfAssumptionKey, number>>;

export type CustomDcfYear = {
  year: number;
  revenue: number | null;
  ebitda: number | null;
  ebit: number | null;
  depreciation: number | null;
  capitalExpenditure: number | null;
  ufcf: number | null;
  pvUfcf: number | null;
};

export type CustomDcfPayload = {
  yahooTicker: string;
  fmpSymbol: string;
  model: 'unlevered';
  price: number | null;
  equityValuePerShare: number | null;
  premiumPct: number | null;
  enterpriseValue: number | null;
  equityValue: number | null;
  netDebt: number | null;
  terminalValue: number | null;
  presentTerminalValue: number | null;
  sumPvUfcf: number | null;
  dilutedShares: number | null;
  wacc: number | null;
  beta: number | null;
  costOfEquity: number | null;
  costOfDebt: number | null;
  afterTaxCostOfDebt: number | null;
  taxRate: number | null;
  riskFreeRate: number | null;
  marketRiskPremium: number | null;
  debtWeighting: number | null;
  equityWeighting: number | null;
  longTermGrowthRate: number | null;
  revenueGrowthPct: number | null;
  ebitdaPct: number | null;
  capitalExpenditurePct: number | null;
  operatingCashFlowPct: number | null;
  years: CustomDcfYear[];
  /** True when WACC − g is under 1pp — Gordon growth is unstable. */
  fragile: boolean;
  /** presentTerminalValue / enterpriseValue × 100. */
  terminalSharePct: number | null;
  asOf: string;
  cached: boolean;
};

/** Normalize UI / query input to a decimal (0.08 for 8%). Accept 8 or 0.08. */
export function toFmpDecimal(n: number): number {
  return Math.abs(n) > 1.5 ? n / 100 : n;
}

function roundFmpQuery(n: number): number {
  return Math.round(n * 1e10) / 1e10;
}

const FMP_PERCENT_QUERY_KEY_SET = new Set<string>(FMP_PERCENT_QUERY_KEYS);

/**
 * Convert app-decimal assumptions back to FMP query units.
 * ERP 0.0472 → 4.72; long-term g 0.04 → 4; revenueGrowthPct 0.0409 stays 0.0409.
 */
export function encodeAssumptionsForFmp(assumptions: CustomDcfAssumptions): Record<string, number> {
  const out: Record<string, number> = {};
  for (const key of CUSTOM_DCF_ASSUMPTION_KEYS) {
    const v = assumptions[key];
    if (v == null || !Number.isFinite(v)) continue;
    out[key] = FMP_PERCENT_QUERY_KEY_SET.has(key) ? roundFmpQuery(v * 100) : v;
  }
  return out;
}

export function sanitizeCustomDcfAssumptions(raw: Record<string, unknown>): CustomDcfAssumptions {
  const out: CustomDcfAssumptions = {};
  for (const key of CUSTOM_DCF_ASSUMPTION_KEYS) {
    const n = num(raw[key]);
    if (n == null) continue;
    out[key] = toFmpDecimal(n);
  }
  return out;
}

export function customDcfCacheKey(symbol: string, assumptions: CustomDcfAssumptions): string {
  const parts = CUSTOM_DCF_ASSUMPTION_KEYS.map((k) => {
    const v = assumptions[k];
    return v == null || !Number.isFinite(v) ? '' : `${k}=${v}`;
  });
  return `${symbol.toUpperCase()}|${parts.join('&')}`;
}

export function emptyCustomDcf(yahooTicker: string, fmpSymbol: string): CustomDcfPayload {
  return {
    yahooTicker,
    fmpSymbol,
    model: 'unlevered',
    price: null,
    equityValuePerShare: null,
    premiumPct: null,
    enterpriseValue: null,
    equityValue: null,
    netDebt: null,
    terminalValue: null,
    presentTerminalValue: null,
    sumPvUfcf: null,
    dilutedShares: null,
    wacc: null,
    beta: null,
    costOfEquity: null,
    costOfDebt: null,
    afterTaxCostOfDebt: null,
    taxRate: null,
    riskFreeRate: null,
    marketRiskPremium: null,
    debtWeighting: null,
    equityWeighting: null,
    longTermGrowthRate: null,
    revenueGrowthPct: null,
    ebitdaPct: null,
    capitalExpenditurePct: null,
    operatingCashFlowPct: null,
    years: [],
    fragile: false,
    terminalSharePct: null,
    asOf: new Date().toISOString(),
    cached: false,
  };
}

function asRows(v: unknown): Json[] {
  if (Array.isArray(v)) return v as Json[];
  if (v && typeof v === 'object') {
    const obj = v as Json;
    if (Array.isArray(obj.data)) return obj.data as Json[];
    if (obj.year != null || obj.symbol != null || obj.equityValuePerShare != null || obj.dcf != null) {
      return [obj];
    }
  }
  return [];
}

function pickNum(r: Json, ...keys: string[]): number | null {
  for (const k of keys) {
    const n = num(r[k]);
    if (n != null) return n;
  }
  return null;
}

/** Rates in the FMP payload mix 0.08 and 8. Store as decimals. */
function asDecimal(n: number | null): number | null {
  if (n == null) return null;
  return toFmpDecimal(n);
}

export function mapCustomDcf(
  yahooTicker: string,
  fmpSymbol: string,
  raw: unknown,
  requested: CustomDcfAssumptions,
): CustomDcfPayload {
  const rows = asRows(raw)
    .slice()
    .sort((a, b) => (fmpYearNum(a.year) ?? 0) - (fmpYearNum(b.year) ?? 0));
  const empty = emptyCustomDcf(yahooTicker, fmpSymbol);
  if (!rows.length) return empty;

  const last = rows[rows.length - 1] ?? {};
  const first = rows[0] ?? {};
  const wacc = asDecimal(pickNum(last, 'wacc', 'WACC'));
  const years: CustomDcfYear[] = rows.map((r, i) => {
    const year = fmpYearNum(r.year) ?? num(r.year) ?? i + 1;
    const ufcf =
      pickNum(r, 'ufcf', 'unleveredFreeCashFlow', 'freeCashFlow', 'fcf') ?? null;
    const pvGiven = pickNum(r, 'pvUfcf', 'presentValueUfcf', 'pvFreeCashFlow');
    const pvUfcf =
      pvGiven ??
      (ufcf != null && wacc != null && wacc > -0.99
        ? ufcf / Math.pow(1 + wacc, i + 1)
        : null);
    return {
      year,
      revenue: pickNum(r, 'revenue'),
      ebitda: pickNum(r, 'ebitda'),
      ebit: pickNum(r, 'ebit'),
      depreciation: pickNum(r, 'depreciation'),
      capitalExpenditure: pickNum(r, 'capitalExpenditure', 'capex'),
      ufcf,
      pvUfcf,
    };
  });

  const price = pickNum(last, 'price', 'Stock Price') ?? pickNum(first, 'price', 'Stock Price');
  // Today’s DCF lives on the first / headline row. Last-row equityValuePerShare can be
  // the terminal-year FV and would plot a V against the local roll-forward path.
  const equityValuePerShare =
    pickNum(first, 'equityValuePerShareToday', 'dcf', 'equityValuePerShare') ??
    pickNum(last, 'equityValuePerShareToday', 'dcf');
  const enterpriseValue = pickNum(last, 'enterpriseValue') ?? pickNum(first, 'enterpriseValue');
  const presentTerminalValue =
    pickNum(last, 'presentTerminalValue', 'pvTerminalValue') ??
    pickNum(first, 'presentTerminalValue', 'pvTerminalValue');
  const g = asDecimal(
    requested.longTermGrowthRate ?? pickNum(last, 'longTermGrowthRate', 'terminalGrowthRate'),
  );
  const premiumPct =
    price != null && equityValuePerShare != null && equityValuePerShare > 0
      ? ((price - equityValuePerShare) / equityValuePerShare) * 100
      : null;
  const terminalSharePct =
    presentTerminalValue != null && enterpriseValue != null && enterpriseValue !== 0
      ? (presentTerminalValue / enterpriseValue) * 100
      : null;
  const fragile = wacc != null && g != null && wacc - g < 0.01;

  return {
    ...empty,
    price,
    equityValuePerShare,
    premiumPct,
    enterpriseValue,
    equityValue: pickNum(last, 'equityValue') ?? pickNum(first, 'equityValue'),
    netDebt: pickNum(last, 'netDebt') ?? pickNum(first, 'netDebt'),
    terminalValue: pickNum(last, 'terminalValue') ?? pickNum(first, 'terminalValue'),
    presentTerminalValue,
    sumPvUfcf: pickNum(last, 'sumPvUfcf') ?? pickNum(first, 'sumPvUfcf'),
    dilutedShares:
      pickNum(last, 'dilutedSharesOutstanding', 'dilutedShares', 'sharesOutstanding') ??
      pickNum(first, 'dilutedSharesOutstanding', 'dilutedShares', 'sharesOutstanding'),
    wacc,
    beta: pickNum(last, 'beta') ?? pickNum(first, 'beta'),
    costOfEquity: asDecimal(
      requested.costOfEquity ?? pickNum(last, 'costOfEquity') ?? pickNum(first, 'costOfEquity'),
    ),
    costOfDebt: asDecimal(
      requested.costOfDebt ?? pickNum(last, 'costOfDebt') ?? pickNum(first, 'costOfDebt'),
    ),
    afterTaxCostOfDebt: asDecimal(
      pickNum(last, 'afterTaxCostOfDebt') ?? pickNum(first, 'afterTaxCostOfDebt'),
    ),
    taxRate: asDecimal(requested.taxRate ?? pickNum(last, 'taxRate') ?? pickNum(first, 'taxRate')),
    riskFreeRate: asDecimal(
      requested.riskFreeRate ?? pickNum(last, 'riskFreeRate') ?? pickNum(first, 'riskFreeRate'),
    ),
    marketRiskPremium: asDecimal(
      requested.marketRiskPremium ??
        pickNum(last, 'marketRiskPremium') ??
        pickNum(first, 'marketRiskPremium'),
    ),
    debtWeighting: asDecimal(pickNum(last, 'debtWeighting') ?? pickNum(first, 'debtWeighting')),
    equityWeighting: asDecimal(pickNum(last, 'equityWeighting') ?? pickNum(first, 'equityWeighting')),
    longTermGrowthRate: g,
    revenueGrowthPct: asDecimal(
      requested.revenueGrowthPct ??
        pickNum(first, 'revenuePercentage', 'revenueGrowthPct', 'revenueGrowth'),
    ),
    ebitdaPct: asDecimal(
      requested.ebitdaPct ?? pickNum(first, 'ebitdaPercentage', 'ebitdaPct', 'ebitdaMargin'),
    ),
    capitalExpenditurePct: asDecimal(
      requested.capitalExpenditurePct ??
        pickNum(first, 'capitalExpenditurePercentage', 'capitalExpenditurePct'),
    ),
    operatingCashFlowPct: asDecimal(
      requested.operatingCashFlowPct ??
        pickNum(first, 'operatingCashFlowPercentage', 'operatingCashFlowPct'),
    ),
    years,
    fragile,
    terminalSharePct,
  };
}
