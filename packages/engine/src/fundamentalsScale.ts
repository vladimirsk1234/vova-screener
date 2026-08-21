/**
 * Align FMP per-share figures with the listing price (USD ADS, etc.).
 *
 * FMP often leaves EPS in the filing currency and/or on ordinary shares, and
 * sometimes applies an ADR ratio twice. The listing price is already per ADS
 * in the exchange currency — without this layer, PE15/Lynch runs on garbage.
 */

export const FUNDAMENTALS_SCALE_VERSION = 3;

/** Ordinary shares represented by one ADS. 1 = not an ADR / already per listing share. */
export const KNOWN_ADR_RATIO: Record<string, number> = {
  XYF: 6,
  TSM: 5,
  BABA: 8,
  BIDU: 8,
  JD: 2,
  HDB: 3,
  IBN: 1,
  INFY: 1,
  PDD: 4,
  NTES: 5,
};

/** Filing-currency units per 1 USD, used when FMP forex is missing. */
export const FALLBACK_FOREIGN_PER_USD: Record<string, number> = {
  USD: 1,
  CNY: 7.2,
  RMB: 7.2,
  CNH: 7.2,
  TWD: 32,
  HKD: 7.8,
  JPY: 150,
  KRW: 1350,
  INR: 84,
  ARS: 1100,
  BRL: 5.5,
  EUR: 0.92,
  GBP: 0.79,
  CAD: 1.37,
  AUD: 1.55,
  CHF: 0.88,
  SGD: 1.34,
  MXN: 18,
  ZAR: 18,
  IDR: 16000,
  TRY: 34,
  CLP: 950,
  COP: 4100,
  PHP: 58,
  THB: 35,
  MYR: 4.5,
  DKK: 6.9,
  SEK: 10.5,
  NOK: 10.7,
  ILS: 3.7,
  PLN: 4.0,
  CZK: 23,
  HUF: 360,
  RON: 4.6,
};

const COMMON_ADR = [2, 3, 4, 5, 6, 8, 10, 20, 25, 40] as const;
const REL_TOL = 0.15;

export type ShareScale = 'ordinary' | 'ads' | 'double_adr' | 'unknown';

export type FundamentalsScale = {
  version: number;
  reportedCurrency: string | null;
  listingCurrency: string | null;
  /** Multiply filing-currency amounts by this to reach listing currency. */
  fxToListing: number;
  adrRatio: number;
  shareScale: ShareScale;
  /** Multiply FMP per-share fields (after FX) by this to reach per-ADS listing units. */
  perShareFactor: number;
  reliable: boolean;
};

export function normalizeCurrency(code: string | null | undefined): string | null {
  if (!code) return null;
  const raw = String(code).trim().toUpperCase();
  if (!raw) return null;
  if (raw === 'RMB' || raw === 'CNH' || raw === 'CNY') return 'CNY';
  if (raw === 'GBX' || raw === 'GBP' || raw === 'GBp') return raw === 'GBX' || raw === 'GBp' ? 'GBX' : 'GBP';
  if (raw === 'ZAC') return 'ZAC';
  return raw;
}

export function knownAdrRatio(ticker: string | null | undefined): number {
  if (!ticker) return 1;
  const key = String(ticker)
    .trim()
    .toUpperCase()
    .replace(/-.*$/, '');
  const n = KNOWN_ADR_RATIO[key];
  return n != null && n > 0 ? n : 1;
}

export function fallbackForeignPerUsd(currency: string | null | undefined): number {
  const c = normalizeCurrency(currency);
  if (!c) return 1;
  if (c === 'GBX') return (FALLBACK_FOREIGN_PER_USD.GBP ?? 0.79) * 100;
  if (c === 'ZAC') return (FALLBACK_FOREIGN_PER_USD.ZAR ?? 18) * 100;
  return FALLBACK_FOREIGN_PER_USD[c] ?? 1;
}

/**
 * `foreignPerUsd` is "how many units of `currency` equal 1 USD" (USDCNY ≈ 7).
 * Result multiplies a `reported` amount into `listing` units.
 */
export function fxToListingMultiplier(
  reported: string | null | undefined,
  listing: string | null | undefined,
  foreignPerUsd: (currency: string) => number | null,
): number {
  const from = normalizeCurrency(reported) ?? 'USD';
  const to = normalizeCurrency(listing) ?? 'USD';
  if (from === to) return 1;
  const fromPerUsd = foreignPerUsd(from) ?? fallbackForeignPerUsd(from);
  const toPerUsd = foreignPerUsd(to) ?? fallbackForeignPerUsd(to);
  if (!(fromPerUsd > 0) || !(toPerUsd > 0)) return 1;
  return toPerUsd / fromPerUsd;
}

export function relClose(a: number, b: number, tol = REL_TOL): boolean {
  if (!Number.isFinite(a) || !Number.isFinite(b)) return false;
  const denom = Math.max(Math.abs(a), Math.abs(b), 1e-12);
  return Math.abs(a - b) / denom <= tol;
}

export function inferShareScale(opts: {
  netIncome: number | null;
  fmpEps: number | null;
  dilutedShares: number | null;
  adrRatio: number;
}): ShareScale {
  const { netIncome, fmpEps, dilutedShares, adrRatio } = opts;
  if (
    netIncome == null ||
    fmpEps == null ||
    dilutedShares == null ||
    !Number.isFinite(netIncome) ||
    !Number.isFinite(fmpEps) ||
    !Number.isFinite(dilutedShares) ||
    dilutedShares <= 0 ||
    fmpEps === 0
  ) {
    return 'unknown';
  }
  const ordinary = netIncome / dilutedShares;
  if (relClose(fmpEps, ordinary)) return adrRatio > 1 ? 'ordinary' : 'ads';
  if (adrRatio > 1) {
    const ads = ordinary * adrRatio;
    if (relClose(fmpEps, ads)) return 'ads';
    if (relClose(fmpEps, ads * adrRatio) || relClose(fmpEps, ordinary * adrRatio * adrRatio)) {
      return 'double_adr';
    }
  }
  return 'unknown';
}

export function inferAdrRatio(opts: {
  ticker?: string | null;
  netIncome: number | null;
  fmpEps: number | null;
  dilutedShares: number | null;
}): number {
  const known = knownAdrRatio(opts.ticker);
  const { netIncome, fmpEps, dilutedShares } = opts;
  if (
    netIncome != null &&
    fmpEps != null &&
    dilutedShares != null &&
    dilutedShares > 0 &&
    fmpEps !== 0
  ) {
    const ordinary = netIncome / dilutedShares;
    const candidates = known > 1 ? [known, ...COMMON_ADR.filter((r) => r !== known)] : [...COMMON_ADR];
    for (const r of candidates) {
      if (relClose(fmpEps, ordinary * r) || relClose(fmpEps, ordinary * r * r)) return r;
    }
  }
  return known;
}

export function perShareFactor(shareScale: ShareScale, adrRatio: number): number {
  const r = adrRatio > 0 ? adrRatio : 1;
  if (shareScale === 'ordinary') return r;
  if (shareScale === 'double_adr') return 1 / r;
  return 1;
}

export function scaleAmount(value: number | null | undefined, factor: number): number | null {
  if (value == null || !Number.isFinite(value)) return null;
  if (!Number.isFinite(factor) || factor === 0) return value;
  return value * factor;
}

export function impliedPe(price: number | null | undefined, eps: number | null | undefined): number | null {
  if (price == null || eps == null || !(price > 0) || !(eps > 0)) return null;
  const pe = price / eps;
  return Number.isFinite(pe) ? pe : null;
}

export function peInBand(pe: number | null, lo = 0.15, hi = 200): boolean {
  return pe != null && Number.isFinite(pe) && pe >= lo && pe <= hi;
}

/** Typical profitable-listing PE; used to reject ADR ×N that implies PE ~2. */
const PLAUSIBLE_PE_LO = 5;
const PLAUSIBLE_PE_HI = 40;

/**
 * Lower is better. Prefer implied PE near FMP `peTTM`; if that vendor PE is in
 * 8–25, treat a candidate PE below ~5 as a unit error (PDD ordinary×4 → 2.4).
 * Without `peTTM`, prefer 5–40 and break ties toward ~15.
 */
export function listingPeScore(pe: number | null, peTtm: number | null | undefined): number {
  if (pe == null || !peInBand(pe)) return Number.POSITIVE_INFINITY;
  const vendor = peTtm != null && peInBand(peTtm) ? peTtm : null;
  if (vendor != null) {
    let score = Math.abs(Math.log(pe) - Math.log(vendor));
    if (vendor >= 8 && vendor <= 25 && pe < PLAUSIBLE_PE_LO) score += 10;
    return score;
  }
  if (pe >= PLAUSIBLE_PE_LO && pe <= PLAUSIBLE_PE_HI) return Math.abs(pe - 15) / 100;
  if (pe < PLAUSIBLE_PE_LO) return 2 + (PLAUSIBLE_PE_LO - pe);
  return 1 + (pe - PLAUSIBLE_PE_HI) / PLAUSIBLE_PE_HI;
}

export function pickShareScaleForListingPe(opts: {
  price: number | null;
  fmpEps: number | null;
  fxToListing: number;
  adrRatio: number;
  peTtm?: number | null;
  inferred: ShareScale;
}): ShareScale {
  const { price, fmpEps, fxToListing, adrRatio, peTtm, inferred } = opts;
  if (!(adrRatio > 1)) return inferred;
  const fxEps = scaleAmount(fmpEps, fxToListing);
  if (fxEps == null || price == null || !(price > 0)) return inferred;

  const candidates: ShareScale[] = ['ads', 'ordinary', 'double_adr'];
  let best: ShareScale = inferred === 'unknown' ? 'ads' : inferred;
  let bestScore = listingPeScore(
    impliedPe(price, scaleAmount(fxEps, perShareFactor(best, adrRatio))),
    peTtm ?? null,
  );
  for (const s of candidates) {
    const score = listingPeScore(
      impliedPe(price, scaleAmount(fxEps, perShareFactor(s, adrRatio))),
      peTtm ?? null,
    );
    if (score < bestScore) {
      bestScore = score;
      best = s;
    }
  }
  if (!Number.isFinite(bestScore)) return inferred;
  return best;
}

/**
 * Prefer net-income / diluted ADS shares when FMP `epsdiluted` is on the wrong share class.
 * `dilutedShares` are treated as ordinary when `adrRatio > 1`.
 */
export function epsFromIncome(opts: {
  netIncome: number | null;
  dilutedShares: number | null;
  fxToListing: number;
  adrRatio: number;
}): number | null {
  const { netIncome, dilutedShares, fxToListing, adrRatio } = opts;
  if (netIncome == null || dilutedShares == null || !(dilutedShares > 0)) return null;
  const ni = netIncome * (Number.isFinite(fxToListing) && fxToListing > 0 ? fxToListing : 1);
  const adsShares = adrRatio > 1 ? dilutedShares / adrRatio : dilutedShares;
  if (!(adsShares > 0)) return null;
  const eps = ni / adsShares;
  return Number.isFinite(eps) ? eps : null;
}

export function peSanityOk(opts: {
  price: number | null;
  eps: number | null;
  peTtm?: number | null;
}): boolean {
  const pe = impliedPe(opts.price, opts.eps);
  if (!peInBand(pe)) return false;
  const vendor = opts.peTtm;
  if (vendor != null && vendor > 0 && pe != null) {
    const gap = Math.abs(Math.log(pe) - Math.log(vendor));
    // Vendor PE is often on a different unit basis; only fail when both look
    // internally consistent yet still disagree by more than 3×.
    if (peInBand(vendor) && gap > Math.log(3)) return false;
  }
  return true;
}

export function buildFundamentalsScale(input: {
  ticker?: string | null;
  reportedCurrency: string | null;
  listingCurrency: string | null;
  netIncome: number | null;
  fmpEps: number | null;
  dilutedShares: number | null;
  price: number | null;
  fxToListing: number;
  knownAdr?: number;
  peTtm?: number | null;
}): FundamentalsScale {
  const reportedCurrency = normalizeCurrency(input.reportedCurrency);
  const listingCurrency = normalizeCurrency(input.listingCurrency);
  const fx = input.fxToListing > 0 && Number.isFinite(input.fxToListing) ? input.fxToListing : 1;
  const adrRatio =
    input.knownAdr != null && input.knownAdr > 0
      ? input.knownAdr
      : inferAdrRatio({
          ticker: input.ticker,
          netIncome: input.netIncome,
          fmpEps: input.fmpEps,
          dilutedShares: input.dilutedShares,
        });

  let shareScale = inferShareScale({
    netIncome: input.netIncome,
    fmpEps: input.fmpEps,
    dilutedShares: input.dilutedShares,
    adrRatio,
  });

  if (adrRatio > 1) {
    shareScale = pickShareScaleForListingPe({
      price: input.price,
      fmpEps: input.fmpEps,
      fxToListing: fx,
      adrRatio,
      peTtm: input.peTtm,
      inferred: shareScale,
    });
  }

  const factor = perShareFactor(shareScale, adrRatio);
  const scaledFmp = scaleAmount(input.fmpEps, fx * factor);
  const fromIncome = epsFromIncome({
    netIncome: input.netIncome,
    dilutedShares: input.dilutedShares,
    fxToListing: fx,
    adrRatio,
  });

  const peFmp = impliedPe(input.price, scaledFmp);
  const peInc = impliedPe(input.price, fromIncome);
  let reliable = peInBand(peFmp) || peInBand(peInc);
  if (peInBand(peInc) && !peInBand(peFmp)) reliable = true;
  if (!peInBand(peFmp) && !peInBand(peInc)) reliable = false;

  return {
    version: FUNDAMENTALS_SCALE_VERSION,
    reportedCurrency,
    listingCurrency,
    fxToListing: fx,
    adrRatio,
    shareScale,
    perShareFactor: factor,
    reliable,
  };
}

export function pickScaledEps(opts: {
  fmpEps: number | null;
  netIncome: number | null;
  dilutedShares: number | null;
  scale: FundamentalsScale;
  price?: number | null;
  peTtm?: number | null;
}): number | null {
  const { fmpEps, netIncome, dilutedShares, scale, price } = opts;
  const scaledFmp = scaleAmount(fmpEps, scale.fxToListing * scale.perShareFactor);
  const fromIncome = epsFromIncome({
    netIncome,
    dilutedShares,
    fxToListing: scale.fxToListing,
    adrRatio: scale.adrRatio,
  });
  const peFmp = impliedPe(price ?? null, scaledFmp);
  const peInc = impliedPe(price ?? null, fromIncome);
  if (fromIncome != null && peInBand(peInc) && !peInBand(peFmp)) return fromIncome;
  if (
    fromIncome != null &&
    scaledFmp != null &&
    Math.abs(fromIncome) > 0 &&
    !relClose(fromIncome, scaledFmp, 0.3) &&
    peInBand(peInc)
  ) {
    if (
      peInBand(peFmp) &&
      listingPeScore(peFmp, opts.peTtm) <= listingPeScore(peInc, opts.peTtm)
    ) {
      return scaledFmp;
    }
    return fromIncome;
  }
  return scaledFmp ?? fromIncome;
}

/**
 * Company FCF in listing currency per ADS — same share-class path as `epsFromIncome`.
 */
export function fcfFromCashFlow(opts: {
  freeCashFlow: number | null;
  dilutedShares: number | null;
  fxToListing: number;
  adrRatio: number;
}): number | null {
  const { freeCashFlow, dilutedShares, fxToListing, adrRatio } = opts;
  if (freeCashFlow == null || dilutedShares == null || !(dilutedShares > 0)) return null;
  const fcf = freeCashFlow * (Number.isFinite(fxToListing) && fxToListing > 0 ? fxToListing : 1);
  const adsShares = adrRatio > 1 ? dilutedShares / adrRatio : dilutedShares;
  if (!(adsShares > 0)) return null;
  const per = fcf / adsShares;
  return Number.isFinite(per) ? per : null;
}

/**
 * Prefer company FCF / diluted ADS shares when FMP `freeCashFlowPerShare` is on
 * the wrong share class. Scores implied P/FCF the same way `pickScaledEps` scores PE.
 */
export function pickScaledFcf(opts: {
  fmpFcfPerShare: number | null;
  freeCashFlow: number | null;
  dilutedShares: number | null;
  scale: FundamentalsScale;
  price?: number | null;
}): number | null {
  const { fmpFcfPerShare, freeCashFlow, dilutedShares, scale, price } = opts;
  const scaledFmp = scaleAmount(fmpFcfPerShare, scale.fxToListing * scale.perShareFactor);
  const fromCf = fcfFromCashFlow({
    freeCashFlow,
    dilutedShares,
    fxToListing: scale.fxToListing,
    adrRatio: scale.adrRatio,
  });
  const pFmp = impliedPe(price ?? null, scaledFmp);
  const pCf = impliedPe(price ?? null, fromCf);
  if (fromCf != null && peInBand(pCf) && !peInBand(pFmp)) return fromCf;
  if (
    fromCf != null &&
    scaledFmp != null &&
    Math.abs(fromCf) > 0 &&
    !relClose(fromCf, scaledFmp, 0.3) &&
    peInBand(pCf)
  ) {
    if (peInBand(pFmp) && listingPeScore(pFmp, null) <= listingPeScore(pCf, null)) {
      return scaledFmp;
    }
    return fromCf;
  }
  return scaledFmp ?? fromCf;
}

export function scalePerShare(
  value: number | null | undefined,
  scale: FundamentalsScale,
): number | null {
  return scaleAmount(value, scale.fxToListing * scale.perShareFactor);
}

export function scaleCompany(
  value: number | null | undefined,
  scale: FundamentalsScale,
): number | null {
  return scaleAmount(value, scale.fxToListing);
}

/** Dividends from FMP are often already USD/ADS even when EPS is not. */
export function scaleDividend(
  value: number | null | undefined,
  price: number | null | undefined,
  scale: FundamentalsScale,
): number | null {
  if (value == null || !Number.isFinite(value)) return null;
  if (price != null && price > 0 && value > 0 && value / price <= 0.4) return value;
  return scalePerShare(value, scale);
}

export function scaleTev(
  tev: number | null | undefined,
  mktCap: number | null | undefined,
  scale: FundamentalsScale,
): number | null {
  if (tev == null || !Number.isFinite(tev)) return null;
  let v = tev;
  if (scale.fxToListing !== 1) v = tev * scale.fxToListing;
  if (mktCap != null && mktCap > 0 && Math.abs(v) / mktCap > 80) return null;
  return v;
}

export function scaleDcf(dcf: number | null | undefined, scale: FundamentalsScale, price?: number | null): number | null {
  const v = scalePerShare(dcf, scale);
  if (v == null) return null;
  if (price != null && price > 0 && Math.abs(v / price) > 40) return null;
  return v;
}

export function formatScaleCaption(scale: FundamentalsScale | null | undefined): string | null {
  if (!scale) return null;
  const listing = scale.listingCurrency ?? 'USD';
  const bits = [listing];
  if (scale.adrRatio > 1) bits.push(`1 ADS = ${scale.adrRatio} ord`);
  if (scale.reportedCurrency && scale.reportedCurrency !== listing) {
    bits.push(`from ${scale.reportedCurrency}`);
  }
  return bits.join(' · ');
}
