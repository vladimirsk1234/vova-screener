/**
 * Star premia on the slim card payload. `premiumPct` stays the Settings default (EPS);
 * EPS/FCF/DCF are the three Value-tab numbers; `bestPremiumPct` is the most undervalued.
 */

export type CardPremia = {
  premiumPct: number | null;
  epsPremiumPct: number | null;
  fcfPremiumPct: number | null;
  dcfPremiumPct: number | null;
  bestPremiumPct: number | null;
};

function finiteNum(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

/** Most negative finite premium — same rule as engine `bestValuePremium`. */
export function bestCardPremium(premia: {
  epsPremiumPct: number | null;
  fcfPremiumPct: number | null;
  dcfPremiumPct: number | null;
}): number | null {
  const vals = [premia.epsPremiumPct, premia.fcfPremiumPct, premia.dcfPremiumPct].filter(
    (n): n is number => n != null && Number.isFinite(n),
  );
  if (!vals.length) return null;
  return Math.min(...vals);
}

/**
 * Map stored instrumentFundamentals star fields onto CardFundamentals premia.
 * Missing `epsPremiumPct` falls back to default `premiumPct`; missing `bestPremiumPct`
 * is recomputed from the three.
 */
export function cardPremiaFromStored(doc: {
  premiumPct?: unknown;
  epsPremiumPct?: unknown;
  fcfPremiumPct?: unknown;
  dcfPremiumPct?: unknown;
  bestPremiumPct?: unknown;
}): CardPremia {
  const premiumPct = finiteNum(doc.premiumPct);
  const epsPremiumPct = finiteNum(doc.epsPremiumPct) ?? premiumPct;
  const fcfPremiumPct = finiteNum(doc.fcfPremiumPct);
  const dcfPremiumPct = finiteNum(doc.dcfPremiumPct);
  const bestPremiumPct =
    finiteNum(doc.bestPremiumPct) ?? bestCardPremium({ epsPremiumPct, fcfPremiumPct, dcfPremiumPct });
  return { premiumPct, epsPremiumPct, fcfPremiumPct, dcfPremiumPct, bestPremiumPct };
}

/** Full slim card from a stored doc that already has card + star fields (no payload). */
export function cardFundamentalsFromStoredDoc(doc: {
  fairValue?: unknown;
  premiumPct?: unknown;
  growthRatePct?: unknown;
  blendedPe?: unknown;
  ltDebtToCapitalTTM?: unknown;
  epsPremiumPct?: unknown;
  fcfPremiumPct?: unknown;
  dcfPremiumPct?: unknown;
  bestPremiumPct?: unknown;
}): {
  fairValue: number | null;
  growthRatePct: number | null;
  blendedPe: number | null;
  ltDebtToCapitalTTM: number | null;
} & CardPremia {
  return {
    fairValue: finiteNum(doc.fairValue),
    growthRatePct: finiteNum(doc.growthRatePct),
    blendedPe: finiteNum(doc.blendedPe),
    ltDebtToCapitalTTM: finiteNum(doc.ltDebtToCapitalTTM),
    ...cardPremiaFromStored(doc),
  };
}
