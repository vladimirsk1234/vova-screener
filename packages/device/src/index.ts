export * from './types';
export * from './money';
export * from './yahoo';
export * from './store';
export * from './tracker';
export * from './results';
export * from './history';
export * from './chart';
export * from './scan';
export * from './rebuild';

/** Fields the iPhone app must never persist or render. */
export const FMP_FIELDS = [
  'epsAtEntry',
  'epsPositiveAtEntry',
  'premiumPctAtEntry',
  'undervaluedAtEntry',
  'fairValue',
  'peTTM',
  'pegTTM',
] as const;

export function assertNoFundamentalFields(value: unknown): void {
  const text = JSON.stringify(value);
  for (const field of FMP_FIELDS) {
    if (text.includes(`"${field}"`)) {
      throw new Error(`FMP field ${field} leaked into device data`);
    }
  }
}
