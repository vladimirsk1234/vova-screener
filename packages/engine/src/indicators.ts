/** Chart overlay math — port of sequence_vova.py compute_* helpers. */

function ema(values: Float64Array, length: number): Float64Array {
  const n = values.length;
  const out = new Float64Array(n);
  if (n === 0) return out;
  const alpha = 2 / (length + 1);
  out[0] = values[0];
  for (let i = 1; i < n; i++) {
    out[i] = alpha * values[i] + (1 - alpha) * out[i - 1];
  }
  return out;
}

function sma(values: Float64Array, length: number): Float64Array {
  const n = values.length;
  const out = new Float64Array(n);
  let sum = 0;
  for (let i = 0; i < n; i++) {
    sum += values[i];
    if (i >= length) sum -= values[i - length];
    const count = Math.min(i + 1, length);
    out[i] = sum / count;
  }
  return out;
}

/** Sample std matching pandas rolling(min_periods=1).std() (ddof=1). */
function rollingStd(values: Float64Array, length: number): Float64Array {
  const n = values.length;
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const start = Math.max(0, i + 1 - length);
    const count = i - start + 1;
    if (count <= 1) {
      out[i] = 0;
      continue;
    }
    let sum = 0;
    for (let j = start; j <= i; j++) sum += values[j];
    const mean = sum / count;
    let varSum = 0;
    for (let j = start; j <= i; j++) {
      const d = values[j] - mean;
      varSum += d * d;
    }
    out[i] = Math.sqrt(varSum / (count - 1));
  }
  return out;
}

export function calcMacd(
  close: Float64Array,
  fast = 12,
  slow = 26,
  signal = 9,
): { macd: Float64Array; signal: Float64Array; hist: Float64Array } {
  const emaFast = ema(close, fast);
  const emaSlow = ema(close, slow);
  const macdLine = new Float64Array(close.length);
  for (let i = 0; i < close.length; i++) macdLine[i] = emaFast[i] - emaSlow[i];
  const signalLine = ema(macdLine, signal);
  const hist = new Float64Array(close.length);
  for (let i = 0; i < close.length; i++) hist[i] = macdLine[i] - signalLine[i];
  return { macd: macdLine, signal: signalLine, hist };
}

export function calcDmi(
  highs: Float64Array,
  lows: Float64Array,
  closes: Float64Array,
  length: number,
): { plusDi: Float64Array; minusDi: Float64Array; adx: Float64Array } {
  const n = closes.length;
  const plusDm = new Float64Array(n);
  const minusDm = new Float64Array(n);
  const tr = new Float64Array(n);
  tr[0] = highs[0] - lows[0];
  for (let i = 1; i < n; i++) {
    const up = highs[i] - highs[i - 1];
    const down = lows[i - 1] - lows[i];
    plusDm[i] = up > down && up > 0 ? up : 0;
    minusDm[i] = down > up && down > 0 ? down : 0;
    tr[i] = Math.max(
      highs[i] - lows[i],
      Math.abs(highs[i] - closes[i - 1]),
      Math.abs(lows[i] - closes[i - 1]),
    );
  }
  const alpha = 1 / length;
  const atr = new Float64Array(n);
  const plusSm = new Float64Array(n);
  const minusSm = new Float64Array(n);
  atr[0] = tr[0];
  plusSm[0] = plusDm[0];
  minusSm[0] = minusDm[0];
  for (let i = 1; i < n; i++) {
    atr[i] = alpha * tr[i] + (1 - alpha) * atr[i - 1];
    plusSm[i] = alpha * plusDm[i] + (1 - alpha) * plusSm[i - 1];
    minusSm[i] = alpha * minusDm[i] + (1 - alpha) * minusSm[i - 1];
  }
  const plusDi = new Float64Array(n);
  const minusDi = new Float64Array(n);
  const dx = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    plusDi[i] = atr[i] !== 0 ? (100 * plusSm[i]) / atr[i] : 0;
    minusDi[i] = atr[i] !== 0 ? (100 * minusSm[i]) / atr[i] : 0;
    const den = plusDi[i] + minusDi[i];
    dx[i] = den !== 0 ? (100 * Math.abs(plusDi[i] - minusDi[i])) / den : Number.NaN;
  }
  const adx = new Float64Array(n);
  adx[0] = Number.isFinite(dx[0]) ? dx[0] : 0;
  for (let i = 1; i < n; i++) {
    const prev = Number.isFinite(adx[i - 1]) ? adx[i - 1] : 0;
    const cur = Number.isFinite(dx[i]) ? dx[i] : prev;
    adx[i] = alpha * cur + (1 - alpha) * prev;
  }
  return { plusDi, minusDi, adx };
}

export function computeElderEnvelope(
  close: Float64Array,
  lenSlow: number,
  lookback: number,
  multiplier: number,
): { emaSlow: Float64Array; envUpper: Float64Array; envLower: Float64Array } {
  const emaSlow = ema(close, lenSlow);
  const n = close.length;
  const myvar = new Float64Array(n);
  for (let i = 0; i < n; i++) myvar[i] = Math.abs(close[i] - emaSlow[i]);
  const myvars = new Float64Array(n);
  for (let i = 0; i < n; i++) myvars[i] = myvar[i] * myvar[i];
  const mymov = sma(myvars, lookback);
  for (let i = 0; i < n; i++) mymov[i] = Math.sqrt(mymov[i]);
  const newmax = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let mx = mymov[i];
    for (let lag = 1; lag <= 5; lag++) {
      const v = i - lag >= 0 ? mymov[i - lag] : 0;
      if (v > mx) mx = v;
    }
    newmax[i] = mx;
  }
  const envUpper = new Float64Array(n);
  const envLower = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    envUpper[i] = emaSlow[i] + newmax[i] * multiplier;
    envLower[i] = emaSlow[i] - newmax[i] * multiplier;
  }
  return { emaSlow, envUpper, envLower };
}

export function computeBollinger(
  close: Float64Array,
  length: number,
  mult: number,
): { basis: Float64Array; upper: Float64Array; lower: Float64Array } {
  const basis = sma(close, length);
  const std = rollingStd(close, length);
  const upper = new Float64Array(close.length);
  const lower = new Float64Array(close.length);
  for (let i = 0; i < close.length; i++) {
    upper[i] = basis[i] + mult * std[i];
    lower[i] = basis[i] - mult * std[i];
  }
  return { basis, upper, lower };
}

export function computeImpulseColors(
  close: Float64Array,
  lenFast: number,
  bullColor: string,
  bearColor: string,
  neutColor: string,
): string[] {
  const emaFast = ema(close, lenFast);
  const { hist } = calcMacd(close);
  const colors = new Array<string>(close.length);
  colors[0] = neutColor;
  for (let i = 1; i < close.length; i++) {
    const bulls = emaFast[i] > emaFast[i - 1] && hist[i] > hist[i - 1];
    const bears = emaFast[i] < emaFast[i - 1] && hist[i] < hist[i - 1];
    if (bulls) colors[i] = bullColor;
    else if (bears) colors[i] = bearColor;
    else colors[i] = neutColor;
  }
  return colors;
}

function toNullable(arr: Float64Array): (number | null)[] {
  return Array.from(arr, (v) => (Number.isFinite(v) ? v : null));
}

export type OverlaySeries = {
  ema_fast: (number | null)[];
  ema_slow: (number | null)[];
  sma_major: (number | null)[];
  env_upper: (number | null)[];
  env_lower: (number | null)[];
  bb_basis: (number | null)[];
  bb_upper: (number | null)[];
  bb_lower: (number | null)[];
};

export function computeOverlays(
  close: Float64Array,
  opts: {
    len_fast?: number;
    len_slow?: number;
    length_major?: number;
    lookback?: number;
    multiplier?: number;
    bb_length?: number;
    bb_mult?: number;
    elder_bull_color?: string;
    elder_bear_color?: string;
    elder_neut_color?: string;
  } = {},
): { overlays: OverlaySeries; impulse_colors: string[] } {
  const lenFast = opts.len_fast ?? 20;
  const lenSlow = opts.len_slow ?? 40;
  const lengthMajor = opts.length_major ?? 200;
  const lookback = opts.lookback ?? 100;
  const multiplier = opts.multiplier ?? 2.0;
  const bbLength = opts.bb_length ?? 20;
  const bbMult = opts.bb_mult ?? 2.0;
  const bull = opts.elder_bull_color ?? '#00c853';
  const bear = opts.elder_bear_color ?? '#ff1744';
  const neut = opts.elder_neut_color ?? '#4eadfc';

  const emaFast = ema(close, lenFast);
  const { emaSlow, envUpper, envLower } = computeElderEnvelope(
    close,
    lenSlow,
    lookback,
    multiplier,
  );
  const smaMajor = sma(close, lengthMajor);
  const bb = computeBollinger(close, bbLength, bbMult);
  const impulse_colors = computeImpulseColors(close, lenFast, bull, bear, neut);

  return {
    overlays: {
      ema_fast: toNullable(emaFast),
      ema_slow: toNullable(emaSlow),
      sma_major: toNullable(smaMajor),
      env_upper: toNullable(envUpper),
      env_lower: toNullable(envLower),
      bb_basis: toNullable(bb.basis),
      bb_upper: toNullable(bb.upper),
      bb_lower: toNullable(bb.lower),
    },
    impulse_colors,
  };
}

export { ema, sma };
