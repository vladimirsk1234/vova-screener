/** TradingView helpers — port of tradingview_embed.py (no yfinance). */

const YAHOO_EXCHANGE_TO_TV: Record<string, string> = {
  NMS: 'NASDAQ',
  NGM: 'NASDAQ',
  NCM: 'NASDAQ',
  NAS: 'NASDAQ',
  NASDAQ: 'NASDAQ',
  NASDAQGS: 'NASDAQ',
  NASDAQCM: 'NASDAQ',
  NASDAQGM: 'NASDAQ',
  NYQ: 'NYSE',
  NYS: 'NYSE',
  NYSE: 'NYSE',
  ASE: 'AMEX',
  AMEX: 'AMEX',
  BTS: 'BATS',
  BAT: 'BATS',
  BATS: 'BATS',
  PCX: 'ARCA',
  ARCA: 'ARCA',
  TOR: 'TSX',
  TSX: 'TSX',
  VAN: 'TSXV',
  TSXV: 'TSXV',
};

export function tfToTvInterval(tf: string): string {
  return ({ Daily: 'D', Weekly: 'W', Monthly: 'M' } as Record<string, string>)[tf] ?? 'D';
}

export function normalizeTvSymbol(symbol: string): string {
  const s = String(symbol || '').trim();
  if (!s) return 'NASDAQ:AAPL';
  return s.includes(':') ? s.toUpperCase() : s.toUpperCase();
}

export function inferTvSymbol(yahooTicker: string, exchangeName?: string): string {
  const t = String(yahooTicker || '')
    .trim()
    .toUpperCase();
  if (!t) return t;
  if (t.includes(':')) return t;
  if (t.endsWith('.TO')) return `TSX:${t.replace(/\.TO$/, '')}`;
  if (t.endsWith('.V')) return `TSXV:${t.replace(/\.V$/, '')}`;
  if (t.endsWith('.NE')) return `NEO:${t.replace(/\.NE$/, '')}`;
  if (t.endsWith('.CN')) return `CSE:${t.replace(/\.CN$/, '')}`;
  const ex = (exchangeName || '').toUpperCase();
  const tvEx = YAHOO_EXCHANGE_TO_TV[ex];
  if (tvEx) return `${tvEx}:${t}`;
  return t;
}

export function buildChartUrl(tvSymbol: string, interval: string): string {
  const sym = encodeURIComponent(normalizeTvSymbol(tvSymbol));
  return `https://www.tradingview.com/chart/?symbol=${sym}&interval=${encodeURIComponent(interval)}`;
}
