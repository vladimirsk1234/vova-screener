/**
 * Run the app's SELL TO CLOSE scan over real symbols and write down what it saw.
 *
 * The synthetic parity checks prove the replay is the same arithmetic as `sequence_vova`. They
 * cannot prove the app and Streamlit are looking at the same bars, and that is the half of the
 * comparison that decides whether the CLOSED list agrees with the Streamlit table: a series that
 * ends one bar earlier, or keeps a bar Streamlit drops, moves every break by a bar and empties
 * the list.
 *
 * So this writes out both halves — the bars the app fetched and the close it read off them — and
 * `scripts/check_live_close_parity.py` replays the same bars through Streamlit's own function and
 * through yfinance's own download, and reports where the two disagree.
 *
 * Run: npx tsx packages/engine/scripts/export_live_close_scan.ts [--tf Daily] [--limit 400]
 */
import * as fs from 'node:fs';
import * as path from 'node:path';
import { evaluateClose, type EvaluateParams } from '../src/evaluate';
import {
  collapseInProgressPeriodBars,
  dropIncompleteBars,
  fillLastBarOhlc,
  intervalAndPeriod,
  toOhlcSeries,
} from '../src/dataUtils';
import { parseListText } from '../src/tickers';
import type { OhlcSeries, Timeframe } from '../src/types';

const ROOT = path.join(__dirname, '..', '..', '..');
const UA =
  'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36';
const CONCURRENCY = 8;

/** The background scan's settings: RR sorts the list, it does not filter it. */
const PARAMS: Omit<EvaluateParams, 'tf'> = {
  direction: 'buy',
  minRr: 0,
  riskPerTrade: 100,
  noRrReq: true,
  useLastHlSl: true,
  newOnly: false,
};

function arg(name: string, fallback: string): string {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 && process.argv[i + 1] ? process.argv[i + 1] : fallback;
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/** The same call `YahooClient` makes, down to the query string, so the bars are the app's bars. */
async function fetchBars(ticker: string, tf: Timeframe): Promise<OhlcSeries | null> {
  const { interval, range } = intervalAndPeriod(tf);
  const url =
    `https://query1.finance.yahoo.com/v8/finance/chart/${encodeURIComponent(ticker)}` +
    `?interval=${interval}&range=${range}&includePrePost=false&events=div%2Csplit`;

  for (let attempt = 0; attempt < 4; attempt++) {
    try {
      const res = await fetch(url, { headers: { 'User-Agent': UA, Accept: 'application/json' } });
      if (res.status === 429 || res.status >= 500) {
        await sleep(800 * (attempt + 1));
        continue;
      }
      if (!res.ok) return null;
      const json: any = await res.json();
      const result = json?.chart?.result?.[0];
      const quote = result?.indicators?.quote?.[0];
      if (!result?.timestamp?.length || !quote) return null;
      const n = result.timestamp.length;
      const col = (a: (number | null)[] | undefined) =>
        Array.from({ length: n }, (_, i) => (a?.[i] == null ? Number.NaN : Number(a[i])));
      let bars = toOhlcSeries(
        result.timestamp,
        col(quote.open),
        col(quote.high),
        col(quote.low),
        col(quote.close),
        col(quote.volume),
      );
      bars = fillLastBarOhlc(bars);
      bars = dropIncompleteBars(bars);
      bars = collapseInProgressPeriodBars(bars, tf);
      return bars.length ? bars : null;
    } catch {
      await sleep(500 * (attempt + 1));
    }
  }
  return null;
}

async function main() {
  const tf = arg('tf', 'Daily') as Timeframe;
  const limit = Number(arg('limit', '400'));
  const listFile = arg('list', 'STOCK-TICKERS.txt');
  const out = arg('out', path.join(ROOT, 'reports', `live_close_${tf.toLowerCase()}.json`));

  const parsed = parseListText(fs.readFileSync(path.join(ROOT, listFile), 'utf8'));
  const entries = parsed.entries.slice(0, limit > 0 ? limit : undefined);
  const params: EvaluateParams = { ...PARAMS, tf };

  const symbols: Record<string, unknown> = {};
  let closes = 0;
  let noData = 0;
  let done = 0;

  const queue = [...entries];
  const worker = async () => {
    while (queue.length) {
      const entry = queue.shift();
      if (!entry) return;
      const bars = await fetchBars(entry.yahoo, tf);
      done += 1;
      if (done % 50 === 0) process.stderr.write(`  ${done}/${entries.length}\n`);
      if (!bars) {
        noData += 1;
        continue;
      }
      const close = evaluateClose({
        bars,
        yahooTicker: entry.yahoo,
        tvSymbol: entry.tv,
        companyName: entry.name ?? undefined,
        params,
      });
      if (close) closes += 1;
      symbols[entry.yahoo] = { bars, close };
    }
  };
  await Promise.all(Array.from({ length: CONCURRENCY }, () => worker()));

  fs.mkdirSync(path.dirname(out), { recursive: true });
  fs.writeFileSync(
    out,
    JSON.stringify({
      tf,
      params,
      atr_len: 14,
      scannedAt: new Date().toISOString(),
      symbols,
    }),
  );
  console.log(
    `${tf}: ${entries.length} symbols, ${Object.keys(symbols).length} with bars, ` +
      `${noData} without, ${closes} closes → ${out}`,
  );
}

void main();
