/**
 * Weekly chart-API collapse + pine parity vs Python (Streamlit hits).
 * Run: npx tsx packages/engine/scripts/check_weekly_hits_parity.ts
 */
import * as fs from 'node:fs';
import * as path from 'node:path';
import { collapseInProgressPeriodBars } from '../src/dataUtils';
import { evaluateSymbol } from '../src/evaluate';
import { runSequenceVovaPine } from '../src/sequenceVova';
import type { OhlcSeries } from '../src/types';

const fixturePath = path.join(__dirname, '..', 'fixtures', 'weekly_hits_parity.json');

function main() {
  if (!fs.existsSync(fixturePath)) {
    console.error('Missing fixture. Run: python scripts/compare_weekly_parity.py');
    process.exit(1);
  }
  const data = JSON.parse(fs.readFileSync(fixturePath, 'utf8')) as {
    opts: { atr_len: number; min_rr: number; use_last_hl_sl: boolean; risk_dollars: number };
    tickers: Record<
      string,
      {
        bars: OhlcSeries;
        pine: { Valid: boolean; New: boolean; RR: number | null; asOf: string };
        yf_pine: { Valid: boolean; New: boolean; RR: number | null; asOf: string };
      }
    >;
  };

  let ok = true;
  for (const [ticker, row] of Object.entries(data.tickers)) {
    const raw = row.bars;
    const collapsed = collapseInProgressPeriodBars(raw, 'Weekly');
    if (collapsed.length !== raw.length - 1 && collapsed.length !== raw.length) {
      // Allow same length when chart API already matches yfinance (weekends).
      console.warn(`${ticker}: unexpected length raw=${raw.length} collapsed=${collapsed.length}`);
    }
    if (raw.length > collapsed.length) {
      const rawLast = raw[raw.length - 1].date;
      const colLast = collapsed[collapsed.length - 1].date;
      if (rawLast === colLast) {
        ok = false;
        console.error(`FAIL ${ticker}: collapse did not drop mid-week stamp ${rawLast}`);
      }
    }

    const pine = runSequenceVovaPine(collapsed, data.opts);
    const exp = row.yf_pine;
    const checks: Array<[string, unknown, unknown]> = [
      ['Valid', pine?.Valid, exp.Valid],
      ['New', pine?.New, exp.New],
      ['asOf', collapsed[collapsed.length - 1]?.date, exp.asOf],
    ];
    for (const [label, got, want] of checks) {
      if (got !== want) {
        ok = false;
        console.error(`FAIL ${ticker}.${label}: got=${JSON.stringify(got)} exp=${JSON.stringify(want)}`);
      }
    }

    const evalNewOnly = evaluateSymbol({
      bars: collapsed,
      yahooTicker: ticker,
      params: {
        minRr: data.opts.min_rr,
        riskPerTrade: data.opts.risk_dollars,
        noRrReq: false,
        useLastHlSl: data.opts.use_last_hl_sl,
        newOnly: true,
        tf: 'Weekly',
      },
    });
    if (exp.Valid && exp.New) {
      if (evalNewOnly.status !== 'signal') {
        ok = false;
        console.error(`FAIL ${ticker} evaluate newOnly: ${JSON.stringify(evalNewOnly)}`);
      }
    }

    console.log(
      `${ticker}: raw=${raw.length} → ${collapsed.length} asOf=${collapsed.at(-1)?.date} ` +
        `Valid=${pine?.Valid} New=${pine?.New} (yf New=${exp.New})`,
    );
  }

  // Unit: same-week merge
  const sample: OhlcSeries = [
    { date: '2026-07-20', open: 1, high: 2, low: 0.5, close: 1.5, volume: 10 },
    { date: '2026-07-27', open: 1.5, high: 3, low: 1, close: 2, volume: 20 },
    { date: '2026-07-30', open: 2, high: 2.5, low: 1.2, close: 2.1, volume: 5 },
  ];
  const merged = collapseInProgressPeriodBars(sample, 'Weekly');
  if (merged.length !== 2 || merged[1].date !== '2026-07-27') {
    ok = false;
    console.error('FAIL unit collapse length/date', merged);
  } else if (merged[1].high !== 3 || merged[1].low !== 1 || merged[1].close !== 2.1) {
    ok = false;
    console.error('FAIL unit collapse OHLC', merged[1]);
  } else {
    console.log('OK unit collapse mid-week stamp');
  }

  if (!ok) process.exit(1);
  console.log('Weekly hits parity OK');
}

main();
