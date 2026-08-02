/**
 * The close-scan replay must match `sequence_vova` trade for trade.
 *
 * `check_parity.ts` compares the answer the Streamlit table renders — the trade that gives up on
 * the last bar. Tracked positions read the rest of the replay as well, so every trade in the
 * series is compared here: the bar it was entered on, the SL and RR that bar produced, and the
 * exit. A single extra or missing trade shifts every entry after it, which is exactly the failure
 * that would misprice a closed position.
 *
 * Fixture: python scripts/export_close_ledger_fixture.py
 * Run: npm run parity:ledger
 */
import * as fs from 'node:fs';
import * as path from 'node:path';
import { runCloseLedger, runSequenceVovaCloseScan } from '../src/sequenceVova';
import type { CloseTrade, OhlcSeries } from '../src/types';

const fixturePath = path.join(__dirname, '..', 'fixtures', 'close_ledger_parity.json');

type ExpectedTrade = Record<keyof CloseTrade, number | string | null>;

type Case = {
  label: string;
  opts: {
    atr_len: number;
    min_rr: number;
    use_last_hl_sl: boolean;
    risk_dollars: number;
    no_rr_req: boolean;
  };
  bars: OhlcSeries;
  trades: ExpectedTrade[];
  scan: Record<string, number | boolean | null>;
};

const NUMERIC: Array<keyof CloseTrade> = [
  'entry_index',
  'entry_price',
  'entry_sl',
  'entry_rr',
  'position_size',
  'exit_index',
  'exit_price',
  'close_rr',
  'pnl_dollars',
  'pnl_pct',
];

let failures = 0;

function fail(label: string, got: unknown, expected: unknown) {
  failures += 1;
  console.error(`FAIL ${label}: got=${JSON.stringify(got)} exp=${JSON.stringify(expected)}`);
}

/** Python writes NaN out as null, so "no number" has to compare equal either way. */
function sameNumber(got: unknown, expected: unknown, eps = 1e-9): boolean {
  const gotEmpty = got == null || (typeof got === 'number' && Number.isNaN(got));
  const expEmpty = expected == null || (typeof expected === 'number' && Number.isNaN(expected));
  if (gotEmpty || expEmpty) return gotEmpty && expEmpty;
  return Math.abs((got as number) - (expected as number)) <= eps;
}

function checkNumber(label: string, got: unknown, expected: unknown) {
  if (!sameNumber(got, expected)) fail(label, got, expected);
}

function checkExact(label: string, got: unknown, expected: unknown) {
  if (got !== expected) fail(label, got, expected);
}

function main() {
  if (!fs.existsSync(fixturePath)) {
    console.error('Missing fixture. Run: python scripts/export_close_ledger_fixture.py');
    process.exit(1);
  }
  const cases = JSON.parse(fs.readFileSync(fixturePath, 'utf8')).cases as Case[];
  let trades = 0;

  for (const testCase of cases) {
    const { label, bars, opts } = testCase;
    const ledger = runCloseLedger(bars, opts);
    if (!ledger) {
      fail(`${label} ledger`, null, 'a ledger');
      continue;
    }

    checkExact(`${label} trade count`, ledger.trades.length, testCase.trades.length);
    checkExact(`${label} asOf`, ledger.asOf, bars[bars.length - 1].date);

    const count = Math.min(ledger.trades.length, testCase.trades.length);
    for (let i = 0; i < count; i++) {
      const got = ledger.trades[i];
      const expected = testCase.trades[i];
      trades += 1;
      for (const key of NUMERIC) checkNumber(`${label} #${i} ${key}`, got[key], expected[key]);
      checkExact(`${label} #${i} entry_date`, got.entry_date, expected.entry_date);
      checkExact(`${label} #${i} exit_date`, got.exit_date, expected.exit_date);
    }

    // The open trade is the tail of the ledger, never a copy of it: a position the app is
    // carrying has to be the same object the next scan closes.
    const last = ledger.trades[ledger.trades.length - 1] ?? null;
    const endsOpen = Boolean(last && last.exit_index === null);
    checkExact(`${label} open trade`, ledger.open, endsOpen ? last : null);

    // Whatever else changes, the answer Streamlit renders has to be read off this same replay.
    const scan = runSequenceVovaCloseScan(bars, opts);
    checkExact(`${label} scan.Valid`, scan?.Valid, testCase.scan.Valid);
    for (const key of ['entry_price', 'exit_price', 'entry_rr', 'close_rr', 'pnl_dollars']) {
      checkNumber(
        `${label} scan.${key}`,
        (scan as unknown as Record<string, number>)?.[key],
        testCase.scan[key],
      );
    }
  }

  if (failures) process.exit(1);
  console.log(`Close ledger parity OK (${cases.length} series, ${trades} trades)`);
}

main();
