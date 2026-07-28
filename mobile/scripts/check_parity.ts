/**
 * Compare TS engine vs Python-exported fixture.
 * Run: npx tsx scripts/check_parity.ts
 */
import * as fs from 'fs';
import * as path from 'path';
import {
  runSequenceVovaCloseScan,
  runSequenceVovaPine,
} from '../src/engine/sequenceVova';
import type { OhlcSeries } from '../src/types';

const fixturePath = path.join(__dirname, '..', 'fixtures', 'parity_sample.json');

function almostEqual(a: number | null | undefined, b: number | null | undefined, eps = 1e-6) {
  const aEmpty = a == null || (typeof a === 'number' && Number.isNaN(a));
  const bEmpty = b == null || (typeof b === 'number' && Number.isNaN(b));
  if (aEmpty && bEmpty) return true;
  if (aEmpty || bEmpty) return false;
  return Math.abs((a as number) - (b as number)) <= eps;
}

function main() {
  if (!fs.existsSync(fixturePath)) {
    console.error('Missing fixture. Run: python scripts/export_parity_fixture.py');
    process.exit(1);
  }
  const data = JSON.parse(fs.readFileSync(fixturePath, 'utf8'));
  const bars = data.bars as OhlcSeries;
  const opts = data.opts;
  const pine = runSequenceVovaPine(bars, opts);
  const close = runSequenceVovaCloseScan(bars, opts);

  let ok = true;
  const check = (label: string, got: unknown, exp: unknown) => {
    const g = got as number | boolean | null;
    const e = exp as number | boolean | null;
    let pass = false;
    if (typeof g === 'boolean' || typeof e === 'boolean') pass = g === e;
    else pass = almostEqual(g as number, e as number);
    if (!pass) {
      ok = false;
      console.error(`FAIL ${label}: got=${JSON.stringify(got)} exp=${JSON.stringify(exp)}`);
    }
  };

  for (const key of ['Valid', 'New', 'Strong', 'RR', 'TP', 'SL', 'Close', 'ATR']) {
    check(`pine.${key}`, (pine as any)?.[key], data.pine?.[key]);
  }
  for (const key of ['Valid', 'New', 'entry_price', 'exit_price', 'pnl_dollars', 'entry_rr']) {
    check(`close.${key}`, (close as any)?.[key], data.close?.[key]);
  }

  if (!ok) process.exit(1);
  console.log('Parity OK');
}

main();
