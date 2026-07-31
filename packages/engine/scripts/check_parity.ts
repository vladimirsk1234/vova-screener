/**
 * Overlay / full-result self-consistency checks against pine last-bar.
 * Run: npm run parity
 */
import * as fs from 'node:fs';
import * as path from 'node:path';
import { runSequenceVovaCloseScan, runSequenceVovaPine } from '../src/sequenceVova';
import { runSequenceVovaFull, runStructureOverlay } from '../src/sequenceVovaFull';
import { buildTradeLine, buildDwmLines } from '../src/watermark';
import { defaultIndicatorParams } from '../src/indicatorParams';
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
  const full = runSequenceVovaFull(bars, opts);
  const overlay = runStructureOverlay(bars, opts);

  let ok = true;
  const check = (label: string, got: unknown, exp: unknown) => {
    const g = got as number | boolean | null;
    const e = exp as number | boolean | null;
    const pass =
      typeof g === 'boolean' || typeof e === 'boolean'
        ? g === e
        : almostEqual(g as number, e as number);
    if (!pass) {
      ok = false;
      console.error(`FAIL ${label}: got=${JSON.stringify(got)} exp=${JSON.stringify(exp)}`);
    }
  };

  for (const key of ['Valid', 'New', 'Strong', 'RR', 'TP', 'SL', 'Close', 'ATR']) {
    check(`pine.${key}`, (pine as unknown as Record<string, unknown>)?.[key], data.pine?.[key]);
  }
  for (const key of ['Valid', 'New', 'entry_price', 'exit_price', 'pnl_dollars', 'entry_rr']) {
    check(`close.${key}`, (close as unknown as Record<string, unknown>)?.[key], data.close?.[key]);
  }

  if (!full) {
    ok = false;
    console.error('FAIL full: null');
  } else {
    check('full.Valid', full.Valid, pine?.Valid);
    check('full.New', full.New, pine?.New);
    check('full.Strong', full.Strong, pine?.Strong);
    check('full.RR', full.RR, pine?.RR);
    check('full.TP', full.TP, pine?.TP);
    check('full.SL', full.SL, pine?.SL);
    check('full.Close', full.Close, pine?.Close);
    check('full.ATR', full.ATR, pine?.ATR);

    if (full.critical_level_series.length !== bars.length) {
      ok = false;
      console.error('FAIL full.critical length');
    }
    if (full.overlays.ema_fast.length !== bars.length) {
      ok = false;
      console.error('FAIL overlays.ema_fast length');
    }
    if (full.impulse_colors.length !== bars.length) {
      ok = false;
      console.error('FAIL impulse_colors length');
    }
    if (!Array.isArray(full.peaks) || !Array.isArray(full.troughs)) {
      ok = false;
      console.error('FAIL peaks/troughs');
    }
    if (!Number.isFinite(full.ADX) || !Number.isFinite(full.ATR_pct)) {
      ok = false;
      console.error('FAIL ADX/ATR_pct');
    }

    const params = defaultIndicatorParams();
    const trade = buildTradeLine(full, params, bars.length - 1);
    if (!trade || typeof trade !== 'string') {
      ok = false;
      console.error('FAIL trade line');
    }
    const dwm = buildDwmLines({ chartBars: bars, chartTf: 'Weekly', params });
    if (!dwm.weekly) {
      ok = false;
      console.error('FAIL dwm weekly');
    }
  }

  if (!overlay) {
    ok = false;
    console.error('FAIL overlay null');
  } else if (full) {
    check('overlay.last_peak', overlay.last_peak, full.last_peak);
    check('overlay.last_trough', overlay.last_trough, full.last_trough);
    if (overlay.critical.length !== full.critical_level_series.length) {
      ok = false;
      console.error('FAIL overlay critical length');
    }
  }

  if (!ok) process.exit(1);
  console.log('Parity OK (pine + close + full overlays)');
}

main();
