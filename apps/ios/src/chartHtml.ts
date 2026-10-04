import type { ChartPayload, ChartPrefs } from '@vova/device';

/** Self-contained candlestick chart. No network, no fundamental series. */
export function chartHtml(payload: ChartPayload, prefs: ChartPrefs): string {
  const data = JSON.stringify({ payload, prefs }).replace(/</g, '\\u003c');
  return `<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1" />
<style>
  html, body { margin: 0; height: 100%; background: #131722; color: #d1d4dc; font: 12px -apple-system, sans-serif; }
  canvas { width: 100%; height: 100%; display: block; touch-action: none; }
</style>
</head>
<body>
<canvas id="c"></canvas>
<script>
const INPUT = ${data};
const payload = INPUT.payload;
const prefs = INPUT.prefs;
const canvas = document.getElementById('c');
const ctx = canvas.getContext('2d');
const bars = payload.bars || [];
let start = Math.max(0, bars.length - 80);
function resize() {
  const dpr = window.devicePixelRatio || 1;
  canvas.width = canvas.clientWidth * dpr;
  canvas.height = canvas.clientHeight * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  draw();
}
function draw() {
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = '#131722';
  ctx.fillRect(0, 0, w, h);
  const view = bars.slice(start);
  if (!view.length) return;
  let min = Infinity, max = -Infinity;
  const consider = (v) => { if (v != null && isFinite(v)) { min = Math.min(min, v); max = Math.max(max, v); } };
  view.forEach((b) => { consider(b.low); consider(b.high); });
  const overlay = payload.overlay;
  const off = start;
  if (overlay) {
    for (let i = off; i < off + view.length; i++) consider(overlay.critical[i]);
    if (prefs.showTpSl && payload.pine) { consider(payload.pine.tp); consider(payload.pine.sl); }
  }
  if (!isFinite(min) || !isFinite(max) || min === max) { min -= 1; max += 1; }
  const pad = (max - min) * 0.08;
  min -= pad; max += pad;
  const slot = w / view.length;
  const y = (v) => h - ((v - min) / (max - min)) * (h - 8) - 4;
  const x = (i) => i * slot + slot / 2;
  const line = (values, color) => {
    ctx.beginPath();
    let moved = false;
    values.forEach((v, i) => {
      if (v == null || !isFinite(v)) { moved = false; return; }
      const px = x(i), py = y(v);
      if (!moved) { ctx.moveTo(px, py); moved = true; } else ctx.lineTo(px, py);
    });
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.4;
    ctx.stroke();
  };
  view.forEach((b, i) => {
    const up = b.close >= b.open;
    ctx.strokeStyle = up ? '#089981' : '#f23645';
    ctx.fillStyle = ctx.strokeStyle;
    ctx.beginPath();
    ctx.moveTo(x(i), y(b.high));
    ctx.lineTo(x(i), y(b.low));
    ctx.stroke();
    const top = y(Math.max(b.open, b.close));
    const bot = y(Math.min(b.open, b.close));
    ctx.fillRect(x(i) - Math.max(1, slot * 0.3), top, Math.max(2, slot * 0.6), Math.max(1, bot - top));
  });
  if (overlay) {
    line(overlay.critical.slice(off, off + view.length), '#00c853');
    if (prefs.showEma) {
      line(overlay.emaFast.slice(off, off + view.length), '#2196f3');
      line(overlay.emaSlow.slice(off, off + view.length), '#f44336');
    }
    if (prefs.showBb) {
      line(overlay.bbUpper.slice(off, off + view.length), '#9e9e9e');
      line(overlay.bbLower.slice(off, off + view.length), '#9e9e9e');
    }
    ctx.fillStyle = '#e0e0e0';
    ctx.font = '10px -apple-system, sans-serif';
    for (const p of overlay.peaks) {
      if (p.idx < off || p.idx >= off + view.length) continue;
      ctx.fillText(p.label, x(p.idx - off) - 8, y(p.price) - 6);
    }
    for (const t of overlay.troughs) {
      if (t.idx < off || t.idx >= off + view.length) continue;
      ctx.fillText(t.label, x(t.idx - off) - 8, y(t.price) + 12);
    }
    if (prefs.showFib && overlay.fib) {
      ctx.setLineDash([4, 4]);
      for (const level of [overlay.fib.fib382, overlay.fib.fib500, overlay.fib.fib618]) {
        ctx.strokeStyle = '#787b86';
        ctx.beginPath();
        ctx.moveTo(0, y(level));
        ctx.lineTo(w, y(level));
        ctx.stroke();
      }
      ctx.setLineDash([]);
    }
    if (prefs.showTpSl && payload.pine) {
      ctx.setLineDash([2, 3]);
      ctx.strokeStyle = '#089981';
      if (payload.pine.tp != null) { ctx.beginPath(); ctx.moveTo(0, y(payload.pine.tp)); ctx.lineTo(w, y(payload.pine.tp)); ctx.stroke(); }
      ctx.strokeStyle = '#f23645';
      if (payload.pine.sl != null) { ctx.beginPath(); ctx.moveTo(0, y(payload.pine.sl)); ctx.lineTo(w, y(payload.pine.sl)); ctx.stroke(); }
      ctx.setLineDash([]);
    }
  }
}
let drag = null;
canvas.addEventListener('pointerdown', (e) => { drag = e.clientX; });
canvas.addEventListener('pointermove', (e) => {
  if (drag == null) return;
  const dx = e.clientX - drag;
  if (Math.abs(dx) < 12) return;
  drag = e.clientX;
  const step = dx > 0 ? -3 : 3;
  start = Math.max(0, Math.min(bars.length - 10, start + step));
  draw();
});
canvas.addEventListener('pointerup', () => { drag = null; });
window.addEventListener('resize', resize);
resize();
</script>
</body>
</html>`;
}
