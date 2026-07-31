import type { ChartSettings } from '../lib/api';
import { DEFAULT_CHART_SETTINGS } from '../lib/chartSettings';

type Props = {
  open: boolean;
  value: ChartSettings;
  onChange: (next: ChartSettings) => void;
  onClose: () => void;
  onSave: () => void;
  onReset: () => void;
};

function Toggle({
  label,
  checked,
  onChange,
}: {
  label: string;
  checked: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <label className="chart-toggle">
      <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} />
      <span>{label}</span>
    </label>
  );
}

function Num({
  label,
  value,
  onChange,
  step = 1,
  min,
  max,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
  step?: number;
  min?: number;
  max?: number;
}) {
  return (
    <label className="chart-field">
      <span>{label}</span>
      <input
        type="number"
        value={value}
        step={step}
        min={min}
        max={max}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </label>
  );
}

function Color({
  label,
  value,
  onChange,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
}) {
  const hex = value.startsWith('#') && value.length >= 7 ? value.slice(0, 7) : '#888888';
  return (
    <label className="chart-field">
      <span>{label}</span>
      <input type="color" value={hex} onChange={(e) => onChange(e.target.value)} />
    </label>
  );
}

export function ChartSettingsPanel({ open, value, onChange, onClose, onSave, onReset }: Props) {
  if (!open) return null;
  const set = <K extends keyof ChartSettings>(key: K, v: ChartSettings[K]) =>
    onChange({ ...value, [key]: v });

  return (
    <div className="chart-settings-sheet" role="dialog" aria-label="Chart settings">
      <div className="chart-settings-head">
        <strong>Chart indicator settings</strong>
        <button type="button" className="btn-sm ghost" onClick={onClose}>
          Close
        </button>
      </div>
      <p className="muted small">Changes apply immediately (no re-scan). Numeric MA/BB params recompute overlays.</p>

      <div className="chart-settings-grid">
        <section>
          <h4>Visibility</h4>
          <Toggle label="Fibonacci" checked={value.show_fib} onChange={(v) => set('show_fib', v)} />
          <Toggle
            label="Short EMA"
            checked={value.show_short_ema}
            onChange={(v) => set('show_short_ema', v)}
          />
          <Toggle
            label="Center EMA"
            checked={value.show_center_ema}
            onChange={(v) => set('show_center_ema', v)}
          />
          <Toggle
            label={`SMA ${value.length_major}`}
            checked={value.show_sma_major}
            onChange={(v) => set('show_sma_major', v)}
          />
          <Toggle label="Bollinger Bands" checked={value.show_bb} onChange={(v) => set('show_bb', v)} />
          <Toggle label="TP / SL lines" checked={value.show_tp_sl} onChange={(v) => set('show_tp_sl', v)} />
        </section>

        <section>
          <h4>Moving averages</h4>
          <Num label="Fast EMA" value={value.len_fast} onChange={(v) => set('len_fast', v)} />
          <Num label="Center EMA" value={value.len_slow} onChange={(v) => set('len_slow', v)} />
          <Num label="Major SMA" value={value.length_major} onChange={(v) => set('length_major', v)} />
        </section>

        <section>
          <h4>Theme</h4>
          <Color label="Background" value={value.bg_color} onChange={(v) => set('bg_color', v)} />
          <Color label="Grid" value={value.grid_color} onChange={(v) => set('grid_color', v)} />
          <Color label="Candle up" value={value.candle_up} onChange={(v) => set('candle_up', v)} />
          <Color label="Candle down" value={value.candle_down} onChange={(v) => set('candle_down', v)} />
        </section>

        <section>
          <h4>Overlays</h4>
          <Color label="HH/LL" value={value.hhll_color} onChange={(v) => set('hhll_color', v)} />
          <Color
            label="Critical up"
            value={value.crit_stop_color_up}
            onChange={(v) => set('crit_stop_color_up', v)}
          />
          <Color
            label="Critical down"
            value={value.crit_stop_color_down}
            onChange={(v) => set('crit_stop_color_down', v)}
          />
          <Color label="Fibonacci" value={value.fib_color} onChange={(v) => set('fib_color', v)} />
          <Num
            label="Fib width"
            value={value.fib_width}
            min={1}
            max={5}
            onChange={(v) => set('fib_width', v)}
          />
          <Color label="Short EMA" value={value.short_ema_color} onChange={(v) => set('short_ema_color', v)} />
          <Color
            label="Center EMA"
            value={value.center_ema_color}
            onChange={(v) => set('center_ema_color', v)}
          />
          <Color
            label="Major SMA"
            value={value.sma_major_color}
            onChange={(v) => set('sma_major_color', v)}
          />
          <Color label="BB basis" value={value.bb_basis_color} onChange={(v) => set('bb_basis_color', v)} />
          <Color label="BB upper" value={value.bb_upper_color} onChange={(v) => set('bb_upper_color', v)} />
          <Color label="BB lower" value={value.bb_lower_color} onChange={(v) => set('bb_lower_color', v)} />
          <Color
            label="Envelope upper"
            value={value.env_upper_color}
            onChange={(v) => set('env_upper_color', v)}
          />
          <Color
            label="Envelope lower"
            value={value.env_lower_color}
            onChange={(v) => set('env_lower_color', v)}
          />
        </section>

        <section>
          <h4>Watermark</h4>
          <Color
            label="Text color"
            value={value.wm_text_color}
            onChange={(v) => set('wm_text_color', v)}
          />
          <Num
            label="Text size"
            value={value.wm_font_size}
            min={8}
            max={32}
            onChange={(v) => set('wm_font_size', v)}
          />
        </section>
      </div>

      <div className="chart-settings-actions">
        <button type="button" className="btn-sm" onClick={onReset}>
          Reset to defaults
        </button>
        <button type="button" className="btn-sm btn-accent" onClick={onSave}>
          Save preset
        </button>
        <button
          type="button"
          className="btn-sm ghost"
          onClick={() => onChange({ ...value, ...DEFAULT_CHART_SETTINGS, show_tp_sl: true, show_fib: true })}
        >
          Show all optional
        </button>
      </div>
    </div>
  );
}
