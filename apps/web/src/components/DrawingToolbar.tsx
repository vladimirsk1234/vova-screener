import type { ChartDrawing } from '../lib/api';

export type DrawingTool =
  | 'cursor'
  | 'trend'
  | 'ray'
  | 'hline'
  | 'vline'
  | 'fib'
  | 'text'
  | 'erase';

const TOOLS: Array<{ id: DrawingTool; label: string; title: string }> = [
  { id: 'cursor', label: '↖', title: 'Select / pan' },
  { id: 'trend', label: '/', title: 'Trend line' },
  { id: 'ray', label: '↗', title: 'Ray' },
  { id: 'hline', label: '—', title: 'Horizontal line' },
  { id: 'vline', label: '|', title: 'Vertical line' },
  { id: 'fib', label: 'Fib', title: 'Fibonacci retracement' },
  { id: 'text', label: 'T', title: 'Text note' },
  { id: 'erase', label: '⌫', title: 'Delete last' },
];

type Props = {
  tool: DrawingTool;
  onTool: (t: DrawingTool) => void;
  onUndo: () => void;
  onRedo: () => void;
  canUndo: boolean;
  canRedo: boolean;
  magnet: boolean;
  onMagnet: (v: boolean) => void;
  count: number;
};

export function DrawingToolbar({
  tool,
  onTool,
  onUndo,
  onRedo,
  canUndo,
  canRedo,
  magnet,
  onMagnet,
  count,
}: Props) {
  return (
    <div className="drawing-toolbar" role="toolbar" aria-label="Drawing tools">
      {TOOLS.map((t) => (
        <button
          key={t.id}
          type="button"
          className={`draw-btn ${tool === t.id ? 'active' : ''}`}
          title={t.title}
          aria-label={t.title}
          onClick={() => onTool(t.id)}
        >
          {t.label}
        </button>
      ))}
      <button type="button" className="draw-btn" title="Undo" aria-label="Undo" disabled={!canUndo} onClick={onUndo}>
        ↶
      </button>
      <button type="button" className="draw-btn" title="Redo" aria-label="Redo" disabled={!canRedo} onClick={onRedo}>
        ↷
      </button>
      <button
        type="button"
        className={`draw-btn ${magnet ? 'active' : ''}`}
        title="Magnet snap"
        aria-label="Magnet snap"
        onClick={() => onMagnet(!magnet)}
      >
        ⌇
      </button>
      <span className="muted small draw-count">{count}</span>
    </div>
  );
}

export function newDrawingId() {
  return `d_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 7)}`;
}

export function pushDrawingHistory(
  drawings: ChartDrawing[],
  past: ChartDrawing[][],
  _future: ChartDrawing[][],
  next: ChartDrawing[],
): { drawings: ChartDrawing[]; past: ChartDrawing[][]; future: ChartDrawing[][] } {
  return {
    drawings: next,
    past: [...past, drawings].slice(-50),
    future: [],
  };
}
