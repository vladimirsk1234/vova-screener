export type SortDir = 'asc' | 'desc';

export type SortOption<T extends string> = {
  value: T;
  label: string;
  /** Direction a fresh click starts from. Descending suits RR, P&L and dates; A-Z does not. */
  from?: SortDir;
};

/**
 * Sort selector shared by every list in the app, so RR sorts the same way whether you are on
 * Results, Manual or History. Clicking the active key flips the direction.
 */
export function SortChips<T extends string>({
  options,
  value,
  dir,
  onChange,
  label = 'Sort',
}: {
  options: ReadonlyArray<SortOption<T>>;
  value: T;
  dir: SortDir;
  onChange: (next: T, dir: SortDir) => void;
  label?: string;
}) {
  return (
    <div className="sort-row" role="group" aria-label={label}>
      {options.map((option) => {
        const active = value === option.value;
        const from = option.from ?? 'desc';
        return (
          <button
            key={option.value}
            type="button"
            className={`sort-chip${active ? ' active' : ''}`}
            aria-pressed={active}
            onClick={() =>
              onChange(option.value, !active ? from : dir === 'asc' ? 'desc' : 'asc')
            }
          >
            {option.label}
            {active ? (dir === 'desc' ? ' ↓' : ' ↑') : ''}
          </button>
        );
      })}
    </div>
  );
}
