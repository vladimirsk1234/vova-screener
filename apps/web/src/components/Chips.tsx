type ChipsProps<T extends string> = {
  label?: string;
  value: T;
  options: readonly T[];
  onChange: (value: T) => void;
  disabled?: boolean;
  format?: (value: T) => string;
};

export function Chips<T extends string>({
  label,
  value,
  options,
  onChange,
  disabled,
  format,
}: ChipsProps<T>) {
  return (
    <div>
      {label ? <span className="field-label">{label}</span> : null}
      <div className="chip-row">
        {options.map((opt) => (
          <button
            key={opt}
            type="button"
            disabled={disabled}
            className={`chip${value === opt ? ' active' : ''}`}
            onClick={() => onChange(opt)}
          >
            {format ? format(opt) : opt}
          </button>
        ))}
      </div>
    </div>
  );
}

export function Switch({
  label,
  checked,
  onChange,
  disabled,
}: {
  label: string;
  checked: boolean;
  onChange: (v: boolean) => void;
  disabled?: boolean;
}) {
  return (
    <label className="switch-row">
      <span>{label}</span>
      <input
        type="checkbox"
        checked={checked}
        disabled={disabled}
        onChange={(e) => onChange(e.target.checked)}
      />
    </label>
  );
}
