import { NavLink } from 'react-router-dom';

export type Segment<T extends string> = {
  value: T;
  to: string;
  label: string;
  /** Rendered as a small count next to the label. */
  badge?: number;
};

/**
 * Router-driven tab row. Results nests three of these (universe, timeframe, bucket) and keeping
 * them in the URL means every tab is linkable and survives a reload.
 */
export function SegmentedTabs<T extends string>({
  segments,
  label,
  size = 'md',
}: {
  segments: ReadonlyArray<Segment<T>>;
  label: string;
  size?: 'md' | 'sm';
}) {
  return (
    <div className={`segmented segmented--${size}`} role="tablist" aria-label={label}>
      {segments.map((segment) => (
        <NavLink
          key={segment.value}
          to={segment.to}
          role="tab"
          className={({ isActive }) => `segment${isActive ? ' active' : ''}`}
          replace
        >
          <span>{segment.label}</span>
          {segment.badge != null ? <span className="segment-badge">{segment.badge}</span> : null}
        </NavLink>
      ))}
    </div>
  );
}
