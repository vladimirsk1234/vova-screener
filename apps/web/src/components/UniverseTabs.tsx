import { UNIVERSES, type Bucket, type Timeframe } from '../lib/api';
import { resultsPathForUniverse } from '../lib/tabMemory';
import { SegmentedTabs } from './SegmentedTabs';

/**
 * Stocks · ETF · Value · Manual. Stocks/ETF keep the current timeframe and bucket when those
 * are on the URL; otherwise they restore the last Results path for that universe.
 */
export function UniverseTabs({
  tf,
  bucket,
}: {
  tf?: Timeframe;
  bucket?: Bucket;
}) {
  const stocksEtf =
    tf && bucket
      ? UNIVERSES.map((u) => ({
          value: u,
          to: `/results/${u}/${tf}/${bucket}`,
          label: u,
        }))
      : UNIVERSES.map((u) => ({
          value: u,
          to: resultsPathForUniverse(u),
          label: u,
        }));
  return (
    <SegmentedTabs
      label="Universe"
      segments={[
        ...stocksEtf,
        { value: 'value' as const, to: '/results/value', label: 'Value' },
        { value: 'manual' as const, to: '/results/manual', label: 'Manual' },
      ]}
    />
  );
}
