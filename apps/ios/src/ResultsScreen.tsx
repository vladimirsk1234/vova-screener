import {
  BUCKETS,
  TIMEFRAMES,
  UNIVERSES,
  bucketCounts,
  listResults,
  setInterest,
  type Bucket,
  type ResultSort,
  type ScanMeta,
  type SortDir,
  type Universe,
  type UserTf,
} from '@vova/device';
import { useEffect, useState } from 'react';
import { ActivityIndicator, FlatList, Pressable, StyleSheet, Text, View } from 'react-native';
import { formatAge } from './format';
import { useScreener } from './state';
import { colors } from './theme';
import { Chips, SignalCard } from './ui';

const SORTS: ResultSort[] = ['rr', 'pnl', 'interest', 'symbol'];
const EMPTY_SCAN: ScanMeta = {
  periodKey: '',
  asOf: null,
  newestAsOf: null,
  finishedAt: null,
  running: false,
  status: null,
};

export function ResultsScreen({
  onOpenChart,
  onManual,
}: {
  onOpenChart: (ticker: string, tf: UserTf, asOf?: string) => void;
  onManual: () => void;
}) {
  const { store, rev, refresh, settings } = useScreener();
  const [universe, setUniverse] = useState<Universe | 'Manual'>('Stocks');
  const [tf, setTf] = useState<UserTf>('Weekly');
  const [bucket, setBucket] = useState<Bucket>('new');
  const [sort, setSort] = useState<ResultSort>('rr');
  const [dir, setDir] = useState<SortDir>('desc');
  const [rows, setRows] = useState<Awaited<ReturnType<typeof load>> | null>(null);

  useEffect(() => {
    if (!store || universe === 'Manual') return;
    let cancelled = false;
    load(store, universe, tf, bucket, sort, dir, settings.minRr).then((page) => {
      if (!cancelled) setRows(page);
    });
    return () => {
      cancelled = true;
    };
  }, [store, universe, tf, bucket, sort, dir, settings.minRr, rev]);

  if (universe === 'Manual') {
    return (
      <View style={styles.page}>
        <Chips
          label="Universe"
          value={universe}
          options={[...UNIVERSES, 'Manual'] as const}
          onChange={setUniverse}
        />
        <Pressable style={styles.primary} onPress={onManual}>
          <Text style={styles.primaryText}>Open manual scan</Text>
        </Pressable>
      </View>
    );
  }

  const page = rows;
  const scanAge = formatAge(page?.scan.finishedAt);
  return (
    <View style={styles.page}>
      <Chips
        label="Universe"
        value={universe}
        options={[...UNIVERSES, 'Manual'] as const}
        onChange={(next) => {
          setUniverse(next);
          if (next === 'Manual') onManual();
        }}
      />
      <Chips label="Timeframe" value={tf} options={TIMEFRAMES} onChange={setTf} format={(v) => (v === 'Weekly' ? 'W' : 'M')} />
      <Chips
        label="Bucket"
        value={bucket}
        options={BUCKETS}
        onChange={setBucket}
        format={(v) => `${v[0].toUpperCase()}${v.slice(1)}${page ? ` ${page.counts[v]}` : ''}`}
      />
      <View style={styles.meta}>
        <Text style={styles.muted}>
          {page?.scan.running ? 'Scanning now' : scanAge ? `Scanned ${scanAge}` : 'No scan yet'}
          {page?.scan.newestAsOf ? ` · bar ${page.scan.newestAsOf}` : ''}
        </Text>
        <Chips
          label="Sort"
          value={sort}
          options={SORTS}
          onChange={(next) => {
            if (next === sort) setDir(dir === 'desc' ? 'asc' : 'desc');
            else {
              setSort(next);
              setDir(next === 'symbol' ? 'asc' : 'desc');
            }
          }}
          format={(v) => `${v === sort ? (dir === 'desc' ? '↓ ' : '↑ ') : ''}${v === 'rr' ? 'RR' : v === 'pnl' ? 'P&L' : v === 'interest' ? 'Marked' : 'A-Z'}`}
        />
      </View>
      {!page ? <ActivityIndicator color={colors.accent} /> : null}
      <FlatList
        style={{ flex: 1 }}
        data={page?.rows ?? []}
        keyExtractor={(item) => item.id}
        contentContainerStyle={{ paddingBottom: 24 }}
        ListEmptyComponent={
          page ? (
            <Text style={styles.muted}>
              {page.scan.finishedAt
                ? 'Nothing here.'
                : 'No scan has produced data yet. Start one from Settings. The pass runs on this iPhone and needs the app to stay open.'}
            </Text>
          ) : null
        }
        renderItem={({ item }) => (
          <SignalCard
            row={item}
            bucket={bucket}
            onOpen={() => onOpenChart(item.yahooTicker, item.tf, bucket === 'closed' ? (item.exitDate ?? undefined) : undefined)}
            onInterest={async (next) => {
              if (!store) return;
              const all = await store.listSignals();
              await store.saveSignals(setInterest(all, item.id, next));
              refresh();
            }}
          />
        )}
      />
    </View>
  );
}

async function load(
  store: NonNullable<ReturnType<typeof useScreener>['store']>,
  universe: Universe,
  tf: UserTf,
  bucket: Bucket,
  sort: ResultSort,
  dir: SortDir,
  minRr: number,
) {
  const [signals, meta] = await Promise.all([store.listSignals(), store.getScanMeta(universe, tf)]);
  const scan = meta ?? EMPTY_SCAN;
  const counts = bucketCounts(signals, universe, tf, minRr, scan);
  const page = listResults({ signals, universe, tf, bucket, sort, dir, minRr, scan, limit: 300 });
  return { ...page, counts, scan };
}

const styles = StyleSheet.create({
  page: { flex: 1, padding: 12, backgroundColor: colors.bg },
  meta: { marginBottom: 8 },
  muted: { color: colors.muted, fontSize: 13, marginVertical: 8 },
  primary: {
    minHeight: 48,
    borderRadius: 10,
    backgroundColor: colors.accent,
    alignItems: 'center',
    justifyContent: 'center',
  },
  primaryText: { color: '#fff', fontWeight: '700' },
});
