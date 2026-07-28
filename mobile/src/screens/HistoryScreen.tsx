import React, { useCallback, useState } from 'react';
import {
  FlatList,
  Pressable,
  RefreshControl,
  StyleSheet,
  Text,
  View,
} from 'react-native';
import { listScanRuns, listSignalsForRun, type ScanRunRow } from '../db/journal';
import type { BuyRow, ResultRow, SellRow } from '../types';

export function HistoryScreen() {
  const [runs, setRuns] = useState<ScanRunRow[]>([]);
  const [selected, setSelected] = useState<number | null>(null);
  const [signals, setSignals] = useState<ResultRow[]>([]);
  const [loading, setLoading] = useState(false);

  const reload = useCallback(async () => {
    setLoading(true);
    try {
      setRuns(await listScanRuns());
    } finally {
      setLoading(false);
    }
  }, []);

  React.useEffect(() => {
    void reload();
  }, [reload]);

  return (
    <View style={styles.root}>
      <FlatList
        data={runs}
        refreshControl={<RefreshControl refreshing={loading} onRefresh={reload} />}
        keyExtractor={(r) => String(r.id)}
        ListEmptyComponent={<Text style={styles.empty}>No saved scans yet. Run a scan first.</Text>}
        renderItem={({ item }) => (
          <Pressable
            style={styles.card}
            onPress={async () => {
              setSelected(item.id);
              setSignals(await listSignalsForRun(item.id));
            }}
          >
            <Text style={styles.title}>
              #{item.id} · {item.direction.toUpperCase()} · {item.tf}
            </Text>
            <Text style={styles.sub}>
              {item.created_at.slice(0, 19).replace('T', ' ')} · {item.source} ·{' '}
              {item.signal_count} signals
              {item.as_of ? ` · as of ${item.as_of}` : ''}
            </Text>
          </Pressable>
        )}
        ListFooterComponent={
          selected == null ? null : (
            <View style={styles.detail}>
              <Text style={styles.detailTitle}>Signals in run #{selected}</Text>
              {signals.map((s, i) => {
                const buy = s as BuyRow;
                const sell = s as SellRow;
                return (
                  <Text key={i} style={styles.signal}>
                    {s.tv_symbol}
                    {'TP' in buy
                      ? ` · TP ${buy.TP} SL ${buy.SL} RR ${buy.RR}`
                      : ` · P&L ${sell['P&L ($)']}`}
                  </Text>
                );
              })}
            </View>
          )
        }
      />
    </View>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: '#1e222d' },
  empty: { color: '#9aa4b2', padding: 24, textAlign: 'center' },
  card: {
    margin: 12,
    marginBottom: 0,
    backgroundColor: '#2a2e39',
    borderRadius: 10,
    padding: 12,
  },
  title: { color: '#fff', fontWeight: '700' },
  sub: { color: '#9aa4b2', marginTop: 4, fontSize: 12 },
  detail: { padding: 16 },
  detailTitle: { color: '#fff', fontWeight: '600', marginBottom: 8 },
  signal: { color: '#cfd6e0', marginBottom: 4 },
});
