import React, { useCallback, useState } from 'react';
import {
  ActivityIndicator,
  Alert,
  FlatList,
  Pressable,
  RefreshControl,
  StyleSheet,
  Text,
  TextInput,
  View,
} from 'react-native';
import {
  listTrades,
  manualCloseTrade,
  updateOpenTrades,
  type TradeRow,
} from '../db/journal';

export function TradesScreen() {
  const [trades, setTrades] = useState<TradeRow[]>([]);
  const [loading, setLoading] = useState(false);
  const [updating, setUpdating] = useState(false);
  const [manualId, setManualId] = useState<number | null>(null);
  const [exitPx, setExitPx] = useState('');

  const reload = useCallback(async () => {
    setLoading(true);
    try {
      setTrades(await listTrades());
    } finally {
      setLoading(false);
    }
  }, []);

  React.useEffect(() => {
    void reload();
  }, [reload]);

  const onUpdate = async () => {
    setUpdating(true);
    try {
      const r = await updateOpenTrades();
      Alert.alert('Updated', `Closed ${r.closed}, still open ${r.stillOpen}`);
      await reload();
    } catch (e) {
      Alert.alert('Error', e instanceof Error ? e.message : String(e));
    } finally {
      setUpdating(false);
    }
  };

  return (
    <View style={styles.root}>
      <Pressable style={styles.updateBtn} onPress={onUpdate} disabled={updating}>
        {updating ? (
          <ActivityIndicator color="#fff" />
        ) : (
          <Text style={styles.updateText}>Update open trades (Yahoo TP/SL)</Text>
        )}
      </Pressable>
      <FlatList
        data={trades}
        refreshControl={<RefreshControl refreshing={loading} onRefresh={reload} />}
        keyExtractor={(t) => String(t.id)}
        ListEmptyComponent={<Text style={styles.empty}>No journaled trades yet.</Text>}
        renderItem={({ item }) => (
          <View style={styles.card}>
            <Text style={styles.title}>
              {item.symbol} · {item.status.toUpperCase()} · {item.tf}
            </Text>
            <Text style={styles.sub}>
              Entry {item.entry} · TP {item.tp} · SL {item.sl} · RR {item.rr_at_entry} ·{' '}
              {item.shares} sh
            </Text>
            {item.status === 'closed' && (
              <Text
                style={{
                  color: (item.pnl_usd ?? 0) >= 0 ? '#089981' : '#f23645',
                  marginTop: 4,
                }}
              >
                {item.exit_reason} @ {item.exit_price} on {item.exit_date} · P&L $
                {item.pnl_usd} ({item.pnl_r}R)
              </Text>
            )}
            {item.status === 'open' && (
              <Pressable
                onPress={() => {
                  setManualId(item.id);
                  setExitPx(String(item.entry));
                }}
              >
                <Text style={styles.link}>Manual close…</Text>
              </Pressable>
            )}
          </View>
        )}
      />
      {manualId != null && (
        <View style={styles.modal}>
          <Text style={styles.title}>Manual close #{manualId}</Text>
          <TextInput
            style={styles.input}
            keyboardType="decimal-pad"
            value={exitPx}
            onChangeText={setExitPx}
            placeholder="Exit price"
          />
          <Pressable
            style={styles.updateBtn}
            onPress={async () => {
              await manualCloseTrade(
                manualId,
                Number(exitPx),
                new Date().toISOString().slice(0, 10),
              );
              setManualId(null);
              await reload();
            }}
          >
            <Text style={styles.updateText}>Confirm close</Text>
          </Pressable>
          <Pressable onPress={() => setManualId(null)}>
            <Text style={styles.link}>Cancel</Text>
          </Pressable>
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: '#1e222d' },
  empty: { color: '#9aa4b2', padding: 24, textAlign: 'center' },
  updateBtn: {
    margin: 12,
    backgroundColor: '#2962ff',
    padding: 12,
    borderRadius: 8,
    alignItems: 'center',
  },
  updateText: { color: '#fff', fontWeight: '600' },
  card: {
    marginHorizontal: 12,
    marginBottom: 10,
    backgroundColor: '#2a2e39',
    borderRadius: 10,
    padding: 12,
  },
  title: { color: '#fff', fontWeight: '700' },
  sub: { color: '#9aa4b2', marginTop: 4, fontSize: 12 },
  link: { color: '#4ea1ff', marginTop: 8 },
  modal: {
    position: 'absolute',
    left: 16,
    right: 16,
    bottom: 40,
    backgroundColor: '#2a2e39',
    borderRadius: 12,
    padding: 16,
    borderWidth: 1,
    borderColor: '#3a4150',
  },
  input: {
    backgroundColor: '#1e222d',
    color: '#fff',
    borderRadius: 8,
    padding: 10,
    marginVertical: 10,
  },
});
