import React from 'react';
import { FlatList, StyleSheet, Text, View } from 'react-native';
import { useScan } from '../store/scanStore';

export function RejectedScreen() {
  const { rejected } = useScan();
  return (
    <View style={styles.root}>
      <FlatList
        data={rejected}
        keyExtractor={(item, i) => `${item.Symbol}-${i}`}
        ListEmptyComponent={<Text style={styles.empty}>No rejected symbols.</Text>}
        renderItem={({ item }) => (
          <View style={styles.row}>
            <Text style={styles.sym}>{item.Symbol}</Text>
            <Text style={styles.reason}>{item.Reason}</Text>
          </View>
        )}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: '#1e222d' },
  empty: { color: '#9aa4b2', padding: 24, textAlign: 'center' },
  row: {
    padding: 12,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: '#3a4150',
  },
  sym: { color: '#fff', fontWeight: '600' },
  reason: { color: '#ffb74d', marginTop: 4 },
});
