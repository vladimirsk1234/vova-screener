import type { Bucket, TrackedSignal } from '@vova/device';
import { Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';
import { barsLabel, exitLabel, money, num, signedMoney } from './format';
import { colors } from './theme';

export function Chips<T extends string>({
  label,
  value,
  options,
  onChange,
  format,
}: {
  label: string;
  value: T;
  options: readonly T[];
  onChange: (next: T) => void;
  format?: (value: T) => string;
}) {
  return (
    <View style={styles.chipBlock}>
      <Text style={styles.chipLabel}>{label}</Text>
      <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={styles.chipRow}>
        {options.map((option) => {
          const active = option === value;
          return (
            <Pressable key={option} onPress={() => onChange(option)} style={[styles.chip, active && styles.chipOn]}>
              <Text style={[styles.chipText, active && styles.chipTextOn]}>{format ? format(option) : option}</Text>
            </Pressable>
          );
        })}
      </ScrollView>
    </View>
  );
}

export function SignalCard({
  row,
  bucket,
  onOpen,
  onInterest,
}: {
  row: TrackedSignal;
  bucket: Bucket;
  onOpen: () => void;
  onInterest: (next: TrackedSignal['interest']) => void;
}) {
  const showPnl = bucket !== 'new' && (row.pnlUsd != null || row.unrealizedUsd != null);
  const pnl = row.status === 'closed' || row.provisionalClose ? row.pnlUsd : row.unrealizedUsd;
  const positive = (pnl ?? 0) >= 0;
  const rr = bucket === 'closed' ? row.rrAtEntry : (row.lastRr ?? row.rrAtEntry);
  return (
    <Pressable onPress={onOpen} style={styles.card}>
      <View style={styles.row}>
        <View style={{ flex: 1 }}>
          <Text style={styles.symbol}>{row.symbol}</Text>
          <Text style={styles.muted} numberOfLines={1}>{row.companyName}</Text>
        </View>
        {showPnl ? (
          <Text style={[styles.badge, positive ? styles.up : styles.down]}>
            {signedMoney(pnl)}
          </Text>
        ) : null}
      </View>
      <View style={styles.badges}>
        {row.interest === 'not_interested' ? <Text style={[styles.badge, styles.down]}>NO INTEREST</Text> : null}
        {row.isStrong ? <Text style={styles.badge}>STRONG</Text> : null}
        {row.provisional && bucket === 'new' ? <Text style={[styles.badge, styles.warn]}>LIVE</Text> : null}
        {bucket === 'closed' && row.exitReason ? <Text style={styles.badge}>{exitLabel(row.exitReason)}</Text> : null}
        {row.provisionalClose ? <Text style={[styles.badge, styles.warn]}>CLOSING</Text> : null}
      </View>
      <Text style={styles.metrics}>
        E {money(row.entry)} · TP {money(row.tp)} · SL {money(row.sl)} · Sh {row.shares} · RR {num(rr)}
      </Text>
      <Text style={styles.foot}>
        {bucket === 'closed'
          ? `${row.openedAsOf} → ${row.exitDate ?? '—'} · exit ${money(row.exitPrice)} · ${num(row.pnlR)}R`
          : bucket === 'valid'
            ? `valid ${barsLabel(row.barsSinceValid)} · since ${row.validSinceAsOf ?? row.openedAsOf} · now ${money(row.lastPrice)}`
            : `signal bar ${row.validSinceAsOf ?? row.openedAsOf}`}
      </Text>
      <View style={styles.actions}>
        <Pressable
          style={[styles.smallBtn, row.interest === 'interested' && styles.smallBtnOn]}
          onPress={() => onInterest(row.interest === 'interested' ? null : 'interested')}
        >
          <Text style={styles.smallBtnText}>Interested</Text>
        </Pressable>
        <Pressable
          style={[styles.smallBtn, row.interest === 'not_interested' && styles.smallBtnDanger]}
          onPress={() => onInterest(row.interest === 'not_interested' ? null : 'not_interested')}
        >
          <Text style={styles.smallBtnText}>Not Interested</Text>
        </Pressable>
      </View>
    </Pressable>
  );
}

export function Sparkline({ points }: { points: Array<{ equity: number }> }) {
  if (points.length < 2) return null;
  const values = points.map((p) => p.equity);
  const min = Math.min(...values, 0);
  const max = Math.max(...values, 0);
  const span = max - min || 1;
  return (
    <View style={styles.spark}>
      {values.map((value, index) => (
        <View
          key={`${index}-${value}`}
          style={{
            flex: 1,
            height: Math.max(2, ((value - min) / span) * 46),
            backgroundColor: values[values.length - 1] >= 0 ? colors.up : colors.down,
            marginHorizontal: 0.5,
          }}
        />
      ))}
    </View>
  );
}

export function Stat({ label, value, tone }: { label: string; value: string; tone?: 'up' | 'down' }) {
  return (
    <View style={styles.stat}>
      <Text style={styles.muted}>{label}</Text>
      <Text style={[styles.statValue, tone === 'up' && styles.upText, tone === 'down' && styles.downText]}>{value}</Text>
    </View>
  );
}

export function toneOf(n: number | null | undefined): 'up' | 'down' | undefined {
  if (n == null || !Number.isFinite(n) || n === 0) return undefined;
  return n > 0 ? 'up' : 'down';
}

const styles = StyleSheet.create({
  chipBlock: { marginBottom: 8 },
  chipLabel: { color: colors.muted, fontSize: 12, marginBottom: 4 },
  chipRow: { gap: 6 },
  chip: {
    minHeight: 36,
    paddingHorizontal: 12,
    borderRadius: 8,
    backgroundColor: colors.surface,
    alignItems: 'center',
    justifyContent: 'center',
  },
  chipOn: { backgroundColor: colors.accent },
  chipText: { color: colors.text, fontSize: 14 },
  chipTextOn: { color: '#fff', fontWeight: '600' },
  card: {
    backgroundColor: colors.surface,
    borderRadius: 12,
    padding: 12,
    marginBottom: 10,
    gap: 6,
  },
  row: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  symbol: { color: colors.text, fontSize: 17, fontWeight: '700' },
  muted: { color: colors.muted, fontSize: 12 },
  badge: {
    color: colors.text,
    backgroundColor: colors.chip,
    overflow: 'hidden',
    borderRadius: 6,
    paddingHorizontal: 6,
    paddingVertical: 2,
    fontSize: 11,
    fontWeight: '700',
  },
  badges: { flexDirection: 'row', flexWrap: 'wrap', gap: 6 },
  up: { color: colors.up },
  down: { color: colors.down },
  warn: { color: '#f0b429' },
  upText: { color: colors.up },
  downText: { color: colors.down },
  metrics: { color: colors.text, fontSize: 13 },
  foot: { color: colors.muted, fontSize: 12 },
  actions: { flexDirection: 'row', gap: 8, marginTop: 4 },
  smallBtn: {
    minHeight: 36,
    paddingHorizontal: 10,
    borderRadius: 8,
    backgroundColor: colors.chip,
    alignItems: 'center',
    justifyContent: 'center',
  },
  smallBtnOn: { backgroundColor: colors.accent },
  smallBtnDanger: { backgroundColor: '#6b2430' },
  smallBtnText: { color: colors.text, fontSize: 13 },
  spark: { height: 48, flexDirection: 'row', alignItems: 'flex-end', marginVertical: 8 },
  stat: { width: '48%', marginBottom: 8 },
  statValue: { color: colors.text, fontSize: 16, fontWeight: '700' },
});
