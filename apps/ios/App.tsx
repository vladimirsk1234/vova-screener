import {
  bridgeReplyScript,
  createCachedStore,
  createDeviceApi,
  parseUniverse,
} from '@vova/device';
import { StatusBar } from 'expo-status-bar';
import { useEffect, useMemo, useRef, useState } from 'react';
import { Linking, StyleSheet, Text, View } from 'react-native';
import { WebView, type WebViewMessageEvent } from 'react-native-webview';
import { openPhoneStore } from './src/db';
import { phoneNotifier } from './src/notifier';
import { ETF_TEXT, STOCKS_TEXT } from './src/universeData';
import { WEB_APP_HTML } from './src/webApp.generated';

/** Origin the inline page runs under, so its localStorage (tab memory, filters) persists. */
const BASE_URL = 'https://sv-screener.app/';

type DeviceApi = ReturnType<typeof createDeviceApi>;

/**
 * The web UI (apps/web) running in a WebView, with its `/api` answered on this phone:
 * bars, signals and History live in SQLite here; Yahoo is the only network source.
 */
export function App() {
  const webRef = useRef<WebView>(null);
  const [error, setError] = useState<string | null>(null);
  const universe = useMemo(() => parseUniverse(STOCKS_TEXT, ETF_TEXT), []);

  const apiPromise = useMemo<Promise<DeviceApi>>(
    () =>
      openPhoneStore().then(async (disk) => {
        const api = createDeviceApi({
          store: createCachedStore(disk),
          universe,
          notifier: phoneNotifier,
          onDataChanged: () =>
            webRef.current?.injectJavaScript('window.__vovaDataChanged && window.__vovaDataChanged(); true;'),
        });
        // Before the WebView's first request, so the chart never reads a preset mid-migration.
        await api.migratePresets().catch(() => undefined);
        return api;
      }),
    [universe],
  );

  useEffect(() => {
    apiPromise
      .then((api) => api.restoreAlerts().catch(() => undefined))
      .catch((err: Error) => setError(err.message));
  }, [apiPromise]);

  const onMessage = async (event: WebViewMessageEvent) => {
    const script = await bridgeReplyScript(await apiPromise, event.nativeEvent.data);
    if (script) webRef.current?.injectJavaScript(script);
  };

  if (error) {
    return (
      <View style={styles.fallback}>
        <Text style={styles.error}>{error}</Text>
      </View>
    );
  }

  return (
    <View style={styles.root}>
      <StatusBar style="light" />
      <WebView
        ref={webRef}
        style={styles.root}
        source={{ html: WEB_APP_HTML, baseUrl: BASE_URL }}
        originWhitelist={['*']}
        onMessage={(event) => void onMessage(event)}
        contentInsetAdjustmentBehavior="never"
        automaticallyAdjustContentInsets={false}
        allowsBackForwardNavigationGestures
        setSupportMultipleWindows
        javaScriptEnabled
        domStorageEnabled
        onShouldStartLoadWithRequest={(request) => {
          if (request.url.startsWith(BASE_URL) || request.url.startsWith('about:')) return true;
          void Linking.openURL(request.url);
          return false;
        }}
        onOpenWindow={(event) => {
          void Linking.openURL(event.nativeEvent.targetUrl);
        }}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: '#050505' },
  fallback: { flex: 1, backgroundColor: '#050505', justifyContent: 'center', padding: 24 },
  error: { color: '#f23645' },
});
