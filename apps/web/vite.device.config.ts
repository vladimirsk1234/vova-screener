import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'node:path';

/**
 * The same UI, built for the iPhone app's WebView: one HTML file with everything inline, `/api`
 * answered on the phone through the native bridge (see `src/lib/deviceBridge.ts`).
 */
export default defineConfig({
  plugins: [react()],
  base: './',
  publicDir: false,
  define: {
    'import.meta.env.VITE_TARGET': JSON.stringify('device'),
  },
  resolve: {
    alias: {
      '@vova/engine': path.resolve(__dirname, '../../packages/engine/src/index.ts'),
    },
  },
  build: {
    outDir: 'dist-device',
    emptyOutDir: true,
    assetsInlineLimit: 100_000_000,
    cssCodeSplit: false,
    modulePreload: false,
    rollupOptions: {
      output: { inlineDynamicImports: true },
    },
  },
});
