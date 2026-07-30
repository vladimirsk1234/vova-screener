import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'node:path';

const apiTarget = process.env.VITE_API_TARGET ?? 'http://localhost:3001';

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@vova/engine': path.resolve(__dirname, '../../packages/engine/src/index.ts'),
    },
  },
  server: {
    port: 5173,
    host: true,
    // Same-origin /api keeps phone access working over LAN without CORS config.
    proxy: {
      '/api': { target: apiTarget, changeOrigin: true },
    },
  },
});
