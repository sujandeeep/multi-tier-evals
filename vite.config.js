import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

const backendUrl = process.env.VITE_BACKEND_URL || 'http://127.0.0.1:8000';
const prometheusUrl = process.env.VITE_PROMETHEUS_URL || 'http://127.0.0.1:9090';

export default defineConfig({
  plugins: [react()],
  server: {
    host: process.env.VITE_DEV_HOST || '0.0.0.0',
    port: 3001,
    proxy: {
      '/api': {
        target: backendUrl,
        changeOrigin: true,
      },
      '/metrics': {
        target: backendUrl,
        changeOrigin: true,
      },
      '/backend-metrics': {
        target: backendUrl,
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/backend-metrics/, '/metrics'),
      },
      '/prom-api': {
        target: prometheusUrl,
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/prom-api/, '/api/v1'),
      },
    },
  },
});
