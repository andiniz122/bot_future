import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0', // Permite conexões externas
    port: 5173,
    strictPort: true, // Falha se a porta não estiver disponível
    cors: true,
    proxy: {
      // Proxy para API requests
      '/api': {
        target: 'http://62.72.1.122:8000',
        changeOrigin: true,
        secure: false,
        ws: true, // Proxy WebSocket
      },
      // Proxy para WebSocket
      '/ws': {
        target: 'ws://62.72.1.122:8000',
        ws: true,
        changeOrigin: true,
      }
    }
  },
  preview: {
    host: '0.0.0.0',
    port: 5173,
    strictPort: true
  },
  build: {
    outDir: 'dist',
    sourcemap: false, // Desabilita sourcemaps em produção
    minify: 'terser',
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['react', 'react-dom'],
          charts: ['recharts'],
        }
      }
    }
  },
  define: {
    // Substitui process.env no build
    'process.env': process.env
  }
})