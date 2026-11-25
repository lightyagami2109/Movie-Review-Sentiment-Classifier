import { defineConfig } from 'vite';

export default defineConfig({
  envDir: '..', // read the single .env in the project folder
  server: {
    proxy: { '/api': process.env.SENTIMENT_API_TARGET || 'http://127.0.0.1:8000' },
  },
});
