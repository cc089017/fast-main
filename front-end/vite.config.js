// front-end/vite.config.js
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      // 프론트에서 "/api/..." 로 호출하면 FastAPI(127.0.0.1:8000)로 프록시
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
        secure: false,
      },
    },
  },
  resolve: {
    alias: { "@": "/src" },
  },
});
