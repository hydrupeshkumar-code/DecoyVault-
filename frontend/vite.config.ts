import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      "/upload": "http://localhost:8000",
      "/detect": "http://localhost:8000",
      "/reconstruct": "http://localhost:8000",
      "/metrics": "http://localhost:8000",
      "/report": "http://localhost:8000",
    },
  },
});
