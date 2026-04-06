import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";
import path from "path";

// Default configuration
const defaultConfig = {
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    host: "::",
    port: 5173,
    hmr: {
      overlay: false,
    },
    proxy: {
      "/api": {
        target: "http://localhost:8080",
        changeOrigin: true,
        secure: false,
      },
    },
  },
};

// Advanced configuration with proxy functions
export default defineConfig(({ command }) => {
  if (command !== "serve") {
    // Return simpler config for build
    return defaultConfig;
  }

  // Development config with proxy middleware
  return {
    ...defaultConfig,
    server: {
      ...defaultConfig.server,
      proxy: {
        // ... (keep original proxy rules if any exist)
        // Add custom middleware for RSS and Translation
        "/proxy/gnews": {
          target: "https://gnews.io",
          changeOrigin: true,
          rewrite: (path) => path.replace(/^\/proxy\/gnews/, "/api/v4"),
        },
        "/proxy/rss2json": {
          target: "https://api.rss2json.com",
          changeOrigin: true,
          rewrite: (path) => path.replace(/^\/proxy\/rss2json/, "/v1"),
        },
        "/proxy/lingva": {
          target: "https://lingva.ml", // A public instance of Lingva Translate
          changeOrigin: true,
          rewrite: (path) => path.replace(/^\/proxy\/lingva/, "/api/v1"),
        },
        "/proxy/mymemory": {
          target: "https://api.mymemory.translated.net",
          changeOrigin: true,
          rewrite: (path) => path.replace(/^\/proxy\/mymemory/, "/get"),
        },
      },
    },
  };
});
