/// <reference types="vite/client" />

interface ImportMetaEnv {
  /**
   * Optional HTTPS origin of your translation proxy (no trailing slash).
   * Expose POST `/translate` with body `{ text, target }` → `{ translation }` using Google Cloud Translation server-side.
   */
  readonly VITE_TRANSLATION_PROXY_URL?: string;
  /** Optional rss2json.com API key for higher rate limits. */
  readonly VITE_RSS2JSON_API_KEY?: string;
  /** Optional GNews API key (https://gnews.io) — works in production without the Vite RSS proxy. */
  readonly VITE_GNEWS_API_KEY?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
