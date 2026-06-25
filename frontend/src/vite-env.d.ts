/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Backend base URL injected at build time. Empty → use the Vite dev proxy. */
  readonly VITE_API_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
