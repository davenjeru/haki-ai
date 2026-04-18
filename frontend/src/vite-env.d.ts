/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** e.g. https://xxxx.execute-api.region.amazonaws.com/prod — no trailing slash */
  readonly VITE_API_BASE_URL?: string
  /** When true, uses local mock responses (also used when VITE_API_BASE_URL is unset) */
  readonly VITE_USE_MOCK?: string
  /** Path appended to base URL for POST chat. Default `/chat` */
  readonly VITE_CHAT_PATH?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
