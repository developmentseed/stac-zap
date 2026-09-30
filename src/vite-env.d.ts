/// <reference types="vite/client" />

interface Window {
  /** Plausible analytics, from the script in index.html. */
  plausible?: (event: string, options?: { props?: Record<string, string> }) => void;
}

interface ImportMetaEnv {
  /** The jev model on OpenRouter. Defaults to `typesafe/jev-1.13`. */
  readonly VITE_JEV_MODEL?: string;
  /** The decisions endpoint of the deployed site: the Worker in `worker/`. */
  readonly VITE_DECIDE_URL?: string;
}
