// Serves the phase 0 browser bench. COOP/COEP make the page cross-origin
// isolated, so ONNX Runtime Web can use threads (?threads=N).
import { defineConfig } from "vite";

const headers = {
  "Cross-Origin-Opener-Policy": "same-origin",
  "Cross-Origin-Embedder-Policy": "require-corp",
};
export default defineConfig({
  root: __dirname,
  server: { headers, port: 5199, strictPort: true },
  preview: { headers, port: 5199, strictPort: true },
  optimizeDeps: { exclude: ["@huggingface/transformers"] },
});
