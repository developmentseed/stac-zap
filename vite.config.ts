import react from "@vitejs/plugin-react";
import { readFileSync } from "node:fs";
import { defineConfig, loadEnv, type Plugin } from "vite";
import wasm from "vite-plugin-wasm";

const stacMap = "@developmentseed/stac-map";
// Its WebAssembly packages load at runtime, and stac-ts is types only.
const NOT_PREBUNDLED = new Set(["cql2-wasm", "stac-wasm", "@duckdb/duckdb-wasm", "shiki", "stac-ts"]);
// The package exports do not include package.json, so read it as a file.
const stacMapDependencies = Object.keys(
  JSON.parse(readFileSync(`node_modules/${stacMap}/package.json`, "utf-8"))
    .dependencies,
).filter((name) => !NOT_PREBUNDLED.has(name));

/** The decision APIs, with the `.env` variable that holds each key. */
const PROVIDERS = {
  openrouter: {
    url: "https://openrouter.ai/api/alpha/decisions",
    keyName: "OPENROUTER_API_KEY",
  },
  codiv: { url: "https://api.codiv.ai/v1/systemone", keyName: "CODIV_API_KEY" },
} as const;

/**
 * Serves `POST /api/decide` in the dev server: it forwards the body to a jev
 * decisions API with the key from `.env`. `?provider=codiv` selects Codiv;
 * the default is OpenRouter. Both take the same body. The key stays on the
 * server and never goes into the bundle. `.env` is read on each request, so a
 * new key works without a restart.
 */
function decideProxy(): Plugin {
  return {
    name: "stac-zap-decide",
    configureServer(server) {
      server.middlewares.use("/api/decide", async (req, res) => {
        if (req.method !== "POST") {
          res.statusCode = 405;
          return res.end();
        }
        const name = new URL(req.url ?? "", "http://localhost").searchParams.get("provider") ?? "openrouter";
        const provider = PROVIDERS[name as keyof typeof PROVIDERS];
        if (!provider) {
          res.statusCode = 400;
          return res.end(`Unknown provider: ${name}`);
        }
        const key = loadEnv(server.config.mode, server.config.envDir || process.cwd(), "")[
          provider.keyName
        ];
        if (!key) {
          res.statusCode = 503;
          return res.end(`Set ${provider.keyName} in .env`);
        }
        const chunks: Buffer[] = [];
        for await (const chunk of req) chunks.push(chunk as Buffer);
        try {
          const response = await fetch(provider.url, {
            method: "POST",
            headers: {
              Authorization: `Bearer ${key}`,
              "Content-Type": "application/json",
              "X-Title": "stac-zap",
            },
            body: Buffer.concat(chunks),
          });
          res.statusCode = response.status;
          res.setHeader("Content-Type", "application/json");
          res.end(await response.text());
        } catch (error) {
          res.statusCode = 502;
          res.end(String(error));
        }
      });
    },
  };
}

export default defineConfig({
  // GitHub Pages serves the site at /<repo>/; the workflow sets BASE_PATH.
  base: process.env.BASE_PATH ?? "/",
  build: { target: "esnext" },
  // stac-map imports the maplibre worker with `?worker&url`, which the
  // dependency pre-bundler cannot load. So Vite serves stac-map as source, and
  // pre-bundles its dependencies, some of which are CommonJS.
  optimizeDeps: {
    exclude: [stacMap],
    include: stacMapDependencies.map((name) => `${stacMap} > ${name}`),
  },
  worker: { format: "es" },
  plugins: [react(), wasm(), decideProxy()],
});
