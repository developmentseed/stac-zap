/**
 * Measures what the browser downloads for one zap with the local model: the
 * bytes on the wire (after Content-Encoding) of each response, by group.
 * The dev server must run. Usage: URL=http://localhost:5176/ pnpm tsx scripts/measure-download.ts
 */
import { chromium } from "playwright";

const url = process.env.URL ?? "http://localhost:5173/";
const browser = await chromium.launch();
const page = await browser.newPage();
await page.addInitScript(() => {
  localStorage.setItem("stac-zap:backend", "local");
  localStorage.setItem("stac-zap:mode", "zap");
});
const rows: { url: string; bytes: number; encoding: string }[] = [];
page.on("requestfinished", async (req) => {
  const res = await req.response();
  if (!res) return;
  const sizes = await req.sizes().catch(() => null);
  rows.push({ url: req.url(), bytes: sizes?.responseBodySize ?? 0, encoding: (await res.allHeaders())["content-encoding"] ?? "" });
});
await page.goto(url, { timeout: 180_000, waitUntil: "domcontentloaded" });
const input = page.getByLabel("Zap prompt");
await input.waitFor({ timeout: 60_000 });
await input.fill("Sentinel-2 over Lisbon last summer with few clouds");
await input.press("Enter");
await page.waitForFunction(() => (document.querySelector('[aria-label="Zap prompt"]') as HTMLInputElement)?.readOnly, null, { timeout: 10_000 }).catch(() => {});
await page.waitForFunction(() => !(document.querySelector('[aria-label="Zap prompt"]') as HTMLInputElement)?.readOnly, null, { timeout: 240_000 });
await page.waitForTimeout(2000);
const big = rows.filter((r) => /onnx|wasm|vectors|tokenizer|stac-zap\.json|config\.json|ort|transformers/i.test(r.url)).sort((a, b) => b.bytes - a.bytes);
for (const r of big) console.log(`${(r.bytes / 1e6).toFixed(2).padStart(7)} MB  ${r.encoding.padEnd(5)} ${r.url.slice(0, 150)}`);
console.log(`model-related total: ${(big.reduce((a, r) => a + r.bytes, 0) / 1e6).toFixed(1)} MB; all responses: ${(rows.reduce((a, r) => a + r.bytes, 0) / 1e6).toFixed(1)} MB`);
await browser.close();
