/**
 * Runs the browser bench in headless Chromium for each model and thread
 * count and input length. Start the server first: pnpm vite --config eval/local/bench/vite.config.ts
 * Usage: pnpm tsx eval/local/bench/run.ts
 */
import { chromium } from "playwright";

const MODELS = [
  "onnx-community/granite-embedding-30m-english-ONNX",
  "onnx-community/granite-embedding-small-english-r2-ONNX",
];
const browser = await chromium.launch();
for (const model of MODELS) {
  for (const [threads, maxlen] of [[1, 128], [4, 128], [1, 64], [4, 64]]) {
    const page = await (await browser.newContext()).newPage();
    page.on("pageerror", (e) => console.error("page error:", e.message));
    await page.goto(`http://localhost:5199/?model=${encodeURIComponent(model)}&threads=${threads}&maxlen=${maxlen}`);
    await page.waitForFunction(() => (window as any).__result, null, { timeout: 600_000 });
    console.log(JSON.stringify(await page.evaluate(() => (window as any).__result)));
    await page.close();
  }
}
await browser.close();
