/**
 * Runs one zap with the local models in headless Chromium and prints what the
 * panel shows, the page errors, and the times. The dev server must run.
 *
 * Usage: pnpm tsx scripts/local-smoke.ts ["a prompt" ...]
 * URL=http://localhost:5173/stac-zap/ by default.
 */
import { chromium } from "playwright";

const url = process.env.URL ?? "http://localhost:5173/stac-zap/";
const prompts = process.argv.slice(2).length ? process.argv.slice(2) : ["Sentinel-2 over Lisbon last summer with few clouds"];
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1400, height: 900 }, deviceScaleFactor: Number(process.env.DPR ?? 1) });
await page.addInitScript(() => {
  localStorage.setItem("stac-zap:backend", "local");
  localStorage.setItem("stac-zap:mode", "zap");
});
const errors: string[] = [];
page.on("pageerror", (e) => errors.push(`pageerror: ${e.message}`));
page.on("console", (m) => {
  if (m.type() === "error") errors.push(`console: ${m.text().slice(0, 300)}`);
});
const t0 = Date.now();
await page.goto(url);
const input = page.getByLabel("Zap prompt");
await input.waitFor({ timeout: 60_000 });
for (const prompt of prompts) {
  const t = Date.now();
  await input.fill(prompt);
  await input.press("Enter");
  // A zap is busy while the input is read-only.
  await page.waitForFunction(() => (document.querySelector('[aria-label="Zap prompt"]') as HTMLInputElement)?.readOnly, null, { timeout: 10_000 }).catch(() => {});
  await page.waitForFunction(() => !(document.querySelector('[aria-label="Zap prompt"]') as HTMLInputElement)?.readOnly, null, { timeout: 240_000 });
  const secs = ((Date.now() - t) / 1000).toFixed(1);
  const panel = (await page.getByTestId("zap-answers").innerText().catch(() => "(no answers panel)")).split("\n").filter(Boolean);
  console.log(`\n== ${prompt} (${secs} s)`);
  console.log(panel.slice(0, 30).join(" | "));
  const href = new URL(await page.evaluate(() => location.href)).searchParams.get("href");
  console.log("opened:", href);
  await page.screenshot({ path: `/tmp/local-smoke-${prompts.indexOf(prompt)}.png` });
}
console.log(`\npage errors (${errors.length}):\n${errors.slice(0, 15).join("\n")}`);
console.log(`total ${((Date.now() - t0) / 1000).toFixed(1)} s`);
await browser.close();
