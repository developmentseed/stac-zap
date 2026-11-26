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
const page = await browser.newPage({ viewport: { width: 1400, height: 900 } });
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
  // Busy: the input is disabled or a spinner shows; wait for the answers, then for the spinner to go.
  await page.waitForTimeout(500);
  await page.waitForFunction(() => !document.querySelector(".chakra-spinner"), null, { timeout: 180_000 }).catch(() => {});
  const panel = await page.locator("body").innerText();
  const start = panel.indexOf(prompt);
  console.log(`\n== ${prompt} (${((Date.now() - t) / 1000).toFixed(1)} s)`);
  console.log(panel.slice(Math.max(0, start), start + 1500).split("\n").filter(Boolean).slice(0, 40).join("\n"));
  console.log("map href:", await page.evaluate(() => location.href));
  await page.screenshot({ path: `/tmp/local-smoke-${prompts.indexOf(prompt)}.png` });
}
console.log(`\npage errors (${errors.length}):\n${errors.slice(0, 15).join("\n")}`);
console.log(`total ${((Date.now() - t0) / 1000).toFixed(1)} s`);
await browser.close();
