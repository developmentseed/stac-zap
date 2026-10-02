/**
 * Prints the results of a run as Markdown tables: the field test, the field
 * test split into prompts that ask for a change and prompts that do not, and
 * the world knowledge test.
 *
 * Usage: pnpm tsx eval/summary.ts [eval/results/<date>]
 */
import { readdirSync, readFileSync } from "node:fs";
import type { Row } from "./fields";

const dir = process.argv[2] ?? `eval/results/${readdirSync("eval/results").sort().at(-1)}`;
const FIELDS = ["catalog", "collection", "place", "period", "season", "cloud"] as const;
const NO_CHANGE = new Set(["keep", "all-time", "whole-year", "any"]);

const read = <T>(prefix: string) =>
  readdirSync(dir)
    .filter((f) => f.startsWith(prefix))
    .map((f) => JSON.parse(readFileSync(`${dir}/${f}`, "utf-8")) as T);
const median = (values: number[]) => {
  const s = [...values].sort((a, b) => a - b);
  return s.length ? s[s.length >> 1]! : NaN;
};
const where = (provider: string) => (provider === "codiv" ? "Codiv" : "OpenRouter");

type Fields = { provider: string; model: string; rows: Row[] };
const fields = read<Fields>("fields-").map((f) => {
  const ok = f.rows.filter((r) => r.ok).length;
  return { ...f, ok };
}).sort((a, b) => b.ok - a.ok);

const cell = (rows: Row[]) => {
  const errors = rows.filter((r) => r.error).length;
  const ok = rows.filter((r) => r.ok).length;
  return `${ok}/${rows.length}${errors ? ` (${errors} rejected)` : ""}`;
};

console.log(`## Field test (${dir})\n`);
console.log(`| Model | Where | ${FIELDS.join(" | ")} | Total | Median latency | Cost |`);
console.log(`|---|---|${FIELDS.map(() => "---").join("|")}|---|---|---|`);
for (const f of fields) {
  const ms = median(f.rows.flatMap((r) => (r.ms ? [r.ms] : [])));
  const cost = f.rows.reduce((s, r) => s + (r.cost ?? 0), 0);
  const cells = FIELDS.map((k) => cell(f.rows.filter((r) => r.field === k)));
  console.log(`| ${f.model} | ${where(f.provider)} | ${cells.join(" | ")} | ${f.ok}/${f.rows.length} | ${ms} ms | $${cost.toFixed(3)} |`);
}

console.log("\n## Field test: change asked or not\n");
const SPLIT = ["place", "period", "season", "cloud"] as const;
console.log(`| Model | ${SPLIT.map((k) => `${k}: change`).join(" | ")} | ${SPLIT.slice(1).map((k) => `${k}: no change`).join(" | ")} |`);
console.log(`|---|${[...SPLIT, ...SPLIT.slice(1)].map(() => "---").join("|")}|`);
for (const f of fields) {
  const part = (k: string, change: boolean) =>
    cell(f.rows.filter((r) => r.field === k && r.expect.some((e) => !NO_CHANGE.has(e)) === change));
  console.log(`| ${f.model} | ${SPLIT.map((k) => part(k, true)).join(" | ")} | ${SPLIT.slice(1).map((k) => part(k, false)).join(" | ")} |`);
}

type World = {
  provider: string;
  model: string;
  cost: number;
  latencies: number[];
  results: { name: string; flat?: number; fine?: number; error?: string }[];
};
console.log("\n## World knowledge test\n");
console.log("| Model | Where | 100 bands: median error | Coarse to fine: median error | Within 100 km | Within 250 km | Median latency | Cost |");
console.log("|---|---|---|---|---|---|---|---|");
const world = read<World>("world-").map((w) => {
  const done = w.results.filter((r) => r.fine !== undefined);
  return { ...w, done, fine: median(done.map((r) => r.fine!)) };
}).sort((a, b) => (a.fine || Infinity) - (b.fine || Infinity));
for (const w of world) {
  if (w.done.length === 0) {
    const error = w.results[0]?.error?.replace(/\s+/g, " ").slice(0, 90);
    console.log(`| ${w.model} | ${where(w.provider)} | rejected: ${error} | | | | | |`);
    continue;
  }
  const flat = median(w.done.map((r) => r.flat!));
  const under = (km: number) => `${w.done.filter((r) => r.fine! < km).length}/${w.results.length}`;
  console.log(`| ${w.model} | ${where(w.provider)} | ${flat.toLocaleString("en")} km | ${w.fine.toLocaleString("en")} km | ${under(100)} | ${under(250)} | ${median(w.latencies)} ms | $${w.cost.toFixed(3)} |`);
}
