/**
 * Prints the results of a run as Markdown tables: the field test, the field
 * test split into prompts that ask for a change and prompts that do not, the
 * world knowledge test, the extraction test and the unparsed prompts.
 *
 * Usage: pnpm tsx eval/summary.ts [eval/results/<date>]
 */
import { readdirSync, readFileSync } from "node:fs";
import { UNPARSED_CASES } from "./cases";
import type { ExtractRow } from "./extract";
import type { Row } from "./fields";
import type { UnparsedRow } from "./unparsed";

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

// The extraction test: place, time and cloud cover as values. Jev is scored
// from its field test rows in the same folder: time is correct when the
// period and the part of the year are both correct.
type Extract = { model: string; set: string; variant?: string; oneCall?: boolean; rows: ExtractRow[] };
const label = (e: Extract) => `${e.model} (${e.variant ?? "plain"}${e.oneCall ? ", one call" : ""})`;
const GROUPS = ["place", "time", "cloud"] as const;
const extracts = read<Extract>("extract-");
const fieldOf = (e: Extract) => e.set === "field";
const jevFields = fields.find((f) => f.model.startsWith("typesafe/jev"));
if (extracts.length || jevFields) {
  console.log("\n## Extraction test: place, time and cloud cover\n");
  console.log("| Model | Place | Time | Cloud cover | Total | Median latency | Cost |");
  console.log("|---|---|---|---|---|---|---|");
  if (jevFields) {
    const rows = jevFields.rows;
    const place = rows.filter((r) => r.field === "place");
    const cloud = rows.filter((r) => r.field === "cloud");
    const prompts = [...new Set(rows.filter((r) => r.field === "period").map((r) => r.prompt))];
    const timeOk = prompts.filter((p) =>
      rows.filter((r) => r.prompt === p && (r.field === "period" || r.field === "season")).every((r) => r.ok),
    ).length;
    const ok = place.filter((r) => r.ok).length + timeOk + cloud.filter((r) => r.ok).length;
    const n = place.length + prompts.length + cloud.length;
    const ms = median(rows.flatMap((r) => (r.ms ? [r.ms] : [])));
    console.log(`| ${jevFields.model} (options) | ${cell(place)} | ${timeOk}/${prompts.length} | ${cell(cloud)} | ${ok}/${n} | ${ms} ms | |`);
  }
  for (const e of extracts.filter(fieldOf)) {
    const part = (g: string) => {
      const r = e.rows.filter((x) => x.group === g);
      const errors = r.filter((x) => x.error).length;
      return `${r.filter((x) => x.ok).length}/${r.length}${errors ? ` (${errors} rejected)` : ""}`;
    };
    const ok = e.rows.filter((r) => r.ok).length;
    const ms = median(e.rows.flatMap((r) => (r.ms ? [r.ms] : [])));
    // With one call, the rows of a prompt share the cost of that call.
    const costs = e.oneCall ? new Map(e.rows.map((r) => [r.prompt, r.cost ?? 0])).values() : e.rows.map((r) => r.cost ?? 0);
    const cost = [...costs].reduce((s, c) => s + c, 0);
    console.log(`| ${label(e)} | ${GROUPS.map(part).join(" | ")} | ${ok}/${e.rows.length} | ${ms} ms | $${cost.toFixed(4)} |`);
  }
}

// The unparsed prompts: one row per prompt, one column per model.
type Unparsed = { model: string; rows: UnparsedRow[] };
const unparsed = [
  ...read<Unparsed>("unparsed-"),
  ...extracts
    .filter((e) => e.set === "unparsed")
    .map((e) => ({ ...e, model: label(e) })),
] as { model: string; rows: (UnparsedRow | ExtractRow)[] }[];
if (unparsed.length) {
  const show = (r: UnparsedRow | ExtractRow | undefined) => {
    if (!r) return "";
    if (r.error) return "rejected";
    const v = r.value;
    const text = v === null || v === undefined ? "none" : Array.isArray(v) ? v.join(" to ") : String(v);
    return `${r.ok ? "✓" : "✗"} ${text}`;
  };
  console.log("\n## Unparsed prompts\n");
  console.log(`| Id | Prompt | Expected | ${unparsed.map((u) => u.model).join(" | ")} |`);
  console.log(`|---|---|---|${unparsed.map(() => "---").join("|")}|`);
  for (const c of UNPARSED_CASES) {
    const cells = unparsed.map((u) => show(u.rows.find((r) => r.id === c.id)));
    const expect = c.range ? c.range.join(" to ") : c.cloud !== undefined ? `${c.cloud}` : c.place;
    console.log(`| ${c.id} | ${c.prompt} | ${expect} | ${cells.join(" | ")} |`);
  }
  console.log(`| | **Correct** | | ${unparsed.map((u) => `${u.rows.filter((r) => r.ok).length}/${u.rows.length}`).join(" | ")} |`);
}
