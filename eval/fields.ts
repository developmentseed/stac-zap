/**
 * The field test: asks one decision model the questions of stac-zap for each
 * prompt in `cases.ts`, and scores the answers as the app applies them.
 *
 * The questions come from the app code (`buildPayload`, the option lists and
 * the cloud cover question), with the catalogs of `data/stac-snapshot.json`.
 * Each field is asked in its own call, so a model with a limit on the number
 * of options can still answer the other fields. The app asks the second-step
 * fields in one call.
 *
 * Usage:
 *   PROVIDER=openrouter MODEL=typesafe/jev-1.14 pnpm tsx eval/fields.ts
 * Optional: FIELDS=cloud,period asks only these fields and replaces them in
 * the result file. RESULTS=eval/results/<date> sets the output folder.
 */
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { KEEP, THRESHOLD } from "../src/zap/decide";
import { cloudChoice } from "../src/zap/options";
import { CASES, NO_CHANGE } from "./cases";
import { ask, modelOfEnv } from "./providers";
import { FIELDS, questionsOf, TODAY, type Field } from "./questions";

/** The probability the app needs to change a field, as in run.ts. */
const THRESHOLDS: Record<Exclude<Field, "cloud">, number> = {
  catalog: 0,
  collection: 0,
  place: THRESHOLD,
  period: 0.35,
  season: THRESHOLD,
};
const NO_CHANGE_OF: Record<string, string[]> = {
  period: [KEEP, "all-time"],
  season: [KEEP, "whole-year"],
  cloud: [KEEP, "any"],
};

export type Row = {
  prompt: string;
  field: Field;
  expect: string[];
  options: number;
  error?: string;
  /** The option the model liked most. */
  top?: string;
  p?: number;
  /** What the app does with the answer: the option, or `keep`. */
  choice?: string;
  ok?: boolean;
  /** The five most likely options. */
  probabilities?: Record<string, number>;
  ms?: number;
  cost?: number;
};

const { provider, model, tag } = modelOfEnv();
const only = process.env.FIELDS?.split(",") as Field[] | undefined;
const dir = process.env.RESULTS ?? `eval/results/${new Date().toISOString().slice(0, 10)}`;
const file = `${dir}/fields-${tag}.json`;

const rows: Row[] = [];
for (const c of CASES) {
  const catalogId = c.cat ?? c.catalog[0]!;
  const questions = questionsOf(c.prompt, catalogId);
  for (const field of FIELDS) {
    if (only && !only.includes(field)) continue;
    const expected = field === "catalog" ? c.catalog : c[field];
    const asked = questions[field];
    if (!expected || !asked) continue;
    const options = Object.keys(asked.question.criteria).length;
    // A catalog with one collection leaves no choice.
    if (options < 2) continue;
    const expect = expected === NO_CHANGE ? NO_CHANGE_OF[field]! : expected;
    const row: Row = { prompt: c.prompt, field, expect, options };
    const reply = await ask(provider, model, asked.state, { q: asked.question });
    const answer = "error" in reply ? undefined : reply.answers.q;
    if ("error" in reply || !answer) {
      row.error = "error" in reply ? reply.error : "no answer";
    } else {
      const p = answer.probabilities ?? {};
      row.top = answer.choice;
      row.p = p[answer.choice] ?? answer.confidence ?? 0;
      row.choice =
        field === "cloud"
          ? (cloudChoice(p, KEEP) ?? KEEP)
          : row.p >= THRESHOLDS[field]
            ? row.top
            : KEEP;
      row.ok = expect.includes(row.choice);
      row.probabilities = Object.fromEntries(
        Object.entries(p).sort((a, b) => b[1] - a[1]).slice(0, 5),
      );
      row.ms = reply.ms;
      row.cost = reply.cost;
    }
    rows.push(row);
    process.stderr.write(row.error ? "x" : row.ok ? "." : "-");
  }
}
console.error();

// With FIELDS, the other fields of an earlier run stay.
let all = rows;
if (only && existsSync(file)) {
  const before: { rows: Row[] } = JSON.parse(readFileSync(file, "utf-8"));
  all = [...before.rows.filter((r) => !only.includes(r.field)), ...rows];
}
mkdirSync(dir, { recursive: true });
writeFileSync(file, JSON.stringify({ provider, model, today: TODAY.toISOString().slice(0, 10), rows: all }, null, 1));
const score = (field: Field) => {
  const r = all.filter((x) => x.field === field);
  return `${field} ${r.filter((x) => x.ok).length}/${r.length}`;
};
console.log(`${model}: ${FIELDS.map(score).join(", ")} → ${file}`);
