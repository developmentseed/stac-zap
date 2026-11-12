/**
 * The unparsed test for a decision model: asks the questions of stac-zap for
 * each prompt in `UNPARSED_CASES`, applies the answers as the app does, and
 * scores the values with the rules of the extraction test. The correct
 * answers are not in the option lists, so a decision model is expected to
 * fail. The test shows how far the nearest option is from the request.
 *
 * Usage:
 *   PROVIDER=openrouter MODEL=typesafe/jev-1.13 pnpm tsx eval/unparsed.ts
 * Optional: RESULTS=eval/results/<date> sets the output folder.
 */
import { mkdirSync, writeFileSync } from "node:fs";
import { KEEP, THRESHOLD } from "../src/zap/decide";
import { cloudChoice, latestYearOf, toRange } from "../src/zap/options";
import { UNPARSED_CASES } from "./cases";
import { ask, modelOfEnv, type RawAnswer } from "./providers";
import { questionsOf, TODAY, type Field } from "./questions";
import { cloudOk, placeOk, rangeOk, type Range } from "./score";

const PC = "planetary-computer";

export type UnparsedRow = {
  id: string;
  prompt: string;
  group: "place" | "time" | "cloud";
  expect: string;
  error?: string;
  /** The value the app makes from the answers. */
  value?: string | Range | number | null;
  ok?: boolean;
  /** The option ids the app used, e.g. { period: "2023", season: "spring" }. */
  choices?: Record<string, string>;
  ms?: number;
  cost?: number;
};

const { provider, model, tag } = modelOfEnv();
const dir = process.env.RESULTS ?? `eval/results/${new Date().toISOString().slice(0, 10)}`;
const file = `${dir}/unparsed-${tag}.json`;

const prob = (a: RawAnswer) => a.probabilities?.[a.choice] ?? a.confidence ?? 0;

const rows: UnparsedRow[] = [];
for (const c of UNPARSED_CASES) {
  const questions = questionsOf(c.prompt, PC);
  const asked = async (...fields: Field[]) => {
    const answers: Partial<Record<Field, RawAnswer>> = {};
    let ms = 0;
    let cost = 0;
    for (const field of fields) {
      const q = questions[field]!;
      const reply = await ask(provider, model, q.state, { q: q.question });
      if ("error" in reply) return { error: reply.error };
      const answer = reply.answers.q;
      if (!answer) return { error: "no answer" };
      answers[field] = answer;
      ms += reply.ms;
      cost += reply.cost;
    }
    return { answers, ms, cost };
  };
  const base = { id: c.id, prompt: c.prompt };
  let row: UnparsedRow;
  if (c.range) {
    row = { ...base, group: "time", expect: c.range.join(" to ") };
    const reply = await asked("period", "season");
    if ("error" in reply) {
      row.error = reply.error;
    } else {
      // As run.ts: the period changes at 0.35, the season at THRESHOLD.
      const period = reply.answers.period!;
      const season = reply.answers.season!;
      let p = prob(period) >= 0.35 && period.choice !== KEEP ? period.choice : null;
      const s = prob(season) >= THRESHOLD && season.choice !== KEEP ? season.choice : "whole-year";
      if (s !== "whole-year" && (!p || !/^\d{4}$/.test(p))) p = latestYearOf(s, TODAY);
      const r = p ? toRange(p, s, TODAY) : null;
      const value: Range | null = r ? [r.startDatetime.slice(0, 10), r.endDatetime.slice(0, 10)] : null;
      Object.assign(row, {
        value,
        ok: rangeOk(value, c.range, c.relative),
        choices: { period: period.choice, season: season.choice },
        ms: reply.ms,
        cost: reply.cost,
      });
    }
  } else if (c.cloud !== undefined) {
    row = { ...base, group: "cloud", expect: `${c.cloud}` };
    const reply = await asked("cloud");
    if ("error" in reply) {
      row.error = reply.error;
    } else {
      const answer = reply.answers.cloud!;
      const id = cloudChoice(answer.probabilities ?? {}, KEEP);
      const value = !id || id === "any" ? null : Number(id);
      Object.assign(row, {
        value,
        ok: cloudOk(value, c.cloud),
        choices: { cloud: id ?? KEEP },
        ms: reply.ms,
        cost: reply.cost,
      });
    }
  } else {
    row = { ...base, group: "place", expect: c.place! };
    const reply = await asked("place");
    if ("error" in reply) {
      row.error = reply.error;
    } else {
      const answer = reply.answers.place!;
      const value = prob(answer) >= THRESHOLD && answer.choice !== KEEP ? answer.choice : null;
      Object.assign(row, {
        value,
        ok: placeOk(value, [c.place!]),
        choices: { place: answer.choice },
        ms: reply.ms,
        cost: reply.cost,
      });
    }
  }
  rows.push(row);
  process.stderr.write(row.error ? "x" : row.ok ? "." : "-");
}
console.error();

mkdirSync(dir, { recursive: true });
writeFileSync(file, JSON.stringify({ provider, model, today: TODAY.toISOString().slice(0, 10), rows }, null, 1));
const score = (group: UnparsedRow["group"]) => {
  const r = rows.filter((x) => x.group === group);
  return `${group} ${r.filter((x) => x.ok).length}/${r.length}`;
};
console.log(`${model} (unparsed): ${(["place", "time", "cloud"] as const).map(score).join(", ")} → ${file}`);
