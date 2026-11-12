/**
 * The extraction test: asks an extraction model on OpenRouter for the place,
 * the time range and the maximum cloud cover of each prompt, and scores the
 * values. An extraction model returns values, not options, so it is not
 * limited to the option lists of the app.
 *
 * Each group of fields (place, time, cloud) is asked in its own call, so a
 * model that does not accept one field type still answers the others.
 *
 * Usage:
 *   MODEL=cloudflare/clairic pnpm tsx eval/extract.ts
 * Optional: SET=unparsed uses the prompts that the app cannot parse today
 * (`UNPARSED_CASES`), instead of the prompts of the field test (`CASES`).
 * SET=dev uses the prompts for tuning (`DEV_CASES`).
 * VARIANT selects the state and the instructions (see `extract-fields.ts`).
 * ONE_CALL=1 asks all groups of a prompt in one call, as the app would; the
 * cost of such a call is then in each of its rows.
 * RESULTS=eval/results/<date> sets the output folder.
 */
import { mkdirSync, writeFileSync } from "node:fs";
import { CASES, DEV_CASES, UNPARSED_CASES } from "./cases";
import { calendarInCode, groupsOf, stateOf, VARIANTS, type Group, type Variant } from "./extract-fields";
import { extract, modelOfEnv, type ExtractAnswer, type ExtractField, type ExtractUsage } from "./providers";
import { TODAY } from "./questions";
import {
  cloudInRange,
  cloudsOf,
  cloudOk,
  placeOf,
  placeOk,
  rangeOk,
  rangesOf,
  type Range,
} from "./score";
import { rangeOfAnswers as appRangeOfAnswers } from "../src/zap/extract";

const REFERENCE_DATE = TODAY.toISOString().slice(0, 10);
const SETS = ["field", "unparsed", "dev"] as const;

const { model, tag } = modelOfEnv();
const set = (process.env.SET ?? "field") as (typeof SETS)[number];
if (!SETS.includes(set)) throw new Error(`Unknown SET: ${set}`);
const variant = (process.env.VARIANT ?? "hinted") as Variant;
if (!VARIANTS.includes(variant)) throw new Error(`Unknown VARIANT: ${variant}`);
/** Ask all groups of a prompt in one call, as the app would. */
const oneCall = process.env.ONE_CALL === "1";
const GROUPS = groupsOf(variant);
const codeCalendar = calendarInCode(variant);

export type ExtractRow = {
  id?: string;
  prompt: string;
  group: Group;
  /** The accepted values, as text for the result file. */
  expect: string;
  error?: string;
  /** The value the test scores: a place, a range or a number, or null. */
  value?: string | Range | number | null;
  ok?: boolean;
  answers?: Record<string, ExtractAnswer>;
  ms?: number;
  cost?: number;
  usage?: ExtractUsage;
};

/** The value of an answer, or null when the model says it is not there. */
const valueOf = (a: ExtractAnswer | undefined) => (a?.value === undefined ? null : a.value);

/**
 * The range from the answers. The compact variants use the app's own
 * `rangeOfAnswers` (`src/zap/extract.ts`), which makes a season, a quarter,
 * "the last N units", "recent" or "latest" into a range with code. The other
 * variants use the dates of the model: with no end, the range runs to today,
 * an end with no start is not a range, and a null `when` means no time.
 */
export function rangeOfAnswers(answers: Record<string, ExtractAnswer>): Range | null | undefined {
  if (codeCalendar) return appRangeOfAnswers(answers, TODAY);
  if ("when" in answers && valueOf(answers.when) === null) return null;
  const start = valueOf(answers.start);
  const end = valueOf(answers.end);
  if (start === null && end === null) return null;
  if (start === null) return undefined;
  return [String(start), end === null ? REFERENCE_DATE : String(end)];
}

type Task = {
  id?: string;
  prompt: string;
  group: Group;
  expect: string;
  score: (answers: Record<string, ExtractAnswer>) => { value: ExtractRow["value"]; ok: boolean };
};

/** What to ask and how to score it, for each prompt and group. */
function tasks(set: string): Task[] {
  const out: Task[] = [];
  if (set === "unparsed" || set === "dev") {
    for (const c of set === "dev" ? DEV_CASES : UNPARSED_CASES) {
      const base = { id: c.id, prompt: c.prompt };
      if (c.range !== undefined) {
        const expect = c.range;
        out.push({
          ...base,
          group: "time",
          expect: expect ? expect.join(" to ") : "none",
          score: (a) => {
            const value = rangeOfAnswers(a);
            return { value: value ?? null, ok: value !== undefined && rangeOk(value, expect, c.relative) };
          },
        });
      }
      if (c.cloud !== undefined) {
        const expect = c.cloud;
        out.push({
          ...base,
          group: "cloud",
          expect: expect === null ? "none" : `${expect}`,
          score: (a) => {
            const value = valueOf(a.cloud) as number | null;
            return { value, ok: cloudOk(value, expect) };
          },
        });
      }
      if (c.place) {
        const expect = c.place;
        out.push({
          ...base,
          group: "place",
          expect,
          score: (a) => {
            const value = valueOf(a.place) as string | null;
            return { value, ok: placeOk(value, [expect]) };
          },
        });
      }
    }
    return out;
  }
  for (const c of CASES) {
    if (c.place) {
      const expect = c.place;
      out.push({
        prompt: c.prompt,
        group: "place",
        expect: expect.join(" | "),
        score: (a) => {
          const value = valueOf(a.place) as string | null;
          return { value: value === null ? placeOf(null) : value, ok: placeOk(value, expect) };
        },
      });
    }
    const time = rangesOf(c);
    if (time) {
      out.push({
        prompt: c.prompt,
        group: "time",
        expect: time.ranges.map((r) => (r ? r.join(" to ") : "none")).join(" | "),
        score: (a) => {
          const value = rangeOfAnswers(a);
          const ok = value !== undefined && time.ranges.some((r) => rangeOk(value, r, time.relative));
          return { value: value ?? null, ok };
        },
      });
    }
    const cloud = cloudsOf(c);
    if (cloud !== undefined) {
      out.push({
        prompt: c.prompt,
        group: "cloud",
        expect: cloud ? `${cloud[0]} to ${cloud[1]}` : "none",
        score: (a) => {
          const value = valueOf(a.cloud) as number | null;
          return { value, ok: cloudInRange(value, cloud) };
        },
      });
    }
  }
  return out;
}

const dir = process.env.RESULTS ?? `eval/results/${new Date().toISOString().slice(0, 10)}`;
const file = `${dir}/extract-${set}-${tag}-${variant}${oneCall ? "-onecall" : ""}.json`;

/** Asks the groups of one prompt in one call. Groups with dates send the reference date. */
async function ask(prompt: string, groups: Group[]) {
  const fields: Record<string, ExtractField> = Object.assign({}, ...groups.map((g) => GROUPS[g]));
  const dated = Object.values(fields).some((f) => f.type === "date");
  return extract(model, stateOf(variant, prompt, TODAY), fields, dated ? REFERENCE_DATE : undefined);
}

const all = tasks(set);
// With ONE_CALL, the tasks of a prompt share one reply.
const shared = new Map<string, ReturnType<typeof ask>>();
const rows: ExtractRow[] = [];
for (const task of all) {
  let reply: Awaited<ReturnType<typeof ask>>;
  if (oneCall) {
    if (!shared.has(task.prompt)) {
      const groups = [...new Set(all.filter((t) => t.prompt === task.prompt).map((t) => t.group))];
      shared.set(task.prompt, ask(task.prompt, groups));
    }
    reply = await shared.get(task.prompt)!;
  } else {
    reply = await ask(task.prompt, [task.group]);
  }
  const row: ExtractRow = { id: task.id, prompt: task.prompt, group: task.group, expect: task.expect };
  if ("error" in reply) {
    row.error = reply.error;
  } else {
    Object.assign(row, task.score(reply.answers));
    row.answers = reply.answers;
    row.ms = reply.ms;
    row.cost = reply.cost;
    row.usage = reply.usage;
  }
  rows.push(row);
  process.stderr.write(row.error ? "x" : row.ok ? "." : "-");
}
console.error();

mkdirSync(dir, { recursive: true });
writeFileSync(file, JSON.stringify({ model, set, variant, oneCall, today: REFERENCE_DATE, rows }, null, 1));
const score = (group: Group) => {
  const r = rows.filter((x) => x.group === group);
  const errors = r.filter((x) => x.error).length;
  return `${group} ${r.filter((x) => x.ok).length}/${r.length}${errors ? ` (${errors} rejected)` : ""}`;
};
console.log(`${model} (${set}, ${variant}): ${(["place", "time", "cloud"] as const).map(score).join(", ")} → ${file}`);
