/**
 * Phase 0 of the local model plan: a rules-only extractor, with no model.
 * The rules are in `src/zap/local/rules.ts`; this file answers the fields of
 * the OpenRouter extractions API with them, so the extraction test runs it as
 * a model:
 *   MODEL=local:rules VARIANT=compact APP_CALLS=1 pnpm tsx eval/extract.ts
 */
import type { ExtractAnswer, ExtractField, ExtractReply } from "../providers";
import { readCloud, readPlace, readTime } from "../../src/zap/local/rules";
import { requestOf } from "./decide";

export { readCloud, readPlace, readTime };

/** Answers the requested fields with the rules. */
export async function extractLocal(
  _model: string,
  state: string,
  fields: Record<string, ExtractField>,
  referenceDate?: string,
): Promise<ExtractReply> {
  const t = performance.now();
  const text = requestOf(state);
  const today = referenceDate ? new Date(`${referenceDate}T12:00:00Z`) : new Date();
  const time = readTime(text, today);
  const values: Record<string, string | number | null> = {
    ...time,
    cloud: readCloud(text),
    place: readPlace(text, time.when),
  };
  const answers: Record<string, ExtractAnswer> = {};
  for (const name of Object.keys(fields)) {
    const value = values[name] ?? null;
    answers[name] = { value, p_null: value === null ? 1 : 0 };
  }
  return { answers, ms: Math.round(performance.now() - t), cost: 0, usage: {} };
}
