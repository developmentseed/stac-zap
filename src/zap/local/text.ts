/**
 * The texts the local models read. They must be the same as in training
 * (`training/train_hf_jobs.py`): a model that was trained on one text and
 * reads another makes worse choices.
 */

/** Whitespace collapsed, then cut to `n` characters with an ellipsis. */
export function short(text: string | undefined | null, n: number): string {
  const t = (text ?? "").split(/\s+/).filter(Boolean).join(" ");
  return t.length <= n ? t : `${t.slice(0, n - 1)}…`;
}

export type CollectionLike = { id: string; title?: string; description?: string };

/** A collection as the retriever and the decider read it. */
export function docText(catalogId: string, c: CollectionLike): string {
  return `${catalogId.replaceAll("-", " ")} | ${short(c.title ?? c.id, 100)} (${c.id}): ${short(c.description, 160)}`;
}

/** The query side of a decider question. */
export function query(question: string, request: string, current: string | null = null): string {
  return `${question} | request: ${request} | current: ${current ?? "not set"}`;
}

/** The decider option for "none of these places". */
export const NONE_PLACE = "None of these places";

/**
 * A geocoder result as the decider reads it, close to the training text
 * ("Lisbon, Lisbon, Portugal (city, 545,000 people)"): the label without the
 * size of the box, e.g. "Lisbon, Portugal (city)".
 */
export function placeText(label: string): string {
  return label.replace(/\(([^,()]+),[^()]*\)\s*$/, "($1)");
}
