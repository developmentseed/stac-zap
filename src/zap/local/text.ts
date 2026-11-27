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
