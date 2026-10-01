/**
 * Turns one prompt into map changes with the jev decision model.
 *
 * jev does not write text. It gets a situation and a set of questions, each
 * with a fixed list of options, and returns one option per question with a
 * probability for each option. So each field the prompt can change becomes one
 * question, and each question gets an extra {@link KEEP} option for "do not
 * change this field".
 *
 * Ported from the awii zap mode (`awii-app/api/zap/decide.py`).
 */

export const KEEP = "keep";

/** The minimum probability of an answer to change a field. */
export const THRESHOLD = 0.5;

export type ZapField = {
  /** What the field is, in words jev reads, e.g. "STAC catalog". */
  label: string;
  /**
   * The current option id, or null when nothing is set. A value that is not
   * an option, e.g. a place found by name, is shown to jev as is.
   */
  current: string | null;
  /** Option id to the text jev reads. */
  options: Record<string, string>;
  /**
   * Option id to the text the answers panel shows, when the text before the
   * first ":" of the option does not fit, e.g. "≤ 20%" for a cloud cover.
   */
  short?: Record<string, string>;
  /** False when the field must change, e.g. no catalog is open yet. */
  keepable?: boolean;
  /** The question, when "Which <label> best serves the request?" does not fit. */
  question?: string;
  /**
   * The minimum probability to change the field, instead of {@link THRESHOLD}.
   * With many options, probabilities spread out, so a field such as the
   * catalog changes whenever jev likes an option more than `keep`: set 0.
   */
  threshold?: number;
};

export type ZapFields = Record<string, ZapField>;

export type Answer = {
  choice: string;
  probability: number;
  /** Option id to probability, for every option. */
  probabilities: Record<string, number>;
};

export type ZapResult = {
  /** Field name to the chosen option id, for the fields that change. */
  changes: Record<string, string>;
  answers: Record<string, Answer>;
};

/** What jev returns per question. */
type RawAnswer = {
  choice?: unknown;
  confidence?: number;
  probabilities?: Record<string, number>;
};

// Tells jev what the app is, so it can read indirect requests
// ("cloud-free", "last summer", "elevation") as changes.
const CONTEXT =
  "The user explores earth observation data in a map app with short, " +
  "informal requests. The app opens a STAC catalog, then one collection of " +
  "that catalog, and searches its items for a place and a time. Act on what " +
  "the user means, not only on the words: a question about a topic asks for " +
  "the collection that shows it, and a request can change more than one " +
  "field. Optical imagery such as Sentinel-2, Landsat or NAIP has cloud " +
  "cover; radar, elevation and land cover data do not. Past seasons mean " +
  "the most recent one before today.";

function currentLabel(field: ZapField): string {
  if (!field.current) return "not set";
  return field.options[field.current] ?? field.current;
}

/**
 * Builds the body of one jev decisions call, with one question per field.
 * @param prompt What the user typed.
 * @param fields The fields the prompt can change.
 * @param model The jev model id, e.g. `typesafe/jev-1.13`.
 * @param today The date jev reads relative requests against.
 */
export function buildPayload(
  prompt: string,
  fields: ZapFields,
  model: string,
  today: Date = new Date(),
) {
  const current = Object.values(fields)
    .map((field) => `- ${field.label}: ${currentLabel(field)}`)
    .join("\n");
  const state =
    `${CONTEXT}\n\nToday is ${today.toISOString().slice(0, 10)}.\n\n` +
    `User request: ${prompt}\n\nCurrent map:\n${current}`;
  const questions = Object.fromEntries(
    Object.entries(fields).map(([name, field]) => {
      const keepable = field.keepable ?? true;
      return [
        name,
        {
          type: "choice",
          instructions:
            (field.question ??
              `Which ${field.label} best serves the request? The request ` +
                "can be indirect: pick the option that it implies.") +
            (keepable
              ? ` Pick '${KEEP}' only if the request has nothing to do with ` +
                `the ${field.label}.`
              : ""),
          criteria: keepable
            ? {
                ...field.options,
                [KEEP]: `Keep the current ${field.label} (${currentLabel(field)})`,
              }
            : field.options,
        },
      ];
    }),
  );
  return { model, state, questions };
}

/**
 * Selects the changes from the jev answers. A field changes only when jev
 * chose one of its options (not {@link KEEP}), the option differs from the
 * current value, and its probability is at least `threshold`.
 * @param fields The fields the answers are for.
 * @param answers The `answers` object of the jev response.
 */
export function readAnswers(
  fields: ZapFields,
  answers: Record<string, RawAnswer | undefined>,
  threshold: number = THRESHOLD,
): ZapResult {
  const changes: Record<string, string> = {};
  const read: Record<string, Answer> = {};
  for (const [name, field] of Object.entries(fields)) {
    const answer = answers[name];
    if (typeof answer?.choice !== "string") continue;
    const choice = answer.choice;
    const probability =
      answer.probabilities?.[choice] ?? answer.confidence ?? 0;
    read[name] = {
      choice,
      probability,
      probabilities: answer.probabilities ?? { [choice]: probability },
    };
    if (
      choice in field.options &&
      choice !== field.current &&
      probability >= (field.threshold ?? threshold)
    ) {
      changes[name] = choice;
    }
  }
  return { changes, answers: read };
}

/**
 * Asks jev one set of questions through the `/api/decide` endpoint.
 * @throws If the endpoint cannot answer.
 */
export async function decide(
  prompt: string,
  fields: ZapFields,
): Promise<ZapResult> {
  const model = import.meta.env.VITE_JEV_MODEL || "typesafe/jev-1.13";
  // The deployed site calls the Worker in `worker/`; the dev server has its
  // own `/api/decide`.
  const url =
    import.meta.env.VITE_DECIDE_URL || `${import.meta.env.BASE_URL}api/decide`;
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(buildPayload(prompt, fields, model)),
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(`jev call failed (${response.status}): ${text}`);
  }
  const body = (await response.json()) as {
    answers?: Record<string, RawAnswer>;
  };
  return readAnswers(fields, body.answers ?? {});
}
