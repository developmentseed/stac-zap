import { useStore } from "@developmentseed/stac-map";
import {
  Box,
  CloseButton,
  HStack,
  IconButton,
  Input,
  InputGroup,
  Spinner,
  Stack,
  Switch,
  Text,
} from "@chakra-ui/react";
import { useEffect, useRef, useState } from "react";
import { LuLink, LuZap } from "react-icons/lu";
import { listCatalogs } from "./zap/catalogs";
import { KEEP, THRESHOLD } from "./zap/decide";
import { zap, type ZapStep } from "./zap/run";

const EXAMPLES = [
  "cloud-free Sentinel-2 over Lisbon last summer",
  "Swiss orthophotos of Zurich",
  "elevation of Greenland",
  "crop types around Hanover",
  "Maxar imagery after Hurricane Idalia",
  "land cover in the Amazon",
];

/** stac-map's URL input. stac-map has no slot in its header, so zap covers it. */
const HREF_INPUT = 'input[placeholder^="Enter a url to a STAC API"]';
/** The answers panel is at least this wide, in pixels. */
const MIN_PANEL_WIDTH = 460;
const MODE_KEY = "stac-zap:mode";
const ALL_CATALOGS_KEY = "stac-zap:all-catalogs";

type Mode = "zap" | "url";

/**
 * Zap as a mode of stac-map's search bar. In zap mode, the zap prompt covers
 * the URL input; the button at its start changes the mode.
 *
 * Enter sends the prompt to jev, and the answers change the map (see
 * `src/zap/run.ts`). The panel below the bar shows every answer with its
 * probability. Focus selects the prompt, so typing replaces it. Typing hides
 * the answers of the last prompt. Escape or the close button clears both.
 */
export default function ZapBar() {
  const href = useStore((store) => store.href);
  const [mode, setMode] = useState<Mode>(readMode);
  const [allCatalogs, setAllCatalogs] = useState(
    () => readStored(ALL_CATALOGS_KEY) === "true",
  );
  const [catalogCount, setCatalogCount] = useState<number | null>(null);
  const rect = useHrefInputRect(mode);
  const [prompt, setPrompt] = useState("");
  const [busy, setBusy] = useState(false);
  const [focused, setFocused] = useState(false);
  const [steps, setSteps] = useState<ZapStep[]>([]);
  const [notes, setNotes] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const input = useRef<HTMLInputElement>(null);
  // A click that focuses the input would move the caret and lose the
  // selection, so the mouseup after focus is ignored.
  const justFocused = useRef(false);
  const hasAnswers = steps.length > 0 || notes.length > 0 || error !== null;

  // Checking which catalogs answer takes a few seconds, so it starts now.
  useEffect(() => {
    let current = true;
    setCatalogCount(null);
    void listCatalogs(allCatalogs).then((list) => {
      if (current) setCatalogCount(list.length);
    });
    return () => {
      current = false;
    };
  }, [allCatalogs]);

  function changeMode(next: Mode) {
    setMode(next);
    writeStored(MODE_KEY, next);
    if (next === "zap") setTimeout(() => input.current?.focus());
  }

  function clearAnswers() {
    setSteps([]);
    setNotes([]);
    setError(null);
  }

  function clear() {
    if (busy) return;
    setPrompt("");
    clearAnswers();
  }

  async function submit(text: string) {
    text = text.trim();
    if (!text || busy) return;
    setBusy(true);
    clearAnswers();
    try {
      await zap(text, {
        allCatalogs,
        onStep: (step) => setSteps((steps) => [...steps, step]),
        onNote: (note) => setNotes((notes) => [...notes, note]),
      });
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  if (!rect) return null;

  const toggle = (
    <IconButton
      variant="plain"
      size="xs"
      aria-label={mode === "zap" ? "Enter a URL instead" : "Ask with zap"}
      title={mode === "zap" ? "Enter a URL instead" : "Ask with zap"}
      onClick={() => changeMode(mode === "zap" ? "url" : "zap")}
    >
      {mode === "zap" ? <LuZap /> : <LuLink />}
    </IconButton>
  );

  if (mode === "url") {
    // Only the button, on the start of stac-map's input.
    return (
      <Box
        position="fixed"
        top={`${rect.top}px`}
        left={`${rect.left + 4}px`}
        h={`${rect.height}px`}
        display="flex"
        alignItems="center"
        zIndex={10}
      >
        {toggle}
      </Box>
    );
  }

  const panelWidth = Math.min(
    Math.max(rect.width, MIN_PANEL_WIDTH),
    window.innerWidth - rect.left - 16,
  );
  const showExamples = !prompt && !hasAnswers && !busy && (focused || !href);

  return (
    <Box
      position="fixed"
      top={`${rect.top}px`}
      left={`${rect.left}px`}
      w={`${rect.width}px`}
      zIndex={10}
    >
      <form
        role="search"
        aria-label="Zap"
        onSubmit={(event) => {
          event.preventDefault();
          void submit(prompt);
        }}
      >
        <InputGroup
          startElement={toggle}
          startElementProps={{ pointerEvents: "auto", ps: 1 }}
          endElement={
            busy ? (
              <Spinner size="xs" />
            ) : prompt || hasAnswers ? (
              <CloseButton size="2xs" aria-label="Clear" onClick={clear} />
            ) : undefined
          }
        >
          <Input
            ref={input}
            h={`${rect.height}px`}
            placeholder="Ask for data, or switch to a URL"
            aria-label="Zap prompt"
            value={prompt}
            // Read-only, not disabled, so the input keeps the focus.
            readOnly={busy}
            aria-busy={busy}
            onChange={(event) => {
              setPrompt(event.target.value);
              if (hasAnswers) clearAnswers();
            }}
            onKeyDown={(event) => {
              if (event.key === "Escape") clear();
            }}
            onFocus={(event) => {
              event.target.select();
              justFocused.current = true;
              setFocused(true);
            }}
            onBlur={() => setFocused(false)}
            onMouseUp={(event) => {
              if (justFocused.current) event.preventDefault();
              justFocused.current = false;
            }}
            bg="bg.muted/90"
          />
        </InputGroup>
      </form>
      {hasAnswers && (
        <Stack
          mt={2}
          w={`${panelWidth}px`}
          bg="bg.panel"
          rounded="md"
          shadow="md"
          px={3}
          py={2}
          gap={1}
          fontSize="xs"
        >
          {steps.flatMap(({ fields, result }) =>
            Object.entries(result.answers).map(([name, answer]) => (
              <AnswerRow
                key={`${name}-${answer.choice}`}
                label={fields[name]!.label}
                option={
                  answer.choice === KEEP
                    ? "keep"
                    : shorten(fields[name]!.options[answer.choice] ?? answer.choice)
                }
                probability={answer.probability}
                applied={name in result.changes}
              />
            )),
          )}
          {notes.map((note) => (
            <Text key={note} color="fg.warning">
              {note}
            </Text>
          ))}
          {error && <Text color="fg.error">{error}</Text>}
        </Stack>
      )}
      {showExamples && (
        <HStack
          mt={2}
          w={`${panelWidth}px`}
          bg="bg.panel"
          rounded="md"
          shadow="sm"
          px={3}
          py={1.5}
          fontSize="xs"
          justify="space-between"
          // Keep the focus in the input, so the list stays open.
          onMouseDown={(event) => event.preventDefault()}
        >
          <Text color="fg.muted">
            {catalogCount === null
              ? "Checking catalogs…"
              : `Asks ${catalogCount} ${allCatalogs ? "" : "curated "}catalogs`}
          </Text>
          <Switch.Root
            size="sm"
            checked={allCatalogs}
            onCheckedChange={({ checked }) => {
              setAllCatalogs(checked);
              writeStored(ALL_CATALOGS_KEY, String(checked));
            }}
          >
            <Switch.HiddenInput />
            <Switch.Control>
              <Switch.Thumb />
            </Switch.Control>
            <Switch.Label>All STAC Index catalogs</Switch.Label>
          </Switch.Root>
        </HStack>
      )}
      {showExamples && (
        <HStack mt={2} w={`${panelWidth}px`} gap={1} flexWrap="wrap">
          {EXAMPLES.map((example) => (
            <Box
              as="button"
              key={example}
              // Keep the focus in the input, so the list stays open.
              onMouseDown={(event) => event.preventDefault()}
              onClick={() => {
                setPrompt(example);
                void submit(example);
              }}
              bg="bg.panel"
              rounded="full"
              px={3}
              py={1}
              fontSize="xs"
              shadow="sm"
              cursor="pointer"
              _hover={{ bg: "bg.muted" }}
            >
              {example}
            </Box>
          ))}
        </HStack>
      )}
    </Box>
  );
}

function readMode(): Mode {
  return readStored(MODE_KEY) === "url" ? "url" : "zap";
}

/** Settings are only remembered when browser storage works. */
function readStored(key: string): string | null {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

function writeStored(key: string, value: string): void {
  try {
    localStorage.setItem(key, value);
  } catch {
    // Not remembered.
  }
}

type Rect = { top: number; left: number; width: number; height: number };

/**
 * The position of stac-map's URL input, which the zap bar covers. The header
 * can move (e.g. when the window changes size), so this checks it four times
 * a second. It also hides the URL input in zap mode, and makes space for the
 * mode button in URL mode.
 */
function useHrefInputRect(mode: Mode): Rect | null {
  const [rect, setRect] = useState<Rect | null>(null);
  useEffect(() => {
    function update() {
      const hrefInput = document.querySelector<HTMLInputElement>(HREF_INPUT);
      // stac-map can drop the input for a moment while it renders; the bar
      // stays where it was, so it keeps the focus.
      if (!hrefInput) return;
      hrefInput.style.visibility = mode === "zap" ? "hidden" : "";
      hrefInput.style.paddingInlineStart = mode === "url" ? "2.25rem" : "";
      const r = hrefInput.getBoundingClientRect();
      setRect((last) =>
        last &&
        last.top === r.top &&
        last.left === r.left &&
        last.width === r.width &&
        last.height === r.height
          ? last
          : { top: r.top, left: r.left, width: r.width, height: r.height },
      );
    }
    update();
    const timer = setInterval(update, 250);
    window.addEventListener("resize", update);
    return () => {
      clearInterval(timer);
      window.removeEventListener("resize", update);
    };
  }, [mode]);
  return rect;
}

function AnswerRow({
  label,
  option,
  probability,
  applied,
}: {
  label: string;
  option: string;
  probability: number;
  applied: boolean;
}) {
  const unsure = option !== "keep" && probability < THRESHOLD;
  return (
    <HStack gap={2} opacity={applied ? 1 : 0.55}>
      <Text w="150px" flexShrink={0} color="fg.muted" truncate>
        {label}
      </Text>
      <Text flex={1} truncate title={option} fontWeight={applied ? "medium" : "normal"}>
        {option}
        {unsure && " (unsure)"}
      </Text>
      <Box w="80px" h="6px" bg="bg.muted" rounded="full" flexShrink={0}>
        <Box
          h="100%"
          w={`${Math.round(probability * 100)}%`}
          bg={applied ? "orange.500" : "fg.subtle"}
          rounded="full"
        />
      </Box>
      <Text w="34px" textAlign="right" flexShrink={0}>
        {Math.round(probability * 100)}%
      </Text>
    </HStack>
  );
}

/** The part of an option before its description, e.g. a collection title. */
function shorten(option: string): string {
  return option.split(":")[0]!;
}
