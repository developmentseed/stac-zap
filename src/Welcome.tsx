import { useStore } from "@developmentseed/stac-map";
import { Box, CloseButton, HStack, IconButton, Link, Stack } from "@chakra-ui/react";
import { useEffect, useState } from "react";
import { LuInfo, LuZap } from "react-icons/lu";
import { ZAP_SUBMIT_EVENT } from "./events";
import { useMobile } from "./mobile";

/** Text in stac-map's welcome panel. stac-map has no slot for this panel. */
const STAC_MAP_INTRO = "is a map-first visualization tool for";
/** stac-map's URL input, in its header. */
const HREF_INPUT = 'input[placeholder^="Enter a url to a STAC API"]';

/** `bottom` is the distance from the bottom of the window. */
type Place = { top: number; bottom: number; left: number; width: number };

/** The width of the panel that the info button opens, in pixels. */
const POPUP_WIDTH = 320;

/**
 * The stac-zap welcome panel. Until the first prompt, it covers stac-map's
 * own welcome panel with the same style. On a phone, stac-map's panel is a
 * sheet at the bottom (see `src/mobile.css`), so this panel grows up from the
 * bottom of that sheet, and scrolls when it is too long.
 *
 * After the first prompt, the panel folds into an info button in the lower
 * left corner. A click on the button opens the same text.
 */
export default function Welcome() {
  const href = useStore((store) => store.href);
  const mobile = useMobile();
  const [folded, setFolded] = useState(false);
  const [open, setOpen] = useState(false);
  useEffect(() => {
    const fold = () => {
      setFolded(true);
      setOpen(false);
    };
    window.addEventListener(ZAP_SUBMIT_EVENT, fold);
    return () => window.removeEventListener(ZAP_SUBMIT_EVENT, fold);
  }, []);
  const showPanel = !folded && !href;
  const place = useIntroPlace(showPanel);

  if (showPanel) {
    if (!place) return null;
    return (
      <Box
        position="fixed"
        top={mobile ? undefined : `${place.top}px`}
        bottom={mobile ? `${place.bottom}px` : undefined}
        left={`${place.left}px`}
        w={`${place.width}px`}
        zIndex={10}
        bg="bg.muted"
        rounded={4}
      >
        <Header />
        <WelcomeText maxH={mobile ? "var(--zap-sheet-height)" : undefined} />
      </Box>
    );
  }

  // On a phone, stac-map's sheet covers the bottom, so the button goes to
  // the top left, below the search bar.
  return (
    <Box
      position="fixed"
      left={4}
      top={mobile ? "112px" : undefined}
      bottom={mobile ? undefined : 8}
      zIndex={10}
    >
      {open && (
        <Box
          position="absolute"
          left={0}
          top={mobile ? "calc(100% + 8px)" : undefined}
          bottom={mobile ? undefined : "calc(100% + 8px)"}
          w={`min(${POPUP_WIDTH}px, calc(100vw - 32px))`}
          bg="bg.muted"
          rounded={4}
          shadow="md"
        >
          <Header onClose={() => setOpen(false)} />
          <WelcomeText maxH="60dvh" />
        </Box>
      )}
      <IconButton
        aria-label="About stac-zap"
        aria-expanded={open}
        size="sm"
        variant="surface"
        rounded="full"
        onClick={() => setOpen((o) => !o)}
      >
        <LuInfo />
      </IconButton>
    </Box>
  );
}

function Header({ onClose }: { onClose?: () => void }) {
  return (
    <HStack
      borderBottomWidth={1}
      borderColor="border.subtle"
      py={1.5}
      px={3}
      fontWeight="lighter"
      fontSize="sm"
    >
      <Box color="orange.500">
        <LuZap aria-hidden />
      </Box>
      stac-zap
      {onClose && <CloseButton size="2xs" ms="auto" aria-label="Close" onClick={onClose} />}
    </HStack>
  );
}

function WelcomeText({ maxH }: { maxH?: string }) {
  return (
    <Stack p={3} fontSize="xs" fontWeight="lighter" gap={2} maxH={maxH} overflow="auto">
      <Box>
        Ask for earth observation data in your own words. A decision model
        turns the request into a{" "}
        <Link variant="underline" href="https://stacspec.org" target="_blank">
          STAC
        </Link>{" "}
        search: it selects from the catalogs, collections, places and dates
        that exist, and does not write text. Try an example below the search
        bar.
      </Box>
      <Box>
        By default, the{" "}
        <Link variant="underline" href="https://typesafe.ai/blog/introducing-system-one-models-and-jev" target="_blank">
          jev
        </Link>{" "}
        model answers. It has a small budget: with many users, requests can be
        slow or stop. The model menu below the search bar can also run small
        open models in your browser.
      </Box>
      <Box>
        Code and issues on{" "}
        <Link variant="underline" href="https://github.com/developmentseed/stac-zap" target="_blank">
          GitHub
        </Link>
        . Built on{" "}
        <Link variant="underline" href="https://github.com/developmentseed/stac-map" target="_blank">
          stac-map
        </Link>
        .
      </Box>
    </Stack>
  );
}

/**
 * The position of stac-map's welcome panel, which is hidden while this one
 * shows. The panel can move, so this checks it four times a second.
 */
function useIntroPlace(active: boolean): Place | null {
  const [place, setPlace] = useState<Place | null>(null);
  useEffect(() => {
    if (!active) return;
    let hidden: HTMLElement | null = null;
    function update() {
      const panel = findIntroPanel();
      if (!panel) return;
      panel.style.visibility = "hidden";
      hidden = panel;
      const r = panel.getBoundingClientRect();
      const bottom = window.innerHeight - r.bottom;
      setPlace((last) =>
        last &&
        last.top === r.top &&
        last.bottom === bottom &&
        last.left === r.left &&
        last.width === r.width
          ? last
          : { top: r.top, bottom, left: r.left, width: r.width },
      );
    }
    update();
    const timer = setInterval(update, 250);
    window.addEventListener("resize", update);
    return () => {
      clearInterval(timer);
      window.removeEventListener("resize", update);
      if (hidden) hidden.style.visibility = "";
    };
  }, [active]);
  return place;
}

/**
 * The outermost box of stac-map's welcome panel: the last box before the one
 * that also holds stac-map's header (the URL input).
 */
function findIntroPanel(): HTMLElement | null {
  const text = [...document.querySelectorAll("strong")].find(
    (el) => el.textContent === "stac-map" && el.parentElement?.textContent?.includes(STAC_MAP_INTRO),
  );
  let panel: HTMLElement | null = null;
  for (let el = text?.parentElement ?? null; el && el !== document.body; el = el.parentElement) {
    if (el.querySelector(HREF_INPUT)) break;
    panel = el;
  }
  return panel;
}
