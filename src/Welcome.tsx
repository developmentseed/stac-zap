import { useStore } from "@developmentseed/stac-map";
import { Box, HStack, Link, Stack } from "@chakra-ui/react";
import { useEffect, useState } from "react";
import { LuZap } from "react-icons/lu";
import { useMobile } from "./mobile";

/** Text in stac-map's welcome panel. stac-map has no slot for this panel. */
const STAC_MAP_INTRO = "is a map-first visualization tool for";
/** stac-map's URL input, in its header. */
const HREF_INPUT = 'input[placeholder^="Enter a url to a STAC API"]';

/** `bottom` is the distance from the bottom of the window. */
type Place = { top: number; bottom: number; left: number; width: number };

/**
 * The stac-zap welcome panel. While no value is open, stac-map shows its own
 * welcome panel; this covers it with the same style. On a phone, stac-map's
 * panel is a sheet at the bottom (see `src/mobile.css`), so this panel grows
 * up from the bottom of that sheet, and scrolls when it is too long.
 */
export default function Welcome() {
  const href = useStore((store) => store.href);
  const mobile = useMobile();
  const place = useIntroPlace(!href);
  if (href || !place) return null;
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
      <HStack
        borderBottomWidth={1}
        borderColor="border.subtle"
        py={2}
        px={4}
        fontWeight="lighter"
        fontSize="sm"
      >
        <Box color="orange.500">
          <LuZap aria-hidden />
        </Box>
        stac-zap
      </HStack>
      <Stack
        p={4}
        fontSize="sm"
        fontWeight="lighter"
        gap={3}
        maxH={mobile ? "var(--zap-sheet-height)" : undefined}
        overflow="auto"
      >
        <Box>
          Ask for earth observation data in your own words. The{" "}
          <Link variant="underline" href="https://typesafe.ai/blog/introducing-system-one-models-and-jev" target="_blank">
            jev
          </Link>{" "}
          decision model turns the request into a{" "}
          <Link variant="underline" href="https://stacspec.org" target="_blank">
            STAC
          </Link>{" "}
          search. jev does not write text: it selects from the catalogs,
          collections, places and dates that exist.
        </Box>
        <Box>Try an example below the search bar, or type your own request.</Box>
        <Box color="fg.muted">
          This demo has a small budget. With many users, requests can be slow
          or stop.
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
    </Box>
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
