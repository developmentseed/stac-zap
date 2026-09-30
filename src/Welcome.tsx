import { useStore } from "@developmentseed/stac-map";
import { Box, HStack, Link, Stack } from "@chakra-ui/react";
import { useEffect, useState } from "react";
import { LuZap } from "react-icons/lu";

/** Text in stac-map's welcome panel. stac-map has no slot for this panel. */
const STAC_MAP_INTRO = "is a map-first visualization tool for";
/** stac-map's URL input, in its header. */
const HREF_INPUT = 'input[placeholder^="Enter a url to a STAC API"]';

type Place = { top: number; left: number; width: number };

/**
 * The stac-zap welcome panel. While no value is open, stac-map shows its own
 * welcome panel; this covers it with the same style.
 */
export default function Welcome() {
  const href = useStore((store) => store.href);
  const place = useIntroPlace(!href);
  if (href || !place) return null;
  return (
    <Box
      position="fixed"
      top={`${place.top}px`}
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
      <Stack p={4} fontSize="sm" fontWeight="lighter" gap={3}>
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
      setPlace((last) =>
        last && last.top === r.top && last.left === r.left && last.width === r.width
          ? last
          : { top: r.top, left: r.left, width: r.width },
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
