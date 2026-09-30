import { Box, Button, Stack, Text } from "@chakra-ui/react";
import { useEffect, useState } from "react";
import { LuZap } from "react-icons/lu";

/** Chakra's `md` breakpoint. stac-map's layout does not fit below it. */
const NARROW = "(max-width: 47.99rem)";

/**
 * A notice on narrow screens: stac-map (and so the zap bar) does not have a
 * layout for phones. The user can close it and continue.
 */
export default function MobileNotice() {
  const narrow = useMediaQuery(NARROW);
  const [closed, setClosed] = useState(false);
  if (!narrow || closed) return null;
  return (
    <Box
      position="fixed"
      inset={0}
      zIndex={100}
      bg="bg/95"
      display="flex"
      alignItems="center"
      justifyContent="center"
      p={6}
    >
      <Stack role="alertdialog" aria-label="Not for mobile" gap={4} maxW="sm" textAlign="center" align="center">
        <Box color="orange.500" fontSize="2xl">
          <LuZap aria-hidden />
        </Box>
        <Text fontWeight="medium">This does not work on mobile.</Text>
        <Text fontSize="sm" color="fg.muted">
          stac-zap needs a larger screen. Open it on a computer.
        </Text>
        <Button size="sm" variant="outline" onClick={() => setClosed(true)}>
          Continue anyway
        </Button>
      </Stack>
    </Box>
  );
}

function useMediaQuery(query: string): boolean {
  const [matches, setMatches] = useState(() => window.matchMedia(query).matches);
  useEffect(() => {
    const list = window.matchMedia(query);
    const update = () => setMatches(list.matches);
    update();
    list.addEventListener("change", update);
    return () => list.removeEventListener("change", update);
  }, [query]);
  return matches;
}
