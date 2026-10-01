import { useEffect, useState } from "react";

/**
 * Screens below Chakra's `md` breakpoint. Keep it the same as the query in
 * `src/mobile.css`.
 */
export const MOBILE_QUERY = "(max-width: 47.99rem)";

/** Whether the screen is a phone, by width. Changes with the window. */
export function useMobile(): boolean {
  const [matches, setMatches] = useState(
    () => window.matchMedia(MOBILE_QUERY).matches,
  );
  useEffect(() => {
    const list = window.matchMedia(MOBILE_QUERY);
    const update = () => setMatches(list.matches);
    update();
    list.addEventListener("change", update);
    return () => list.removeEventListener("change", update);
  }, []);
  return matches;
}
