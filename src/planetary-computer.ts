/**
 * Signs Planetary Computer asset URLs, so stac-map can read their COGs.
 *
 * Planetary Computer keeps its assets in Azure Blob Storage, which needs a SAS
 * token on each request. stac-map signs only thumbnails, so its COG layers
 * fail on these assets. This wraps `fetch` and adds the token to each blob
 * URL. It uses the same token API and the same host rule as stac-map's
 * thumbnails (`src/utils/planetary-computer.ts` upstream).
 */

const TOKEN_API = "https://planetarycomputer.microsoft.com/api/sas/v1/token";
/** Get a new token this long before the old one expires, in milliseconds. */
const EXPIRY_MARGIN = 5 * 60 * 1000;

type Token = { token: string; "msft:expiry": string };

const tokens = new Map<string, Promise<Token>>();

/** The storage account and container of a blob URL that needs a token. */
function containerOf(url: URL): string | null {
  if (
    !url.hostname.endsWith(".blob.core.windows.net") ||
    // Public assets, as in stac-map.
    url.hostname.startsWith("ai4edatasetspublicassets") ||
    url.searchParams.has("sig")
  ) {
    return null;
  }
  const account = url.hostname.split(".")[0];
  const container = url.pathname.split("/")[1];
  return account && container ? `${account}/${container}` : null;
}

async function tokenFor(container: string): Promise<Token> {
  const cached = tokens.get(container);
  if (cached) {
    const token = await cached;
    if (Date.parse(token["msft:expiry"]) - EXPIRY_MARGIN > Date.now()) {
      return token;
    }
  }
  const next = fetchToken(container);
  tokens.set(container, next);
  next.catch(() => tokens.delete(container));
  return next;
}

async function fetchToken(container: string): Promise<Token> {
  const response = await unsignedFetch(`${TOKEN_API}/${container}`);
  if (!response.ok) {
    throw new Error(`No SAS token for ${container}: ${response.status}`);
  }
  return (await response.json()) as Token;
}

const unsignedFetch = globalThis.fetch.bind(globalThis);

/** Makes each `fetch` of a Planetary Computer blob carry a SAS token. */
export function signPlanetaryComputerFetches(): void {
  globalThis.fetch = async (input, init) => {
    const href =
      input instanceof Request ? input.url : input instanceof URL ? input.href : input;
    const url = URL.canParse(href) ? new URL(href) : null;
    const container = url && containerOf(url);
    if (!url || !container) return unsignedFetch(input, init);
    const { token } = await tokenFor(container);
    for (const [key, value] of new URLSearchParams(token)) {
      url.searchParams.set(key, value);
    }
    return unsignedFetch(
      input instanceof Request ? new Request(url, input) : url,
      init,
    );
  };
}
