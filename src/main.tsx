import { StacMap } from "@developmentseed/stac-map";
import "@developmentseed/stac-map/style.css";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { normalizeAssetKeys } from "./asset-keys";
import { signPlanetaryComputerFetches } from "./planetary-computer";
import ZapBar from "./ZapBar";

signPlanetaryComputerFetches();
normalizeAssetKeys();

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <div style={{ height: "100dvh", width: "100dvw" }}>
      {/* The footer renders inside stac-map's providers, so the bar gets
          its Chakra theme and color mode. */}
      <StacMap footer={<ZapBar />} />
    </div>
  </StrictMode>,
);
