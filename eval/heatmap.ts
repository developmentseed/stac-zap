/**
 * Draws the field test as a heatmap: models as rows, fields as columns, the
 * percent of answers that the app applies correctly in each cell. Writes an
 * SVG; convert it with `rsvg-convert -w 2400 in.svg -o out.png`.
 *
 * Usage: pnpm tsx eval/heatmap.ts [eval/results/<date>] [out.svg]
 */
import { readdirSync, readFileSync, writeFileSync } from "node:fs";
import type { Row } from "./fields";

const dir = process.argv[2] ?? `eval/results/${readdirSync("eval/results").sort().at(-1)}`;
const out = process.argv[3] ?? "docs/decision-models-heatmap.svg";
const COLUMNS = [
  ["catalog", "Catalog"],
  ["collection", "Collection"],
  ["place", "Place"],
  ["period", "Time period"],
  ["season", "Part of year"],
  ["cloud", "Cloud cover"],
] as const;

// One-hue sequential ramp, light to dark (the dataviz default blue).
const RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"];
const INK = "#1a1a19";
const MUTED = "#6b6a66";
const SURFACE = "#ffffff";
const NEUTRAL = "#f0efec";

/** Short names for the rows, with the maker. */
const NAMES: Record<string, [string, string]> = {
  "typesafe/jev-1.13": ["Jev 1.13", "TypeSafe"],
  "inception/mercury-decide:free": ["Mercury Decide (free)", "Inception"],
  "liquid/d1": ["D1", "Liquid AI"],
  "jevk5-0.2": ["JevK5 0.2", "Alibi Serikbay"],
  "openjev-latest": ["OpenJev 0.1", ""],
  "togethercomputer/tev1-4b-experimental": ["Tev1 4B Experimental", "Together AI"],
  "upstage/solar-decide": ["Solar Decide", "Upstage"],
  "jaredpalmer/kev-4b": ["Kev 4B", "Jared Palmer"],
  "laya-1.0": ["Laya 1.0", "Convai Innovations"],
  "clm-v0.1": ["CLM 0.1", "Contrastive-LM"],
  "verdict-1.4": ["Verdict 1.4", "Heman10x"],
};

type Result = { provider: string; model: string; rows: Row[] };
const results = readdirSync(dir)
  .filter((f) => f.startsWith("fields-"))
  .map((f) => JSON.parse(readFileSync(`${dir}/${f}`, "utf-8")) as Result)
  .map((r) => ({ ...r, ok: r.rows.filter((x) => x.ok).length }))
  .sort((a, b) => b.ok - a.ok);

const esc = (s: string) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;");
const W = 1200;
const LEFT = 330;
const TOP = 170;
const CELL_W = (W - LEFT - 40) / (COLUMNS.length + 1);
const CELL_H = 52;
const GAP = 2;
const H = TOP + results.length * CELL_H + 130;

const parts: string[] = [];
parts.push(
  `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" font-family="Inter, 'Helvetica Neue', Arial, sans-serif">`,
  `<defs><pattern id="hatch" width="8" height="8" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="8" height="8" fill="${NEUTRAL}"/><line x1="0" y1="0" x2="0" y2="8" stroke="#d6d4cf" stroke-width="3"/></pattern></defs>`,
  `<rect width="${W}" height="${H}" fill="${SURFACE}"/>`,
  `<text x="40" y="56" font-size="30" font-weight="700" fill="${INK}">Which decision model sets the map right?</text>`,
  `<text x="40" y="90" font-size="17" fill="${MUTED}">stac-zap field test: 30 map requests, 6 settings. Each cell is the percent of answers the app applies correctly.</text>`,
);

const header = (i: number, label: string, bold = false) =>
  `<text x="${LEFT + i * CELL_W + CELL_W / 2}" y="${TOP - 18}" font-size="15" font-weight="${bold ? 700 : 600}" fill="${INK}" text-anchor="middle">${label}</text>`;
COLUMNS.forEach(([, label], i) => parts.push(header(i, label)));
parts.push(header(COLUMNS.length, "Overall", true));

const color = (pct: number) => RAMP[Math.min(RAMP.length - 1, Math.round((pct / 100) * (RAMP.length - 1)))]!;
const cell = (x: number, y: number, rows: Row[], strong = false) => {
  const errors = rows.filter((r) => r.error).length;
  const w = CELL_W - GAP;
  const h = CELL_H - GAP;
  if (rows.length > 0 && errors / rows.length > 0.5) {
    parts.push(
      `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="4" fill="url(#hatch)"/>`,
      `<text x="${x + w / 2}" y="${y + h / 2 + 5}" font-size="13" fill="${MUTED}" text-anchor="middle">too many options</text>`,
    );
    return;
  }
  const pct = Math.round((100 * rows.filter((r) => r.ok).length) / rows.length);
  const fill = color(pct);
  const dark = RAMP.indexOf(fill) >= 7;
  parts.push(
    `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="4" fill="${fill}"/>`,
    `<text x="${x + w / 2}" y="${y + h / 2 + 6}" font-size="${strong ? 18 : 16}" font-weight="${strong ? 700 : 500}" fill="${dark ? SURFACE : INK}" text-anchor="middle">${pct}%</text>`,
  );
};

results.forEach((r, row) => {
  const y = TOP + row * CELL_H;
  const where = r.provider === "codiv" ? "Codiv" : "OpenRouter";
  const [name, maker] = NAMES[r.model] ?? [r.model, ""];
  parts.push(
    `<text x="40" y="${y + 24}" font-size="17" font-weight="600" fill="${INK}">${esc(name)}</text>`,
    `<text x="40" y="${y + 43}" font-size="13" fill="${MUTED}">${esc(maker ? `${maker} · on ${where}` : `on ${where}`)}</text>`,
  );
  COLUMNS.forEach(([field], i) => cell(LEFT + i * CELL_W, y, r.rows.filter((x) => x.field === field)));
  cell(LEFT + COLUMNS.length * CELL_W + 8, y, r.rows, true);
});

// Legend: the ramp from 0 to 100%.
const ly = TOP + results.length * CELL_H + 34;
const lw = 260;
RAMP.forEach((c, i) => parts.push(`<rect x="${40 + (i * lw) / RAMP.length}" y="${ly}" width="${lw / RAMP.length - 1}" height="12" fill="${c}"/>`));
parts.push(
  `<text x="40" y="${ly + 32}" font-size="13" fill="${MUTED}">0%</text>`,
  `<text x="${40 + lw}" y="${ly + 32}" font-size="13" fill="${MUTED}" text-anchor="end">100%</text>`,
  `<rect x="${60 + lw}" y="${ly - 2}" width="34" height="16" rx="3" fill="url(#hatch)"/>`,
  `<text x="${102 + lw}" y="${ly + 11}" font-size="13" fill="${MUTED}">The model's API takes fewer options than the question has (up to 250 collections)</text>`,
  `<text x="40" y="${ly + 62}" font-size="13" fill="${MUTED}">Tested on 2 October 2026. One question per call. Overall counts rejected questions as wrong. </text>`,
  `<text x="40" y="${ly + 82}" font-size="13" fill="${MUTED}">A small test of one app, not a benchmark. Code and data: github.com/developmentseed/stac-zap (eval/)</text>`,
  "</svg>",
);
writeFileSync(out, parts.join("\n"));
console.log(`${results.length} models → ${out}`);
