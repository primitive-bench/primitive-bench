/**
 * Framework-agnostic leaderboard logic the MCP tools call.
 *
 * Pure, deterministic lookups over the bundled public-snapshot JSON (emitted by
 * `bench results emit`). No LLM, no network — the client model does any
 * natural-language -> slice-key mapping, this layer just answers.
 *
 * The seed JSON is imported (not fs-read) so the bundler always traces it into the
 * serverless function.
 */
import type { PrimitiveReport, SliceReport, SliceResult } from "../types/bench-report";

import websearchJson from "../data/websearch.json";
import extractionJson from "../data/extraction.json";

/** All nine primitives Primitive Bench certifies (frozen enum order). */
export const ALL_PRIMITIVES = [
  "ocr", "websearch", "vectordb", "reranker", "retrieval",
  "extraction", "chunking", "crawl", "memory",
] as const;
export type Primitive = (typeof ALL_PRIMITIVES)[number];

/** Primitives with a published snapshot today; the rest answer "coming soon". */
const REPORTS: Partial<Record<Primitive, PrimitiveReport>> = {
  websearch: websearchJson as unknown as PrimitiveReport,
  extraction: extractionJson as unknown as PrimitiveReport,
};

function findSlice(report: PrimitiveReport, slice: string): SliceReport | undefined {
  return (report.slices ?? []).find((s) => s.slice === slice);
}

function row(r: SliceResult) {
  return {
    adapter: r.adapter,
    rank: r.rank,
    point_estimate: r.point_estimate,
    ci_low: r.ci?.ci_low ?? null,
    ci_high: r.ci?.ci_high ?? null,
    n: r.n,
    separable: r.separable ?? null,
  };
}

/** One row per primitive: published vs. coming-soon, with slice count. */
export function listPrimitives() {
  return ALL_PRIMITIVES.map((p) => {
    const report = REPORTS[p];
    return {
      primitive: p,
      status: report ? ("published" as const) : ("no_published_results" as const),
      slices: report ? (report.slices ?? []).length : 0,
    };
  });
}

/** Discover the slice keys (and their status) for a primitive. */
export function listSlices(primitive: Primitive) {
  const report = REPORTS[primitive];
  if (!report) {
    return { primitive, status: "no_published_results" as const, slices: [] };
  }
  return {
    primitive,
    status: "published" as const,
    slices: (report.slices ?? []).map((s) => ({
      slice: s.slice,
      status: s.status,
      n: s.n,
      metric: s.metric_name,
      winner: s.winner ?? null,
      band: (s.band ?? []),
    })),
  };
}

/**
 * The headline tool: who should I use for `primitive` on `slice`?
 * Honors the separability gate (TIE), the thin-n gate, and saturation.
 */
export function recommend(primitive: Primitive, slice: string) {
  const report = REPORTS[primitive];
  if (!report) {
    return {
      primitive, slice, status: "no_published_results" as const,
      message: `No published results for '${primitive}' yet — coming soon.`,
    };
  }
  const s = findSlice(report, slice);
  if (!s) {
    return {
      primitive, slice, status: "unknown_slice" as const,
      message: `Unknown slice '${slice}'. Call list_slices('${primitive}') for valid keys.`,
      available_slices: (report.slices ?? []).map((x) => x.slice),
    };
  }
  const base = {
    primitive, slice, metric: s.metric_name, n: s.n,
    citation: s.citation ?? null, leaders: (s.results ?? []).map(row),
  };
  if (s.thin) {
    return { ...base, status: "thin_data" as const, winner: null, band: (s.band ?? []),
      message: `Too few items (n=${s.n}) to name a winner on '${slice}'.` };
  }
  if (s.saturated) {
    return { ...base, status: "saturated" as const, winner: null, band: (s.band ?? []),
      message: `'${slice}' is saturated — every provider ties near the top on ${s.metric_name}.` };
  }
  if (s.winner) {
    return { ...base, status: "published" as const, winner: s.winner, band: (s.band ?? []),
      message: `${s.winner} wins '${slice}' — its Wilson interval clears the runner-up's.` };
  }
  return { ...base, status: "published" as const, winner: null, band: (s.band ?? []),
    message: `TIE on '${slice}': ${(s.band ?? []).join(", ")} are statistically indistinguishable here.` };
}

/** Head-to-head separability of two adapters on a slice (Wilson-interval overlap). */
export function compare(primitive: Primitive, adapterA: string, adapterB: string, slice: string) {
  const report = REPORTS[primitive];
  if (!report) {
    return { primitive, slice, status: "no_published_results" as const,
      message: `No published results for '${primitive}' yet — coming soon.` };
  }
  const s = findSlice(report, slice);
  if (!s) {
    return { primitive, slice, status: "unknown_slice" as const,
      message: `Unknown slice '${slice}'. Call list_slices('${primitive}') for valid keys.`,
      available_slices: (report.slices ?? []).map((x) => x.slice) };
  }
  const ra = (s.results ?? []).find((r) => r.adapter === adapterA);
  const rb = (s.results ?? []).find((r) => r.adapter === adapterB);
  if (!ra || !rb) {
    return { primitive, slice, status: "unknown_adapter" as const,
      message: `One of '${adapterA}'/'${adapterB}' is not present on '${slice}'.`,
      adapters: (s.results ?? []).map((r) => r.adapter) };
  }
  const aHi = ra.ci?.ci_high ?? 1, aLo = ra.ci?.ci_low ?? 0;
  const bHi = rb.ci?.ci_high ?? 1, bLo = rb.ci?.ci_low ?? 0;
  const separable = aHi < bLo || bHi < aLo; // no overlap
  const leader = ra.point_estimate >= rb.point_estimate ? adapterA : adapterB;
  return {
    primitive, slice, status: "published" as const,
    method: "wilson_interval_overlap",
    separable,
    a: row(ra), b: row(rb),
    citation: s.citation ?? null,
    message: separable
      ? `${leader} is statistically separable from the other on '${slice}'.`
      : `${adapterA} and ${adapterB} are NOT separable on '${slice}' (Wilson intervals overlap).`,
  };
}

/** Full ranked table for one slice (the structured form of the markdown report). */
export function getSliceLeaderboard(primitive: Primitive, slice: string) {
  const report = REPORTS[primitive];
  if (!report) {
    return { primitive, slice, status: "no_published_results" as const,
      message: `No published results for '${primitive}' yet — coming soon.` };
  }
  const s = findSlice(report, slice);
  if (!s) {
    return { primitive, slice, status: "unknown_slice" as const,
      available_slices: (report.slices ?? []).map((x) => x.slice) };
  }
  return {
    primitive, slice, status: s.status, metric: s.metric_name, n: s.n,
    winner: s.winner ?? null, band: (s.band ?? []), citation: s.citation ?? null,
    rows: (s.results ?? []).map(row),
  };
}

/** The methodology in brief — why CIs, why no global ranking (D-01/D-04/D-10). */
export function methodology() {
  return {
    thesis: "One winner is a lie — results are per-slice, never a single global ranking.",
    points: [
      "Wilson score intervals on every proportion (preferred over CLT at eval-sized n).",
      "A slice names a winner only when its Wilson interval clears the runner-up's; otherwise a TIE band (separability publish-gate).",
      "Thin slices (small n) and saturated slices (everyone ties high) are flagged, never called.",
      "Ground truth is tiered: verified-external, authoritative-registry, sentinel-planted.",
    ],
    repo: "https://github.com/primitive-bench/primitive-bench",
  };
}
