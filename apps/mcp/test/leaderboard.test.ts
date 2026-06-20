import test from "node:test";
import assert from "node:assert/strict";

import {
  recommend,
  compare,
  listPrimitives,
  listSlices,
  getSliceLeaderboard,
} from "../lib/leaderboard";

test("recommend names exa as the sole winner of company_lookup", () => {
  const r = recommend("websearch", "company_lookup");
  assert.equal(r.status, "published");
  assert.equal((r as { winner?: string | null }).winner, "exa");
});

test("government_registry is a TIE band of exa + serpapi", () => {
  const r = recommend("websearch", "government_registry") as {
    winner: string | null; band: string[];
  };
  assert.equal(r.winner, null);
  assert.deepEqual(r.band, ["exa", "serpapi"]);
});

test("a thin (n=1) slice refuses to name a winner", () => {
  const r = recommend("websearch", "b2b_tools");
  assert.equal(r.status, "thin_data");
});

test("an unpublished primitive returns coming-soon", () => {
  const r = recommend("ocr", "anything");
  assert.equal(r.status, "no_published_results");
});

test("compare: firecrawl is separable from exa_live on fed_register", () => {
  const r = compare("extraction", "firecrawl", "exa_live", "fed_register") as {
    separable: boolean;
  };
  assert.equal(r.separable, true);
});

test("listPrimitives returns all 9, exactly 2 published", () => {
  const ps = listPrimitives();
  assert.equal(ps.length, 9);
  assert.equal(ps.filter((p) => p.status === "published").length, 2);
});

test("listSlices exposes websearch slice keys", () => {
  const r = listSlices("websearch") as { slices: { slice: string }[] };
  const keys = r.slices.map((s) => s.slice);
  assert.ok(keys.includes("company_lookup"));
  assert.ok(keys.includes("government_registry"));
});

test("getSliceLeaderboard returns ranked rows", () => {
  const r = getSliceLeaderboard("websearch", "technical_docs") as {
    rows: { rank: number }[];
  };
  assert.deepEqual(r.rows.map((x) => x.rank), [1, 2, 3, 4]);
});
