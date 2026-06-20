/**
 * Primitive Bench MCP server (Streamable HTTP + SSE via mcp-handler).
 *
 * Every tool is a deterministic lookup over the bundled public-snapshot JSON — no
 * LLM, no network on the server side. The `[transport]` segment lets one handler
 * serve both /mcp (Streamable HTTP) and /sse (legacy SSE).
 */
import { createMcpHandler } from "mcp-handler";
import { z } from "zod";

import {
  ALL_PRIMITIVES,
  compare,
  getSliceLeaderboard,
  listPrimitives,
  listSlices,
  methodology,
  recommend,
} from "@/lib/leaderboard";

const primitive = z.enum(ALL_PRIMITIVES);

/** Wrap any JSON-able result as an MCP text content block. */
function json(data: unknown) {
  return { content: [{ type: "text" as const, text: JSON.stringify(data, null, 2) }] };
}

const handler = createMcpHandler((server) => {
  server.tool(
    "list_primitives",
    "List all nine AI-infrastructure primitives Primitive Bench certifies and whether each has published results yet.",
    async () => json(listPrimitives()),
  );

  server.tool(
    "list_slices",
    "List the slice/constraint keys (and their status) for a primitive. Use these keys with `recommend`.",
    { primitive },
    async ({ primitive }) => json(listSlices(primitive)),
  );

  server.tool(
    "recommend",
    "Recommend the best provider for a primitive on a specific slice. Names a sole winner only when it is statistically separable (Wilson interval clears the runner-up's); otherwise returns a TIE band. Refuses thin (small-n) and saturated slices.",
    {
      primitive,
      slice: z.string().describe("A slice key from list_slices, e.g. 'company_lookup'."),
    },
    async ({ primitive, slice }) => json(recommend(primitive, slice)),
  );

  server.tool(
    "compare",
    "Head-to-head: are two providers statistically separable on a slice (Wilson-interval overlap)?",
    {
      primitive,
      slice: z.string(),
      adapter_a: z.string(),
      adapter_b: z.string(),
    },
    async ({ primitive, slice, adapter_a, adapter_b }) =>
      json(compare(primitive, adapter_a, adapter_b, slice)),
  );

  server.tool(
    "get_slice_leaderboard",
    "Full ranked table (point estimates + 95% Wilson CIs) for one slice.",
    { primitive, slice: z.string() },
    async ({ primitive, slice }) => json(getSliceLeaderboard(primitive, slice)),
  );

  server.tool(
    "methodology",
    "Why Primitive Bench reports per-slice confidence intervals and TIE bands instead of a single global ranking.",
    async () => json(methodology()),
  );
});

export const maxDuration = 60;
export { handler as GET, handler as POST, handler as DELETE };
