# @primitive-bench/mcp

The **MCP server** for Primitive Bench — lets an AI agent query per-slice, CI-backed
infra-primitive leaderboards *while it reasons*. This is where "one winner is a lie"
becomes a feature: the agent supplies its constraints, the benchmark returns the
slice-specific winner **or** an honest TIE band.

Built on [`mcp-handler`](https://www.npmjs.com/package/mcp-handler) (Next.js + Vercel),
serving **Streamable HTTP** and **SSE**.

## Tools

| Tool | What it answers |
|---|---|
| `recommend(primitive, slice)` | Best provider for a slice — sole winner only if statistically separable, else a TIE band. Refuses thin/saturated slices. |
| `compare(primitive, slice, adapter_a, adapter_b)` | Are two providers separable here (Wilson-interval overlap)? |
| `get_slice_leaderboard(primitive, slice)` | Full ranked table with point estimates + 95% Wilson CIs. |
| `list_primitives()` | All nine primitives and whether each has published results yet. |
| `list_slices(primitive)` | Slice keys for a primitive (feed these to `recommend`). |
| `methodology()` | Why we report per-slice CIs and TIE bands, not a global ranking. |

Every tool is a **deterministic lookup** over the bundled public-snapshot JSON — no LLM and
no network on the server side, so per-query inference cost is **zero**. The client's model
does any natural-language → slice-key mapping for free.

Live today: **websearch** and **extraction**. The other seven primitives return
`no_published_results — coming soon`.

## Endpoints

**🟢 Live:** `https://benchpublic.vercel.app/mcp`

- Streamable HTTP: `https://benchpublic.vercel.app/mcp`
- SSE (legacy): `https://benchpublic.vercel.app/sse`

## Local development

```bash
pnpm install
pnpm dev            # http://localhost:3000/mcp
pnpm smoke          # end-to-end: connects a real MCP client, exercises the tools
pnpm test           # fast unit tests for the leaderboard logic
pnpm typecheck      # tsc --noEmit
```

## Data + contract (the Python ↔ TS seam)

The seed JSON in `data/` and the TS types in `types/` are **generated** from the Python
side, so they cannot drift from the frozen `bench-schemas` contract. Regenerate from the
repo root:

```bash
uv run bench results emit      # writes data/results/public-snapshot/ AND apps/mcp/data/
uv run bench results schema    # writes apps/mcp/types/bench-report.schema.json
pnpm gen:types                 # schema -> types/bench-report.d.ts
pnpm check:types               # CI gate: fails if the committed types drifted
```

Curated raw counts live with each primitive in `packages/eval-*/snapshots/*.counts.toml`
(transcribed from the published reports); `bench-stats/leaderboard.py` derives the
Wilson CIs and winner/TIE bands.

## Deploy to Vercel

> The official instance is already live at `https://benchpublic.vercel.app/mcp`. To deploy your own:

1. Import the repo in Vercel and set **Root Directory = `apps/mcp`** (Next.js auto-detected).
2. Deploy. Your endpoint is `https://<your-deploy>/mcp`.

Runs on **Fluid Compute**; the route sets `maxDuration = 60`. No env vars required (the data
is bundled). For high-traffic SSE sessions, add an Upstash Redis URL via `mcp-handler` config.

## Add to Claude

**Claude Code**
```bash
claude mcp add --transport http primitive-bench https://benchpublic.vercel.app/mcp
```

**Claude Desktop** (`claude_desktop_config.json`)
```json
{
  "mcpServers": {
    "primitive-bench": { "type": "http", "url": "https://benchpublic.vercel.app/mcp" }
  }
}
```

Then ask: *"Which web search API is best for government registry lookups?"* — Claude calls
`recommend` and answers with the exa/serpapi TIE band, Wilson CIs, and a citation.
