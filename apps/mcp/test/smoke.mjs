/**
 * End-to-end smoke test: connect a real MCP client to the running server over
 * Streamable HTTP, list tools, and exercise the headline cases.
 *
 *   pnpm dev            # in one shell (starts the server on :3000)
 *   pnpm smoke          # in another
 */
import assert from "node:assert/strict";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StreamableHTTPClientTransport } from "@modelcontextprotocol/sdk/client/streamableHttp.js";

const url = new URL(process.env.MCP_URL || "http://localhost:3000/mcp");

const client = new Client({ name: "primitive-bench-smoke", version: "0.0.0" });
await client.connect(new StreamableHTTPClientTransport(url));

const { tools } = await client.listTools();
console.log("TOOLS:", tools.map((t) => t.name).join(", "));

async function call(name, args) {
  const res = await client.callTool({ name, arguments: args });
  return JSON.parse(res.content[0].text);
}

const win = await call("recommend", { primitive: "websearch", slice: "company_lookup" });
console.log(`company_lookup     -> status=${win.status} winner=${win.winner}`);

const tie = await call("recommend", { primitive: "websearch", slice: "government_registry" });
console.log(`government_registry -> status=${tie.status} winner=${tie.winner} band=${JSON.stringify(tie.band)}`);

const thin = await call("recommend", { primitive: "websearch", slice: "b2b_tools" });
console.log(`b2b_tools          -> status=${thin.status}`);

const soon = await call("recommend", { primitive: "ocr", slice: "x" });
console.log(`ocr                -> status=${soon.status}`);

const cmp = await call("compare", {
  primitive: "extraction", slice: "fed_register", adapter_a: "firecrawl", adapter_b: "exa_live",
});
console.log(`compare fed_register firecrawl vs exa_live -> separable=${cmp.separable}`);

const crawl = await call("recommend", { primitive: "crawl", slice: "render:js_rendered" });
console.log(`render:js_rendered  -> status=${crawl.status} winner=${crawl.winner}`);

assert.equal(tools.length, 6, "expected 6 tools");
assert.equal(win.winner, "exa");
assert.equal(tie.winner, null);
assert.deepEqual(tie.band, ["exa", "serpapi"]);
assert.equal(thin.status, "thin_data");
assert.equal(soon.status, "no_published_results");
assert.equal(cmp.separable, true);
assert.equal(crawl.winner, "render-crawl");

console.log("\nSMOKE OK — 6 tools, winner/TIE/thin/coming-soon/compare/crawl all correct.");
await client.close();
