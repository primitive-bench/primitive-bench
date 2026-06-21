# DECISIONS — Primitive Bench

Architecture decision record. Carried forward across methodology versions. Each
decision is append-only; supersede with a new entry rather than editing in place.

| ID | Decision | Rationale |
|----|----------|-----------|
| **D-01** | Per-slice / per-constraint results, never a single global ranking. | "One winner is a lie." Validated by VectorDBBench (15 cases), OmniDocBench (19 layout categories), LMSYS category arenas, MTEB/BEIR per-task. |
| **D-02** | Hybrid repo: public `primitive-bench` (MIT) harness + private `primitivebench-platform` product. | Public repo is the trust anchor; held-out answers can't leak into a public monorepo. |
| **D-03** | `bench-schemas` is THE frozen contract. Additive-only within a MINOR; renames/removals are MAJOR. Every package imports types only from it, writes only files it owns. | This interface boundary is what lets the build lanes run in parallel without colliding. |
| **D-04** | Statistics are canonical & citable only: McNemar (paired), Wilson (proportion CI), seeded bootstrap (continuous), Bradley-Terry/Elo (ranking). | "Real evals, not reviews." Wilson over CLT for eval-sized n (arXiv:2503.01747). |
| **D-05** | Harness shape = dataset → Task → Solver/Adapter → Scorer → result schema. | Converges with lm-eval, Inspect AI, HELM. Reuse the proven pattern. |
| **D-06** | Adapters registered by name (lm-eval registry). OpenAI-compatible default. | String references in configs; clone-friendly across vendors. |
| **D-07** | HMAC-keyed split integrity over public dev + private held-out test. | Cryptographic upgrade of the OmniDocBench/OCRBench public+private pattern. |
| **D-08** | Canary GUID markers embedded in golden files (BIG-bench convention). | Contamination detection. *Verify exact GUID convention before publishing methodology v3.* |
| **D-09** | Three-tier ground truth: verified-external, authoritative-registry, sentinel-planted. | From the WebSearch Golden Evals methodology. Sentinels also detect drift. |
| **D-10** | Separability is a publish gate. No winner shown for a slice unless McNemar-separable at n (CIs non-overlapping). | Methodology guardrail, not a failure mode. Raise n from measured discordance, or merge the slice. |
| **D-11** | Saturation guard: a golden set >~90% with minimal spread is exhausted; rotate fresh sentinel-planted items. | SimpleQA saturation lesson; LMSYS live-prompt policy. |
| **D-12** | DeWitt-clause handling: only publish results for permissively-licensed / API-accessible systems; flag restricted ones. | BenchANT: 4 of 13 vector DBs forbid publishing benchmark results. |
| **D-13** | Per-run directory layout (manifest.json + items.jsonl + slices.jsonl); ingest → DuckDB. | ann-benchmarks pattern; Open ASR Leaderboard `run_eval.py` → JSONL. |
| **D-14** | Deterministic runs: PYTHONHASHSEED=0, per-run master seed in manifest, pinned dataset/task versions, captured env/docker digests. | Reproducibility is the trust differentiator. |
| **D-15** | Ship OCR first, then WebSearch; OCR becomes the template every other primitive clones. | User already has working OCR + WebSearch architectures. Prove the loop once. |
| **D-16** | Don't vendor GPL/commercial-dual code into MIT `primitive-bench` (e.g. marker core). Benchmark it, don't embed it. | License hygiene. Re-check each LICENSE before reuse. |
| **D-17** | Tiering: free public leaderboards → $5K private workflow evals → certification → $25K+ enterprise reports. No pay-to-rank. | Trust ("G2 for AI infra") depends on neutrality; promptfoo-on-Railway shows self-serve is cheap to operate. |
| **D-18** | Relicense the code MIT → Apache-2.0 (supersedes the "(MIT)" in D-02). Public data stays CC-BY-4.0; the D-16 license-hygiene posture is unchanged (still no GPL/commercial-dual vendoring). | Apache-2.0 adds an explicit patent grant + trademark/NOTICE clarity — a more formal footing for a vendor-neutral trust anchor — while staying permissive and compatible with the MIT/Apache-2.0 code already ported in. Contributions are inbound=outbound under Apache-2.0 §5 (no CLA). |
| **D-19** | Crawl primitive scores **page_coverage** — a target page reached from a seed AND its current content delivered (truth_token survives) — unifying the two canonical crawl-quality axes, COVERAGE and FRESHNESS. Misses decompose into `not_reached` (coverage gap) vs `stale` (freshness gap) vs blocked/token_absent/empty. The public, reproducible board is generated **offline from controlled known-structure `.example` sites** (exact coverage ground truth, network-free, CI-safe); the live board adds hosted vendors over real seeds sampled from each site's **sitemap.xml** (authoritative page registry). | Real-site total-coverage truth is unknowable, so the reproducible portion uses controlled sites with known structure (a standard crawler-evaluation design); the sitemap is the webmaster's own declaration of what pages exist. Axes per Olston & Najork, "Web Crawling," FnTIR 2010; freshness/age per Cho & Garcia-Molina, SIGMOD 2000 / ACM TODS 2003. page_coverage is a paired-binary outcome, so the same McNemar/Wilson separability gate (D-10) applies as every other vertical. |
