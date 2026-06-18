# Contributing to Primitive Bench (`primitive-bench`)

Welcome. `primitive-bench` is the public, Apache-2.0-licensed, reproducible eval harness for
AI **infrastructure primitives** — OCR, web search, vector DBs, rerankers,
retrieval, extraction, chunking, crawling, memory. This guide gets a new
contributor from zero to a merged PR.

> **The one thing to internalize first:** our thesis is *"one winner is a lie."*
> No primitive wins every slice, so we never publish a single global ranking — we
> publish **per-slice results with confidence intervals and statistical
> separability**. Everything in this repo serves that. Read
> [`apps/docs/DECISIONS.md`](apps/docs/DECISIONS.md) (D-01…D-17) before a large change.

---

## 1. The golden rule: `bench-schemas` is a frozen contract

`packages/bench-schemas` defines the types every other package speaks:
`RunManifest`, `ItemResult`, `SliceResult`, `ScorerOutput`, `AdapterSpec`,
`StatTest`, `Primitive`, `GroundTruthTier`.

- **Import types only from `bench_schemas`. Write only files your package owns.**
  No shared mutable state — that boundary is what lets independent work proceed in
  parallel without collisions (D-03).
- The contract is **additive-only within a MINOR version**. Adding an *optional*
  field is fine. Renaming/removing a field, or tightening a type, is a **MAJOR**
  bump and a coordinated migration.
- If you touch `bench-schemas`, you **must** bump `SCHEMA_VERSION` in the same
  commit. CI fails the PR otherwise.
- Cross-package names are pinned in [`INTERPACKAGE.md`](INTERPACKAGE.md). Use those
  exact names; don't invent alternates.

If your change seems to *need* a schema change, open an issue first — there is
usually a way to carry it in `metrics`, `slices`, or an optional field instead.

---

## 2. Local setup

We use **[uv](https://docs.astral.sh/uv/)** (workspace) + **Turborepo**. Python ≥ 3.11.

```bash
git clone https://github.com/Primitive-Bench/primitive-bench.git
cd primitive-bench
uv sync --all-packages        # install every workspace package (editable)
uv run pytest -q              # run the test suite
uv run ruff check .           # lint
```

Run a single package's tests: `uv run pytest packages/bench-stats -q`.

Secrets (vendor API keys) live in a git-ignored `.env`; never commit them. Adapter
keys are read from env (see `packages/bench-adapters/README.md` for the per-vendor
variable table).

---

## 3. Repo map (where does my change go?)

| You want to… | Edit |
|---|---|
| Add/adjust a **vendor adapter** (a system-under-test) | `packages/bench-adapters/` |
| Add a **statistical method** | `packages/bench-stats/` |
| Add shared harness infra (fetch, split, golden-set generation) | `packages/bench-core/` |
| Build/extend a **primitive vertical** (scoring + slices) | `packages/eval-<primitive>/` |
| Add **public golden data** | `golden-sets-public/<primitive>/` |
| Change the **contract** | `packages/bench-schemas/` (rare — see §1) |
| Methodology / decisions docs | `apps/docs/` |

The two reference verticals to copy from are **`eval-websearch`** and
**`eval-extraction`** — they are fully implemented; the others
(`eval-ocr`, `eval-vectordb`, `eval-reranker`, `eval-retrieval`, `eval-chunking`,
`eval-crawl`) are stubs waiting to be filled the same way.

---

## 4. Recipes

### Add a vendor adapter
1. In `packages/bench-adapters/src/bench_adapters/<search|extract|…>/`, subclass
   `Adapter` and decorate it:
   ```python
   from bench_adapters import Adapter, register

   @register("my_vendor")
   class MyVendorAdapter(Adapter):
       def invoke(self, item: dict) -> dict:
           # read keys from env; never hardcode
           return {"raw_output": ..., "latency_ms": ..., "cost_usd": ...,
                   "returned_urls": [...]}   # or "main_text": ... for extraction
   ```
2. Make sure importing the subpackage registers it (the package `__init__`
   imports submodules for their side effects).
3. Document the env var(s) it needs in `packages/bench-adapters/README.md`.

### Add or extend a primitive vertical
1. Implement `Task` and `Scorer` (subclass `bench_core.Task` / `bench_core.Scorer`),
   set `primitive = Primitive.<X>`, and emit `bench_schemas.ItemResult` /
   `ScorerOutput` — **decompose misses** (`miss_reason`) rather than just pass/fail.
2. Define the slices in `slices.yaml`. A slice is a constraint that can *separate*
   adapters; if it can't, it's not worth a slice.
3. Add a public dev set under `golden-sets-public/<primitive>/` (see §5).
4. Add tests. Keep all methodology comments — they justify the scoring.

### Add a statistical method
Only **canonical, citable** methods (D-04): McNemar (paired), Wilson (proportion
CI), seeded bootstrap (continuous), Bradley-Terry/Elo (ranking). Public functions
return a `bench_schemas.StatTest` or a documented dataclass, and must be
deterministic (bootstrap takes a required `seed`). Don't break existing signatures.

---

## 5. Golden data rules (read before touching `golden-sets-public/`)

- **Public dev splits only.** Held-out test answers **never** live in this repo —
  they sit behind the private eval server with HMAC-keyed split integrity (D-07).
  If you're unsure whether data is "held out," it is; don't commit it.
- Embed the **canary GUID** (`golden-sets-public/CANARY`) in dev files so trainers
  can exclude them (D-08).
- Three ground-truth tiers (D-09): `verified_external`, `authoritative_registry`,
  `sentinel_planted`. Tag every row.
- Don't fabricate large datasets. Small, honest, human-verifiable samples + clear
  schema docs beat synthetic bulk.
- Published data is **CC-BY-4.0** (`golden-sets-public/LICENSE-DATA`); code is Apache-2.0.

---

## 6. The trust guardrails (don't regress these)

- **Separability gate (D-10):** a slice publishes a winner *only* when the leader is
  statistically separable from the runner-up (McNemar p < α at that n; CIs don't
  overlap). Otherwise report a **TIE band**. Low separability is a correct outcome,
  not a bug — raise n (derived from measured discordance) or merge the slice.
- **Saturation guard (D-11):** if a golden set exceeds ~90% with no spread, it's
  exhausted — rotate in fresh sentinel-planted items.
- **DeWitt clauses (D-12):** some vendor EULAs forbid publishing benchmark results.
  Mark such adapters `publish_restricted=True`; the leaderboard hides them.
- **Determinism (D-14):** `PYTHONHASHSEED=0`, per-run master seed in the manifest,
  pinned dataset/task versions.
- **License hygiene (D-16):** never vendor GPL/commercial-dual code (e.g. marker
  core) into this Apache-2.0 repo. Benchmark it, don't embed it.

---

## 7. License & provenance

- By contributing you agree your contribution is licensed under the repo's
  **Apache-2.0** license (data under CC-BY-4.0). This is the standard inbound=outbound
  rule (Apache-2.0 §5); no separate CLA is required.
- If you port code/data from elsewhere, keep a one-line **provenance note** in the
  package README and confirm the source license is compatible (MIT/Apache-2.0/CC-BY).

---

## 8. PR workflow

1. **Branch** off `main` (e.g. `eval-ocr-scorer`, `adapter-cohere-rerank`). Don't
   commit directly to `main`.
2. Keep PRs focused. Run `uv run pytest -q` and `uv run ruff check .` first.
3. Write a description that states **what slice/metric/adapter** changed and how you
   verified it. Include sample output for scoring/stat changes.
4. CI must be green (tests, lint, and the `SCHEMA_VERSION` guard).
5. A CODEOWNER reviews; we **squash-merge** once approved (and the branch is deleted).

### Commit/PR conventions
- **Conventional Commits** for PR titles and commit subjects: `type: imperative
  summary`, where `type` is one of `feat`, `fix`, `docs`, `chore`, `ci`, `build`,
  `refactor`, `perf`, `test`, `style`, `revert`. The body explains *why*. The
  `pr-title` CI check validates the title.
- Because we **squash-merge**, the **PR title becomes the single commit on `main`**
  and the **PR description becomes its body** — write both as the permanent
  changelog entry for the change.
- One logical change per PR. Schema changes ride alone, never bundled.

---

## 9. Good first issues

- Fill a stub vertical (`eval-ocr` is the highest-priority template; then
  `eval-vectordb`, `eval-reranker`).
- Add a vendor adapter to an existing primitive.
- Add a slice that you believe separates two adapters, with the data to prove it.
- Expand a public golden dev set with verified, canary-marked rows.

Questions? Open an issue. Thanks for helping make AI-infra selection evidence-based
instead of folklore.
