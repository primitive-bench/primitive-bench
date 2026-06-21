# golden-sets-public / vectordb

PUBLIC dev split for the **vectordb** primitive — approximate nearest-neighbor (ANN)
recall@10. Canary-marked; reproducible by anyone.

**Held-out test answers NEVER live here** — they sit behind the private eval server
(primitivebench-platform/apps/eval-server) with HMAC-keyed split integrity.

## What a golden row is

One row = **one query** + its **exact top-K nearest-neighbor ids** over a named corpus.
Ground truth is the exact NN computed by brute force (deterministic, reproducible →
`ground_truth_tier: verified_external`), exactly the ANN-Benchmarks convention.

```json
{
  "row_id": "ex_syn_0000",
  "dataset": "synthetic",            // corpus + embedding family (slice axis)
  "metric": "euclidean",             // euclidean (L2) | angular (cosine)
  "dim": 16,
  "budget": "accurate",              // operating point (fast | accurate)
  "query_vector": [ ... ],           // the query embedding (illustrative)
  "true_neighbors": [105, 1307, ...],// EXACT top-k base indices (ground truth)
  "k": 10,
  "slices": ["dataset:synthetic", "metric:euclidean", "dim:low", "budget:accurate"],
  "ground_truth_tier": "verified_external",
  "source": "self-authored-example"
}
```

The engine returns its top-k base ids; the scorer computes
`recall@10 = |returned ∩ true_neighbors| / k`. Aggregated per slice as `Σhits / ΣK`,
recall@10 is the proportion separability surface (Wilson + McNemar).

## Files

- `dev.example.jsonl` — four illustrative rows (canary GUID embedded), reproducible from
  `eval_vectordb.datasets.synthetic("euclidean", dim=16, n_base=2000, n_query=200, k=10, seed=0)`.
- The real corpora — ANN-Benchmarks HDF5 (`sift-128-euclidean`, `glove-100-angular`,
  `gist-960-euclidean`, …), Cohere `embed-multilingual-v3` (1024d) and OpenAI
  `text-embedding-3` (1536d) Wikipedia/DBpedia — are **built locally** by
  `eval_vectordb.datasets` (downloaded + subsampled to a medium scale, exact neighbors
  recomputed) and **git-ignored** (`_cache/`, `dev.jsonl`), never redistributed here.

## Slice axes (the "one winner is a lie" surface)

`dataset:*` (corpus/embedding family) · `metric:euclidean|angular` ·
`dim:low|mid|high` · `budget:fast|accurate` (the recall-vs-QPS tradeoff). See
`packages/eval-vectordb/slices.yaml`.

## License

Data is CC-BY-4.0 (`golden-sets-public/LICENSE-DATA`); the ANN-Benchmarks datasets are
redistributed by their respective authors under their own licenses — we download, never
vendor, them.
