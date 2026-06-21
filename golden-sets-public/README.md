# golden-sets-public

The **public dev splits** of the Primitive Bench golden datasets. Everything in
this directory is meant to be reproducible by anyone: small, honest samples plus
the schema docs needed to regenerate or replicate the full sets. Data here is
licensed CC-BY-4.0 (see `LICENSE-DATA`); the pipeline code is Apache-2.0.

> This directory is a trust anchor (DECISIONS **D-02**): held-out answers can't
> leak into a public monorepo. What you see here is *only* the public side.

## Public dev splits vs. held-out test splits (D-07)

Each primitive's golden set is partitioned by **HMAC-keyed split integrity**
(**D-07**) into:

- **public dev split** — shipped here. Used to replicate published leaderboard
  numbers. `row_id` is a salt-independent hash of the row's identity, so the same
  fact always lands in the same bucket and re-runs never reshuffle assignments.
- **held-out test split** — lives behind the private eval server
  (`primitivebench-platform/apps/eval-server`). The answers are never published.

The **HMAC salt is never published.** Knowing a `row_id` tells you nothing about
which split it lands in without the salt, which is what keeps the held-out set
held out even though the row identities are deterministic. (Cryptographic upgrade
of the OmniDocBench / OCRBench public+private pattern.)

## The canary (D-08)

`CANARY` carries a fixed, clearly-labelled placeholder GUID
(`PRIMITIVEBENCH-CANARY-GUID:<uuid>`), embedded in the header of each dev split in
this tree. It follows the **BIG-bench contamination convention**: trainers should
**exclude any document containing the GUID** from training corpora, so a model's
later score against this benchmark isn't contaminated.

Caveat carried from **D-08**: the *exact* canary convention must be re-verified
against current community guidance before it is cited in methodology v3. Treat
`CANARY` as documented intent, not a finalized standard.

## Three-tier ground truth (D-09)

Every golden row's answer comes from exactly one tier:

1. **verified-external** — hand-verified against a live third-party page (with a
   content hash to catch later drift). E.g. the `extraction` product rows.
2. **authoritative-registry** — pulled from an authoritative registry's delta
   feed, where a unique `truth_token` (CVE id, SEC accession number, Federal
   Register document number, release tag) must literally appear on the page. E.g.
   the `websearch` `fresh` rows produced by the registry-delta pump.
3. **sentinel-planted** — pages **we deploy and control**, so T0 (the freshness
   clock origin) is authoritative and undisputable. Sentinels double as drift
   detectors. See `sentinel/PROTOCOL.md`.

Saturation guard (**D-11**): a golden set sitting above ~90% with little spread is
exhausted; rotate in fresh sentinel-planted items rather than declaring victory.

## Layout

```
golden-sets-public/
├── README.md            ← this file
├── CANARY               ← contamination canary GUID (D-08)
├── LICENSE-DATA         ← CC-BY-4.0 data license
├── extraction/
│   ├── README.md        ← schema, 4 strata, public-vs-holdout
│   └── dev.jsonl        ← 12-row public web_extraction sample (canary header)
├── websearch/
│   ├── README.md        ← GoldenRow schema, 10 intent slices, the goldgen pump
│   └── dev.example.jsonl← 4 hand-written illustrative rows (NOT a scored cohort)
├── crawl/
│   ├── README.md        ← target schema, render/depth/site_type/freshness slices
│   └── dev.example.jsonl← 56 controlled known-structure targets (the snapshot's scored set)
└── sentinel/
    └── PROTOCOL.md      ← planted-page protocol (authoritative T0)
```

`.jsonl` files may begin with `#` comment lines (canary header + provenance);
skip those when parsing. Validate with:

```
python -c "import json,glob; [json.loads(l) for f in glob.glob('golden-sets-public/**/*.jsonl', recursive=True) for l in open(f) if l.strip() and not l.startswith('#')]; print('golden jsonl OK')"
```

Other primitive subdirectories (`ocr/`, `retrieval/`, `reranker/`, `chunking/`,
`vectordb/`) carry their own READMEs and follow the same public-split conventions.
