<!--
The PR title becomes the squash commit on `main` — use Conventional Commits:
  type: imperative summary
types: feat, fix, docs, chore, ci, build, refactor, perf, test, style, revert
-->

## What changed & why

<!-- State exactly what slice / metric / adapter / package changed, and why. -->

## How verified

<!-- Paste sample output for any scoring / stat / adapter change. -->

- [ ] `uv run pytest -q` passes
- [ ] `uv run ruff check .` and `uv run ruff format --check .` are clean

## Scope checks

- [ ] One logical change; schema changes ride alone (not bundled).
- [ ] **If `bench-schemas` changed:** `SCHEMA_VERSION` bumped in the same commit (D-03).
- [ ] **If `golden-sets-public/` changed:** rows are a PUBLIC dev split — canary-marked (D-08), tier-tagged (D-09), CC-BY-4.0 — with no held-out answers (D-07).
- [ ] Ported code/data keeps a one-line provenance note and a compatible license (D-16).

Closes #
