"""bench-schemas — the frozen contract for Primitive Bench.

THIS IS THE GATE. Everything in primitive-bench and primitivebench-platform codes
against the types defined here. Breaking changes require a new MAJOR version and
a coordinated migration across all lanes. See DECISIONS.md (D-03).

Contract surface (v0.1.0):
    RunManifest   — describes one reproducible eval run (seed, versions, env)
    ItemResult    — per-item scoring record (the atomic unit)
    SliceResult   — aggregated result for one slice/constraint
    ScorerOutput  — what a scorer emits for a single item
    AdapterSpec   — declares a provider/primitive adapter and its capabilities
"""

from bench_schemas.models import (
    AdapterSpec,
    GroundTruthTier,
    ItemResult,
    Primitive,
    RunManifest,
    ScorerOutput,
    SliceResult,
    StatTest,
    SCHEMA_VERSION,
)

__all__ = [
    "AdapterSpec",
    "GroundTruthTier",
    "ItemResult",
    "Primitive",
    "RunManifest",
    "ScorerOutput",
    "SliceResult",
    "StatTest",
    "SCHEMA_VERSION",
]
