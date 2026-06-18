"""web_search probe/scoring harness.

Ported from arlenk2021/GoldenEvalsWebSearch `src/probe/main.py`, adapted to the
primitive-bench layout:

  * vendors are bench-adapters search adapters (`bench_adapters.get('exa')(spec)`),
    invoked through the uniform `adapter.invoke(item) -> {returned_urls, ...}`
    contract instead of the source's async `vendor.search(query, k)`.
  * the liveness gate is `bench_core.verify.liveness_gate` (batch-wide exclusion,
    never charged as a miss).
  * the URL equivalence class is `bench_core.urls.EquivalenceClass`.
  * each scored (row, vendor) pair is emitted as a `bench_schemas.ItemResult`,
    with `output.miss_reason` set from the miss taxonomy, `slices` from the row's
    intent tags, and `ground_truth_tier` mapped from the row Stratum.

For each golden row in the chosen split:
  1. Liveness gate (batch-wide exclusion, never charged as a miss).
  2. Query each vendor with the chosen QUERY FORM (default: descriptive — the
     honest discriminator; token_in_query is the easy navigational form).
  3. Score hit@{1,5} as set membership against the equivalence class.
  4. On a miss, run mirror AUTO-PROMOTION: any returned URL holding the truth
     token is promoted into the class (logged to adjudication) so an
     unanticipated mirror is not a false miss. Then classify the miss.

Vendors see only the query — never the gold URL or its metadata.
k is PINNED per snapshot (see KS).
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from bench_adapters import get
from bench_core.domain import Stratum, stratum_to_tier
from bench_core.urls import EquivalenceClass
from bench_core.verify import liveness_gate
from bench_schemas import ItemResult, ScorerOutput
from bench_schemas.models import Primitive

from eval_websearch.queries import DEFAULT_FORM
from eval_websearch.scoring import (
    SENTINEL_NOT_INDEXED,
    SENTINEL_RANKED_BELOW_K,
    classify_miss,
    find_promotions,
    first_correct_rank,
    hit_at_k,
)

KS = (1, 5)                         # PINNED per snapshot; do not vary mid-cohort
PROMOTE_DEPTH = max(KS)            # only fetch/promote within the scored window
DEAD_REASON_ABORT_FRACTION = 0.10


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _query_for(row: dict, form: str) -> str | None:
    variants = row.get("query_variants") or {}
    return variants.get(form) or (row.get("query") if form == DEFAULT_FORM else None)


def _search(adapter, query: str, k: int) -> tuple[list[str], dict]:
    """Invoke a bench-adapters search adapter and return (returned_urls, raw)."""
    raw = adapter.invoke({"query": query, "k": k})
    return list(raw.get("returned_urls", [])), raw


def _sentinel_verdict(adapter, row: dict, eq: EquivalenceClass) -> str:
    """Split a sentinel `not_found` miss into its real cause via a SECOND search.

    The bench OWNS the sentinel page and minted a globally-unique truth token, so
    the token_in_query variant is a pure navigational/indexing probe: if even that
    surfaces the canonical URL anywhere in the returned results, the page is
    indexed and the descriptive query just ranked it below k; otherwise the vendor
    has not indexed the page at all. This is ONE extra search — no further retries.
    Vendor errors propagate to the caller, which records them as today.
    """
    token_query = _query_for(row, "token_in_query")
    if not token_query:
        return SENTINEL_NOT_INDEXED
    urls, _ = _search(adapter, token_query, max(KS))
    return SENTINEL_RANKED_BELOW_K if first_correct_rank(urls, eq) != -1 else SENTINEL_NOT_INDEXED


async def probe_row(
    row: dict,
    adapters: dict[str, object],
    run_id: str,
    reps: int,
    form: str,
    adjudication: list[dict] | None = None,
) -> list[ItemResult]:
    """Probe one golden row against each vendor adapter, emitting ItemResults.

    `adapters` is {vendor_name -> bench-adapters adapter instance}. One
    equivalence class per row is shared across vendors so a mirror promoted by
    vendor A counts for the rest of the batch. Mirror promotions are appended to
    `adjudication` (the source's data/adjudication.jsonl ledger) when provided.
    """
    query = _query_for(row, form)
    if query is None:
        return []
    members = row["equivalence_members"]
    eq = EquivalenceClass(members[0], members[1:])
    token = row["truth_token"]
    tier = stratum_to_tier(row["stratum"])
    out: list[ItemResult] = []

    for name, adapter in adapters.items():
        for rep in range(reps):
            try:
                urls, raw = _search(adapter, query, max(KS))
            except Exception as exc:  # adapter failure isolates to this (row, vendor, rep)
                out.append(ItemResult(
                    run_id=run_id, adapter=name, item_id=row["row_id"],
                    primitive=Primitive.WEBSEARCH, slices=row.get("slices", []),
                    ground_truth_tier=tier,
                    output=ScorerOutput(correct=None, miss_reason="adapter_error"),
                    error=repr(exc)[:200],
                ))
                continue

            h = hit_at_k(urls, eq.members, KS)
            promoted_here: list[str] = []
            if not h[max(KS)]:
                for ret_url, final_url in await find_promotions(urls, eq, token, PROMOTE_DEPTH):
                    eq.add(ret_url)
                    eq.add(final_url)
                    promoted_here.append(ret_url)
                    if adjudication is not None:
                        adjudication.append({
                            "row_id": row["row_id"], "event": "mirror_promoted",
                            "url": ret_url, "final": final_url, "vendor": name, "ts": _now(),
                        })
                if promoted_here:
                    h = hit_at_k(urls, eq.members, KS)  # rescore with the promoted mirror

            miss_reason = None if h[max(KS)] else classify_miss(urls, eq, bool(promoted_here), KS)
            # Sentinel discriminator (Layer 1): only on a sentinel row whose
            # descriptive probe truly missed (no class member in top-k, nothing
            # promoted), issue ONE token_in_query indexing probe to split the
            # `not_found` miss into not_indexed vs ranked_below_k. `miss_reason`
            # stays `not_found`; sentinel_verdict carries the split. Null for all
            # non-sentinel rows and for any sentinel hit/promotion.
            sentinel_verdict = None
            if (row["stratum"] == Stratum.SENTINEL and not h[max(KS)]
                    and not promoted_here and form == "descriptive"):
                try:
                    sentinel_verdict = _sentinel_verdict(adapter, row, eq)
                except Exception:
                    # token-probe failed; keep the descriptive record honest and
                    # leave the verdict unresolved rather than dropping the row.
                    sentinel_verdict = None

            metrics = {f"hit@{k}": float(h[k]) for k in KS}
            metrics["first_rank"] = float(first_correct_rank(urls, eq))
            metrics["n_results"] = float(len(urls))
            if raw.get("latency_ms") is not None:
                metrics["latency_ms"] = float(raw["latency_ms"])
            out.append(ItemResult(
                run_id=run_id, adapter=name, item_id=row["row_id"],
                primitive=Primitive.WEBSEARCH, slices=row.get("slices", []),
                ground_truth_tier=tier,
                output=ScorerOutput(
                    # `correct` drives the proportion stats; the pinned hit@max(KS)
                    # is the binary outcome (descriptive form = honest discriminator).
                    correct=bool(h[max(KS)]),
                    score=float(h[max(KS)]),
                    metrics=metrics,
                    miss_reason=miss_reason,
                    equivalence_class=eq.canonical,
                    rationale=(f"query_form={form}; sentinel_verdict={sentinel_verdict}"
                               if sentinel_verdict else f"query_form={form}"),
                ),
                raw_output=raw.get("raw_output"),
                latency_ms=raw.get("latency_ms"),
                cost_usd=raw.get("cost_usd"),
            ))
    return out


async def run(
    rows: list[dict],
    vendors: list[str],
    run_id: str,
    reps: int = 1,
    form: str = DEFAULT_FORM,
    adapter_specs: dict[str, object] | None = None,
) -> list[ItemResult]:
    """Probe a batch of golden rows. Returns the emitted ItemResults.

    1) liveness gate, batch-wide exclusion: a row that fails re-fetch + token
       re-assertion is excluded FOR ALL VENDORS and never charged as a miss. If
       too large a fraction fail for one reason, abort rather than publish a
       biased slice.
    2) probe each live row against each vendor adapter.

    `adapter_specs` maps vendor name -> AdapterSpec (or pre-built adapter). When a
    plain spec is given, the registered adapter class is instantiated with it.
    """
    # 1) liveness gate, batch-wide exclusion
    live_rows: list[dict] = []
    dead_reasons: dict[str, int] = {}
    for r in rows:
        lv = await liveness_gate({"golden_url": r["canonical_url"], "truth_token": r["truth_token"],
                                  "source": r.get("source")})
        if lv.live:
            live_rows.append(r)
        else:
            dead_reasons[lv.reason] = dead_reasons.get(lv.reason, 0) + 1
    if rows:
        for reason, cnt in dead_reasons.items():
            if cnt / len(rows) > DEAD_REASON_ABORT_FRACTION:
                print(f"[probe] ABORT: {cnt}/{len(rows)} rows failed liveness for "
                      f"'{reason}' (>{DEAD_REASON_ABORT_FRACTION:.0%}); batch biased.")
                return []

    adapters = _build_adapters(vendors, adapter_specs)
    adjudication: list[dict] = []
    records: list[ItemResult] = []
    for r in live_rows:
        records.extend(await probe_row(r, adapters, run_id, reps, form, adjudication))
    print(f"[probe] {len(live_rows)}/{len(rows)} rows live; form={form}; "
          f"{len(records)} records; dead={dead_reasons}; promotions={len(adjudication)}")
    return records


def _build_adapters(vendors: list[str], adapter_specs: dict[str, object] | None) -> dict[str, object]:
    """Instantiate the requested bench-adapters search adapters by name.

    If `adapter_specs[name]` already exposes `.invoke`, it is used as-is (handy
    for tests with fakes); otherwise it is treated as an AdapterSpec and passed to
    the registered adapter class constructor.
    """
    adapter_specs = adapter_specs or {}
    out: dict[str, object] = {}
    for name in vendors:
        spec = adapter_specs.get(name)
        if spec is not None and hasattr(spec, "invoke"):
            out[name] = spec
        else:
            cls = get(name)
            out[name] = cls(spec) if spec is not None else cls.__new__(cls)
    return out


def run_sync(
    rows: list[dict],
    vendors: list[str],
    run_id: str,
    reps: int = 1,
    form: str = DEFAULT_FORM,
    adapter_specs: dict[str, object] | None = None,
) -> list[ItemResult]:
    return asyncio.run(run(rows, vendors, run_id, reps=reps, form=form,
                           adapter_specs=adapter_specs))
