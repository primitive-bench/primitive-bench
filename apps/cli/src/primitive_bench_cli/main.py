"""The `bench` CLI (promptfoo / lm-eval ergonomics).

    bench init <primitive>     scaffold a config for a primitive
    bench run --config ...      run an eval, write runs/<run_id>/
    bench view <run_dir>        summarize slices + separability
    bench submit <run_dir>      submit to the held-out eval server (scores only)
    bench results emit          derive public leaderboard JSON from curated snapshots

The eval commands are stubs for v0.1.0; `results emit` is live (it feeds the MCP server).
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import typer
from bench_schemas.models import Primitive
from bench_stats.leaderboard import PrimitiveReport, build_primitive_report

app = typer.Typer(help="Primitive Bench — vendor-neutral eval harness for AI infra primitives.")


@app.command()
def init(primitive: str) -> None:
    """Scaffold a run config for PRIMITIVE (ocr, websearch, vectordb, ...)."""
    typer.echo(f"[stub] would scaffold config for primitive={primitive}")


@app.command()
def run(config: str = typer.Option(..., help="Path to run config YAML"),
        seed: int = typer.Option(0, help="Master deterministic seed")) -> None:
    """Run an eval and write a per-run result directory."""
    typer.echo(f"[stub] would run config={config} seed={seed}")


@app.command()
def view(run_dir: str) -> None:
    """Summarize a run: per-slice point estimates, CIs, separability badges."""
    typer.echo(f"[stub] would summarize {run_dir}")


@app.command()
def submit(run_dir: str) -> None:
    """Submit a run to the held-out eval server (returns scores, never answers)."""
    typer.echo(f"[stub] would submit {run_dir} to the held-out eval server")


# --------------------------------------------------------------------------- #
# results — derive the public leaderboard JSON the MCP server serves.
# --------------------------------------------------------------------------- #
results_app = typer.Typer(help="Derive and inspect public leaderboard results.")
app.add_typer(results_app, name="results")


def _find_repo_root(start: Path) -> Path:
    """Walk up from `start` to the workspace root (the dir holding packages/ + apps/).

    data/ is intentionally not used as a marker — it holds only generated artifacts and
    is absent on a fresh checkout (git does not track empty dirs).
    """
    for p in (start, *start.parents):
        if (p / "packages").is_dir() and (p / "apps").is_dir():
            return p
    raise typer.BadParameter("could not locate repo root (no packages/ + apps/ above cwd)")


@results_app.command("emit")
def results_emit(
    root: Path = typer.Option(None, help="Repo root (default: auto-detect from cwd)."),
) -> None:
    """Build SliceResult-backed leaderboard JSON from packages/*/snapshots/*.counts.toml.

    Writes the canonical artifact to data/results/public-snapshot/ AND a deploy-reachable
    copy to apps/mcp/data/ (the cross-language seam the MCP server reads).
    """
    repo = root or _find_repo_root(Path.cwd())
    count_files = sorted(repo.glob("packages/*/snapshots/*.counts.toml"))
    if not count_files:
        raise typer.BadParameter(f"no *.counts.toml found under {repo}/packages/*/snapshots/")

    canonical = repo / "data" / "results" / "public-snapshot"
    bundled = repo / "apps" / "mcp" / "data"
    canonical.mkdir(parents=True, exist_ok=True)
    bundled.mkdir(parents=True, exist_ok=True)

    for cf in count_files:
        spec = tomllib.loads(cf.read_text())
        primitive = Primitive(spec["primitive"])
        slices_raw = {
            key: [tuple(row) for row in rows] for key, rows in spec["slices"].items()
        }
        report = build_primitive_report(
            primitive=primitive,
            run_id=spec["run_id"],
            slices_raw=slices_raw,
            metric_name=spec["metric_name"],
            citation=spec.get("citation"),
        )
        payload = json.dumps(report.model_dump(mode="json"), indent=2) + "\n"
        for out_dir in (canonical, bundled):
            (out_dir / f"{primitive.value}.json").write_text(payload)

        called = [s.slice for s in report.slices if s.winner]
        ties = [s.slice for s in report.slices if not s.winner and s.status == "published"]
        typer.echo(
            f"{primitive.value}: {len(report.slices)} slices "
            f"({len(called)} called, {len(ties)} TIE) -> {cf.name}"
        )

    typer.echo(f"wrote canonical -> {canonical}")
    typer.echo(f"wrote bundled   -> {bundled}")


@results_app.command("schema")
def results_schema(
    root: Path = typer.Option(None, help="Repo root (default: auto-detect from cwd)."),
) -> None:
    """Emit the JSON Schema of PrimitiveReport (the contract the MCP types generate from).

    The schema is derived from the pydantic models — including the frozen bench-schemas
    types — so the TS side can never drift from the contract (D-03).
    """
    repo = root or _find_repo_root(Path.cwd())
    out = repo / "apps" / "mcp" / "types" / "bench-report.schema.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    schema = PrimitiveReport.model_json_schema()
    out.write_text(json.dumps(schema, indent=2) + "\n")
    typer.echo(f"wrote JSON Schema -> {out}")


if __name__ == "__main__":
    app()
