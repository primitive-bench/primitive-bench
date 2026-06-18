"""The `bench` CLI (promptfoo / lm-eval ergonomics).

    bench init <primitive>     scaffold a config for a primitive
    bench run --config ...      run an eval, write runs/<run_id>/
    bench view <run_dir>        summarize slices + separability
    bench submit <run_dir>      submit to the held-out eval server (scores only)

Stub — commands print intent for v0.1.0 so downstream lanes can wire against the surface.
"""

from __future__ import annotations

import typer

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


if __name__ == "__main__":
    app()
