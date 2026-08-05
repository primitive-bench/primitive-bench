"""Bradley-Terry ranking from pairwise outcomes — the canonical ranking primitive.

This is the ranking method the schema already reserves (`StatTest.method ==
"bradley_terry"`) and the methodology names alongside Elo. It turns a pile of
head-to-head results ("adapter A beat adapter B on this item") into a strength
per system plus a confidence interval, which is what lets a slice say "A leads,
and the lead is real" instead of forcing a single global #1.

Design (matches the repo's conventions):
  * Stdlib-only (no scipy/statsmodels), like proportions/resampling/retrieval, so
    the harness keeps no hard scipy dep (D-04).
  * The fit is the standard Bradley-Terry MM / iterative-scaling update
    (Hunter 2004), which is monotone and deterministic.
  * CIs follow the same seeded percentile-bootstrap stance as `bootstrap_ci`
    (LMSYS/FastChat pattern): resample the game list, refit, take percentiles.
    `seed` is REQUIRED and echoed on every StatTest for reproducibility.
  * Returns a documented dataclass wrapping per-system `StatTest`s, mirroring how
    `proportions.separable` wraps a StatTest in a `Separability` dataclass.

Strengths are normalized to sum to 1, so they read as comparable shares within
one call. Two well-known degenerate cases are handled gracefully rather than
raising: a system that wins every game pushes toward the upper boundary, and a
system that never wins gets strength 0 (the Bradley-Terry MLE is not finite in
those cases, but the *ranking* stays correct and monotone).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Hashable, Sequence

from bench_schemas import StatTest


@dataclass
class BradleyTerryRanking:
    """Result of a Bradley-Terry fit over pairwise games.

    `systems` is every system that played, ordered strongest first (ties broken
    by name for determinism). `strengths` maps each system to its point strength
    (the values sum to 1). `tests` maps each system to a `StatTest(method=
    "bradley_terry")` carrying that strength as `statistic` plus the bootstrap
    `ci_low`/`ci_high`. `n_games` and `seed` are echoed for reproducibility.
    """

    systems: list[Hashable] = field(default_factory=list)
    strengths: dict[Hashable, float] = field(default_factory=dict)
    tests: dict[Hashable, StatTest] = field(default_factory=dict)
    n_games: int = 0
    seed: int = 0

    @property
    def winner(self) -> Hashable | None:
        """The single strongest system, or None when there were no games."""
        return self.systems[0] if self.systems else None


def _fit(
    games: Sequence[tuple[Hashable, Hashable]],
    systems: list[Hashable],
    max_iter: int,
    tol: float,
) -> dict[Hashable, float]:
    """Bradley-Terry MM fit (Hunter 2004) over a fixed system set.

    `p_i <- W_i / sum_j n_ij / (p_i + p_j)`, iterated to convergence, where
    `W_i` is i's win count and `n_ij` the number of games between i and j.
    Normalized to sum to 1 each iteration so the scale-free strengths stay
    comparable. The system set is fixed by the caller so bootstrap resamples
    return aligned vectors even when a resample omits some system.
    """
    wins: dict[Hashable, float] = {s: 0.0 for s in systems}
    beat: dict[tuple[Hashable, Hashable], int] = {}
    for winner, loser in games:
        if winner == loser:
            continue  # a system cannot beat itself; ignore degenerate rows
        wins[winner] += 1
        beat[(winner, loser)] = beat.get((winner, loser), 0) + 1

    p: dict[Hashable, float] = {s: 1.0 / len(systems) for s in systems}
    for _ in range(max_iter):
        nxt: dict[Hashable, float] = {}
        for i in systems:
            denom = 0.0
            for j in systems:
                if i == j:
                    continue
                n_ij = beat.get((i, j), 0) + beat.get((j, i), 0)
                if n_ij:
                    denom += n_ij / (p[i] + p[j])
            nxt[i] = wins[i] / denom if denom > 0 else 0.0
        total = sum(nxt.values())
        if total > 0:
            for i in systems:
                nxt[i] /= total
        delta = max((abs(nxt[i] - p[i]) for i in systems), default=0.0)
        p = nxt
        if delta < tol:
            break
    return p


def bradley_terry(
    games: Sequence[tuple[Hashable, Hashable]],
    *,
    seed: int,
    resamples: int = 1000,
    alpha: float = 0.05,
    max_iter: int = 1000,
    tol: float = 1e-9,
) -> BradleyTerryRanking:
    """Rank systems from pairwise wins via Bradley-Terry with bootstrap CIs.

    `games` is a sequence of `(winner, loser)` pairs (one per head-to-head
    outcome; rows where winner == loser are ignored). The point strengths come
    from a single MM fit on all games; the CI for each system comes from a
    seeded percentile bootstrap over the game list (refit on each resample),
    matching `bootstrap_ci`. `seed` is mandatory and stored on every StatTest.

    Returns a `BradleyTerryRanking`. With no games it returns an empty ranking.
    """
    systems = sorted({s for g in games for s in g[:2] if g[0] != g[1]}, key=repr)
    n_games = sum(1 for a, b in games if a != b)
    if not systems:
        return BradleyTerryRanking(n_games=0, seed=seed)

    point = _fit(games, systems, max_iter, tol)

    # Seeded percentile bootstrap: resample games, refit over the SAME system set
    # (a system absent from a resample fits to strength 0), collect per-system
    # strengths, take percentiles. Same seed -> identical intervals.
    rng = random.Random(seed)
    scored = [g for g in games if g[0] != g[1]]
    m = len(scored)
    draws: dict[Hashable, list[float]] = {s: [] for s in systems}
    for _ in range(resamples):
        resample = [scored[rng.randrange(m)] for _ in range(m)]
        fit = _fit(resample, systems, max_iter, tol)
        for s in systems:
            draws[s].append(fit[s])

    lo_idx = int((alpha / 2) * resamples)
    hi_idx = min(resamples - 1, int((1 - alpha / 2) * resamples))
    tests: dict[Hashable, StatTest] = {}
    for s in systems:
        samples = sorted(draws[s])
        tests[s] = StatTest(
            method="bradley_terry",
            statistic=point[s],
            ci_low=samples[lo_idx],
            ci_high=samples[hi_idx],
            n=n_games,
            seed=seed,
        )

    order = sorted(systems, key=lambda s: (-point[s], repr(s)))
    return BradleyTerryRanking(
        systems=order,
        strengths=point,
        tests=tests,
        n_games=n_games,
        seed=seed,
    )
