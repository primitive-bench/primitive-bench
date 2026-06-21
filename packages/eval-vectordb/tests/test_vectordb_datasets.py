"""Dataset registry tests — synthetic path only (HDF5/HF need network + extras)."""
from __future__ import annotations

import pytest

from eval_vectordb import datasets


def test_parse_ann_name():
    assert datasets.parse_ann_name("sift-128-euclidean") == (128, "euclidean")
    assert datasets.parse_ann_name("glove-100-angular") == (100, "angular")
    with pytest.raises(ValueError):
        datasets.parse_ann_name("kosarak-27983-jaccard")  # unsupported metric


def test_synthetic_shapes_and_metric():
    ds = datasets.synthetic("euclidean", dim=16, n_base=500, n_query=50, k=10, seed=0)
    assert ds.train.shape == (500, 16)
    assert ds.test.shape == (50, 16)
    assert ds.neighbors.shape == (50, 10)
    assert ds.metric == "euclidean" and ds.name == "synthetic"


def test_bruteforce_recovers_exact_neighbors():
    """The bruteforce-numpy engine reproduces the dataset's exact ground truth."""
    from bench_adapters.registry import get

    ds = datasets.synthetic("angular", dim=24, n_base=800, n_query=60, k=10, seed=1)
    eng = get("bruteforce-numpy")()
    eng.build(ds.train, ds.metric, {})
    hits = tot = 0
    for i in range(ds.test.shape[0]):
        got = set(eng.query(ds.test[i], ds.k))
        truth = {int(x) for x in ds.neighbors[i]}
        hits += len(got & truth)
        tot += ds.k
    assert hits / tot >= 0.99  # exact == exact (modulo measure-zero ties)


def test_load_dispatches_synthetic():
    ds = datasets.load("synthetic-angular", base_limit=None, query_limit=None, seed=0)
    assert ds.metric == "angular" and ds.name == "synthetic-angular"
