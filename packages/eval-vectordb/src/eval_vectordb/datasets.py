"""Dataset registry for the vectordb primitive.

Three sources, one `Dataset` shape (base/train vectors, query/test vectors, and the
exact top-K ground-truth neighbor ids):

  * **ANN-Benchmarks HDF5** (the reproducibility gold standard): `sift-128-euclidean`,
    `glove-100-angular`, `gist-960-euclidean`, `fashion-mnist-784-euclidean`,
    `nytimes-256-angular`, … Downloaded from ann-benchmarks.com; files carry
    `train`/`test`/`neighbors`/`distances`. Ground truth = exact top-100 NN.
  * **Modern RAG text embeddings** (best/latest models): Cohere `embed-multilingual-v3`
    (1024d) and `-v2` (768d) Wikipedia, OpenAI `text-embedding-3` (1536d) DBpedia —
    pulled from HuggingFace, subsampled, exact NN recomputed.
  * **Synthetic** (offline, fixed seed): tiny clustered vectors for tests/smoke; no
    download, ground truth computed on the fly.

To keep a run honest and tractable, the base is subsampled to a **medium** scale
(default 100k base / 1k queries / k=10). When the base is subsampled, ANN-Benchmarks'
precomputed neighbors (indices into the *full* base) are invalid, so we **recompute
exact neighbors by brute force** on the subsample (`bruteforce-numpy`-equivalent,
faiss-accelerated when available). Built datasets are cached as `.npz` keyed by
(name, base_limit, query_limit, k, seed).

Ground truth is exact and reproducible → `GroundTruthTier.VERIFIED_EXTERNAL`.
"""
from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from typing import Any, Optional

from eval_vectordb._paths import CACHE_DIR

EUCLIDEAN = "euclidean"
ANGULAR = "angular"

DEFAULT_BASE_LIMIT = 100_000
DEFAULT_QUERY_LIMIT = 1_000
DEFAULT_K = 10


@dataclass
class Dataset:
    """One ANN benchmark dataset, ready to build/query/score."""

    name: str            # slice label, e.g. "sift-128-euclidean", "cohere-wiki-v3-1024"
    metric: str          # "euclidean" | "angular"
    dim: int
    train: Any           # np.ndarray [N, dim] float32 — base vectors
    test: Any            # np.ndarray [Q, dim] float32 — query vectors
    neighbors: Any       # np.ndarray [Q, k] int64 — exact top-k base indices
    k: int


# --------------------------------------------------------------------------- #
# Registry: name -> spec. `kind` selects the loader.
# --------------------------------------------------------------------------- #
# ANN-Benchmarks HDF5 (metric/dim parsed from the canonical name).
_ANN_HDF5 = [
    "sift-128-euclidean",
    "gist-960-euclidean",
    "glove-25-angular",
    "glove-100-angular",
    "glove-200-angular",
    "nytimes-256-angular",
    "fashion-mnist-784-euclidean",
    "mnist-784-euclidean",
    "deep-image-96-angular",
]

# Modern text-embedding sets on HuggingFace (best/latest embedding models).
_HF: dict[str, dict[str, Any]] = {
    "cohere-wiki-v3-1024": {
        "repo": "Cohere/wikipedia-2023-11-embed-multilingual-v3",
        "config": "en",
        "split": "train",
        "emb_columns": ["emb", "embedding"],
        "metric": ANGULAR,
        "dim": 1024,
        "note": "Cohere embed-multilingual-v3.0",
    },
    "cohere-wiki-768": {
        "repo": "Cohere/wikipedia-22-12-en-embeddings",
        "config": None,
        "split": "train",
        "emb_columns": ["emb", "embedding"],
        "metric": ANGULAR,
        "dim": 768,
        "note": "Cohere embed-multilingual-v2.0",
    },
    "openai3-1536": {
        "repo": "Qdrant/dbpedia-entities-openai3-text-embedding-3-small-1536-100K",
        "config": None,
        "split": "train",
        "emb_columns": [
            "text-embedding-3-small-1536-embedding",
            "openai",
            "embedding",
            "emb",
        ],
        "metric": ANGULAR,
        "dim": 1536,
        "note": "OpenAI text-embedding-3-small",
    },
}


def list_datasets() -> list[str]:
    return ["synthetic", "synthetic-angular", *_ANN_HDF5, *_HF.keys()]


def parse_ann_name(name: str) -> tuple[int, str]:
    """('sift-128-euclidean') -> (128, 'euclidean'). Raises on an unknown metric."""
    parts = name.rsplit("-", 2)
    if len(parts) != 3:
        raise ValueError(f"cannot parse ANN-Benchmarks name {name!r} (expected <set>-<dim>-<metric>)")
    _set, dim, metric = parts
    if metric not in (EUCLIDEAN, ANGULAR):
        raise ValueError(f"unsupported metric {metric!r} for {name!r} (need euclidean|angular)")
    return int(dim), metric


# --------------------------------------------------------------------------- #
# Exact ground-truth neighbors (faiss-accelerated, numpy fallback).
# --------------------------------------------------------------------------- #
def _normalize(x):
    import numpy as np

    a = np.ascontiguousarray(np.asarray(x, dtype="float32"))
    norms = np.linalg.norm(a, axis=-1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (a / norms).astype("float32")


def exact_neighbors(train, test, k: int, metric: str):
    """Exact top-k base indices for every query — the verified-external ground truth."""
    import numpy as np

    train = np.ascontiguousarray(np.asarray(train, dtype="float32"))
    test = np.ascontiguousarray(np.asarray(test, dtype="float32"))
    try:  # faiss flat is ~exact and fast at medium scale
        import faiss

        if metric == ANGULAR:
            tr, te = _normalize(train), _normalize(test)
            index = faiss.IndexFlatIP(tr.shape[1])
            index.add(tr)
            _d, idx = index.search(te, k)
        else:
            index = faiss.IndexFlatL2(train.shape[1])
            index.add(train)
            _d, idx = index.search(test, k)
        return idx.astype("int64")
    except Exception:
        pass
    # numpy fallback, batched over queries to bound memory.
    if metric == ANGULAR:
        tr, te = _normalize(train), _normalize(test)
        out = np.empty((te.shape[0], k), dtype="int64")
        for s in range(0, te.shape[0], 256):
            sims = te[s : s + 256] @ tr.T
            out[s : s + 256] = np.argsort(-sims, axis=1, kind="stable")[:, :k]
        return out
    out = np.empty((test.shape[0], k), dtype="int64")
    tr_sq = np.einsum("ij,ij->i", train, train)
    for s in range(0, test.shape[0], 256):
        q = test[s : s + 256]
        # ||t||^2 - 2 t·q  (drop +||q||^2, constant per row -> same argsort)
        d = tr_sq[None, :] - 2.0 * (q @ train.T)
        out[s : s + 256] = np.argsort(d, axis=1, kind="stable")[:, :k]
    return out


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #
def _cache_path(name: str, base_limit: Optional[int], query_limit: Optional[int], k: int, seed: int):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{name}.b{base_limit or 'all'}.q{query_limit or 'all'}.k{k}.s{seed}.npz"
    return CACHE_DIR / tag


def _load_cached(path):
    import numpy as np

    if not path.exists():
        return None
    z = np.load(path, allow_pickle=False)
    return z["train"], z["test"], z["neighbors"]


def _save_cached(path, train, test, neighbors):
    import numpy as np

    np.savez(path, train=train, test=test, neighbors=neighbors)


def _subsample(arr, limit, rng):
    if limit is None or limit >= arr.shape[0]:
        return arr, None
    idx = rng.choice(arr.shape[0], size=limit, replace=False)
    idx.sort()
    return arr[idx], idx


def synthetic(metric: str = EUCLIDEAN, *, dim: int = 16, n_base: int = 2000,
              n_query: int = 200, k: int = DEFAULT_K, seed: int = 0,
              n_clusters: int = 20) -> Dataset:
    """Deterministic clustered vectors with exact neighbors — offline tests/smoke."""
    import numpy as np

    rng = np.random.RandomState(seed)
    centers = rng.randn(n_clusters, dim).astype("float32") * 4.0
    def _draw(n):
        c = rng.randint(0, n_clusters, size=n)
        return (centers[c] + rng.randn(n, dim).astype("float32")).astype("float32")
    train = _draw(n_base)
    test = _draw(n_query)
    neighbors = exact_neighbors(train, test, k, metric)
    label = "synthetic" if metric == EUCLIDEAN else "synthetic-angular"
    return Dataset(label, metric, dim, train, test, neighbors, k)


def _download(url: str, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:  # noqa: S310
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    tmp.rename(dest)
    return dest


def load_hdf5(name: str, *, base_limit: Optional[int], query_limit: Optional[int],
              k: int, seed: int) -> Dataset:
    """Load an ANN-Benchmarks HDF5 set, subsample, and (re)compute exact neighbors."""
    import numpy as np

    dim, metric = parse_ann_name(name)
    cache = _cache_path(name, base_limit, query_limit, k, seed)
    cached = _load_cached(cache)
    if cached is not None:
        train, test, neighbors = cached
        return Dataset(name, metric, train.shape[1], train, test, neighbors, k)

    try:
        import h5py
    except Exception as exc:  # dep missing
        raise RuntimeError(f"{name}: h5py required for HDF5 datasets ({exc}); install eval-vectordb[datasets]")

    url = f"https://ann-benchmarks.com/{name}.hdf5"
    hdf = _download(url, CACHE_DIR / f"{name}.hdf5")
    rng = np.random.RandomState(seed)
    with h5py.File(hdf, "r") as f:
        train_full = np.asarray(f["train"], dtype="float32")
        test_full = np.asarray(f["test"], dtype="float32")
        nbr_full = np.asarray(f["neighbors"], dtype="int64") if "neighbors" in f else None

    test, _ti = _subsample(test_full, query_limit, rng)
    train, base_idx = _subsample(train_full, base_limit, rng)

    if base_idx is None and nbr_full is not None:
        # full base -> the file's precomputed neighbors are valid.
        q = _ti if _ti is not None else slice(None)
        neighbors = nbr_full[q][:, :k]
    else:
        # subsampled base -> recompute exact neighbors against the subsample.
        neighbors = exact_neighbors(train, test, k, metric)

    train, test, neighbors = train.astype("float32"), test.astype("float32"), neighbors.astype("int64")
    _save_cached(cache, train, test, neighbors)
    return Dataset(name, metric, train.shape[1], train, test, neighbors, k)


def load_hf(name: str, *, base_limit: Optional[int], query_limit: Optional[int],
            k: int, seed: int) -> Dataset:
    """Load a HuggingFace text-embedding set, subsample, compute exact neighbors."""
    import numpy as np

    spec = _HF[name]
    metric = spec["metric"]
    cache = _cache_path(name, base_limit, query_limit, k, seed)
    cached = _load_cached(cache)
    if cached is not None:
        train, test, neighbors = cached
        return Dataset(name, metric, train.shape[1], train, test, neighbors, k)

    try:
        from datasets import load_dataset
    except Exception as exc:  # dep missing
        raise RuntimeError(f"{name}: `datasets` required for HF sets ({exc}); install eval-vectordb[datasets]")

    n = (base_limit or DEFAULT_BASE_LIMIT) + (query_limit or DEFAULT_QUERY_LIMIT)
    ds = load_dataset(spec["repo"], spec["config"], split=f"{spec['split']}[:{n}]")
    col = next((c for c in spec["emb_columns"] if c in ds.column_names), None)
    if col is None:
        raise RuntimeError(f"{name}: no embedding column among {spec['emb_columns']} in {ds.column_names}")
    emb = np.asarray(ds[col], dtype="float32")
    rng = np.random.RandomState(seed)
    perm = rng.permutation(emb.shape[0])
    q = query_limit or DEFAULT_QUERY_LIMIT
    test = emb[perm[:q]]
    train = emb[perm[q : q + (base_limit or DEFAULT_BASE_LIMIT)]]
    neighbors = exact_neighbors(train, test, k, metric)
    _save_cached(cache, train, test, neighbors)
    return Dataset(name, metric, train.shape[1], train, test, neighbors, k)


def load(name: str, *, base_limit: Optional[int] = DEFAULT_BASE_LIMIT,
         query_limit: Optional[int] = DEFAULT_QUERY_LIMIT, k: int = DEFAULT_K,
         seed: int = 0) -> Dataset:
    """Load a dataset by registry name (synthetic / ANN-Benchmarks / HuggingFace)."""
    if name in ("synthetic", "synthetic-angular"):
        metric = EUCLIDEAN if name == "synthetic" else ANGULAR
        return synthetic(metric, k=k, seed=seed)
    if name in _HF:
        return load_hf(name, base_limit=base_limit, query_limit=query_limit, k=k, seed=seed)
    # default: treat as an ANN-Benchmarks <set>-<dim>-<metric> name.
    return load_hdf5(name, base_limit=base_limit, query_limit=query_limit, k=k, seed=seed)
