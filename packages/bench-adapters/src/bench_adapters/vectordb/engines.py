"""Vector DB / ANN engine adapters: build an index over base vectors, then query it.

Unlike the stateless search/extract/rerank adapters (one `invoke(item)` per item), a
vector engine has a two-phase lifecycle: **build()** an index once over the dataset's
base/train vectors, then **query()** it many times. The eval-vectordb runner drives
that lifecycle; these classes implement it per engine. `invoke()` is intentionally
unsupported — the base `Adapter` Protocol stays satisfied (so the registry is
uniform) but the entrypoints are build()/query().

Three tiers, all registered in the shared `bench_adapters.registry`:

  * OSS in-process (keyless, pip-installable): `bruteforce-numpy` (exact oracle /
    regression sentinel), `hnswlib`, `faiss-flat`, `faiss-hnsw`, `faiss-ivf`,
    `annoy`, `qdrant-local`, `lancedb`.
  * Dockerized servers (connect via env host/port; see docker/docker-compose.vectordb.yml):
    `pgvector`, `milvus`, `weaviate`, `elasticsearch`.
  * Hosted clouds (key from env, billed per query): `pinecone`, `zilliz-cloud`,
    `weaviate-cloud` — marked `publish_restricted` per the DeWitt clause (D-12).

Every heavy import is lazy (inside build()) so `import bench_adapters` and the offline
test suite never require faiss/hnswlib/qdrant/etc. A missing dependency, missing key,
or unreachable service raises `VendorUnavailable` and the runner skips that lane
uncharged — exactly the search/extract/rerank stance.

Distance metrics use the ann-benchmarks vocabulary: ``euclidean`` (L2) and ``angular``
(cosine). For angular we L2-normalize the vectors, so an L2 or inner-product index
ranks identically to cosine — that lets even L2-only indexes serve angular datasets.
"""
from __future__ import annotations

import os
import tempfile
from importlib.metadata import PackageNotFoundError, version as _pkg_version
from typing import Any


class VendorUnavailable(Exception):
    """Raised when a vector engine cannot run (missing dep / key / unreachable service)."""


def _pkg_ver(*pkgs: str) -> str:
    for pkg in pkgs:
        try:
            return _pkg_version(pkg)
        except PackageNotFoundError:
            continue
    return "unknown"


def _need(value: str | None, name: str) -> str:
    if not value:
        raise VendorUnavailable(f"{name}: required setting unset")
    return value


def _env(*names: str) -> str:
    """First non-empty value among the given env var names."""
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    return ""


# Canonical distance metrics (ann-benchmarks vocabulary).
EUCLIDEAN = "euclidean"
ANGULAR = "angular"


def _as_f32(x: Any):
    import numpy as np

    return np.ascontiguousarray(np.asarray(x, dtype="float32"))


def _normalize(x: Any):
    """L2-normalize rows so inner-product / L2 ranking == cosine ranking (angular)."""
    import numpy as np

    a = _as_f32(x)
    norms = np.linalg.norm(a, axis=-1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (a / norms).astype("float32")


class VectorEngine:
    """Base ANN engine: build() an index, then query() top-k base ids (best first).

    Subclasses set the class metadata and implement build()/query()/free(). It
    duck-types the `bench_adapters.registry.Adapter` Protocol (carries `spec`) without
    subclassing it, because the build/query lifecycle replaces `invoke(item)`.
    """

    name: str = ""
    vendor: str = ""
    engine_version: str = "unknown"
    is_sentinel: bool = False
    publish_restricted: bool = False
    supported_metrics: tuple[str, ...] = (EUCLIDEAN, ANGULAR)

    def __init__(self, spec: Any = None):
        self.spec = spec

    # --- lifecycle -------------------------------------------------------- #
    def build(self, base: Any, metric: str, params: dict[str, Any]) -> None:
        raise NotImplementedError

    def query(self, vector: Any, k: int) -> list[int]:
        """Return the engine's top-k base indices for one query vector, best first."""
        raise NotImplementedError

    def free(self) -> None:
        """Release index memory between configs (override if the engine holds state)."""

    def index_memory_bytes(self) -> int | None:
        return None

    def invoke(self, item: dict[str, Any]) -> dict[str, Any]:  # pragma: no cover
        raise NotImplementedError(
            "vectordb engines use build()/query(); the eval-vectordb runner drives them"
        )


# --------------------------------------------------------------------------- #
# OSS in-process — keyless, pip-installable. These power the default run + tests.
# --------------------------------------------------------------------------- #
from bench_adapters.registry import register  # noqa: E402


@register("bruteforce-numpy")
class BruteForceNumpy(VectorEngine):
    """Exact top-k via numpy — the recall=1.0 oracle and regression sentinel.

    numpy only, always available. Also the ground-truth computer used by the
    dataset loader when a subsampled base invalidates precomputed neighbors.
    """

    name = "bruteforce-numpy"
    vendor = "numpy"
    is_sentinel = True

    def build(self, base, metric, params):
        self.metric = metric
        self.engine_version = _pkg_ver("numpy")
        self._base = _normalize(base) if metric == ANGULAR else _as_f32(base)

    def query(self, vector, k):
        import numpy as np

        if self.metric == ANGULAR:
            q = _normalize(vector).reshape(-1)
            scores = self._base @ q
            order = np.argsort(-scores, kind="stable")
        else:
            diff = self._base - _as_f32(vector).reshape(1, -1)
            dist = np.einsum("ij,ij->i", diff, diff)
            order = np.argsort(dist, kind="stable")
        return order[:k].astype(int).tolist()

    def index_memory_bytes(self):
        return int(self._base.nbytes)

    def free(self):
        self._base = None


@register("hnswlib")
class HnswLib(VectorEngine):
    """hnswlib — the reference HNSW. `ef_search` spans the fast/accurate budgets."""

    name = "hnswlib"
    vendor = "hnswlib"

    def build(self, base, metric, params):
        try:
            import hnswlib
        except Exception as exc:  # dep missing
            raise VendorUnavailable(f"hnswlib: not installed ({exc})")
        self.engine_version = _pkg_ver("hnswlib")
        b = _as_f32(base)
        space = "l2" if metric == EUCLIDEAN else "cosine"
        idx = hnswlib.Index(space=space, dim=b.shape[1])
        idx.init_index(
            max_elements=b.shape[0],
            ef_construction=int(params.get("ef_construction", 200)),
            M=int(params.get("M", 16)),
        )
        idx.add_items(b, num_threads=1)
        idx.set_ef(int(params.get("ef_search", 64)))
        self._index = idx
        self._dim = b.shape[1]
        self._n = b.shape[0]

    def query(self, vector, k):
        labels, _ = self._index.knn_query(
            _as_f32(vector).reshape(1, -1), k=k, num_threads=1
        )
        return [int(x) for x in labels[0]]

    def index_memory_bytes(self):
        # ~ HNSW: vectors (4B * dim) + graph (~M*2 links * 4B) per element. Approximate.
        return int(self._n * (self._dim * 4 + 64))

    def free(self):
        self._index = None


class _FaissEngine(VectorEngine):
    vendor = "faiss"

    def _faiss(self):
        try:
            import faiss
        except Exception as exc:  # dep missing
            raise VendorUnavailable(f"{self.name}: faiss not installed ({exc})")
        self.engine_version = _pkg_ver("faiss-cpu", "faiss")
        return faiss

    def _prep(self, base, metric):
        # angular -> normalize so L2/IP ranking == cosine ranking.
        return _normalize(base) if metric == ANGULAR else _as_f32(base)

    def query(self, vector, k):
        v = _normalize(vector) if self.metric == ANGULAR else _as_f32(vector)
        _d, ids = self._index.search(v.reshape(1, -1), k)
        return [int(i) for i in ids[0] if i != -1]

    def free(self):
        self._index = None


@register("faiss-flat")
class FaissFlat(_FaissEngine):
    """FAISS exact flat index — a real-library recall=1.0 reference."""

    name = "faiss-flat"

    def build(self, base, metric, params):
        faiss = self._faiss()
        self.metric = metric
        b = self._prep(base, metric)
        idx = faiss.IndexFlatIP(b.shape[1]) if metric == ANGULAR else faiss.IndexFlatL2(b.shape[1])
        idx.add(b)
        self._index = idx
        self._bytes = int(b.nbytes)

    def index_memory_bytes(self):
        return self._bytes


@register("faiss-hnsw")
class FaissHnsw(_FaissEngine):
    """FAISS HNSW (graph). `M` / `ef_construction` build knobs; `ef_search` at query."""

    name = "faiss-hnsw"

    def build(self, base, metric, params):
        faiss = self._faiss()
        self.metric = metric
        b = self._prep(base, metric)  # normalized for angular -> L2 graph ranks as cosine
        idx = faiss.IndexHNSWFlat(b.shape[1], int(params.get("M", 32)))
        idx.hnsw.efConstruction = int(params.get("ef_construction", 200))
        idx.add(b)
        idx.hnsw.efSearch = int(params.get("ef_search", 64))
        self._index = idx
        self._bytes = int(b.nbytes)

    def index_memory_bytes(self):
        return self._bytes


@register("faiss-ivf")
class FaissIvf(_FaissEngine):
    """FAISS IVF-Flat (inverted lists). `nlist` partitions; `nprobe` spans budgets."""

    name = "faiss-ivf"

    def build(self, base, metric, params):
        faiss = self._faiss()
        self.metric = metric
        b = self._prep(base, metric)
        n = b.shape[0]
        # faiss wants ~39 training points per centroid; clamp nlist so tiny sets train.
        requested = int(params.get("nlist", int(max(1, n**0.5)) * 4))
        nlist = max(1, min(requested, max(1, n // 39)))
        quant = faiss.IndexFlatIP(b.shape[1]) if metric == ANGULAR else faiss.IndexFlatL2(b.shape[1])
        metric_id = faiss.METRIC_INNER_PRODUCT if metric == ANGULAR else faiss.METRIC_L2
        idx = faiss.IndexIVFFlat(quant, b.shape[1], nlist, metric_id)
        idx.train(b)
        idx.add(b)
        idx.nprobe = max(1, min(int(params.get("nprobe", 8)), nlist))
        self._index = idx
        self._bytes = int(b.nbytes)

    def index_memory_bytes(self):
        return self._bytes


@register("annoy")
class Annoy(VectorEngine):
    """Spotify Annoy (random-projection trees). `n_trees`/`search_k` span budgets."""

    name = "annoy"
    vendor = "spotify"

    def build(self, base, metric, params):
        try:
            from annoy import AnnoyIndex
        except Exception as exc:  # dep missing
            raise VendorUnavailable(f"annoy: not installed ({exc})")
        self.engine_version = _pkg_ver("annoy")
        b = _as_f32(base)
        ann_metric = "angular" if metric == ANGULAR else "euclidean"
        idx = AnnoyIndex(b.shape[1], ann_metric)
        for i, v in enumerate(b):
            idx.add_item(i, v.tolist())
        idx.build(int(params.get("n_trees", 50)))
        self._index = idx
        self._search_k = int(params.get("search_k", -1))

    def query(self, vector, k):
        v = _as_f32(vector).reshape(-1).tolist()
        return [int(i) for i in self._index.get_nns_by_vector(v, k, search_k=self._search_k)]

    def free(self):
        self._index = None


@register("qdrant-local")
class QdrantLocal(VectorEngine):
    """Qdrant in-process (`:memory:`). HNSW `m`/`ef_construct`; `hnsw_ef` at query."""

    name = "qdrant-local"
    vendor = "qdrant"

    def build(self, base, metric, params):
        try:
            from qdrant_client import QdrantClient, models
        except Exception as exc:  # dep missing
            raise VendorUnavailable(f"qdrant-local: qdrant-client not installed ({exc})")
        self.engine_version = _pkg_ver("qdrant-client")
        self._models = models
        b = _as_f32(base)
        dist = models.Distance.COSINE if metric == ANGULAR else models.Distance.EUCLID
        client = QdrantClient(location=":memory:")
        client.recreate_collection(
            collection_name="bench",
            vectors_config=models.VectorParams(
                size=int(b.shape[1]),
                distance=dist,
                hnsw_config=models.HnswConfigDiff(
                    m=int(params.get("M", 16)),
                    ef_construct=int(params.get("ef_construction", 200)),
                ),
            ),
        )
        client.upload_collection(
            collection_name="bench",
            vectors=b,
            ids=list(range(b.shape[0])),
            parallel=1,
        )
        self._client = client
        self._ef = int(params.get("ef_search", 64))

    def query(self, vector, k):
        v = _as_f32(vector).reshape(-1).tolist()
        res = self._client.query_points(
            collection_name="bench",
            query=v,
            limit=k,
            search_params=self._models.SearchParams(hnsw_ef=self._ef),
            with_payload=False,
        ).points
        return [int(p.id) for p in res]

    def free(self):
        try:
            self._client.close()
        except Exception:
            pass
        self._client = None


@register("lancedb")
class LanceDB(VectorEngine):
    """LanceDB embedded (IVF-PQ on disk). Builds an on-disk table in a temp dir."""

    name = "lancedb"
    vendor = "lancedb"

    def build(self, base, metric, params):
        try:
            import lancedb
            import pyarrow as pa
        except Exception as exc:  # dep missing
            raise VendorUnavailable(f"lancedb: not installed ({exc})")
        self.engine_version = _pkg_ver("lancedb")
        b = _as_f32(base)
        self._metric = "cosine" if metric == ANGULAR else "l2"
        self._dir = tempfile.mkdtemp(prefix="lancedb-bench-")
        db = lancedb.connect(self._dir)
        dim = int(b.shape[1])
        schema = pa.schema(
            [pa.field("id", pa.int64()), pa.field("vector", pa.list_(pa.float32(), dim))]
        )
        tbl = db.create_table(
            "bench",
            data=[{"id": i, "vector": b[i].tolist()} for i in range(b.shape[0])],
            schema=schema,
        )
        # IVF-PQ needs enough rows; below a threshold LanceDB brute-forces (recall=1.0).
        if b.shape[0] >= 1024:
            try:
                tbl.create_index(
                    metric=self._metric,
                    num_partitions=int(params.get("nlist", 256)),
                    num_sub_vectors=max(1, dim // 16),
                )
            except Exception:
                pass  # fall back to flat search
        self._tbl = tbl
        self._nprobes = int(params.get("nprobe", 8))

    def query(self, vector, k):
        v = _as_f32(vector).reshape(-1).tolist()
        rows = (
            self._tbl.search(v)
            .metric(self._metric)
            .nprobes(self._nprobes)
            .limit(k)
            .select(["id"])
            .to_list()
        )
        return [int(r["id"]) for r in rows]

    def free(self):
        self._tbl = None


# --------------------------------------------------------------------------- #
# Dockerized servers — connect via env host/port (docker/docker-compose.vectordb.yml).
# Implemented against each client's documented API; exercised only when the service
# is up (VendorUnavailable otherwise), so offline CI never touches them.
# --------------------------------------------------------------------------- #
@register("pgvector")
class PgVector(VectorEngine):
    """PostgreSQL + pgvector 0.8 HNSW. DSN from PGVECTOR_DSN / DATABASE_URL."""

    name = "pgvector"
    vendor = "postgres"

    def build(self, base, metric, params):
        try:
            import psycopg
            from pgvector.psycopg import register_vector
        except Exception as exc:
            raise VendorUnavailable(f"pgvector: psycopg/pgvector not installed ({exc})")
        self.engine_version = _pkg_ver("pgvector")
        dsn = _env("PGVECTOR_DSN", "DATABASE_URL") or "postgresql://postgres:postgres@localhost:5432/postgres"
        try:
            conn = psycopg.connect(dsn, connect_timeout=5, autocommit=True)
        except Exception as exc:
            raise VendorUnavailable(f"pgvector: cannot connect ({exc})")
        b = _normalize(base) if metric == ANGULAR else _as_f32(base)
        dim = int(b.shape[1])
        op = "vector_cosine_ops" if metric == ANGULAR else "vector_l2_ops"
        self._order = "<=>" if metric == ANGULAR else "<->"
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute("DROP TABLE IF EXISTS bench")
        conn.execute(f"CREATE TABLE bench (id int, embedding vector({dim}))")
        register_vector(conn)
        with conn.cursor().copy("COPY bench (id, embedding) FROM STDIN WITH (FORMAT BINARY)") as copy:
            copy.set_types(["int4", "vector"])
            for i in range(b.shape[0]):
                copy.write_row([i, b[i]])
        conn.execute(
            f"CREATE INDEX ON bench USING hnsw (embedding {op}) "
            f"WITH (m = {int(params.get('M', 16))}, ef_construction = {int(params.get('ef_construction', 200))})"
        )
        conn.execute(f"SET hnsw.ef_search = {int(params.get('ef_search', 64))}")
        self._conn = conn

    def query(self, vector, k):
        v = (_normalize(vector) if getattr(self, "_order", "<->") == "<=>" else _as_f32(vector)).reshape(-1)
        rows = self._conn.execute(
            f"SELECT id FROM bench ORDER BY embedding {self._order} %s LIMIT %s", (v, k)
        ).fetchall()
        return [int(r[0]) for r in rows]

    def free(self):
        try:
            self._conn.close()
        except Exception:
            pass
        self._conn = None


@register("milvus")
class Milvus(VectorEngine):
    """Milvus 2.5 HNSW via pymilvus. URI from MILVUS_URI (default localhost:19530)."""

    name = "milvus"
    vendor = "milvus"

    def build(self, base, metric, params):
        try:
            from pymilvus import MilvusClient
        except Exception as exc:
            raise VendorUnavailable(f"milvus: pymilvus not installed ({exc})")
        self.engine_version = _pkg_ver("pymilvus")
        uri = _env("MILVUS_URI") or "http://localhost:19530"
        try:
            client = MilvusClient(uri=uri, token=_env("MILVUS_TOKEN"))
        except Exception as exc:
            raise VendorUnavailable(f"milvus: cannot connect ({exc})")
        b = _as_f32(base)
        self._metric = "COSINE" if metric == ANGULAR else "L2"
        if client.has_collection("bench"):
            client.drop_collection("bench")
        client.create_collection("bench", dimension=int(b.shape[1]), metric_type=self._metric)
        client.insert(
            "bench",
            [{"id": i, "vector": b[i].tolist()} for i in range(b.shape[0])],
        )
        client.flush("bench")
        self._client = client
        self._ef = int(params.get("ef_search", 64))

    def query(self, vector, k):
        v = _as_f32(vector).reshape(-1).tolist()
        res = self._client.search(
            "bench", data=[v], limit=k, search_params={"params": {"ef": self._ef}}
        )
        return [int(h["id"]) for h in res[0]]

    def free(self):
        try:
            self._client.close()
        except Exception:
            pass
        self._client = None


@register("weaviate")
class Weaviate(VectorEngine):
    """Weaviate (self-hosted) HNSW. Host from WEAVIATE_HOST (default localhost)."""

    name = "weaviate"
    vendor = "weaviate"

    def _connect(self):
        try:
            import weaviate
        except Exception as exc:
            raise VendorUnavailable(f"weaviate: weaviate-client not installed ({exc})")
        self.engine_version = _pkg_ver("weaviate-client")
        host = _env("WEAVIATE_HOST") or "localhost"
        try:
            return weaviate.connect_to_local(host=host, port=int(_env("WEAVIATE_PORT") or 8080))
        except Exception as exc:
            raise VendorUnavailable(f"weaviate: cannot connect ({exc})")

    def build(self, base, metric, params):
        import weaviate.classes.config as wc

        client = self._connect()
        b = _as_f32(base)
        dist = wc.VectorDistances.COSINE if metric == ANGULAR else wc.VectorDistances.L2_SQUARED
        if client.collections.exists("Bench"):
            client.collections.delete("Bench")
        coll = client.collections.create(
            "Bench",
            vector_index_config=wc.Configure.VectorIndex.hnsw(
                distance_metric=dist,
                ef_construction=int(params.get("ef_construction", 200)),
                max_connections=int(params.get("M", 16)),
            ),
        )
        with coll.batch.dynamic() as batch:
            for i in range(b.shape[0]):
                batch.add_object(properties={"idx": i}, vector=b[i].tolist())
        self._client = client
        self._coll = coll

    def query(self, vector, k):
        v = _as_f32(vector).reshape(-1).tolist()
        res = self._coll.query.near_vector(near_vector=v, limit=k, return_properties=["idx"])
        return [int(o.properties["idx"]) for o in res.objects]

    def free(self):
        try:
            self._client.close()
        except Exception:
            pass
        self._client = None


@register("elasticsearch")
class Elasticsearch(VectorEngine):
    """Elasticsearch dense_vector HNSW (Lucene). URL from ES_URL (default localhost:9200)."""

    name = "elasticsearch"
    vendor = "elastic"

    def build(self, base, metric, params):
        try:
            from elasticsearch import Elasticsearch as ES, helpers
        except Exception as exc:
            raise VendorUnavailable(f"elasticsearch: client not installed ({exc})")
        self.engine_version = _pkg_ver("elasticsearch")
        url = _env("ES_URL") or "http://localhost:9200"
        try:
            es = ES(url, request_timeout=30)
            es.info()
        except Exception as exc:
            raise VendorUnavailable(f"elasticsearch: cannot connect ({exc})")
        b = _as_f32(base)
        sim = "cosine" if metric == ANGULAR else "l2_norm"
        if es.indices.exists(index="bench"):
            es.indices.delete(index="bench")
        es.indices.create(
            index="bench",
            mappings={
                "properties": {
                    "vector": {
                        "type": "dense_vector",
                        "dims": int(b.shape[1]),
                        "index": True,
                        "similarity": sim,
                        "index_options": {
                            "type": "hnsw",
                            "m": int(params.get("M", 16)),
                            "ef_construction": int(params.get("ef_construction", 200)),
                        },
                    }
                }
            },
        )
        helpers.bulk(
            es,
            ({"_index": "bench", "_id": i, "vector": b[i].tolist()} for i in range(b.shape[0])),
        )
        es.indices.refresh(index="bench")
        self._es = es
        self._k_factor = int(params.get("num_candidates_factor", 10))

    def query(self, vector, k):
        v = _as_f32(vector).reshape(-1).tolist()
        res = self._es.search(
            index="bench",
            knn={"field": "vector", "query_vector": v, "k": k, "num_candidates": k * self._k_factor},
            source=False,
            size=k,
        )
        return [int(h["_id"]) for h in res["hits"]["hits"]]

    def free(self):
        self._es = None


# --------------------------------------------------------------------------- #
# Hosted clouds — key from env, billed per query (QP$). publish_restricted (D-12):
# conservative default pending each EULA's publishing terms; the leaderboard hides
# restricted adapters until cleared.
# --------------------------------------------------------------------------- #
@register("pinecone")
class Pinecone(VectorEngine):
    """Pinecone serverless. Key from PINECONE_API_KEY."""

    name = "pinecone"
    vendor = "pinecone"
    publish_restricted = True

    def build(self, base, metric, params):
        try:
            from pinecone import Pinecone as PC, ServerlessSpec
        except Exception as exc:
            raise VendorUnavailable(f"pinecone: client not installed ({exc})")
        self.engine_version = _pkg_ver("pinecone", "pinecone-client")
        key = _need(_env("PINECONE_API_KEY"), "pinecone")
        pc = PC(api_key=key)
        b = _as_f32(base)
        self._metric = "cosine" if metric == ANGULAR else "euclidean"
        index_name = _env("PINECONE_INDEX") or "primitive-bench"
        if not pc.has_index(index_name):
            pc.create_index(
                name=index_name,
                dimension=int(b.shape[1]),
                metric=self._metric,
                spec=ServerlessSpec(
                    cloud=_env("PINECONE_CLOUD") or "aws",
                    region=_env("PINECONE_REGION") or "us-east-1",
                ),
            )
        idx = pc.Index(index_name)
        for s in range(0, b.shape[0], 500):
            idx.upsert(
                vectors=[(str(i), b[i].tolist()) for i in range(s, min(s + 500, b.shape[0]))]
            )
        self._index = idx

    def query(self, vector, k):
        v = _as_f32(vector).reshape(-1).tolist()
        res = self._index.query(vector=v, top_k=k, include_values=False)
        return [int(m["id"]) for m in res["matches"]]

    def free(self):
        self._index = None


@register("zilliz-cloud")
class ZillizCloud(Milvus):
    """Zilliz Cloud (managed Milvus). URI/token from ZILLIZ_URI / ZILLIZ_TOKEN."""

    name = "zilliz-cloud"
    vendor = "zilliz-cloud"
    publish_restricted = True

    def build(self, base, metric, params):
        try:
            from pymilvus import MilvusClient
        except Exception as exc:
            raise VendorUnavailable(f"zilliz-cloud: pymilvus not installed ({exc})")
        self.engine_version = _pkg_ver("pymilvus")
        uri = _need(_env("ZILLIZ_URI"), "zilliz-cloud")
        token = _need(_env("ZILLIZ_TOKEN"), "zilliz-cloud")
        try:
            client = MilvusClient(uri=uri, token=token)
        except Exception as exc:
            raise VendorUnavailable(f"zilliz-cloud: cannot connect ({exc})")
        b = _as_f32(base)
        self._metric = "COSINE" if metric == ANGULAR else "L2"
        if client.has_collection("bench"):
            client.drop_collection("bench")
        client.create_collection("bench", dimension=int(b.shape[1]), metric_type=self._metric)
        client.insert("bench", [{"id": i, "vector": b[i].tolist()} for i in range(b.shape[0])])
        client.flush("bench")
        self._client = client
        self._ef = int(params.get("ef_search", 64))


@register("weaviate-cloud")
class WeaviateCloud(Weaviate):
    """Weaviate Cloud (WCS). URL/key from WEAVIATE_URL / WEAVIATE_API_KEY."""

    name = "weaviate-cloud"
    vendor = "weaviate-cloud"
    publish_restricted = True

    def _connect(self):
        try:
            import weaviate
            from weaviate.classes.init import Auth
        except Exception as exc:
            raise VendorUnavailable(f"weaviate-cloud: weaviate-client not installed ({exc})")
        self.engine_version = _pkg_ver("weaviate-client")
        url = _need(_env("WEAVIATE_URL"), "weaviate-cloud")
        key = _need(_env("WEAVIATE_API_KEY"), "weaviate-cloud")
        try:
            return weaviate.connect_to_weaviate_cloud(cluster_url=url, auth_credentials=Auth.api_key(key))
        except Exception as exc:
            raise VendorUnavailable(f"weaviate-cloud: cannot connect ({exc})")


# Convenience grouping (mirrors rerank.adapters.ALL).
OSS_IN_PROCESS = [
    BruteForceNumpy, HnswLib, FaissFlat, FaissHnsw, FaissIvf, Annoy, QdrantLocal, LanceDB,
]
DOCKER_SERVERS = [PgVector, Milvus, Weaviate, Elasticsearch]
HOSTED_CLOUDS = [Pinecone, ZillizCloud, WeaviateCloud]
ALL = OSS_IN_PROCESS + DOCKER_SERVERS + HOSTED_CLOUDS
