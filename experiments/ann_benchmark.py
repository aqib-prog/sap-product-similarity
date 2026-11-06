"""Compare fast nearest-neighbour indexes with the exact search.

Run from the project folder:  python experiments/ann_benchmark.py
For the picture vectors (CLIP) and the sentence vectors, it builds FAISS HNSW, FAISS IVF and
Annoy indexes with several settings, and measures build time, index size, time per query and
recall@10 against the exact top 10 saved by ann_exact_baseline.py. One thread and one query at a
time, so the timings are comparable.
The results are saved in results/ann_benchmark.json; log_ann_benchmark.py records them in MLflow.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
from ann_data import load_spaces

from product_similarity.vector_search import exact_top_k, recall_at_k_by_score

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
K = 10
KEEP_ALIVE: list = []
TIMED_QUERIES = 1000  # the same 1,000 products are used for time and for recall
REPEATS = 5  # runs of each timed query; the fastest is kept


def timed_search(search, positions: np.ndarray) -> tuple[np.ndarray, float]:
    """Search for every position, one query at a time.

    The one rule for every method: each of the 1,000 products is searched 5 times and the
    fastest run is kept; the time reported is the median of the 1,000. Recall uses the same
    1,000 products.
    """
    found = np.empty((len(positions), K), dtype=np.int64)
    times = []
    for i, q in enumerate(positions):
        runs = REPEATS if i < TIMED_QUERIES else 1
        best = np.inf
        for _ in range(runs):
            t0 = time.perf_counter()
            ids = search(int(q))
            best = min(best, time.perf_counter() - t0)
        if i < TIMED_QUERIES:
            times.append(best * 1000)
        found[i] = [j for j in ids if j != q][:K]
    return found, float(np.median(times))


def measure(name, params, build, search_for, positions, recall, extra=None):
    t0 = time.perf_counter()
    index, size_mb = build()
    build_s = time.perf_counter() - t0
    rows = []
    for setting, search in search_for(index):
        found, ms = timed_search(search, positions)
        rows.append(
            {
                "index": name,
                **params,
                **setting,
                "build_seconds": round(build_s, 2),
                "index_mb": round(size_mb, 1),
                "ms_per_query": round(ms, 3),
                "recall_at_10": round(recall(found), 4),
                **(extra or {}),
            }
        )
        print(rows[-1], flush=True)
    return rows


def make_recall(matrix, positions, exact):
    """Recall@10 of a found list, against the exact answers (ties are not punished)."""

    def recall(found):
        return recall_at_k_by_score(found, matrix, positions, exact)

    return recall


def run_space(space, positions, recall):
    import faiss
    from annoy import AnnoyIndex

    matrix = space.matrix
    n, dim = matrix.shape
    out = []

    def flat_build():
        index = faiss.IndexFlatIP(dim)
        index.add(matrix)
        return index, matrix.nbytes / 1e6

    out += measure(
        "faiss_flat",
        {},
        flat_build,
        lambda idx: [({}, lambda q: idx.search(matrix[q : q + 1], K + 1)[1][0].tolist())],
        positions,
        recall,
    )

    for m in (16, 32):

        def hnsw_build(m=m):
            index = faiss.IndexHNSWFlat(dim, m, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = 200
            index.add(matrix)
            return index, len(faiss.serialize_index(index)) / 1e6

        def hnsw_search(idx):
            settings = []
            for ef in (16, 32, 64, 128, 256):

                def search(q, ef=ef):
                    idx.hnsw.efSearch = ef
                    return idx.search(matrix[q : q + 1], K + 1)[1][0].tolist()

                settings.append(({"efSearch": ef}, search))
            return settings

        out += measure(
            "faiss_hnsw",
            {"M": m, "efConstruction": 200},
            hnsw_build,
            hnsw_search,
            positions,
            recall,
        )

    nlist = int(np.sqrt(n))

    def ivf_build():
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(matrix)
        index.add(matrix)
        KEEP_ALIVE.append(quantizer)  # the index needs its quantizer to stay in memory
        return index, len(faiss.serialize_index(index)) / 1e6

    def ivf_search(idx):
        settings = []
        for nprobe in (1, 4, 16, 32):

            def search(q, nprobe=nprobe):
                idx.nprobe = nprobe
                return idx.search(matrix[q : q + 1], K + 1)[1][0].tolist()

            settings.append(({"nprobe": nprobe}, search))
        return settings

    out += measure("faiss_ivf", {"nlist": nlist}, ivf_build, ivf_search, positions, recall)

    for trees in (10, 50, 100):

        def annoy_build(trees=trees):
            index = AnnoyIndex(dim, "angular")
            for i, vector in enumerate(matrix):
                index.add_item(i, vector.tolist())
            index.build(trees)
            path = os.path.join(tempfile.mkdtemp(), "a.ann")
            index.save(path)
            return index, os.path.getsize(path) / 1e6

        def annoy_search(idx, trees=trees):
            settings = []
            for factor in (1, 5, 20):
                search_k = factor * trees * (K + 1)
                settings.append(
                    (
                        {"search_k": search_k},
                        lambda q, sk=search_k: idx.get_nns_by_item(q, K + 1, search_k=sk),
                    )
                )
            return settings

        out += measure("annoy", {"n_trees": trees}, annoy_build, annoy_search, positions, recall)
    return out


def main() -> None:
    import faiss

    faiss.omp_set_num_threads(1)
    spaces, query_rows, _ = load_spaces()
    saved = np.load(ROOT / "data" / "embeddings" / "ann_exact_top10.npz")
    report = {"k": K, "threads": 1, "spaces": {}}
    for name, space in spaces.items():
        position = space.position
        qrows = query_rows["random1000"]
        positions = np.array([position[int(r)] for r in qrows])
        exact = saved[f"{name}_random1000"]
        check = exact_top_k(space.matrix, positions[:5], K)
        assert np.array_equal(check, exact[:5]), "saved exact answers differ"
        recall = make_recall(space.matrix, positions, exact)
        print(f"== {name}: {space.matrix.shape}", flush=True)
        report["spaces"][name] = {
            "products": int(space.matrix.shape[0]),
            "dimensions": int(space.matrix.shape[1]),
            "indexes": run_space(space, positions, recall),
        }
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "ann_benchmark.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("saved", out)


if __name__ == "__main__":
    main()
