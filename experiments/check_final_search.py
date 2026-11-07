"""Check that the real search (`HybridEngine`) gives the same results as the notebooks.

Run from the project folder:  python experiments/check_final_search.py
For the 30 development products it compares the top 10 of the real search (exact mode) with
the top 10 of the evaluation code behind notebook 08, and scores both with my ratings
(nDCG@10 and precision@10). The 20 locked products are hidden from the search, so they are
never returned. Results: results/final_search_check.json.
"""

from __future__ import annotations

import json
from pathlib import Path

from hybrid_setup import build_evaluator

from product_similarity import evaluation as ev
from product_similarity.hybrid import FINAL_MIX, HybridEngine, expand_four, load_final_vectors

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def main() -> None:
    setup = build_evaluator()
    evaluator, base, grades, queries = setup.evaluator, setup.engine, setup.grades, setup.queries
    weights = expand_four(FINAL_MIX)
    tuned = json.loads((RESULTS / "hybrid_tuning.json").read_text(encoding="utf-8"))
    assert {k: v for k, v in tuned["four_weights"]["weights"].items()} == FINAL_MIX, (
        "FINAL_MIX differs from results/hybrid_tuning.json"
    )
    vectors = load_final_vectors(
        base.products,
        ROOT / "data" / "embeddings",
        ROOT / "data" / "images" / "shared" / "excluded.json",
    )
    engine = HybridEngine(
        base.products, base.id_map, vectors, mode="exact", hidden_ids=setup.locked
    )

    notebook_top = {no: evaluator.top_ids(weights, no) for no in queries}
    real_top = {no: engine.find_similar(setup.dev[no], 10) for no in queries}
    same = [no for no in queries if real_top[no] == notebook_top[no]]
    same_set = [no for no in queries if set(real_top[no]) == set(notebook_top[no])]
    differing = {
        no: {"notebook": notebook_top[no], "real": real_top[no]}
        for no in queries
        if real_top[no] != notebook_top[no]
    }
    notebook_scores = ev.summarise(ev.score_method(notebook_top, grades))
    real_scores = ev.summarise(ev.score_method(real_top, grades))
    report = {
        "products": len(queries),
        "same_list_in_same_order": len(same),
        "same_products_any_order": len(same_set),
        "notebook": {
            "ndcg_at_10": round(notebook_scores["ndcg_at_10"], 4),
            "precision_at_10": round(notebook_scores["precision_at_10"], 4),
        },
        "real_search_exact": {
            "ndcg_at_10": round(real_scores["ndcg_at_10"], 4),
            "precision_at_10": round(real_scores["precision_at_10"], 4),
        },
        "differences": differing,
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "final_search_check.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in report.items() if k != "differences"}, indent=2))
    for no in differing:
        print(no, "differs")


if __name__ == "__main__":
    main()
