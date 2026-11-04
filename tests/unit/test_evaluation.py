"""Scoring against ratings: nDCG, precision, confidence ranges and file loading."""

import math

import pytest

from product_similarity.evaluation import (
    bootstrap_mean,
    dcg,
    load_grades,
    ndcg_at_k,
    paired_difference,
    precision_at_k,
    score_method,
    summarise,
)


def test_dcg_rewards_good_results_near_the_top():
    assert dcg([2, 0]) > dcg([0, 2])
    assert dcg([2]) == pytest.approx(3.0)
    assert dcg([0, 2]) == pytest.approx(3 / math.log2(3))


def test_ndcg_is_one_for_the_best_possible_order():
    pool = [2, 2, 1, 0, 0]
    assert ndcg_at_k([2, 2, 1, 0, 0], pool) == pytest.approx(1.0)
    assert ndcg_at_k([0, 0, 1, 2, 2], pool) < 1.0


def test_ndcg_cannot_exceed_the_pool_and_is_undefined_without_matches():
    assert ndcg_at_k([0, 0, 0], [0, 0, 0]) is None
    assert 0 <= ndcg_at_k([1, 0, 0], [2, 1, 0]) <= 1


def test_precision_counts_results_at_or_above_a_grade():
    ranked = [2, 1, 0, 0, 2, 0, 0, 0, 1, 0]
    assert precision_at_k(ranked, 10, 1) == 0.4
    assert precision_at_k(ranked, 10, 2) == 0.2
    assert precision_at_k([2, 2], 10, 1) == 0.2


def test_bootstrap_of_identical_values_has_no_spread():
    assert bootstrap_mean([0.5] * 10) == (0.5, 0.5, 0.5)


def test_bootstrap_range_contains_the_mean_and_is_repeatable():
    values = [0.1, 0.9, 0.4, 0.6, 0.8, 0.2, 0.7]
    mean, low, high = bootstrap_mean(values)
    assert low <= mean <= high
    assert bootstrap_mean(values) == (mean, low, high)


def test_paired_difference_has_the_right_sign_and_uses_shared_products():
    better = {"a": 0.9, "b": 0.8, "c": 0.7, "only_here": 1.0}
    worse = {"a": 0.5, "b": 0.4, "c": 0.3}
    mean, low, high = paired_difference(better, worse)
    assert mean == pytest.approx(0.4) and low > 0 and high >= low


def test_load_grades_skips_blanks_and_rejects_bad_values(tmp_path):
    good = tmp_path / "g.csv"
    good.write_text("query_no,candidate_no,candidate_id,grade\nQ01,01,a,2\nQ01,02,b,\nQ02,01,c,0\n")
    assert load_grades(good) == {"Q01": {"a": 2}, "Q02": {"c": 0}}
    bad = tmp_path / "bad.csv"
    bad.write_text("query_no,candidate_no,candidate_id,grade\nQ01,01,a,3\n")
    with pytest.raises(ValueError):
        load_grades(bad)


def test_score_method_counts_unrated_results_as_zero_and_reports_them():
    grades = {"Q01": {"a": 2, "b": 1, "c": 0}}
    scores = score_method({"Q01": ["a", "b", "zzz"]}, grades, k=3)["Q01"]
    assert scores["unrated"] == 1
    assert scores["precision"] == pytest.approx(2 / 3)
    assert scores["ndcg"] == pytest.approx(1.0, abs=0.2)


def test_summarise_reports_means_ranges_and_counts():
    per_query = {
        f"Q{i}": {"ndcg": 0.5 + i / 20, "precision": 0.5, "strong_precision": 0.2, "unrated": 1}
        for i in range(6)
    }
    per_query["Q9"] = {"ndcg": None, "precision": 0.0, "strong_precision": 0.0, "unrated": 0}
    summary = summarise(per_query)
    assert summary["products_scored"] == 6 and summary["unrated_in_top_10"] == 6
    assert summary["ndcg_at_10_low"] <= summary["ndcg_at_10"] <= summary["ndcg_at_10_high"]
