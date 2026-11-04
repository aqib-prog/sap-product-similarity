"""Ordering rules: ties, rounding and exclusion."""

import numpy as np

from product_similarity.ranking import rank_indices


def test_equal_scores_are_ordered_by_id_rank():
    scores = np.array([0.5, 0.5, 0.5])
    id_rank = np.array([2, 0, 1])
    assert rank_indices(scores, id_rank, exclude=-1).tolist() == [1, 2, 0]


def test_query_is_excluded():
    scores = np.array([0.9, 0.5, 0.7])
    assert rank_indices(scores, np.arange(3), exclude=0).tolist() == [2, 1]


def test_tiny_float_differences_do_not_decide_the_order():
    scores = np.array([0.1 + 0.2, 0.3])
    assert scores[0] != scores[1]
    assert rank_indices(scores, np.array([1, 0]), exclude=-1).tolist() == [1, 0]


def test_unscored_products_go_last():
    scores = np.array([np.nan, 0.1, 0.9])
    assert rank_indices(scores, np.arange(3), exclude=-1).tolist() == [2, 1, 0]
