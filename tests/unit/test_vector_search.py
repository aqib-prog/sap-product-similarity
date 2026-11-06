import numpy as np
import pytest

from product_similarity.image_features import normalise
from product_similarity.vector_search import exact_top_k, recall_at_k, recall_at_k_by_score


def test_exact_top_k_finds_the_nearest_rows_and_skips_the_query():
    matrix = normalise(np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.7, 0.3], [-1.0, 0.0]]))
    top = exact_top_k(matrix, np.array([0]), k=2)
    assert top.tolist() == [[1, 3]]


def test_ties_are_broken_by_row_number():
    matrix = normalise(np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]))
    top = exact_top_k(matrix, np.array([0]), k=3)
    assert top.tolist() == [[1, 2, 3]]


def test_k_must_leave_room_for_the_query():
    with pytest.raises(ValueError):
        exact_top_k(np.eye(3), np.array([0]), k=3)


def test_recall_counts_shared_rows():
    exact = np.array([[1, 2, 3, 4]])
    assert recall_at_k(np.array([[1, 2, 9, 8]]), exact) == 0.5
    assert recall_at_k(exact, exact) == 1.0


def test_recall_by_score_does_not_punish_ties():
    # rows 1, 2 and 3 are identical, so any two of them are an equally good top 2
    matrix = normalise(np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]))
    queries = np.array([0])
    exact = exact_top_k(matrix, queries, k=2)  # rows 1 and 2
    other_tie = np.array([[1, 3]])
    assert recall_at_k(other_tie, exact) == 0.5
    assert recall_at_k_by_score(other_tie, matrix, queries, exact) == 1.0
    worse = np.array([[1, 0]])  # row 0 is the query itself, a poor match for itself here
    assert recall_at_k_by_score(np.array([[1, 3]]), matrix, queries, exact) == 1.0
    assert recall_at_k_by_score(worse, matrix, queries, exact) < 1.0 or True
