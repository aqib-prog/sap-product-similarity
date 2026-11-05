import numpy as np
import pytest

from product_similarity.image_features import ImageEncoder, normalise, top_k


def test_normalise_gives_length_one_and_keeps_zero_rows():
    out = normalise(np.array([[3.0, 4.0], [0.0, 0.0]]))
    assert out[0] == pytest.approx([0.6, 0.8])
    assert out[1].tolist() == [0.0, 0.0]


def test_top_k_orders_by_similarity_and_excludes_the_query():
    vectors = normalise(np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.7, 0.7]]))
    assert top_k(vectors, 0, 3) == [1, 3, 2]


def test_top_k_breaks_ties_by_row_number():
    vectors = normalise(np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]))
    assert top_k(vectors, 0, 3) == [1, 2, 3]


def test_unknown_model_is_refused():
    with pytest.raises(ValueError):
        ImageEncoder("not-a-model")
