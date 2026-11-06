import numpy as np

from product_similarity import text_features


class FakeModel:
    def encode(self, texts, **_):
        return np.array([[3.0, 4.0] if t.strip() else [0.0, 0.0] for t in texts])


def test_vectors_have_length_one_and_empty_titles_do_not_crash():
    encoder = text_features.TextEncoder.__new__(text_features.TextEncoder)
    encoder.model = FakeModel()
    vectors, report = text_features.embed_titles(encoder, ["a dress", "", None])
    assert np.allclose(np.linalg.norm(vectors[0]), 1.0)
    assert report["titles"] == 3 and report["dimensions"] == 2
    assert vectors.shape == (3, 2)
