import numpy as np

from product_similarity.dedupe import pick_distinct


def unit(*v):
    a = np.array(v, dtype=float)
    return a / np.linalg.norm(a)


def test_copies_of_query_and_of_results_are_dropped():
    vectors = np.array(
        [unit(1, 0, 0), unit(1, 0, 0), unit(0, 1, 0), unit(0, 1, 0.01), unit(0, 0, 1)]
    )
    links = ["q", "q", "a", "b", "c"]
    # row 1 shares the query's link, row 3 is a near copy of row 2
    kept, dropped = pick_distinct([0, 1, 2, 3, 4], vectors, links, query=0, k=10)
    assert kept == [2, 4]
    assert dropped == {"copy_of_query": 1, "copy_of_result": 1}


def test_stops_at_k():
    vectors = np.eye(4)
    kept, _ = pick_distinct([0, 1, 2, 3], vectors, ["a", "b", "c", "d"], query=0, k=2)
    assert kept == [1, 2]


def test_picture_table_gives_products_without_pictures_their_own_link():
    from types import SimpleNamespace

    from product_similarity.dedupe import picture_table

    products = [
        SimpleNamespace(uniq_id="a", image_url="u1"),
        SimpleNamespace(uniq_id="b", image_url=None),
    ]
    matrix, links = picture_table(products, {"u1": 0}, np.array([[1.0, 0.0]]))
    assert links == ["u1", "no-picture-b"]
    assert matrix[1].tolist() == [0.0, 0.0]
