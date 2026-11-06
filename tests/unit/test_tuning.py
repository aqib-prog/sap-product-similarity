import pytest

from product_similarity import tuning


def score_near(target):
    """A score that is highest when the weights match `target` (as shares of the total)."""

    def score(weights):
        total = sum(weights.values())
        shares = {k: v / total for k, v in weights.items()}
        return -sum((shares[k] - target[k]) ** 2 for k in target)

    return score


def test_coordinate_ascent_finds_the_best_mix_on_the_grid():
    target = {"a": 0.5, "b": 0.25, "c": 0.25}
    result = tuning.coordinate_ascent(score_near(target), {"a": 0.1, "b": 0.1, "c": 0.1})
    shares = tuning.normalise(result.weights)
    assert shares["a"] == pytest.approx(0.5, abs=0.05)
    assert result.score > score_near(target)({"a": 0.1, "b": 0.1, "c": 0.1})


def test_a_tie_keeps_the_smaller_weight():
    result = tuning.coordinate_ascent(lambda w: 1.0, {"a": 0.2, "b": 0.2})
    assert result.weights == {"a": 0.2, "b": 0.2}
    assert result.rounds == 1


def test_all_zero_weights_are_never_tried():
    seen = []

    def score(weights):
        seen.append(dict(weights))
        return -weights["a"]

    tuning.coordinate_ascent(score, {"a": 0.2})
    assert all(sum(w.values()) > 0 for w in seen)


def test_random_start_is_repeatable_and_not_all_zero():
    one = tuning.random_start(["a", "b", "c"], seed=1)
    assert one == tuning.random_start(["a", "b", "c"], seed=1)
    assert sum(one.values()) > 0


def test_folds_cover_every_item_once_and_repeat():
    items = [f"q{i}" for i in range(30)]
    folds = tuning.make_folds(items, 5, seed=7)
    assert sorted(sum(folds, [])) == sorted(items)
    assert {len(f) for f in folds} == {6}
    assert folds == tuning.make_folds(items, 5, seed=7)


def test_cross_validation_scores_on_left_out_products():
    queries = [f"q{i}" for i in range(10)]
    calls = []

    def score_on(weights, subset):
        calls.append(tuple(subset))
        return -abs(weights["a"] - 0.3) - 0.0 * len(subset)

    rows = tuning.cross_validate(
        score_on, queries, [{"a": 0.1}], reference={"a": 0.1}, n_folds=5, seed=1
    )
    assert len(rows) == 5
    for row in rows:
        assert row["test_products"] == 2 and row["train_products"] == 8
        assert row["test_score"] >= row["reference_test_score"]
        assert sum(row["weights"].values()) == pytest.approx(1.0)


def test_importance_orders_by_the_drop():
    score = lambda w: 10 * w["big"] + 1 * w["small"]  # noqa: E731
    rows = tuning.importance(score, {"big": 0.5, "small": 0.5, "off": 0.0})
    assert [r["part"] for r in rows] == ["big", "small"]
    assert rows[0]["drop"] == pytest.approx(5.0)


def test_normalise_rejects_all_zero():
    with pytest.raises(ValueError):
        tuning.normalise({"a": 0.0})
