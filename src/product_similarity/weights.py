"""The weights a user can set for one search.

A weight says how much one part counts in the similarity score. There are two ways to set them:
the **simple** way with 4 names (title, text_vectors, image, attributes), where `attributes` is
the seven attributes together, and the **advanced** way with the 10 parts. The two ways cannot
be mixed. What the user sends is what is used: a part that is left out gets 0. Every weight is a
number from 0 to 1 and at least one must be above 0. The numbers do not have to add up to 1,
because the score is a weighted average and only the proportions matter.
"""

from __future__ import annotations

from collections.abc import Mapping

from product_similarity.hybrid import ATTRIBUTES, FINAL_MIX, expand_four
from product_similarity.hybrid_eval import PARTS

SIMPLE_NAMES = ("title", "text_vectors", "image", "attributes")
ADVANCED_NAMES = tuple(PARTS)


class WeightsError(ValueError):
    """The weights sent with a request are not valid. The message says what is wrong."""


def parse_weights(text: str) -> dict[str, float]:
    """Read `name:value,name:value` into weights for the 10 parts, or raise WeightsError."""
    if text is None or not text.strip():
        raise WeightsError(
            "weights is empty; use name:value pairs, for example image:0.6,title:0.4"
        )
    given: dict[str, float] = {}
    for piece in text.split(","):
        name, separator, number = piece.partition(":")
        name, number = name.strip(), number.strip()
        if not separator or not name or not number:
            raise WeightsError(f"'{piece.strip()}' is not a name:value pair")
        if name not in set(SIMPLE_NAMES) | set(ADVANCED_NAMES):
            raise WeightsError(
                f"unknown weight '{name}'; the simple names are {', '.join(SIMPLE_NAMES)} "
                f"and the advanced names are {', '.join(ADVANCED_NAMES)}"
            )
        if name in given:
            raise WeightsError(f"weight '{name}' is given twice")
        try:
            value = float(number)
        except ValueError:
            raise WeightsError(f"the weight of '{name}' is not a number: '{number}'") from None
        if not 0.0 <= value <= 1.0 or value != value:
            raise WeightsError(f"the weight of '{name}' must be from 0 to 1, not {number}")
        given[name] = value
    if "attributes" in given and any(name in ATTRIBUTES for name in given):
        raise WeightsError(
            "do not mix the simple way ('attributes') with the advanced attribute names "
            f"({', '.join(ATTRIBUTES)})"
        )
    if not any(value > 0 for value in given.values()):
        raise WeightsError("at least one weight must be above 0")
    if "attributes" in given:  # the simple way: leave out means 0
        return expand_four({name: given.get(name, 0.0) for name in SIMPLE_NAMES})
    return {name: given.get(name, 0.0) for name in ADVANCED_NAMES}


def weights_key(weights: Mapping[str, float] | None) -> tuple | None:
    """A fixed, hashable form of the weights, for the cache question."""
    if weights is None:
        return None
    return tuple(round(float(weights.get(name, 0.0)), 6) for name in ADVANCED_NAMES)


def tuned_weights() -> dict[str, dict[str, float]]:
    """The tuned weights, in the simple and in the advanced way."""
    return {
        "simple": {name: round(FINAL_MIX[name], 6) for name in SIMPLE_NAMES},
        "advanced": {name: round(value, 6) for name, value in expand_four(FINAL_MIX).items()},
    }
