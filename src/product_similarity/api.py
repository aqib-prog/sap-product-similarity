"""The web service: FastAPI routes for finding similar products."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from product_similarity.cache import SearchCache
from product_similarity.hybrid import MODES
from product_similarity.metrics import Metrics
from product_similarity.scoring import (
    ModeNotAvailableError,
    ProductNotFoundError,
    SimilarityEngine,
)
from product_similarity.service import default_engine
from product_similarity.settings import get_settings
from product_similarity.weights import (
    ADVANCED_NAMES,
    SIMPLE_NAMES,
    WeightsError,
    parse_weights,
    tuned_weights,
    weights_key,
)

logger = logging.getLogger("product_similarity.api")

# Every ID in the dataset is 32 lowercase hexadecimal characters.
ID_PATTERN = r"^[0-9a-f]{32}$"

# Only answers of up to this many products are cached, so that a few huge answers cannot fill
# the memory.
MAX_CACHED_RESULTS = 100


class ApiError(Exception):
    """An error with an HTTP code, a short code and a sentence for the caller."""

    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


def error_body(code: str, message: str, **extra) -> dict:
    return {"error": code, "message": message, **extra}


def create_app(
    load_engine: Callable[[], SimilarityEngine] = default_engine,
    cache_size: int | None = None,
) -> FastAPI:
    """Build the web app. `load_engine` runs once at startup (tests pass a small one).

    `cache_size` is how many answers the in-process cache keeps (0 turns it off); the default
    comes from the settings.
    """
    settings = get_settings()
    cache = SearchCache(settings.cache_size if cache_size is None else cache_size)
    max_results = settings.max_results
    metrics = Metrics(
        cache_entries=lambda: cache.stats()["entries"],
        is_ready=lambda: app.state.engine is not None,
    )

    def ranked_answer(
        engine: SimilarityEngine,
        product_id: str,
        num_similar: int,
        mode: str | None,
        weights: dict[str, float] | None,
    ) -> tuple[tuple[tuple[str, float], ...], str, bool]:
        """The best matches as (ID, score) pairs, the mode that ran, and whether the answer
        came from the cache.

        The question that the cache remembers is the product, the number of results, the mode
        and the weights: another mode or other weights is another question.
        """
        row = engine.resolve(product_id)  # an old ID and the product that was kept share an entry
        key = (row, num_similar, mode or getattr(engine, "mode", None), weights_key(weights))
        cacheable = num_similar <= MAX_CACHED_RESULTS
        if cacheable:
            saved = cache.get(key)
            if saved is not None:
                metrics.searches.labels(mode=saved[1], cache="hit").inc()
                return saved[0], saved[1], True
        started = time.perf_counter()
        try:
            ranked, used = engine.search(product_id, num_similar, mode, weights)
        except ModeNotAvailableError as problem:
            raise ApiError(422, "mode_not_available", str(problem)) from None
        metrics.search_seconds.labels(mode=used).observe(time.perf_counter() - started)
        metrics.searches.labels(mode=used, cache="miss").inc()
        ranked = tuple(ranked)
        if cacheable:
            cache.put(key, (ranked, used))
        return ranked, used, False

    def read_options(
        engine: SimilarityEngine, mode: str | None, weights: str | None
    ) -> tuple[str | None, dict[str, float] | None]:
        """Check the mode and the weights of a request."""
        if mode is not None and mode not in MODES:
            raise ApiError(422, "invalid_mode", f"mode must be one of: {', '.join(MODES)}")
        parsed = None
        if weights is not None:
            try:
                parsed = parse_weights(weights)
            except WeightsError as problem:
                raise ApiError(422, "invalid_weights", str(problem)) from None
        if (mode is not None or parsed is not None) and not engine.supports_options:
            raise ApiError(
                422,
                "option_not_available",
                "this service runs the baseline search, which has no mode or weights option",
            )
        return mode, parsed

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.engine = None
        try:
            app.state.engine = load_engine()
        except Exception:
            logger.exception("Could not load the similarity engine")
            if get_settings().require_vectors:
                raise  # stop the service: it must not run without the final search
        yield

    app = FastAPI(title="Product similarity", lifespan=lifespan)
    app.state.cache = cache
    app.state.metrics = metrics

    @app.middleware("http")
    async def count_requests(request: Request, call_next):
        """Count and time every request, except the reads of `/metrics` itself."""
        if request.url.path == "/metrics":
            return await call_next(request)
        started = time.perf_counter()
        status = 500  # an error that no handler caught is a 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            route = request.scope.get("route")
            endpoint = getattr(route, "path", None) or "unmatched"
            metrics.requests.labels(endpoint=endpoint, status=str(status)).inc()
            metrics.request_seconds.labels(endpoint=endpoint).observe(time.perf_counter() - started)

    def get_engine(request: Request) -> SimilarityEngine:
        engine = request.app.state.engine
        if engine is None:
            raise ApiError(503, "not_ready", "the service is not ready yet")
        return engine

    @app.get("/find_similar_products", response_model=list[str])
    def find_similar_products(
        request: Request,
        response: Response,
        product_id: str = Query(..., pattern=ID_PATTERN),
        num_similar: int = Query(..., ge=1, le=max_results),
        mode: Optional[str] = Query(None),  # noqa: UP045 - FastAPI reads this on Python 3.9
        weights: Optional[str] = Query(None),  # noqa: UP045
    ) -> list[str]:
        engine = get_engine(request)
        mode, parsed = read_options(engine, mode, weights)
        try:
            ranked, used, cached = ranked_answer(engine, product_id, num_similar, mode, parsed)
        except ProductNotFoundError:
            raise ApiError(
                404, "product_not_found", f"no product has the ID {product_id}"
            ) from None
        response.headers["X-Cache"] = "hit" if cached else "miss"
        response.headers["X-Search-Mode"] = used
        return [uid for uid, _ in ranked]

    @app.get("/find_similar_products/details")
    def find_similar_products_details(
        request: Request,
        product_id: str = Query(..., pattern=ID_PATTERN),
        num_similar: int = Query(..., ge=1, le=max_results),
        mode: Optional[str] = Query(None),  # noqa: UP045 - FastAPI reads this on Python 3.9
        weights: Optional[str] = Query(None),  # noqa: UP045
    ) -> dict:
        """The same search as `/find_similar_products`, with details for display.

        Returns the question that was asked, the query product, each result with its
        title, price and score, the mode that ran and the weights used, whether the answer
        came from the cache, and how long the search took in milliseconds.
        """
        engine = get_engine(request)
        mode, parsed = read_options(engine, mode, weights)
        started = time.perf_counter()
        try:
            ranked, used, cached = ranked_answer(engine, product_id, num_similar, mode, parsed)
            found = engine.describe_ranked(product_id, ranked)
        except ProductNotFoundError:
            raise ApiError(
                404, "product_not_found", f"no product has the ID {product_id}"
            ) from None
        elapsed_ms = (time.perf_counter() - started) * 1000
        asked = mode or getattr(engine, "mode", None)
        body = {
            "question": {
                "product_id": product_id,
                "num_similar": num_similar,
                "mode": asked,
                "weights": weights,
            },
            **found,
            "mode": used,
            "weights": "tuned" if parsed is None else parsed,
            "cached": cached,
            "response_time_ms": round(elapsed_ms, 2),
        }
        if asked is not None and used != asked:
            body["mode_note"] = (
                f"the {asked} mode needs a sentence or picture weight above 0, "
                f"so the {used} mode ran"
            )
        return body

    @app.get("/weights")
    def weights_and_modes(request: Request) -> dict:
        """The tuned weights, the modes and the limits, for a page that sets them."""
        engine = get_engine(request)
        return {
            "default_mode": getattr(engine, "mode", None),
            "modes": list(MODES) if engine.supports_options else [],
            "fast_ready": bool(getattr(engine, "fast_ready", False)),
            "max_results": max_results,
            "weight_range": {"min": 0, "max": 1},
            "simple_names": list(SIMPLE_NAMES),
            "advanced_names": list(ADVANCED_NAMES),
            "tuned_weights": tuned_weights(),
        }

    @app.get("/cache")
    def cache_stats() -> dict:
        """How the in-process cache is doing: entries, hits, misses and the hit rate."""
        return cache.stats()

    @app.get("/metrics", include_in_schema=False)
    def metrics_page() -> Response:
        """The numbers for Prometheus: requests, times, cache hits, errors and memory."""
        body, content_type = metrics.render()
        return Response(body, media_type=content_type)

    @app.get("/live")
    def live() -> dict:
        """The process is running."""
        return {"status": "ok"}

    @app.get("/ready")
    def ready(request: Request) -> JSONResponse:
        """The data is loaded and the service can answer."""
        engine = request.app.state.engine
        if engine is None:
            return JSONResponse(error_body("not_ready", "the service is not ready yet"), 503)
        return JSONResponse(
            {
                "status": "ready",
                "products": len(engine.products),
                "search": engine.search_name,
                "modes": list(MODES) if engine.supports_options else [],
                "fast_ready": bool(getattr(engine, "fast_ready", False)),
            }
        )

    # ---- every error has the same shape: {"error": code, "message": sentence} ----

    @app.exception_handler(ApiError)
    async def api_error(request: Request, exc: ApiError) -> JSONResponse:
        metrics.errors.labels(error=exc.code).inc()
        return JSONResponse(error_body(exc.code, exc.message), status_code=exc.status)

    @app.exception_handler(RequestValidationError)
    async def invalid_input(request: Request, exc: RequestValidationError) -> JSONResponse:
        metrics.errors.labels(error="invalid_input").inc()
        problems = []
        for item in exc.errors():
            field = str(item["loc"][-1])
            kind = item["type"]
            if kind == "missing":
                problems.append(f"{field} is required")
            elif field == "product_id":
                problems.append("product_id must be 32 lowercase hexadecimal characters")
            elif field == "num_similar":
                problems.append(f"num_similar must be a whole number from 1 to {max_results}")
            else:
                problems.append(f"{field}: {item['msg']}")
        return JSONResponse(
            error_body("invalid_input", "; ".join(problems), problems=problems), status_code=422
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_problem(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        codes = {404: "not_found", 405: "method_not_allowed"}
        code = codes.get(exc.status_code, "error")
        metrics.errors.labels(error=code).inc()
        return JSONResponse(
            error_body(code, str(exc.detail)),
            status_code=exc.status_code,
        )

    @app.exception_handler(Exception)
    async def unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        # The full error goes to the log; the caller only sees a safe message.
        logger.exception("Unhandled error on %s", request.url.path)
        metrics.errors.labels(error="internal_error").inc()
        return JSONResponse(
            error_body("internal_error", "something went wrong on the server"), status_code=500
        )

    return app
