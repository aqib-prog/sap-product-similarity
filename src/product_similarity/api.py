"""The web service: FastAPI routes for finding similar products."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse

from product_similarity.cache import SearchCache
from product_similarity.scoring import ProductNotFoundError, SimilarityEngine
from product_similarity.service import default_engine
from product_similarity.settings import get_settings

logger = logging.getLogger("product_similarity.api")

# Every ID in the dataset is 32 lowercase hexadecimal characters.
ID_PATTERN = r"^[0-9a-f]{32}$"

# Only answers of up to this many products are cached, so that a few huge answers cannot fill
# the memory.
MAX_CACHED_RESULTS = 100


def create_app(
    load_engine: Callable[[], SimilarityEngine] = default_engine,
    cache_size: int | None = None,
) -> FastAPI:
    """Build the web app. `load_engine` runs once at startup (tests pass a small one).

    `cache_size` is how many answers the in-process cache keeps (0 turns it off); the default
    comes from the settings.
    """
    cache = SearchCache(get_settings().cache_size if cache_size is None else cache_size)

    def ranked_answer(
        engine: SimilarityEngine, product_id: str, num_similar: int
    ) -> tuple[tuple[tuple[str, float], ...], bool]:
        """The best matches as (ID, score) pairs, and whether they came from the cache."""
        key = (
            engine.resolve(product_id),
            num_similar,
        )  # an old ID and its kept product share an entry
        cacheable = num_similar <= MAX_CACHED_RESULTS
        if cacheable:
            saved = cache.get(key)
            if saved is not None:
                return saved, True
        ranked = tuple(engine.find_similar_with_scores(product_id, num_similar))
        if cacheable:
            cache.put(key, ranked)
        return ranked, False

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

    def get_engine(request: Request) -> SimilarityEngine:
        engine = request.app.state.engine
        if engine is None:
            raise HTTPException(status_code=503, detail="Service not ready")
        return engine

    @app.get("/find_similar_products", response_model=list[str])
    def find_similar_products(
        request: Request,
        response: Response,
        product_id: str = Query(..., pattern=ID_PATTERN),
        num_similar: int = Query(..., ge=1),
    ) -> list[str]:
        engine = get_engine(request)
        try:
            ranked, cached = ranked_answer(engine, product_id, num_similar)
            response.headers["X-Cache"] = "hit" if cached else "miss"
            return [uid for uid, _ in ranked]
        except ProductNotFoundError:
            raise HTTPException(status_code=404, detail="Product not found") from None

    @app.get("/find_similar_products/details")
    def find_similar_products_details(
        request: Request,
        product_id: str = Query(..., pattern=ID_PATTERN),
        num_similar: int = Query(..., ge=1),
    ) -> dict:
        """The same search as `/find_similar_products`, with details for display.

        Returns the question that was asked, the query product, each result with its
        title, price and score, whether the answer came from the cache, and how long the
        search took in milliseconds.
        """
        engine = get_engine(request)
        started = time.perf_counter()
        try:
            ranked, cached = ranked_answer(engine, product_id, num_similar)
            found = engine.describe_ranked(product_id, ranked)
        except ProductNotFoundError:
            raise HTTPException(status_code=404, detail="Product not found") from None
        elapsed_ms = (time.perf_counter() - started) * 1000
        return {
            "question": {"product_id": product_id, "num_similar": num_similar},
            **found,
            "cached": cached,
            "response_time_ms": round(elapsed_ms, 2),
        }

    @app.get("/cache")
    def cache_stats() -> dict:
        """How the in-process cache is doing: entries, hits, misses and the hit rate."""
        return cache.stats()

    @app.get("/live")
    def live() -> dict:
        """The process is running."""
        return {"status": "ok"}

    @app.get("/ready")
    def ready(request: Request) -> JSONResponse:
        """The data is loaded and the service can answer."""
        engine = request.app.state.engine
        if engine is None:
            return JSONResponse({"status": "not ready"}, status_code=503)
        return JSONResponse(
            {
                "status": "ready",
                "products": len(engine.products),
                "search": engine.search_name,
            }
        )

    @app.exception_handler(Exception)
    async def unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        # The full error goes to the log; the caller only sees a safe message.
        logger.exception("Unhandled error on %s", request.url.path)
        return JSONResponse({"detail": "Internal server error"}, status_code=500)

    return app
