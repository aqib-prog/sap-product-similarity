"""The web service: FastAPI routes for finding similar products."""

from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from product_similarity.scoring import ProductNotFoundError, SimilarityEngine
from product_similarity.service import default_engine

logger = logging.getLogger("product_similarity.api")

# Every ID in the dataset is 32 lowercase hexadecimal characters.
ID_PATTERN = r"^[0-9a-f]{32}$"


def create_app(load_engine: Callable[[], SimilarityEngine] = default_engine) -> FastAPI:
    """Build the web app. `load_engine` runs once at startup (tests pass a small one)."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.engine = None
        try:
            app.state.engine = load_engine()
        except Exception:
            logger.exception("Could not load the similarity engine")
        yield

    app = FastAPI(title="Product similarity", lifespan=lifespan)

    def get_engine(request: Request) -> SimilarityEngine:
        engine = request.app.state.engine
        if engine is None:
            raise HTTPException(status_code=503, detail="Service not ready")
        return engine

    @app.get("/find_similar_products", response_model=list[str])
    def find_similar_products(
        request: Request,
        product_id: str = Query(..., pattern=ID_PATTERN),
        num_similar: int = Query(..., ge=1),
    ) -> list[str]:
        engine = get_engine(request)
        try:
            return engine.find_similar(product_id, num_similar)
        except ProductNotFoundError:
            raise HTTPException(status_code=404, detail="Product not found") from None

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
        return JSONResponse({"status": "ready", "products": len(engine.products)})

    @app.exception_handler(Exception)
    async def unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        # The full error goes to the log; the caller only sees a safe message.
        logger.exception("Unhandled error on %s", request.url.path)
        return JSONResponse({"detail": "Internal server error"}, status_code=500)

    return app
