"""Entry point for `uvicorn app:app`. The data loads when the server starts."""

from product_similarity.api import create_app

app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
