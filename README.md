# Product Similarity Search

A multimodal product similarity engine for the Amazon Fashion 2020 dataset (30,000 products). Given a product's `uniq_id`, it returns the most similar products using structured attributes, product titles and product images.

The project is built for the SAP technical exercise. The original task description is in the [assignment repository](https://github.com/punitmishra/sap-cxii-tech-ex-01).

> **Status: work in progress.** The table below lists only what is implemented and tested today.

## Status

| Area | State |
|---|---|
| Data loading and validation (reads the archive in place, integrity check) | Done, tested |
| Exploratory data analysis notebook (6 explained charts, 24 reproduced figures) | Done |
| Preprocessing pipeline | In progress |
| Similarity function `find_similar_products` | Planned |
| Image-aware similarity | Planned |
| FastAPI service and Docker image | Planned |
| Vector search benchmarks (FAISS Flat and HNSW, Annoy) | Planned |
| Kubernetes manifests, CI/CD, monitoring | Planned |

## Dataset

`data/archive.zip` holds one line-delimited JSON file with 30,000 products. It is read directly from the archive and never modified. The exploration notebook reproduces these findings:

- Product IDs are unique, and 471 rows are recrawls of the same ASIN.
- Only about 21% of products have a real weight (the rest carry the placeholder `999999999`), and about 20% have a colour value.
- Price is strongly right-skewed (median 590, mean 862 INR).
- Ratings are unreliable on their own: 6,611 products have a rating but no reviews.
- About a third of products share an exact title with another product.
- 99.99% of products have image URLs.

## Getting started

Requires Python 3.9 or newer (CI runs Python 3.10).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,notebooks]"
make test
make lint
```

To open the exploration notebook:

```bash
jupyter lab notebooks/01_data_exploration.ipynb
```

## Project layout

```text
src/product_similarity/   Python package
  data.py                 Reads the archive, checks its hash, converts numbers
  validation.py           Reports data problems without changing the data
  settings.py             File locations, overridable with environment variables
tests/unit/               Unit tests
notebooks/                Analysis notebooks (import logic from the package)
data/archive.zip          Dataset supplied with the exercise
.github/workflows/        Continuous integration
```

## License

Apache-2.0, see [LICENSE](LICENSE).
