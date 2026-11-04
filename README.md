# Product Similarity Search

A multimodal product similarity engine for the Amazon Fashion 2020 dataset (30,000 products). Given a product's `uniq_id`, it returns the most similar products using structured attributes, product titles and product images.

The project is built for the SAP technical exercise. The original task description is in the [assignment repository](https://github.com/punitmishra/sap-cxii-tech-ex-01).

> **Status: work in progress.** The table below lists only what is implemented and tested today.

## Status

| Area | State |
|---|---|
| Data loading and validation (reads the archive in place, integrity check) | Done, tested |
| Exploratory data analysis notebook (6 explained charts, 24 reproduced figures) | Done |
| Preprocessing pipeline (cleaning, repeated products, derived fields) | Done, tested |
| Baseline similarity function `find_similar_products` (title and attributes, untuned weights) | Done, tested |
| Evaluation: 50 test products, blind rating page, scoring, MLflow tracking | Done for the 30 development products; the 20 final products stay locked |
| Image-aware similarity | Planned (the engine already has a slot for image features) |
| FastAPI service | Done, tested |
| Docker image | Done: built and tested in CI (starts, answers requests, runs as a non-root user) |
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

To open the notebooks:

```bash
jupyter lab notebooks/
```

## Run the service

```bash
PYTHONPATH=src uvicorn app:app --port 8000
```

```bash
curl "http://localhost:8000/find_similar_products?product_id=26d41bdc1495de290bc8e6062d927729&num_similar=5"
```

The response is a JSON list of product IDs. An unknown ID returns 404, invalid input returns 422, and `/live` and `/ready` report service health. `/find_similar_products/details` returns the same search with titles, prices, scores and timing.

## Example

The required endpoint returns a plain JSON list of product IDs:

```bash
curl "http://localhost:8000/find_similar_products?product_id=26d41bdc1495de290bc8e6062d927729&num_similar=3"
```

```json
["46b2e244b4fd48e0a66f74a6031b10a4", "ec83e50a5c1e1b4f4a669e07266cdc3c", "2572b4aa6bf45ad494f75556d597e4cb"]
```

For display, `/find_similar_products/details` returns the same results with the question asked, readable product details, scores and the response time:

```bash
curl "http://localhost:8000/find_similar_products/details?product_id=26d41bdc1495de290bc8e6062d927729&num_similar=3"
```

```json
{
  "question": {
    "product_id": "26d41bdc1495de290bc8e6062d927729",
    "num_similar": 3
  },
  "query": {
    "id": "26d41bdc1495de290bc8e6062d927729",
    "title": "LA' Facon Cotton Kalamkari Handblock Saree Blouse Fabric 100 cms Black Base Dancers (Cotton)",
    "brand": "LA' Facon",
    "category": "WomensKurtasKurtis",
    "audience": "women",
    "price": 200.0,
    "rating": 5.0,
    "image_url": "https://images-na.ssl-images-amazon.com/images/I/51Wj2WownyL.jpg"
  },
  "results": [
    {
      "id": "46b2e244b4fd48e0a66f74a6031b10a4",
      "title": "Janasya Women's Black Cotton Anarkali Printed Kurta",
      "brand": null,
      "category": "WomensKurtasKurtis",
      "audience": "women",
      "price": null,
      "rating": 5.0,
      "image_url": "https://images-na.ssl-images-amazon.com/images/I/415FAR7SieL.jpg",
      "score": 0.571378
    },
    {
      "id": "ec83e50a5c1e1b4f4a669e07266cdc3c",
      "title": "Janasya Women's Black Cotton A-Line Printed Kurta",
      "brand": null,
      "category": "WomensKurtasKurtis",
      "audience": "women",
      "price": 339.0,
      "rating": 3.0,
      "image_url": "https://images-na.ssl-images-amazon.com/images/I/41eWb3tffyL.jpg",
      "score": 0.567306
    },
    {
      "id": "2572b4aa6bf45ad494f75556d597e4cb",
      "title": "Janasya Women's Black Pure Cotton Kurta",
      "brand": null,
      "category": "WomensKurtasKurtis",
      "audience": "women",
      "price": 549.0,
      "rating": 4.4,
      "image_url": "https://images-na.ssl-images-amazon.com/images/I/410E0-LxWAL.jpg",
      "score": 0.562516
    }
  ],
  "response_time_ms": 8.47
}
```

The query is a saree blouse fabric that the dataset files under the kurta category, so every result is a kurta. Category-based matching carries over that label. The response time varies by machine.

With Docker:

```bash
docker build -t product-similarity .
docker run -p 8000:8000 product-similarity
```

## Analysis notebooks

| Notebook | What it does | Main findings |
|---|---|---|
| `01_data_exploration` | Loads the 30,000 products, checks the file is the original, and charts each attribute. It recomputes 24 figures to confirm them | Name, category and image are almost always present, and brand and price mostly. Weight is mostly a placeholder, colour is a list of options, and ratings often have no reviews. About a third of products share a title, and 471 rows are repeats |
| `02_preprocessing` | Cleans the data step by step with before-and-after charts: repeated products, weight, brands, categories, colours, audience, reviews and price | One record per product, with every ID still valid. Missing categories are filled from the products on the same shelf (about 97% matched in a hide-and-guess check). Guessing brands from titles is built but switched off |
| `03_baseline_examples` | Runs the first `find_similar_products` on five real queries, explains each score, times it, and compares it with simpler methods | Results come back in about 6 ms. Title similarity (TF-IDF) and attribute similarity give very different lists. Two weaknesses are noted: near-copies in the results, and products with little data scoring too high |
| `04_evaluation_baseline` | Scores the three methods against the ratings with nDCG@10 and precision@10, with 95% ranges | The title plus attributes beats either alone, and both gaps are larger than luck. Title only versus attributes only cannot be told apart with 30 products |

Each notebook ends with a conclusion in plain words. The logic lives in the Python package, and the notebooks only call it and show results.

## How the results were evaluated

The data has no answer key saying which products are similar, so I made one. A method is judged by how well its top 10 results match my ratings.

1. **Test products.** 50 products chosen with a fixed seed from seven groups (common and rare categories, sparse and complete data, repeated titles, low and high prices). 30 are for development, and 20 are **locked** and rated only once at the end.
2. **Blind pooling.** For each development product, the top 10 results of three methods (baseline, title only, attributes only) were mixed into one shuffled list. The page never says which method found a result.
3. **Rating.** Each result gets 0 (not a match), 1 (acceptable) or 2 (strong match), judged as a shopper looking for a substitute. The rules and worked examples are in [`evaluation/rubric.md`](evaluation/rubric.md). The 729 results were rated in a browser page that shows the picture, title, brand, category and price:

![The rating page: each result is a card with a picture and 0, 1, 2 buttons](evaluation/rating_page.png)

4. **Scoring.** Each method's top 10 is scored with **nDCG@10** (strong matches near the top count most) and **precision@10**, with 95% ranges from resampling the products.
5. **Tracking.** Every run is recorded in MLflow with its settings, scores, data fingerprints and limitations, and the results are exported to [`results/`](results/).

### First results (30 development products)

| Method | nDCG@10 | 95% range | Precision@10 |
|---|---|---|---|
| **Baseline** (title and attributes) | **0.72** | 0.64 to 0.81 | 0.74 |
| Title only | 0.57 | 0.48 to 0.67 | 0.55 |
| Attributes only | 0.48 | 0.38 to 0.56 | 0.62 |

The baseline beats each part alone by more than luck. Title only and attributes only cannot be told apart with 30 products. The full table, with limitations, is in [`results/dev_evaluation_report.md`](results/dev_evaluation_report.md).

### Recorded in MLflow

Every scoring run is saved in [MLflow](https://mlflow.org), which keeps a log of experiments so each result can be traced and compared later. For each method, a run records:

- **Settings:** the method and its weights, the number of results scored, and the number of products.
- **Scores:** nDCG@10 and precision@10, each with its 95% range.
- **Fingerprints:** short hashes of the dataset and the ratings file, plus the git commit, so the result points to the exact files and code that produced it.
- **Notes:** the limitations, saved as tags on the run, and the exported result files.

MLflow is used only for tracking. The web service does not use it, and it is not in the Docker image. This is MLflow's comparison page for the three methods:

![MLflow comparing the baseline, title_only and attributes_only runs: settings, weights, fingerprints and scores side by side](evaluation/mlflow_runs.png)

### Reproduce it

Rate again (this replaces the rating page, so keep your downloaded ratings file):

```bash
PYTHONPATH=src python -m product_similarity.judging
```

Open `evaluation/judgments/dev_review.html` in a browser, rate the cards, press **Download my ratings (CSV)**, and save the file as `evaluation/judgments/dev_grades.csv`. Then score and record the run:

```bash
pip install -e ".[experiments]"
python experiments/run_evaluation.py
```

View the recorded runs in MLflow:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5055
```

Open http://127.0.0.1:5055, click **Model training** at the top left, and open the `dev-ratings` experiment.

### Limits

- One person rated, so the labels reflect one judgment.
- Only 30 products were rated, so small differences cannot be told apart.

## Project layout

```text
app.py                    Entry point for uvicorn
src/product_similarity/   Python package
  data.py                 Reads the archive, checks its hash, keeps one record per product
  validation.py           Reports data problems without changing the data
  preprocessing.py        Cleaning and derived fields
  scoring.py              Per-attribute similarity scores and the weighted mix
  ranking.py              Fixed ordering of results
  service.py              The public find_similar_products function
  api.py                  Web routes (FastAPI)
  judging.py              Chooses the test products and builds the blind rating sheet
  review_page.py          The browser page used to rate results
  evaluation.py           nDCG, precision and confidence ranges
  tracking.py             Optional MLflow logging (the service never uses it)
  settings.py             File locations, overridable with environment variables
experiments/              Scripts that score methods and record runs
evaluation/               Rating rules, test products and ratings
results/                  Exported scores and the evaluation report
tests/                    Unit and integration tests
notebooks/                Analysis notebooks (import logic from the package)
data/archive.zip          Dataset supplied with the exercise
.github/workflows/        Continuous integration
```

## License

Apache-2.0, see [LICENSE](LICENSE).
