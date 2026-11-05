# Product Similarity Search

A multimodal product similarity engine for the Amazon Fashion 2020 dataset (30,000 products). Given a product's `uniq_id`, it returns the most similar products using structured attributes, product titles and product images.

The project is built for the SAP technical exercise. The original task description is in the [assignment repository](https://github.com/punitmishra/sap-cxii-tech-ex-01).

> **Status: work in progress.** I am building this one step at a time. This page describes what is done and tested today, what I found, and what comes next.

## Status

| Area | State |
|---|---|
| Data loading and validation (reads the archive in place, integrity check) | Done, tested |
| Exploratory data analysis notebook (6 explained charts, 24 reproduced figures) | Done |
| Preprocessing pipeline (cleaning, repeated products, derived fields) | Done, tested |
| Baseline similarity function `find_similar_products` (title and attributes, untuned weights) | Done, tested |
| Evaluation: 50 test products, blind rating page, scoring, MLflow tracking | Done for the 30 development products; the 20 final products stay locked |
| Image sample check (100 images) | Done: 99 of 100 downloaded |
| Image similarity with ResNet and CLIP vectors | Tested on 708 development pictures (CLIP ties the text baseline); the full test on all 30,000 pictures is next |
| FastAPI service | Done, tested |
| Docker image | Done: built and tested in CI (starts, answers requests, runs as a non-root user) |
| CI: lint, tests, Docker build and smoke test on every push | Done, passing |
| Vector search benchmarks (FAISS Flat and HNSW, Annoy) | Planned |
| Kubernetes manifests, automatic deployment, monitoring | Planned |

## What I did so far

I work in small steps. After each step I check the result, and I write down what I found. I build on a `dev` branch and merge into `main` with a pull request.

1. **Looked at the data.** I read all 30,000 products and checked the columns, missing values, repeated products and prices. ([notebook 01](notebooks/01_data_exploration.ipynb))
2. **Cleaned the data.** One record per product, with every ID still valid. ([notebook 02](notebooks/02_preprocessing.ipynb))
3. **Built a first similarity search.** It compares titles and attributes and returns product IDs, with a FastAPI service and a Docker image on top. ([notebook 03](notebooks/03_baseline_examples.ipynb))
4. **Made a fair way to measure it.** There is no answer key, so I made one from my own ratings, and I record every run in MLflow. ([notebook 04](notebooks/04_evaluation_baseline.ipynb))
5. **Checked the pictures.** I tested 100 sample images before any bulk download.
6. **Tried pictures for similarity.** I turned pictures into vectors with ResNet and CLIP and scored them with the same ratings. ([notebook 05](notebooks/05_image_similarity.ipynb))
7. **Automated the checks (CI/CD).** Every push runs the linter, the tests and a Docker smoke test.

The sections below explain each step: what I did, why, and what I found.

## Getting started

Requires Python 3.9 or newer (CI runs Python 3.10).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,notebooks]"
make test
make lint
```

Optional extras: `experiments` (MLflow) and `images` (ResNet and CLIP). Neither is needed to run the service or the tests.

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

## The data

### What is in it

`data/archive.zip` holds one line-delimited JSON file with 30,000 products. It is read directly from the archive and never modified or extracted. When it loads, the code first checks the file's SHA-256 fingerprint, so I know it is exactly the file SAP supplied. A few fields that are stored as text but are really numbers (price, rating, review counts) are converted while reading. Everything else is left as it came, and cleaning happens later.

### What I found

The exploration notebook reproduces these findings:

- Product IDs are unique, and 471 rows are recrawls of the same ASIN.
- Only about 21% of products have a real weight (the rest carry the placeholder `999999999`), and about 20% have a colour value.
- Price is strongly right-skewed (median 590, mean 862 INR).
- Ratings are unreliable on their own: 6,611 products have a rating but no reviews.
- About a third of products share an exact title with another product.
- 99.99% of products have image URLs.

### Why it matters

Several attributes are often missing, so a similarity search cannot rely on one field. This is why the first search skips missing attributes instead of guessing them (see below).

## Cleaning the data

The cleaning pipeline works step by step, with before-and-after charts in [notebook 02](notebooks/02_preprocessing.ipynb):

- **Repeated products.** The 471 repeated rows are merged: I keep the newest crawl, and the lowest ID if they tie. All 30,000 original IDs still resolve to a product, so asking about an old ID still works.
- **Titles and brands.** Broken text characters are repaired, and brand spellings are tidied into one form. Guessing a missing brand from the title is built but switched off, because I could not show that it helps.
- **Weight.** The placeholder value is treated as missing, not as a real weight.
- **Categories.** A missing category is filled from products on the same shelf. In a hide-and-guess check, about 97% matched.
- **Other fields.** Audience (men, women, kids) is derived from the title and category, colours are extracted from the text, and price and review fields are cleaned.

## How the first similarity search works

`find_similar_products(product_id, num_similar)` scores one product against all the others and returns the best matches. Each comparison gives a score between 0 and 1, and the scores are mixed with weights.

### Titles

Titles are turned into sparse vectors with TF-IDF, and two titles are compared with cosine similarity. Words that are rare across the catalogue count more than common ones.

### Attributes

Each attribute is compared directly:

- **Category, brand, audience:** same or different.
- **Price and weight:** closeness on a log scale, so a 10 times difference scores 0.
- **Rating:** closeness in stars, reaching 0 at 4 stars apart.
- **Colour:** how much the two colour lists overlap (Jaccard).

### Mixing the scores

| Part | Weight |
|---|---|
| Title | 0.35 |
| Category | 0.20 |
| Brand | 0.10 |
| Audience | 0.10 |
| Price | 0.10 |
| Colour | 0.05 |
| Weight | 0.05 |
| Rating | 0.05 |
| Image | 0 (off until picture vectors are added) |

These starting weights are **not tuned**. They are a placeholder until experiments choose better ones.

### Missing data

If either product lacks an attribute, that attribute is skipped and the remaining weights are rescaled so they still add up to 1. Missing data is never filled in with a guess.

### A fixed order

Scores are rounded to 6 decimals, and ties are broken by product ID, so the same question always gives the same list.

## The service and Docker

The service is a small FastAPI app. The engine is built once at startup (about 3 seconds), and then each request takes a few milliseconds. `num_similar` must be at least 1.

```bash
docker build -t product-similarity .
docker run -p 8000:8000 product-similarity
```

The Docker image is built on `python:3.10-slim`, installs only the pinned serving packages from `requirements.txt` (no MLflow, no notebooks, no image models), and runs as a non-root user.

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
| `05_image_similarity` | Turns pictures into vectors with ResNet and CLIP, searches among the 708 development pictures, and scores the results with the same ratings as the text methods | CLIP ties the text baseline (nDCG@10 0.695 against 0.681), and ResNet is clearly lower (0.569). Image search only looked at pictures the text methods had already picked, so the full test comes next |

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

### Recorded in MLflow (text methods)

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

## Image sample check

Before downloading images in bulk, I checked a sample of 100 (kept locally, never committed). Details are in [`results/image_sample_report.md`](results/image_sample_report.md).

- **First images are good product photos.** 99 of 100 downloaded, and the one failure is a link that no longer exists (error 404).
- **Pictures shared by many products are mostly size charts**, which would make unrelated products look alike. So the plan is to use the **first image only** and filter out shared pictures.
- **Polite downloading.** One request at a time, at most 4 per second, https only, one allowed host, a time limit and a size limit, and no redirects followed.
- **Full download:** about 29,000 pictures, roughly 860 MB. Not started.

## Image similarity (small test)

I turned the pictures of the 708 development products (the 30 test products and the results I had rated) into **vectors**, which are lists of numbers. Pictures that look alike get vectors that are close together. I used two ready-made models as they are, with no training: ResNet and CLIP. For each of the 30 test products, I found the 10 closest pictures, rated the new results myself, and scored everything with the same ratings.

| Method | nDCG@10 | Precision@10 |
|---|---|---|
| CLIP (pictures) | 0.695 | 0.74 |
| Text and attributes (my baseline) | 0.681 | 0.74 |
| ResNet (pictures) | 0.569 | 0.63 |
| Title only | 0.545 | 0.55 |
| Attributes only | 0.445 | 0.62 |

- CLIP is as good as my best text method. The gap is too small to call a winner.
- CLIP is clearly better than ResNet, and ResNet is worse than my text method.
- Pictures alone find good matches, so combining text and picture vectors is worth trying next.
- **Limits:** image search only looked at 708 pictures that the text methods had already picked, which helps it. There were 30 test products and one rater. The full table is in [`results/dev_image_evaluation_report.md`](results/dev_image_evaluation_report.md).

The runs are saved in MLflow. This is its comparison page for the two image methods:

![MLflow comparing the image_resnet and image_clip runs: nDCG@10 and precision@10 side by side](evaluation/mlflow_image_runs.png)

### Reproduce it

The pictures and vectors stay on my computer and are not in Git.

```bash
pip install -e ".[dev,experiments,images]"
python experiments/download_dev_images.py      # about 700 pictures, one request at a time
python experiments/extract_image_features.py   # ResNet and CLIP vectors
python experiments/run_image_search.py         # finds results, writes the rating page for new ones
```

Open `evaluation/judgments/dev_review_images.html`, rate the cards, press **Download my ratings (CSV)**, and save the file as `evaluation/judgments/dev_grades_images.csv`. Then score everything together:

```bash
python experiments/run_image_evaluation.py
```

## Tests and CI/CD

### Tests

There are 137 tests. The unit tests cover loading, cleaning, scoring, ranking, the service, the API, the rating tools, the scoring maths, MLflow logging and the image code. The integration tests run on the real dataset and on the real API.

### CI/CD with GitHub Actions

On every push and pull request, GitHub Actions runs two jobs:

1. **test:** installs the project, checks the code style with ruff, and runs all the tests.
2. **docker:** builds the image, starts the container, waits until it is ready, asks it for similar products, checks that an unknown ID returns 404, and checks that the container does not run as root.

The second job only starts if the first one passes. I merge into `main` through a pull request. Automatic deployment (for example to Kubernetes) is planned but not built yet.

## Limits so far

- One person rated the results, so the labels reflect one judgment.
- Only 30 development products have been used, so small differences cannot be told apart.
- Image search was only tested on 708 pictures that the text methods had already picked. That helps it, so the full test comes next.
- The 20 locked final products have not been rated or used. They are kept for the final test.
- The attribute weights are untuned starting values.

## What is next

- Download all the pictures (about 29,000) and test CLIP on the whole catalogue.
- Combine text and picture vectors, and check whether the mix beats each part.
- Compare fast search methods (FAISS, HNSW, Annoy) and test reducing the vector size with PCA.
- Add caching, Kubernetes files and automatic deployment.
- Rate the 20 locked products once, at the end, for the final result.

## Project layout

```text
app.py                      Entry point for uvicorn
Dockerfile                  Builds the service image (serving packages only, non-root user)
requirements.txt            Pinned packages used by the service and the Docker image
pyproject.toml              Package settings and the optional extras (dev, notebooks, experiments, images)
Makefile                    Shortcuts: make install, lint, format, test
src/product_similarity/     Python package
  data.py                   Reads the archive, checks its hash, keeps one record per product
  validation.py             Reports data problems without changing the data
  preprocessing.py          Cleaning and derived fields
  scoring.py                Per-attribute similarity scores and the weighted mix
  ranking.py                Fixed ordering of results
  service.py                The public find_similar_products function
  api.py                    Web routes (FastAPI)
  settings.py               File locations, overridable with environment variables
  judging.py                Chooses the test products and builds the blind rating sheet
  review_page.py            The browser page used to rate results
  evaluation.py             nDCG, precision and confidence ranges
  tracking.py               Optional MLflow logging (the service never uses it)
  images.py                 Safe image download, quality checks and sample selection
  image_features.py         Turns pictures into vectors with ResNet and CLIP (offline only)
experiments/                Scripts that I run by hand (nothing here is used by the service)
  run_evaluation.py         Scores the text methods against my ratings and records the runs
  check_images.py           Downloads and checks the 100-image sample
  image_grids.py            Makes contact sheets of the sample images (kept local)
  download_dev_images.py    Downloads the pictures of the development products
  extract_image_features.py Turns those pictures into ResNet and CLIP vectors
  run_image_search.py       Finds similar pictures and writes the rating page for new results
  run_image_evaluation.py   Scores text and image methods together and records the runs
evaluation/                 Test products, rating rules, rating pages and my ratings
  queries.json              The 50 test products (30 development, 20 locked)
  rubric.md                 The rules for rating 0, 1 or 2
  judgments/                Rating sheets, rating pages and my ratings (CSV)
  *.png                     Screenshots of the rating page and MLflow
results/                    Exported scores, reports and image check numbers
tests/                      137 tests
  unit/                     Fast tests for each module
  integration/              Tests on the real dataset and the real API
notebooks/                  Five analysis notebooks (they import logic from the package)
data/archive.zip            Dataset supplied with the exercise
.github/workflows/ci.yml    Continuous integration (lint, tests, Docker smoke test)
```

Not in Git: the downloaded pictures, the saved picture vectors and the local MLflow databases. They stay on my computer.

## License

Apache-2.0, see [LICENSE](LICENSE).
