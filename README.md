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
| Picture download for the whole catalogue | Done: 29,107 pictures (134 broken links) |
| Fast vector search (FAISS HNSW and IVF, Annoy) | Done: HNSW is about 60 times faster than the exact search with 97% recall@10; the exact search is already fast at this size |
| Hybrid search (weights tuned by coordinate ascent, 5-fold cross-validation) | Done: the 4-weight hybrid scores highest (nDCG@10 0.806); the final test on the locked products is still to do |
| Sentence vectors on titles | Done: clearly better than TF-IDF on titles, about the same as my baseline; combining everything is next |
| Image similarity with CLIP vectors | Done on the whole catalogue: CLIP scores highest on the 30 development products; combining it with text is next |
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
7. **Tested pictures on the whole catalogue.** I downloaded about 29,000 pictures, turned them into CLIP vectors, rated the new results and scored them next to the text methods. ([notebook 06](notebooks/06_clip_whole_catalogue.ipynb))
8. **Tried a sentence model on the titles.** I turned all titles into vectors with a ready-made sentence model and scored it with my ratings next to TF-IDF and CLIP. ([notebook 07](notebooks/07_sentence_vectors_on_titles.ipynb))
9. **Mixed everything into one hybrid search.** I combined the title (TF-IDF and sentence vectors), the CLIP picture and the attributes into one score, tuned the weights with coordinate ascent and checked them with 5-fold cross-validation. ([notebook 08](notebooks/08_hybrid_search.ipynb))
10. **Made the search faster.** I timed the exact search, compared FAISS HNSW and IVF and Annoy with it, and used the fast index inside the hybrid. ([notebook 09](notebooks/09_fast_vector_search.ipynb))
11. **Automated the checks (CI/CD).** Every push runs the linter, the tests and a Docker smoke test.

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
| `06_clip_whole_catalogue` | Downloads the first picture of almost every product, turns them all into CLIP vectors, searches the whole catalogue, and scores the results with all my ratings next to the text methods | CLIP over the whole catalogue scores highest (nDCG@10 0.757). It is clearly better than title only and attributes only, and ahead of the text baseline (0.644) by a gap too small to be sure about with 30 products |
| `07_sentence_vectors_on_titles` | Turns every title into a vector with a ready-made sentence model, searches, and scores the results with all my ratings next to TF-IDF, attributes and CLIP | Sentence vectors (nDCG@10 0.662) are clearly better than TF-IDF on titles (0.500) and about the same as my baseline (0.637). CLIP is still highest (0.751), but its gap over sentence vectors could be luck |
| `08_hybrid_search` | Mixes the title (TF-IDF and sentence vectors), CLIP and the attributes into one score, tunes the weights with coordinate ascent, checks them with 5-fold cross-validation, and scores the result next to every single method | The 4-weight hybrid scores highest (nDCG@10 0.806, against 0.750 for CLIP and 0.637 for the baseline). Sentence vectors and pictures carry most of the weight. The gap over CLIP alone could be luck with 30 products |
| `09_fast_vector_search` | Times the exact search, compares FAISS HNSW and IVF and Annoy with it (time, memory, recall@10) and uses a fast index inside the hybrid | HNSW is about 60 times faster than the exact search and finds 97% of the exact top 10. The exact search is already about 1 ms at this size, so the gain matters for a bigger catalogue |

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
- **Pictures shared by several products** could have been size charts, which would make unrelated products look alike. The check on the whole catalogue (below) found almost none.
- **Polite downloading.** One request at a time, at most 4 per second, https only, one allowed host, a time limit and a size limit, and no redirects followed.
- **Full download:** done, see the section on the whole catalogue below.

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

## Image similarity over the whole catalogue

In the small test I only looked at 708 products. Here I downloaded the first picture of almost every product, turned them all into CLIP vectors, and searched the whole catalogue.

- **Download.** 29,107 pictures (763 MB), one request at a time and at most 4 per second. 134 links no longer exist (error 404). None of the 20 locked products were downloaded. The pictures stay on my computer.
- **Shared pictures.** 231 pictures are used by two or more products. I looked at all of them. 229 are normal product photos of the same item listed more than once, for example in other sizes or colours. 2 are not product photos (a shop logo and a sewing-pattern drawing), so I left them out. I found no size charts.
- **One copy per item.** Results that use the same picture, or a nearly identical one, count as the same item, and I show only one of them. Copies of the searched product are not shown. Every method gets the same rule, so the comparison stays fair.
- **Vectors.** CLIP turned all 29,107 pictures into 512 numbers each in under 8 minutes.
- **Rating.** For each of the 30 test products, CLIP found the 10 closest items. 44 of the 300 results already had my rating, and I rated the other 256 without seeing which method found them.

### Conclusion

**What I wanted to know:** do product pictures help me find similar products? This time I searched the whole catalogue.

**What I did.** I took about 29,000 product pictures and turned each one into a list of numbers with CLIP. Pictures that look alike get numbers that are close together. For my 30 test products I found the 10 most similar items. I rated the new results myself, without knowing which method found them. I used one set of my ratings to check every method, so the comparison is fair, and I show only one copy of each item.

| Method | nDCG@10 | Precision@10 |
|---|---|---|
| CLIP, whole catalogue (pictures) | 0.757 | 0.85 |
| Text and attributes (my baseline) | 0.644 | 0.74 |
| CLIP, first 708 pictures only | 0.648 | 0.74 |
| Title only | 0.507 | 0.54 |
| Attributes only | 0.427 | 0.62 |

Precision@10 is how many of the top 10 I rated 1 or 2.

**What I found**

- CLIP beats title only and attributes only. The gap is real.
- CLIP is ahead of my text baseline too, but with only 30 products I can't be sure it is really better.
- When CLIP searched the whole catalogue, it did better than when it searched only 708 pictures.
- Pictures and text both find good matches on their own, so I want to try using them together.

**What I can't say yet**

- I tested only 30 products, and I was the only person rating.
- I haven't touched my 20 final test products.

**What I do next.** I will mix text, CLIP and attributes into one hybrid search, then make it fast with an index.

The five runs are saved in MLflow. This is its comparison page:

![MLflow comparing the five methods: nDCG@10 and precision@10 side by side](evaluation/mlflow_full_image_runs.png)

The full table is in [`results/dev_full_image_evaluation_report.md`](results/dev_full_image_evaluation_report.md), and the walk-through is in [notebook 06](notebooks/06_clip_whole_catalogue.ipynb).

### Reproduce it

```bash
pip install -e ".[dev,experiments,images]"
python experiments/download_images.py             # about 29,000 pictures, one request at a time
python experiments/extract_all_image_features.py  # CLIP vectors for all pictures
python experiments/run_full_image_search.py       # finds results, writes the rating page
```

Open `evaluation/judgments/dev_review_full_images.html`, rate the cards, press **Download my ratings (CSV)**, and save the file as `evaluation/judgments/dev_grades_full_images.csv`. Then score everything together:

```bash
python experiments/run_full_image_evaluation.py
```

## Sentence vectors on titles

TF-IDF only matches words. Here I tried a model that understands the meaning of a title, so that "kurta" and "kurti" count as similar.

- **Model.** `all-MiniLM-L6-v2` from Hugging Face. It is small (about 90 MB), free to use (Apache-2.0 licence) and turns a sentence into 384 numbers. Its authors trained it on over a billion sentence pairs, such as questions with answers, so that pairs which belong together get close vectors. It is meant for search and finding similar sentences, and it cuts text longer than 256 word pieces. My titles are much shorter. It was not trained on shopping titles, so it may be weaker on brand names and size words. I used it as it is, with no training.
- **Vectors.** All 29,529 titles became vectors in 31.5 seconds, with no failures.
- **Search.** For each of the 30 test products the model found the 10 closest titles. The 20 locked products can never be returned. 153 of the 300 results already had my rating, and I rated the other 147 without seeing which method found them.

### Conclusion

**What I wanted to know:** does a model that understands the meaning of titles find better matches than TF-IDF, which only matches words?

**What I did.** I took a ready-made sentence model, `all-MiniLM-L6-v2`, and turned all 29,529 product titles into lists of numbers. Titles with similar meaning get numbers that are close together. I used the model as it is, with no training. For my 30 test products I found the 10 closest titles. I rated the new results myself, without knowing which method found them. I used one set of my ratings to check every method, so the comparison is fair, and I show only one copy of each item.

| Method | nDCG@10 | Precision@10 |
|---|---|---|
| CLIP, whole catalogue (pictures) | 0.751 | 0.85 |
| Sentence vectors (titles) | 0.662 | 0.76 |
| Title (TF-IDF) and attributes (my baseline) | 0.637 | 0.74 |
| TF-IDF, title only | 0.500 | 0.55 |
| Attributes only | 0.422 | 0.62 |

Precision@10 is how many of the top 10 I rated 1 or 2.

**What I found**

- Sentence vectors are clearly better than TF-IDF on titles. Understanding meaning helps more than matching words.
- Sentence vectors and my baseline score about the same. With only 30 products I can't say which is better.
- CLIP is still the highest, but its gap over sentence vectors is small enough to be luck.
- Pictures and titles each find good matches on their own, so I want to try using them together.

**What I can't say yet**

- I tested only 30 products, and I was the only person rating.
- The model was trained on questions, answers and captions, not on shopping titles, so it may miss brand names and size words.
- I haven't touched my 20 final test products.

**What I do next.** I will mix text, CLIP and attributes into one hybrid search, and tune the weights using my ratings.

The five runs are saved in MLflow. This is its comparison page:

![MLflow comparing the five methods: nDCG@10 and precision@10 side by side](evaluation/mlflow_text_vector_runs.png)

The full table is in [`results/dev_text_vector_evaluation_report.md`](results/dev_text_vector_evaluation_report.md), and the walk-through is in [notebook 07](notebooks/07_sentence_vectors_on_titles.ipynb).

### Reproduce it

```bash
pip install -e ".[dev,experiments,images,text]"
python experiments/extract_text_features.py     # downloads the model once, makes the title vectors
python experiments/run_text_vector_search.py    # finds results, writes the rating page
```

Open `evaluation/judgments/dev_review_text_vectors.html`, rate the cards, press **Download my ratings (CSV)**, and save the file as `evaluation/judgments/dev_grades_text_vectors.csv`. Then score everything together:

```bash
python experiments/run_text_vector_evaluation.py
```

## Hybrid search

So far I tested each part on its own. Here I mix them into one **hybrid search** and choose how much each part counts (its weight) from my ratings.

- **The parts.** The TF-IDF title, the sentence-vector title, the CLIP picture, category, brand, audience, price, colour, weight and rating. Each gives a score from 0 to 1 for a pair of products, and a weighted sum of the scores is the final score (a weighted linear combination, as used for hybrid retrieval by Bruch et al., 2023). If a value is missing, that part is skipped and the other weights are scaled up. The sentence and picture scores are cosine similarities shifted so that 0 means "as similar as two random products".
- **Choosing the weights: coordinate ascent.** It changes one weight at a time, keeps the value with the best nDCG@10 on my ratings, and repeats until a full round changes nothing (Metzler and Croft, 2007). I started it from five starting points and kept the best result.
- **Checking for luck: 5-fold cross-validation.** I tuned on 24 of my 30 development products and scored the 6 left out, five times. So every hybrid score below comes from products the weights had not seen (an out-of-fold score).
- **Two mixes.** One with 10 weights (one for each part), and one with 4 weights (TF-IDF title, sentence title, picture, and all attributes together).
- **First try.** The tuned weights did not beat my untuned baseline on unseen products (0.614 for 10 weights and 0.635 for 4 weights, against 0.637). The hybrid had found 107 new results that were not in my ratings yet, which made the tuning unfair.
- **Rating the new results.** I rated the 107 results myself on a blind page (shuffled for each product, no method names), and tuned again. The 20 locked products were never used.

### Results

| Method | nDCG@10 | Precision@10 |
|---|---|---|
| Hybrid, 4 weights | 0.806 | 0.87 |
| Hybrid, 10 weights | 0.764 | 0.82 |
| CLIP (pictures) | 0.750 | 0.85 |
| Sentence vectors (titles) | 0.659 | 0.76 |
| Title (TF-IDF) and attributes (my baseline) | 0.637 | 0.74 |
| TF-IDF, title only | 0.498 | 0.55 |
| Attributes only | 0.420 | 0.62 |

**Final weights of the 4-weight mix:** sentence title 46%, picture 29%, attributes together 23%, TF-IDF title 3%. When I remove one part at a time from the 10-weight mix (an ablation), the sentence vectors and the picture cost the most (about 0.165 of nDCG@10 each), and category is the most useful attribute.

### Conclusion

**What I wanted to know:** if I mix all the parts into one search, is it better than each part on its own?

**What I did.** I took ten parts: the TF-IDF title, the sentence title, the CLIP picture, category, brand, audience, price, colour, weight and rating. Each part scores a pair of products. I added the scores with weights (a weighted linear combination), and a weight says how much a part counts.

To choose the weights, I used **coordinate ascent**. I change one weight at a time, keep the change if nDCG@10 goes up, and repeat until nothing improves. I started from several starting points and kept the best.

To make sure the weights are not just lucky for my 30 products, I used **5-fold cross-validation**. I tuned the weights on 24 products and tested them on the 6 I had left out, five times. All the hybrid scores below come from products the weights had not seen.

I tried two mixes: one with 10 weights, and one with only 4 weights (TF-IDF title, sentence title, picture, and all attributes together).

My first try did not beat my baseline, because the hybrid found 107 new results that were not in my ratings yet. I rated those myself, without knowing which mix found them, and tuned again. I used one set of my ratings to check every method, so the comparison is fair, and I show only one copy of each item.

| Method | nDCG@10 | Precision@10 |
|---|---|---|
| Hybrid, 4 weights | 0.806 | 0.87 |
| Hybrid, 10 weights | 0.764 | 0.82 |
| CLIP (pictures) | 0.750 | 0.85 |
| Sentence vectors (titles) | 0.659 | 0.76 |
| Title (TF-IDF) and attributes (my baseline) | 0.637 | 0.74 |
| TF-IDF, title only | 0.498 | 0.55 |
| Attributes only | 0.420 | 0.62 |

Precision@10 is how many of the top 10 I rated 1 or 2.

**What I found**

- The 4-weight hybrid is clearly better than my baseline, the sentence vectors, TF-IDF and the attributes alone.
- It is ahead of CLIP, but with only 30 products I can't be sure it is really better.
- The mix with 4 weights did better than the one with 10, because fewer weights means less chance of fitting my 30 products by luck.
- Sentence vectors and pictures matter most. If I remove either one, the score drops by about 0.165. Category is the most useful attribute.

**What I can't say yet**

- I tested only 30 products, and I was the only person rating.
- I tuned the weights on these 30 products, so I will test them once on my 20 final products at the end.

**What I do next.** I will make the search fast with an index (FAISS, HNSW, Annoy) and test making the vectors smaller with PCA.

### Recorded in MLflow

The check for luck from the first try (untuned, 10 weights, 4 weights, on all 30 products and on unseen products):

![MLflow comparing the untuned, 10-weight and 4-weight mixes](evaluation/mlflow_hybrid_tuning_runs.png)

All seven methods side by side:

![MLflow comparing the seven methods: nDCG@10 and precision@10](evaluation/mlflow_hybrid_runs.png)

The full table is in [`results/dev_hybrid_evaluation_report.md`](results/dev_hybrid_evaluation_report.md), the weights and the cross-validation folds are in [`results/hybrid_tuning.json`](results/hybrid_tuning.json), and the walk-through is in [notebook 08](notebooks/08_hybrid_search.ipynb).

References: Metzler and Croft, [Linear feature-based models for information retrieval](https://link.springer.com/content/pdf/10.1007/s10791-006-9019-z.pdf) (2007); Bruch, Gai and Ingber, [An Analysis of Fusion Functions for Hybrid Retrieval](https://arxiv.org/abs/2210.11934) (2023).

### Reproduce it

```bash
pip install -e ".[dev,experiments,images,text]"
python experiments/tune_hybrid_weights.py     # coordinate ascent and 5-fold cross-validation
python experiments/run_hybrid_search.py       # finds results that have no rating, writes the rating page
```

Open `evaluation/judgments/dev_review_hybrid.html`, rate the cards, press **Download my ratings (CSV)**, and save the file as `evaluation/judgments/dev_grades_hybrid.csv`. Then tune again and score everything together:

```bash
python experiments/tune_hybrid_weights.py
python experiments/run_hybrid_evaluation.py
```

## Fast vector search

To find similar products I compare one product with all the others. That is fine for 29,500 products, but a bigger catalogue needs a faster way. Here I time the exact search first, then compare fast **approximate nearest-neighbour** indexes with it.

- **One rule for every method.** 1,000 random products, picked with seed 2020, never a locked product and never one of my 30 development products. Each product is searched 5 times and the fastest run is kept. The time is the median of the 1,000. Recall@10 is the average over the same 1,000 products. One thread, one query at a time.
- **Recall@10.** How many of the exact top 10 a fast method also finds. The exact top 10 comes from the brute-force search by cosine similarity. A different member of a tie is not counted as a miss. Recall@10 does not use my ratings. My ratings are used for nDCG@10.
- **The baseline.** The exact search takes about 0.85 ms for the sentence vectors and 1.5 ms for the picture vectors, among about 29,500 products. The record is [`results/ann_exact_baseline.json`](results/ann_exact_baseline.json).
- **The fast methods** (all settings are in [`results/ann_benchmark.json`](results/ann_benchmark.json)): FAISS Flat (exact), FAISS HNSW (a graph search, Malkov and Yashunin, 2018), FAISS IVF (a search over groups) and Annoy (random trees, from Spotify).
- **Inside the hybrid.** A fast index picks 200 candidates from the combined sentence and picture vectors, and the hybrid with 4 weights picks the final 10. The record is [`results/ann_hybrid_check.json`](results/ann_hybrid_check.json).

### Conclusion

**What I wanted to know:** can I make the similarity search faster without losing much quality, and does it matter at my size?

**What I did.** I used one rule for every method. I took 1,000 random products, picked with seed 2020 and never a locked product. For each product I ran the search 5 times and kept the fastest run, and I report the median time of the 1,000. Recall@10 is the average over the same 1,000 products. I counted how many of the exact top 10 each fast method found. The exact top 10 comes from the brute-force search, which compares a product with all the others by cosine similarity. I tried FAISS HNSW, FAISS IVF and Annoy on the picture vectors (CLIP) and on the sentence vectors. I recorded every setting in MLflow, one run for each. Then I used the fast index inside my hybrid search: the index picks 200 candidates, and the hybrid with 4 weights picks the final 10.

| Method (sentence vectors, 384 numbers) | Time per query | Memory | Recall@10 |
|---|---|---|---|
| Exact search (brute force) | 0.85 ms | 45.3 MB | 100% |
| FAISS HNSW | 0.014 ms | 49.6 MB | 97% |
| FAISS IVF | 0.11 ms | 45.8 MB | 98% |
| Annoy | 1.7 ms | 67.1 MB | 97% |

| Method (picture vectors, 512 numbers) | Time per query | Memory | Recall@10 |
|---|---|---|---|
| Exact search (brute force) | 1.5 ms | 60.1 MB | 100% |
| FAISS HNSW | 0.018 ms | 64.4 MB | 97% |
| FAISS IVF | 0.12 ms | 60.7 MB | 99% |
| Annoy | 1.4 ms | 104.9 MB | 97% |

For each fast method I show its fastest setting that still finds at least 95% of the exact top 10.

| Hybrid with the fast index | nDCG@10 | Top 10 shared with the exact hybrid |
|---|---|---|
| Exact hybrid (scores all products, 6.3 ms) | 0.808 | 100% |
| HNSW picks 200 candidates, then the hybrid (0.07 ms for the candidates) | 0.784 | 94% |
| Exact search picks 200 candidates, then the hybrid | 0.807 | 97% |

**What I found**

- The exact search is already fast at my size: about 1 ms for one product among 29,500, with 45 to 60 MB of memory.
- HNSW is about 60 times faster than the exact search and still finds 97% of the exact top 10. It needs about 10% more memory and 6 to 7 seconds to build. IVF is about 8 times faster with almost no extra memory. Annoy was not faster than the exact search at this size and needs more memory.
- Inside the hybrid, the fast index costs a little quality: nDCG@10 goes from 0.808 to 0.784 with HNSW. The cost comes from the candidates it misses, because exact candidates lose almost nothing (0.807).
- I chose HNSW (Malkov and Yashunin, 2018) because it gave the best speed for the recall.

**What I can't say yet**

- I tested only about 29,500 products. At this size the exact search is already fast, so the gain from a fast index will matter more for a much larger catalogue. I did not test that.
- The timings are from one laptop, one thread, one query at a time. I did not time the final reranking of the 200 candidates with the hybrid.

**What I do next.** I will test whether shrinking the vectors with PCA helps, for the exact search and for every fast method, and draw a t-SNE picture of the vectors.

### Recorded in MLflow

Every setting of every method is one run in the experiment `ann-benchmark` (50 runs). The ten runs tagged `headline` are the exact search and the fastest setting of each method that finds at least 95% of the exact top 10:

![MLflow comparing the exact search and the fast methods on the sentence and picture vectors](evaluation/mlflow_ann_benchmark_runs.png)

The fast index inside the hybrid is the experiment `ann-hybrid-check`:

![MLflow comparing the exact hybrid with the fast pipelines](evaluation/mlflow_ann_hybrid_runs.png)

The walk-through, with every table and chart, is in [notebook 09](notebooks/09_fast_vector_search.ipynb).

References: Malkov and Yashunin, [Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs](https://arxiv.org/abs/1603.09320) (HNSW); Johnson, Douze and Jégou, [Billion-scale similarity search with GPUs](https://arxiv.org/abs/1702.08734) (FAISS); [Annoy](https://github.com/spotify/annoy).

### Reproduce it

```bash
pip install -e ".[dev,experiments,images,text,ann,notebooks]"
python experiments/ann_exact_baseline.py     # times the exact search, saves its answers
python experiments/ann_benchmark.py          # FAISS Flat, HNSW, IVF and Annoy
python experiments/ann_hybrid_check.py       # the fast index inside the hybrid
python experiments/log_ann_benchmark.py      # records the runs in MLflow
```

## Where the numbers come from

Every result in this page can be traced to a notebook, a results file and an MLflow experiment. The MLflow databases stay on my computer (they are not in Git), so the screenshots in `evaluation/` show the runs.

| Result | Notebook | Results files | MLflow experiment |
|---|---|---|---|
| The first text methods (TF-IDF, attributes) | [04](notebooks/04_evaluation_baseline.ipynb) | [`dev_scoreboard.csv`](results/dev_scoreboard.csv), [`dev_evaluation_report.md`](results/dev_evaluation_report.md) | `dev-ratings` |
| Pictures, small test (ResNet, CLIP) | [05](notebooks/05_image_similarity.ipynb) | [`dev_image_scoreboard.csv`](results/dev_image_scoreboard.csv), [`dev_image_evaluation_report.md`](results/dev_image_evaluation_report.md) | `dev-images`, `dev-ratings-with-images` |
| CLIP over the whole catalogue | [06](notebooks/06_clip_whole_catalogue.ipynb) | [`dev_full_image_scoreboard.csv`](results/dev_full_image_scoreboard.csv), [`dev_full_image_evaluation_report.md`](results/dev_full_image_evaluation_report.md) | `dev-ratings-full-images` |
| Sentence vectors on titles | [07](notebooks/07_sentence_vectors_on_titles.ipynb) | [`dev_text_vector_scoreboard.csv`](results/dev_text_vector_scoreboard.csv), [`dev_text_vector_evaluation_report.md`](results/dev_text_vector_evaluation_report.md) | `dev-ratings-text-vectors` |
| Hybrid search and its weights | [08](notebooks/08_hybrid_search.ipynb) | [`dev_hybrid_scoreboard.csv`](results/dev_hybrid_scoreboard.csv), [`hybrid_tuning.json`](results/hybrid_tuning.json), [`hybrid_tuning_first_round.json`](results/hybrid_tuning_first_round.json) | `dev-hybrid-tuning`, `dev-ratings-hybrid` |
| Speed of the exact and the fast search | [09](notebooks/09_fast_vector_search.ipynb) | [`ann_exact_baseline.json`](results/ann_exact_baseline.json), [`ann_benchmark.json`](results/ann_benchmark.json), [`ann_hybrid_check.json`](results/ann_hybrid_check.json) | `ann-benchmark`, `ann-hybrid-check` |

## Tests and CI/CD

### Tests

There are 156 tests. The unit tests cover loading, cleaning, scoring, ranking, the service, the API, the rating tools, the scoring maths, MLflow logging and the image code. The integration tests run on the real dataset and on the real API.

### CI/CD with GitHub Actions

On every push and pull request, GitHub Actions runs two jobs:

1. **test:** installs the project, checks the code style with ruff, and runs all the tests.
2. **docker:** builds the image, starts the container, waits until it is ready, asks it for similar products, checks that an unknown ID returns 404, and checks that the container does not run as root.

The second job only starts if the first one passes. I merge into `main` through a pull request. Automatic deployment (for example to Kubernetes) is planned but not built yet.

## Limits so far

- One person rated the results, so the labels reflect one judgment.
- Only 30 development products have been used, so small differences cannot be told apart.
- The hybrid weights were tuned on the same 30 products that I report on. The cross-validation scores come from products the weights had not seen, but the 20 locked products give the honest final test.
- The sentence model was not trained on shopping titles, so it may be weaker on brand names and size words.
- The 20 locked final products have not been rated or used. They are kept for the final test.
- The attribute weights are untuned starting values.

## What is next

- Make the service stronger: checks on the input and proper error codes.
- Add caching, Kubernetes files and automatic deployment.
- A page where people set each weight with sliders, with the tuned weights as the best setting.
- Test shortening the vectors with PCA for the exact search and every fast method, and draw a t-SNE picture of the vectors.
- Add caching, Kubernetes files and automatic deployment.
- Rate the 20 locked products once, at the end, for the final result.

## Project layout

```text
app.py                      Entry point for uvicorn
Dockerfile                  Builds the service image (serving packages only, non-root user)
requirements.txt            Pinned packages used by the service and the Docker image
pyproject.toml              Package settings and the optional extras (dev, notebooks, experiments, images, text, ann)
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
  dedupe.py                 Shows one copy of each item in a result list
  text_features.py          Turns titles into vectors with a sentence model (offline only)
  tuning.py                 Coordinate ascent, cross-validation and part importance for the weights
  vector_search.py          Exact (brute-force) vector search and recall, the reference for fast indexes
  hybrid_eval.py            Scores a set of hybrid weights against my ratings quickly
experiments/                Scripts that I run by hand (nothing here is used by the service)
  run_evaluation.py         Scores the text methods against my ratings and records the runs
  check_images.py           Downloads and checks the 100-image sample
  image_grids.py            Makes contact sheets of the sample images (kept local)
  download_dev_images.py    Downloads the pictures of the development products
  extract_image_features.py Turns those pictures into ResNet and CLIP vectors
  run_image_search.py       Finds similar pictures and writes the rating page for new results
  run_image_evaluation.py   Scores text and image methods together and records the runs
  download_images.py        Downloads the first picture of every product (not the locked ones)
  extract_all_image_features.py  Turns all pictures into CLIP vectors
  run_full_image_search.py  CLIP search over the whole catalogue, writes the rating page
  run_full_image_evaluation.py  Scores text and CLIP search together and records the runs
  extract_text_features.py  Turns all titles into sentence vectors
  run_text_vector_search.py Finds similar titles and writes the rating page for new results
  run_text_vector_evaluation.py  Scores TF-IDF, sentence vectors and CLIP together and records the runs
  hybrid_setup.py           Shared set-up for the hybrid scripts (ratings, vectors, scorer)
  tune_hybrid_weights.py    Tunes the hybrid weights and checks them with 5-fold cross-validation
  run_hybrid_search.py      Finds hybrid results without a rating and writes the rating page
  run_hybrid_evaluation.py  Scores the hybrid next to the single methods and records the runs
  log_hybrid_tuning.py      Records the tuning runs in MLflow
  ann_data.py               Shared set-up for the fast-search scripts (vectors and test products)
  ann_exact_baseline.py     Times the exact search and saves its answers
  ann_benchmark.py          Compares FAISS Flat, HNSW, IVF and Annoy with the exact search
  ann_hybrid_check.py       Uses a fast index to pick candidates for the hybrid
  log_ann_benchmark.py      Records the fast-search runs in MLflow
evaluation/                 Test products, rating rules, rating pages and my ratings
  queries.json              The 50 test products (30 development, 20 locked)
  rubric.md                 The rules for rating 0, 1 or 2
  judgments/                Rating sheets, rating pages and my ratings (CSV)
  *.png                     Screenshots of the rating page and MLflow
results/                    Exported scores, reports and image check numbers
tests/                      156 tests
  unit/                     Fast tests for each module
  integration/              Tests on the real dataset and the real API
notebooks/                  Nine analysis notebooks (they import logic from the package)
data/archive.zip            Dataset supplied with the exercise
.github/workflows/ci.yml    Continuous integration (lint, tests, Docker smoke test)
```

Not in Git: the downloaded pictures, the saved picture vectors and the local MLflow databases. They stay on my computer.

## License

Apache-2.0, see [LICENSE](LICENSE).
