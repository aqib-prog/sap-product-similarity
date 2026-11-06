# Development evaluation: hybrid search and the single methods

Seven methods scored against the same human ratings of 30 development products. The hybrid scores are out-of-fold: each product is scored with weights tuned on the other products. Every method shows one copy of each item.

| Method | nDCG@10 | 95% range | Precision@10 |
|---|---|---|---|
| Hybrid, 4 weights | 0.806 | 0.73 to 0.87 | 0.87 |
| Hybrid, 10 weights | 0.764 | 0.68 to 0.84 | 0.82 |
| CLIP (pictures) | 0.750 | 0.65 to 0.84 | 0.85 |
| Sentence vectors (titles) | 0.659 | 0.56 to 0.75 | 0.76 |
| Title (TF-IDF) and attributes (my baseline) | 0.637 | 0.53 to 0.74 | 0.74 |
| TF-IDF, title only | 0.498 | 0.40 to 0.60 | 0.55 |
| Attributes only | 0.420 | 0.33 to 0.51 | 0.62 |

## Differences (nDCG@10, same products)

| Comparison | Difference | 95% range | Verdict |
|---|---|---|---|
| Hybrid, 4 weights minus Hybrid, 10 weights | +0.042 | +0.014 to +0.075 | real |
| Hybrid, 4 weights minus CLIP (pictures) | +0.055 | -0.040 to +0.155 | could be luck |
| Hybrid, 4 weights minus Sentence vectors (titles) | +0.146 | +0.082 to +0.219 | real |
| Hybrid, 4 weights minus Title (TF-IDF) and attributes (my baseline) | +0.168 | +0.082 to +0.255 | real |
| Hybrid, 4 weights minus TF-IDF, title only | +0.308 | +0.218 to +0.402 | real |
| Hybrid, 4 weights minus Attributes only | +0.386 | +0.295 to +0.475 | real |
| Hybrid, 10 weights minus CLIP (pictures) | +0.013 | -0.081 to +0.111 | could be luck |
| Hybrid, 10 weights minus Sentence vectors (titles) | +0.104 | +0.037 to +0.180 | real |
| Hybrid, 10 weights minus Title (TF-IDF) and attributes (my baseline) | +0.126 | +0.049 to +0.206 | real |
| Hybrid, 10 weights minus TF-IDF, title only | +0.266 | +0.178 to +0.361 | real |
| Hybrid, 10 weights minus Attributes only | +0.343 | +0.259 to +0.433 | real |
| CLIP (pictures) minus Sentence vectors (titles) | +0.091 | -0.026 to +0.204 | could be luck |
| CLIP (pictures) minus Title (TF-IDF) and attributes (my baseline) | +0.113 | -0.003 to +0.223 | could be luck |
| CLIP (pictures) minus TF-IDF, title only | +0.253 | +0.134 to +0.359 | real |
| CLIP (pictures) minus Attributes only | +0.330 | +0.207 to +0.449 | real |
| Sentence vectors (titles) minus Title (TF-IDF) and attributes (my baseline) | +0.022 | -0.075 to +0.108 | could be luck |
| Sentence vectors (titles) minus TF-IDF, title only | +0.162 | +0.093 to +0.232 | real |
| Sentence vectors (titles) minus Attributes only | +0.239 | +0.130 to +0.344 | real |
| Title (TF-IDF) and attributes (my baseline) minus TF-IDF, title only | +0.140 | +0.045 to +0.232 | real |
| Title (TF-IDF) and attributes (my baseline) minus Attributes only | +0.217 | +0.144 to +0.297 | real |
| TF-IDF, title only minus Attributes only | +0.077 | -0.035 to +0.190 | could be luck |

## Final weights (tuned on all 30 products)

| Part | 4-weight mix | 10-weight mix |
|---|---|---|
| title | 0.029 | 0.062 |
| text_vectors | 0.457 | 0.375 |
| image | 0.286 | 0.312 |
| category | 0.070 | 0.062 |
| brand | 0.035 | 0.031 |
| audience | 0.035 | 0.031 |
| price | 0.035 | 0.031 |
| colour | 0.018 | 0.016 |
| weight | 0.018 | 0.047 |
| rating | 0.018 | 0.031 |

## Limitations

- One person rated the results, so the labels reflect one judgment.
- Only 30 development products were used, so small differences cannot be told apart.
- The 20 locked final products have not been rated or used.
