# Development evaluation: text and image similarity

Five methods scored against the same human ratings of 30 development products (first ratings plus the ratings of the new image results).

| Method | nDCG@10 | 95% range | Precision@10 | Strong matches in top 10 |
|---|---|---|---|---|
| baseline | 0.681 | 0.59 to 0.77 | 0.74 | 54% |
| title_only | 0.545 | 0.45 to 0.65 | 0.55 | 42% |
| attributes_only | 0.445 | 0.36 to 0.53 | 0.62 | 32% |
| image_resnet | 0.569 | 0.47 to 0.66 | 0.63 | 43% |
| image_clip | 0.695 | 0.62 to 0.77 | 0.74 | 52% |

## Differences (nDCG@10, same products)

| Comparison | Difference | 95% range | Verdict |
|---|---|---|---|
| baseline minus title_only | +0.136 | +0.030 to +0.237 | real |
| baseline minus attributes_only | +0.236 | +0.160 to +0.317 | real |
| baseline minus image_resnet | +0.112 | +0.016 to +0.215 | real |
| baseline minus image_clip | -0.014 | -0.107 to +0.084 | could be luck |
| title_only minus attributes_only | +0.100 | -0.019 to +0.219 | could be luck |
| title_only minus image_resnet | -0.024 | -0.136 to +0.091 | could be luck |
| title_only minus image_clip | -0.150 | -0.253 to -0.050 | real |
| attributes_only minus image_resnet | -0.124 | -0.233 to -0.015 | real |
| attributes_only minus image_clip | -0.250 | -0.362 to -0.131 | real |
| image_resnet minus image_clip | -0.126 | -0.205 to -0.052 | real |

## Notes

- 1 new result was left unrated and is scored as 0 here. Scored as 2 instead, the image scores change by at most 0.003.
- Text scores are lower than in the first report because the answer key now also contains the new ratings.
- Results in the top 10 that nobody rated (scored as 0): baseline 2, title_only 2, attributes_only 6, image_resnet 0, image_clip 1.

## Limitations

- One person rated the results, so the labels reflect one judgment.
- Only 30 development products were used, so small differences cannot be told apart.
- Image search only looked at 708 downloaded pictures; text search looked at all 30,000 products.
- The first ratings were collected for results picked by the text methods, so they may favour text.
- The 20 locked final products have not been rated or used.
