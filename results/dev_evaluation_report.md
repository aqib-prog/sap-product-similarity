# Development evaluation: first scoreboard

Three methods scored against human ratings of 30 development products (nDCG@10 and precision@10, with 95% ranges).

| Method | nDCG@10 | 95% range | Precision@10 | Strong matches in top 10 | Unrated in top 10 |
|---|---|---|---|---|---|
| baseline | 0.723 | 0.64 to 0.81 | 0.74 | 54% | 2 |
| title_only | 0.573 | 0.48 to 0.67 | 0.55 | 42% | 2 |
| attributes_only | 0.475 | 0.38 to 0.56 | 0.62 | 32% | 6 |

## Differences between methods (nDCG@10, same products)

| Comparison | Difference | 95% range | Verdict |
|---|---|---|---|
| baseline minus title_only | +0.150 | +0.037 to +0.254 | real |
| baseline minus attributes_only | +0.248 | +0.168 to +0.334 | real |
| title_only minus attributes_only | +0.098 | -0.028 to +0.223 | could be luck |

## Limitations

- One person rated the results, so the labels reflect one judgment.
- Only 30 development products were rated, so small differences cannot be told apart.
- Only results found by these three methods were rated (pooled ratings); a new method's new results need rating before it can be scored fairly.
- Unrated results count as 0 (2 to 6 per method).
- The 20 locked final products have not been rated or used.
