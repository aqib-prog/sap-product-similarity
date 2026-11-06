# Development evaluation: text search and CLIP over the whole catalogue

Five methods scored against the same human ratings of 30 development products (all ratings so far). Every method shows one copy of each item.

| Method | nDCG@10 | 95% range | Precision@10 | Strong matches in top 10 |
|---|---|---|---|---|
| baseline | 0.644 | 0.55 to 0.74 | 0.74 | 53% |
| title_only | 0.507 | 0.41 to 0.61 | 0.54 | 41% |
| attributes_only | 0.427 | 0.34 to 0.52 | 0.62 | 32% |
| image_clip_all | 0.757 | 0.66 to 0.84 | 0.85 | 66% |
| image_clip_708 | 0.648 | 0.57 to 0.73 | 0.74 | 52% |

## Differences (nDCG@10, same products)

| Comparison | Difference | 95% range | Verdict |
|---|---|---|---|
| baseline minus title_only | +0.137 | +0.039 to +0.231 | real |
| baseline minus attributes_only | +0.216 | +0.141 to +0.294 | real |
| baseline minus image_clip_all | -0.113 | -0.228 to +0.009 | could be luck |
| baseline minus image_clip_708 | -0.005 | -0.092 to +0.086 | could be luck |
| title_only minus attributes_only | +0.079 | -0.034 to +0.190 | could be luck |
| title_only minus image_clip_all | -0.250 | -0.360 to -0.124 | real |
| title_only minus image_clip_708 | -0.142 | -0.239 to -0.046 | real |
| attributes_only minus image_clip_all | -0.329 | -0.452 to -0.200 | real |
| attributes_only minus image_clip_708 | -0.221 | -0.325 to -0.113 | real |
| image_clip_all minus image_clip_708 | +0.108 | +0.033 to +0.184 | real |

## Notes

- One copy per item: baseline lost 3 copies, title_only lost 5 copies, attributes_only lost 1 copies (out of 300 results each). Without the rule the text nDCG@10 would be baseline 0.646, title_only 0.511, attributes_only 0.431.
- image_clip_708 is the earlier test that searched only 708 pictures; image_clip_all searched every downloaded picture.

## Limitations

- One person rated the results, so the labels reflect one judgment.
- Only 30 development products were used, so small differences cannot be told apart.
- The 20 locked final products have not been rated or used.
