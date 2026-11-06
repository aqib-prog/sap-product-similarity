# Development evaluation: TF-IDF, sentence vectors and CLIP

Five methods scored against the same human ratings of 30 development products (all ratings so far). Every method shows one copy of each item.

| Method | nDCG@10 | 95% range | Precision@10 | Strong matches in top 10 |
|---|---|---|---|---|
| baseline | 0.637 | 0.53 to 0.74 | 0.74 | 54% |
| title_only | 0.500 | 0.40 to 0.60 | 0.55 | 41% |
| attributes_only | 0.422 | 0.33 to 0.51 | 0.62 | 32% |
| image_clip_all | 0.751 | 0.65 to 0.84 | 0.85 | 66% |
| text_vectors | 0.662 | 0.56 to 0.75 | 0.76 | 57% |

## Differences (nDCG@10, same products)

| Comparison | Difference | 95% range | Verdict |
|---|---|---|---|
| baseline minus title_only | +0.137 | +0.042 to +0.229 | real |
| baseline minus attributes_only | +0.215 | +0.141 to +0.293 | real |
| baseline minus image_clip_all | -0.114 | -0.225 to +0.004 | could be luck |
| baseline minus text_vectors | -0.026 | -0.114 to +0.072 | could be luck |
| title_only minus attributes_only | +0.078 | -0.035 to +0.191 | could be luck |
| title_only minus image_clip_all | -0.251 | -0.358 to -0.130 | real |
| title_only minus text_vectors | -0.163 | -0.233 to -0.093 | real |
| attributes_only minus image_clip_all | -0.329 | -0.448 to -0.204 | real |
| attributes_only minus text_vectors | -0.241 | -0.346 to -0.131 | real |
| image_clip_all minus text_vectors | +0.088 | -0.030 to +0.203 | could be luck |

## Notes

- One copy per item: baseline lost 3 copies, title_only lost 5 copies, attributes_only lost 1 copies (out of 300 results each). Without the rule the text nDCG@10 would be baseline 0.637, title_only 0.502, attributes_only 0.425.
- text_vectors is the sentence model on titles; title_only is TF-IDF on titles; image_clip_all is CLIP on pictures.

## Limitations

- One person rated the results, so the labels reflect one judgment.
- Only 30 development products were used, so small differences cannot be told apart.
- The 20 locked final products have not been rated or used.
