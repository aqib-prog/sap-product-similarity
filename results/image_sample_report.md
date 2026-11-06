# Image sample check

A first look at 100 product images before any large download. The images stay on this computer (`data/images/` is ignored by git); only the numbers below are kept in the repository.

## What was checked

- **100 images** chosen with a fixed seed: 40 random, 20 from rare categories, 15 widely shared pictures (to find banners and size charts), 15 products with little data, and 10 with repeated titles. The 20 locked final products are excluded.
- Random, rare-category, sparse and repeated-title products use their **first medium image**. The shared-picture group uses a picture that 5 or more products share, taken from any position.
- **Download limits:** one request at a time and at most 4 per second, https only, one allowed host (`images-na.ssl-images-amazon.com`), 15-second timeout, 2 MB cap, no redirects followed, one retry for server errors, and a descriptive user agent. The host's `robots.txt` restricts only two named crawlers and has no blanket rule.
- **Quality checks on every image:** it opens and decodes, its size and shape, near-blank, blurry, and the same picture used by different products.
- **Visual review:** all 100 images were viewed on contact sheets.

## Results

| Check | Result |
|---|---|
| Downloaded | 99 of 100 |
| Failed | 1 (http_404: 1) |
| File size | median 25 KB, mean 29 KB, largest 66 KB |
| Picture size | every image is at most 500 pixels on its longest side; 98 of 99 are exactly 500 |
| Same picture on different products | none in this sample |
| Time | 47.6 seconds for 100 requests (0.48 s each, including the polite pause) |

### Failed downloads

| # | Group | Product | Reason | Attempts |
|---|---|---|---|---|
| 22 | random | Columbia Women's Saturday Trail II Plaid Short, Jewel Plaid, | HTTP 404 (not found) | 1 |

The picture link for this product no longer exists. A 404 is not retried. The link is in `results/image_sample.csv`.

### Automatic quality flags

| Flag | Images | Rule | What it turned out to be |
|---|---|---|---|
| low_resolution | 5 | shorter side under 300 px | Narrow products (a tie, a belt, a vest, trousers) and two size charts. The pictures are fine, just narrow |
| extreme_shape | 1 | long side more than 2.5 times the short side | A narrow tie |
| mostly_blank | 1 | almost uniform or more than 98.5% near-white | A white vest on a white background |
| blurry | 0 | sharpness under 15 | none |

**The automatic flags were mostly false alarms, and they missed the real problem (below).**

## What the pictures show

| Group | Images | What they are |
|---|---|---|
| Random | 39 downloaded | All clean product pictures on a white background: models wearing the item, flat lays, multi-packs |
| Rare categories | 20 | All products. A few are close crops or fabric pieces, which is normal for those categories |
| Sparse data | 15 | All products |
| Repeated titles | 10 | All products |
| **Shared pictures** | 15 | **11 size charts**, 1 close-up of black fabric, 2 back views of a black T-shirt, and 1 accessory set. Only the accessory set is a good product picture |

**Key finding:** pictures that many products share are mostly **size charts**, and none of the automatic checks catch them. They are not product pictures, so comparing them would make unrelated products look alike.

**Where they sit:** across all 123,689 image links in the dataset, 258 pictures are shared by 5 or more products. They appear almost only in later positions (second to seventh and beyond). Only **11 products (0.04%)** have a first image shared by 5 or more products, and 561 (1.9%) share their first image with at least one other product.

**Conclusion:** use the **first image only**, and leave out any picture shared by 5 or more products. Combining all images of a product would pull size charts in unless they are filtered first.

## Estimate for the full download (not started)

| Item | Estimate | Basis |
|---|---|---|
| Pictures | 29,261 distinct first images | counted from the dataset |
| Disk | about 856 MB | 29,261 x 29 KB mean, measured |
| Download time | about 3.9 hours | one request at a time at the same polite pace, measured: 0.48 s each |
| Preparing a picture for a model | 3.0 ms | measured (resize to 224 x 224) |
| Features, ResNet-50 (2,048 numbers each) | about 240 MB | arithmetic |
| Features, CLIP ViT-B/32 (512 numbers each) | about 60 MB | arithmetic |
| Running ResNet-50 on a laptop CPU | roughly 10 to 25 minutes | **assumption, not measured.** To be measured on the first 500 images |
| Running CLIP on a laptop CPU | roughly 20 to 50 minutes | **assumption, not measured** |
| Model files to download | about 100 MB (ResNet-50), about 350 MB (CLIP) | approximate, would need approval |

## Access and terms

- `robots.txt` on the image host names four crawlers and has no blanket rule for others. Baiduspider and Googlebot-Image have a few path limits, and GPTBot and CCBot (AI training crawlers) are blocked completely. The requests used a different, descriptive user agent, and the pictures are only turned into vectors by pretrained models (no training), kept locally and never republished.
- I could not read Amazon's conditions of use (the page returned an error), so **I cannot say what they allow.** Amazon's terms generally restrict automated access to its services, so a large download should be a deliberate decision. Nothing here is legal advice.
- What was done: 100 images, slow and one at a time, kept private and never committed.

## Options for the full download (needs a separate decision)

1. **Smaller first step (recommended).** Download only the pictures needed to test whether images help: the 30 development products and their rated results, about 760 pictures (roughly 22 MB, 6 minutes). Compare image similarity with your ratings. Download more only if it helps.
2. **All first images.** About 856 MB and about 4 hours at this pace. Needed only to search the whole catalogue by picture.
3. **No images.** The text and attribute methods already work.

## Limits of this check

- 100 images is a sample. It cannot show how rare a problem is, only that it exists.
- Product relevance was judged by eye by one reviewer (the contact sheets were viewed, not scored).
- Model speed is an assumption until it is measured.
