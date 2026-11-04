# Rating rules

Imagine a shopper looking at the **query product** who wants something they would accept **instead of it** (a substitute). For each result, give one grade.

| Grade | Meaning | Example (query: men's blue cotton round-neck T-shirt) |
|---|---|---|
| **2** | Strong match: same type of product, same audience, and a close style | Another men's blue round-neck T-shirt |
| **1** | Acceptable: same type and audience, but a noticeably different style, colour, pattern or price tier | A men's red V-neck polo |
| **0** | Not a match: a different type of product, a different audience, an accessory, or not clothing | A men's cap, a women's kurta, a belt |

## Rules
- Judge only the result against the query. Do not guess how the result was found.
- Compare the product itself. Do not give extra credit for a famous brand, or lose credit for a missing price.
- The same item listed again (the same title, maybe another size) still gets the grade it deserves, normally a 2.
- If the title and category disagree with what the picture shows, trust what you can see in the picture link.
- If you truly cannot decide, leave the grade empty. Empty grades are counted and left out of the scores.
- Grades are whole numbers: 0, 1 or 2.

## How to rate with the page
Open `evaluation/judgments/dev_review.html` in a browser (double-click it; it needs the internet to show the pictures). The query product is in the blue box, and each result is a card with a picture, title, brand, category and price. Click **0**, **1** or **2** under each card. Click the same number again to undo. Your clicks are saved automatically in that browser, so you can stop and come back, as long as you use the same browser and the same file. Press **Download my ratings (CSV)** now and then as a backup, and save the final file as `evaluation/judgments/dev_grades.csv`.

## Worked examples
These use products that are **not** among the 50 test products. The grades come from titles and categories; when you can see the picture, trust the picture.

**Query: StormBase Men Skull T-Shirt - Live to Win** (men's T-shirt with a skull print, ₹284)

| Result | Grade | Why |
|---|---|---|
| Unisopent Mens Skull Printed Full Sleeves T-Shirt (₹299) | **2** | Same type, same audience, also a skull print, and a similar price. Sleeve length is a small style difference |
| StormBase SB Men's V-Neck T-Shirt (₹300) | **1** | Same brand, type and audience, but a plain V-neck, not a printed graphic tee |
| Hanes Men's 3-Pack Tagless Crew Neck T-Shirt | **1** | A men's T-shirt, but a plain multi-pack, so a different style |
| U.S. Polo Association Men's Skull Cap | **0** | A cap, not a T-shirt. It only matches on the word "skull" |
| Black & White Skull Beanie | **0** | A hat, not a T-shirt |

**Query: Craftsvilla Chiffon with Blouse Piece Saree** (₹1,179)

| Result | Grade | Why |
|---|---|---|
| Craftsvilla Silk with Blouse Piece Saree (₹1,169) | **1** | Same brand, type and price, but silk instead of chiffon |
| Vaamsi Chiffon with Blouse Piece Saree (₹299) | **1** | Same fabric and type, but a much cheaper price tier |

**Query: Hanes Men's Blended Brief**

| Result | Grade | Why |
|---|---|---|
| Hanes Men's Plain Brief (₹199) | **2** | Same brand, same type, same audience |
| Hanes Men's V-Neck T-Shirt | **0** | Same brand, but a different type of garment |
| Kuber Industries Brocade Jewellery Box | **0** | Not clothing at all |

**Query: Sheknows Fusion Wear Cotton Blue Denim Straight Kurta**

| Result | Grade | Why |
|---|---|---|
| The same title listed again (₹720 vs ₹738) | **2** | The same item, so it is a perfect substitute |
| Jockey Men's Cotton Fusion Brief | **0** | Matches only on the word "Fusion", and it is men's underwear |

## Tips
- Ask one question for every card: "Would a shopper be happy to see this instead of the query?"
- A result can look similar on the page yet be the wrong audience. Boys' and men's clothes, or women's and girls', are different audiences.
- You will see near-copies of the same listing. Rate each one on its own, normally the same grade as its twin.
