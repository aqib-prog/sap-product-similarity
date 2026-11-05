"""Make contact sheets of the sample images, for looking at them by eye.

Reads data/images/sample/rows.json (written by check_images.py) and writes PNG
sheets to data/images/sample/grids/. These stay on this computer (git ignores them).
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / "data" / "images" / "sample"
COLUMNS, ROWS, THUMB, CAPTION = 5, 4, 230, 78
RED, AMBER, GREEN = (200, 50, 40), (220, 150, 20), (60, 150, 70)


def main() -> None:
    rows = json.loads((LOCAL / "rows.json").read_text(encoding="utf-8"))
    out = LOCAL / "grids"
    out.mkdir(exist_ok=True)
    per_sheet = COLUMNS * ROWS
    cell_w, cell_h = THUMB + 12, THUMB + CAPTION + 12
    for sheet_no, start in enumerate(range(0, len(rows), per_sheet), start=1):
        sheet = Image.new("RGB", (COLUMNS * cell_w, ROWS * cell_h), (245, 245, 245))
        draw = ImageDraw.Draw(sheet)
        for i, row in enumerate(rows[start : start + per_sheet]):
            x, y = (i % COLUMNS) * cell_w + 6, (i // COLUMNS) * cell_h + 6
            flags = row.get("flags", [])
            colour = (
                RED
                if "download_failed" in flags or "cannot_decode" in flags
                else AMBER
                if flags
                else GREEN
            )
            draw.rectangle(
                [x - 3, y - 3, x + THUMB + 3, y + THUMB + CAPTION - 3], outline=colour, width=3
            )
            if row.get("path") and Path(row["path"]).exists():
                with Image.open(row["path"]) as picture:
                    picture = picture.convert("RGB")
                    picture.thumbnail((THUMB, THUMB))
                    sheet.paste(
                        picture,
                        (x + (THUMB - picture.width) // 2, y + (THUMB - picture.height) // 2),
                    )
            else:
                draw.text((x + 60, y + THUMB // 2), "download failed", fill=RED)
            caption = f"#{row['no']} {row['group']}"
            if flags:
                caption += "  [" + ", ".join(flags) + "]"
            lines = textwrap.wrap(caption, 36)[:2] + textwrap.wrap(row["title"], 36)[:2]
            for k, line in enumerate(lines):
                draw.text((x, y + THUMB + 4 + k * 15), line, fill=(20, 20, 20))
        sheet.save(out / f"grid_{sheet_no}.png", optimize=True)
    print(f"{len(rows)} images in {-(-len(rows) // per_sheet)} sheets: {out}")


if __name__ == "__main__":
    main()
