"""A self-contained web page for rating results, with product pictures.

The page is one HTML file with the data inside it. Open it in a browser, click
0, 1 or 2 on each card, and download the ratings as a CSV. Progress is saved in
the browser, so rating can be done across several sittings. The page never says
which method found a result.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

PAGE_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Rate similar products</title>
<style>
  :root { --bg:#f6f7f9; --card:#fff; --ink:#1f2933; --mute:#667085; --line:#d9dee5;
          --accent:#2b6cb0; --g0:#c0392b; --g1:#d68910; --g2:#1e8449; }
  * { box-sizing: border-box; }
  body { margin:0; font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
         background:var(--bg); color:var(--ink); }
  header { position:sticky; top:0; z-index:5; background:var(--card);
           border-bottom:1px solid var(--line); padding:10px 16px; }
  header h1 { font-size:16px; margin:0 0 4px; }
  .bar { display:flex; gap:12px; align-items:center; flex-wrap:wrap; font-size:14px; }
  .bar button, .bar label.btn { background:var(--accent); color:#fff; border:0;
         border-radius:6px; padding:6px 12px; cursor:pointer; font-size:14px; }
  .bar input[type=file] { display:none; }
  details { margin:6px 0 0; font-size:13px; color:var(--mute); }
  details table { border-collapse:collapse; margin-top:4px; }
  details td { padding:2px 10px 2px 0; vertical-align:top; }
  nav { padding:8px 16px; display:flex; flex-wrap:wrap; gap:6px; }
  nav a { text-decoration:none; font-size:13px; padding:3px 8px; border-radius:12px;
          border:1px solid var(--line); background:var(--card); color:var(--ink); }
  nav a.done { background:#e7f5ec; border-color:#9fd3b0; }
  section.query { margin:16px; }
  .qcard { display:flex; gap:14px; background:#eef4fb; border:1px solid #b9d0ea;
           border-radius:10px; padding:12px; margin-bottom:10px; }
  .qcard img { width:150px; height:150px; object-fit:contain; background:#fff;
               border-radius:6px; }
  .grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(210px, 1fr)); gap:10px; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:10px;
          padding:10px; display:flex; flex-direction:column; gap:6px; }
  .card.rated { border-color:#9fd3b0; }
  .card img { width:100%; height:200px; object-fit:contain; background:#fff; border-radius:6px; }
  .missing { height:200px; display:flex; align-items:center; justify-content:center;
             color:var(--mute); background:#f0f1f3; border-radius:6px; font-size:13px; }
  .title { font-size:13px; line-height:1.3; min-height:34px; }
  .meta { font-size:12px; color:var(--mute); }
  .buttons { display:flex; gap:6px; }
  .buttons button { flex:1; padding:7px 0; border:1px solid var(--line); background:#fff;
                    border-radius:6px; cursor:pointer; font-size:14px; font-weight:600; }
  .buttons button[data-g="0"].on { background:var(--g0); color:#fff; border-color:var(--g0); }
  .buttons button[data-g="1"].on { background:var(--g1); color:#fff; border-color:var(--g1); }
  .buttons button[data-g="2"].on { background:var(--g2); color:#fff; border-color:var(--g2); }
  .qtitle { font-size:15px; font-weight:600; }
</style>
</head>
<body>
<header>
  <h1>Rate each result as a substitute for the query product</h1>
  <div class="bar">
    <span id="progress"></span>
    <button id="download">Download my ratings (CSV)</button>
    <label class="btn">Load saved ratings<input type="file" id="load" accept=".csv"></label>
    <button id="clear" style="background:#8a94a0">Clear all</button>
  </div>
  <details>
    <summary>Rating rules (click to open)</summary>
    <table>
      <tr><td><b style="color:var(--g2)">2</b></td><td>Strong match: same type of product, same audience, close style</td></tr>
      <tr><td><b style="color:var(--g1)">1</b></td><td>Acceptable: same type and audience, but a noticeably different style, colour, pattern or price tier</td></tr>
      <tr><td><b style="color:var(--g0)">0</b></td><td>Not a match: a different type, a different audience, an accessory, or not clothing</td></tr>
    </table>
    <div>Leave a card empty if you truly cannot decide. Your clicks are saved in this browser automatically.</div>
  </details>
</header>
<nav id="nav"></nav>
<main id="main"></main>
<script>
const DATA = __DATA__;
const KEY = "__KEY__";
let ratings = {};
try { ratings = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { ratings = {}; }

function save() { try { localStorage.setItem(KEY, JSON.stringify(ratings)); } catch (e) {} }
function key(q, c) { return q.query_no + "|" + c.no; }
function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined && text !== null) e.textContent = text;
  return e;
}
function picture(url, alt) {
  const holder = el("div");
  if (!url) { holder.appendChild(el("div", "missing", "no picture")); return holder; }
  const img = document.createElement("img");
  img.src = url; img.alt = alt; img.loading = "lazy"; img.referrerPolicy = "no-referrer";
  img.onerror = () => { holder.replaceChildren(el("div", "missing", "picture not available")); };
  holder.appendChild(img);
  return holder;
}
function money(p) { return (p === null || p === undefined || p === "") ? "no price" : "₹" + p; }

function render() {
  const main = document.getElementById("main"); main.replaceChildren();
  const nav = document.getElementById("nav"); nav.replaceChildren();
  DATA.forEach(q => {
    const a = el("a", "", q.query_no); a.href = "#" + q.query_no; a.id = "nav-" + q.query_no; nav.appendChild(a);
    const sec = el("section", "query"); sec.id = q.query_no;
    const qc = el("div", "qcard");
    qc.appendChild(picture(q.image_url, q.title));
    const info = el("div");
    info.appendChild(el("div", "qtitle", q.query_no + ": " + q.title));
    info.appendChild(el("div", "meta", (q.category || "no category") + " | " + money(q.price)));
    info.appendChild(el("div", "meta", "Would a shopper accept each result below instead of this product?"));
    qc.appendChild(info); sec.appendChild(qc);
    const grid = el("div", "grid");
    q.candidates.forEach(c => {
      const card = el("div", "card"); card.dataset.k = key(q, c);
      card.appendChild(picture(c.image_url, c.title));
      card.appendChild(el("div", "title", c.title));
      card.appendChild(el("div", "meta", [c.brand || "no brand", c.category || "no category", money(c.price)].join(" | ")));
      const row = el("div", "buttons");
      [0, 1, 2].forEach(g => {
        const b = el("button", "", String(g)); b.dataset.g = g; b.title = ["Not a match", "Acceptable", "Strong match"][g];
        b.onclick = () => {
          const k = key(q, c);
          if (ratings[k] === g) { delete ratings[k]; } else { ratings[k] = g; }
          save(); refresh();
        };
        row.appendChild(b);
      });
      card.appendChild(row); grid.appendChild(card);
    });
    sec.appendChild(grid); main.appendChild(sec);
  });
  refresh();
}

function refresh() {
  let done = 0, total = 0;
  DATA.forEach(q => {
    let qDone = 0;
    q.candidates.forEach(c => { total++; if (ratings[key(q, c)] !== undefined) { done++; qDone++; } });
    document.getElementById("nav-" + q.query_no).className = qDone === q.candidates.length ? "done" : "";
  });
  document.querySelectorAll(".card").forEach(card => {
    const g = ratings[card.dataset.k];
    card.className = g === undefined ? "card" : "card rated";
    card.querySelectorAll("button").forEach(b => { b.className = (g !== undefined && Number(b.dataset.g) === g) ? "on" : ""; });
  });
  document.getElementById("progress").textContent = done + " of " + total + " rated";
}

document.getElementById("download").onclick = () => {
  const lines = ["query_no,candidate_no,candidate_id,grade"];
  DATA.forEach(q => q.candidates.forEach(c => {
    const g = ratings[key(q, c)];
    lines.push([q.query_no, c.no, c.id, g === undefined ? "" : g].join(","));
  }));
  const blob = new Blob([lines.join("\\n") + "\\n"], { type: "text/csv" });
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = "__FILE__";
  document.body.appendChild(a); a.click(); a.remove();
};
document.getElementById("load").onchange = ev => {
  const file = ev.target.files[0]; if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {
    const byId = {}; DATA.forEach(q => q.candidates.forEach(c => { byId[q.query_no + "|" + c.no] = true; }));
    reader.result.split(/\\r?\\n/).slice(1).forEach(line => {
      const p = line.split(","); if (p.length < 4 || p[3] === "") return;
      const k = p[0] + "|" + p[1]; if (byId[k]) ratings[k] = Number(p[3]);
    });
    save(); refresh();
  };
  reader.readAsText(file);
};
document.getElementById("clear").onclick = () => {
  if (confirm("Remove all ratings from this browser?")) { ratings = {}; save(); refresh(); }
};
render();
</script>
</body>
</html>
"""


def render_review_page(
    rows: Sequence[dict], key: str = "ratings-v1", filename: str = "dev_grades.csv"
) -> str:
    """Build the rating page from the sheet rows (no method information is used).

    `key` names the saved progress in the browser and `filename` the downloaded file,
    so two sheets can be rated without mixing up their ratings.
    """
    queries: list[dict] = []
    index: dict[str, dict] = {}
    for row in rows:
        number = row["query_no"]
        if number not in index:
            index[number] = {
                "query_no": number,
                "title": row["query_title"],
                "category": row["query_category"],
                "price": row["query_price"],
                "image_url": row["query_image_url"],
                "candidates": [],
            }
            queries.append(index[number])
        index[number]["candidates"].append(
            {
                "no": row["candidate_no"],
                "id": row["candidate_id"],
                "title": row["candidate_title"],
                "brand": row["candidate_brand"],
                "category": row["candidate_category"],
                "price": row["candidate_price"],
                "image_url": row["candidate_image_url"],
            }
        )
    data = json.dumps(queries, ensure_ascii=False).replace("</", "<\\/")
    page = PAGE_TEMPLATE.replace("__KEY__", key).replace("__FILE__", filename)
    return page.replace("__DATA__", data)


def write_review_page(
    rows: Sequence[dict],
    path: Path,
    key: str = "ratings-v1",
    filename: str = "dev_grades.csv",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_review_page(rows, key, filename), encoding="utf-8")
