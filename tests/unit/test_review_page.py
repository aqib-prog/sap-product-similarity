"""The rating page: complete, blind and safe."""

import json
import re

from product_similarity.review_page import render_review_page, write_review_page


def _rows():
    rows = []
    for query in ("Q01", "Q02"):
        for number in ("01", "02", "03"):
            rows.append(
                {
                    "query_no": query,
                    "query_title": f"Query {query}",
                    "query_category": "MensT_Shirts",
                    "query_price": 300.0,
                    "query_image_url": "https://example.test/q.jpg",
                    "candidate_no": number,
                    "candidate_title": f"Candidate {query}-{number}",
                    "candidate_brand": None,
                    "candidate_category": "MensT_Shirts",
                    "candidate_price": None,
                    "candidate_image_url": "https://example.test/c.jpg",
                    "candidate_id": f"{query}{number}".ljust(32, "0"),
                    "grade": "",
                }
            )
    return rows


def _embedded(html):
    return json.loads(re.search(r"const DATA = (.*?);\nconst KEY", html, re.S).group(1))


def test_page_contains_every_query_and_candidate():
    data = _embedded(render_review_page(_rows()))
    assert [q["query_no"] for q in data] == ["Q01", "Q02"]
    assert all(len(q["candidates"]) == 3 for q in data)
    assert data[0]["candidates"][0]["id"].startswith("Q0101")


def test_page_never_names_a_method():
    html = render_review_page(_rows()).lower()
    for word in ("baseline", "title_only", "title only", "attributes_only", "attributes only"):
        assert word not in html


def test_titles_cannot_break_out_of_the_script():
    rows = _rows()
    rows[0]["candidate_title"] = "</script><script>alert(1)</script>"
    html = render_review_page(rows)
    assert html.count("</script>") == 1
    assert _embedded(html)[0]["candidates"][0]["title"] == "</script><script>alert(1)</script>"


def test_page_is_written_as_utf8(tmp_path):
    rows = _rows()
    rows[0]["candidate_title"] = "Kurta \u20b9 caf\u00e9"
    write_review_page(rows, tmp_path / "page.html")
    assert "caf\u00e9" in (tmp_path / "page.html").read_text(encoding="utf-8")
