"""Offline regression tests for scraper.py: no network calls.

    pip install requests beautifulsoup4 pytest
    pytest
"""
import json
import os
import sys

import pytest
from bs4 import BeautifulSoup

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import scraper as s  # noqa: E402


def text_info(text, listing=False):
    info = s._blank_info()
    s._from_text(text, info, "page text", listing)
    return info


def candidate(i, cost, title="LM7805 voltage regulator", qty=1, **kw):
    r = {"title": title, "url": f"https://site{i}.com/p/{i}", "snippet": "",
         "search_rank": i, **s._blank_info()}
    r.update(cost=cost, cost_source="json-ld", supplier=f"site{i}", in_stock=True,
             shipping_cost="Free", shipping_time="2 days")
    r.update(kw)
    return s.apply_qty(r, qty)


# ---------------------------------------------------------------- price extraction
def test_tied_json_ld_offers_do_not_crash():
    # Same price on an AggregateOffer high end (url None) and an Offer with a url.
    ld = {"@type": "Product", "name": "LM7805 Regulator", "offers": [
        {"@type": "AggregateOffer", "lowPrice": "0.50", "highPrice": "0.89",
         "priceCurrency": "USD"},
        {"@type": "Offer", "price": "0.89", "priceCurrency": "USD", "url": "https://x.com/a"},
        {"@type": "Offer", "price": "0.50", "priceCurrency": "USD", "url": "https://x.com/b"}]}
    soup = BeautifulSoup(f'<script type="application/ld+json">{json.dumps(ld)}</script>',
                         "html.parser")
    info = s._blank_info()
    s._from_json_ld(soup, info, "https://x.com/c", "LM7805 Regulator")
    assert info["cost"] == 0.50


def test_zero_meta_price_is_ignored():
    soup = BeautifulSoup('<meta property="product:price:amount" content="0.00">', "html.parser")
    info = s._blank_info()
    s._from_meta(soup, info)
    assert info["cost"] is None


def test_zero_json_ld_price_is_ignored():
    ld = {"@type": "Product", "name": "X", "offers": {"@type": "Offer", "price": "0"}}
    soup = BeautifulSoup(f'<script type="application/ld+json">{json.dumps(ld)}</script>',
                         "html.parser")
    info = s._blank_info()
    s._from_json_ld(soup, info, "", "X")
    assert info["cost"] is None


def test_noise_words_need_a_word_boundary():
    assert s._text_prices("Discover $4.99") == [(4.99, "$")]
    assert s._text_prices("Was $9.99 now $4.99") == [(4.99, "$")]
    assert s._text_prices("Free shipping on orders over $50") == []


# ---------------------------------------------------------------- shipping
def test_threshold_free_shipping_is_conditional():
    info = text_info("Free shipping on orders over $100. LM7805 $0.89")
    assert info["shipping_cost"] is None
    assert info["free_ship_min"] == 100.0


def test_threshold_before_free_shipping():
    info = text_info("Spend $50 for free shipping")
    assert info["shipping_cost"] is None
    assert info["free_ship_min"] == 50.0


def test_unconditional_free_shipping():
    assert text_info("This item ships with free shipping")["shipping_cost"] == "Free"


def test_shipping_threshold_is_not_the_fee():
    info = text_info("Free shipping on orders over $100. Standard shipping $5.99")
    assert info["shipping_cost"] == "$5.99"


def test_apply_qty_frees_shipping_only_above_threshold():
    small = candidate(0, 0.89, shipping_cost=None, free_ship_min=100.0)
    assert small["shipping_cost"] is None and small["total"] is None
    big = candidate(0, 0.89, shipping_cost=None, free_ship_min=100.0, qty=200)
    assert big["shipping_cost"] == "Free" and big["total"] == big["subtotal"]


@pytest.mark.parametrize("text,days", [("72 hours", 3), ("24 hours", 1), ("2 hours", 1),
                                       ("3-5 business days", 5), ("2 weeks", 14)])
def test_ship_days(text, days):
    assert s.ship_days(text) == days


# ---------------------------------------------------------------- fetching
class FakeResponse:
    def __init__(self, ctype, body=b"<html><h1>Hi</h1></html>", status=200):
        self.status_code, self.headers, self._body = status, {"Content-Type": ctype}, body
        self.encoding = "utf-8"

    def iter_content(self, size):
        for i in range(0, len(self._body), size):
            yield self._body[i:i + size]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_pdf_results_are_not_parsed(monkeypatch):
    monkeypatch.setattr(s.requests, "get", lambda *a, **k: FakeResponse("application/pdf"))
    r = s.part_info({"title": "LM7805 datasheet", "url": "https://x.com/lm7805.pdf",
                     "snippet": ""})
    assert "not a web page" in r["fetch_error"]


def test_html_is_fetched(monkeypatch):
    monkeypatch.setattr(s.requests, "get",
                        lambda *a, **k: FakeResponse("text/html; charset=utf-8"))
    assert "<h1>Hi</h1>" in s._fetch_html("https://x.com/")


def test_parse_errors_stay_on_their_page(monkeypatch):
    monkeypatch.setattr(s.requests, "get", lambda *a, **k: FakeResponse("text/html"))
    monkeypatch.setattr(s, "_from_meta", lambda *a: 1 / 0)
    r = s.part_info({"title": "t", "url": "https://x.com/p", "snippet": "$1.00"})
    assert r["fetch_error"].startswith("parse error")
    assert r["cost"] == 1.0  # snippet fallback still runs


# ---------------------------------------------------------------- ranking
def test_exact_part_number_has_no_variants():
    _, key_hit, variants = s.relevance("LM317", {"title": "LM317 regulator TO-220",
                                                 "snippet": "", "url": "https://x.com/lm317"})
    assert key_hit and variants == []


def test_longer_part_number_is_flagged_as_variant():
    _, key_hit, variants = s.relevance("LM317", {"title": "LM317L regulator TO-92",
                                                 "snippet": "", "url": "https://x.com/p/1"})
    assert key_hit and variants == ["LM317L"]


def test_variant_ranks_below_exact_match_at_same_price():
    variant = candidate(0, 1.00, title="LM317L regulator")  # better search rank
    exact = candidate(1, 1.00, title="LM317 regulator")
    s.score_all([variant, exact], "LM317", 1)
    assert exact["score"] < variant["score"]
    assert any("check suffix" in n for n in variant["notes"])


def test_zero_price_cannot_top_the_ranking():
    soup = BeautifulSoup('<meta property="product:price:amount" content="0">', "html.parser")
    info = s._blank_info()
    s._from_meta(soup, info)
    rs = [candidate(0, 0.89), candidate(1, info["cost"])]
    s.score_all(rs, "LM7805", 1)
    top = s.pick_top(rs, 2)
    assert top[0]["cost"] == 0.89 and top[0]["score"] == pytest.approx(1.0)
