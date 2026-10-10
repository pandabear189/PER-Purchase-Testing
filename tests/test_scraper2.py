"""Offline tests for scraper2.py: no network calls, the LLM is faked.

    pip install requests beautifulsoup4 pytest
    pytest
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import scraper2 as s  # noqa: E402

PAGE = """<html><head>
<script type="application/ld+json">{"@type": "Product", "name": "LM7805",
  "offers": {"@type": "Offer", "price": "5.90", "priceCurrency": "USD"}}</script>
<style>.x { color: red }</style></head>
<body><nav>Cart $0.00</nav><h1>LM7805 Regulator (10pcs)</h1>
<s>$9.99</s> $5.90 In stock: 120. Free shipping over $50
<footer>Gift cards from $25</footer></body></html>"""


class FakeLLM:
    """Records prompts and returns a canned reply (or raises one)."""

    def __init__(self, reply=None, error=None):
        self.reply, self.error, self.prompts = reply or {}, error, []

    def json(self, system, user):
        self.prompts.append((system, user))
        if self.error:
            raise s.LLMError(self.error)
        return self.reply


def result(url="https://acme.com/p/lm7805", title="LM7805", snippet=""):
    return {"title": title, "url": url, "snippet": snippet, "search_rank": 0}


def info_from(reply, source="llm"):
    return s.to_info(reply, s._blank_info(), source)


def candidate(i, cost, qty=1, **kw):
    r = {**result(f"https://site{i}.com/p/{i}", "LM7805 voltage regulator"),
         "search_rank": i, **s._blank_info()}
    r.update(cost=cost, cost_source="llm", supplier=f"site{i}", in_stock=True,
             shipping_cost="Free", shipping_time="2 days")
    r.update(kw)
    return s.apply_qty(r, qty)


# ---------------------------------------------------------------- page -> prompt
def test_page_text_keeps_product_text_and_json_ld():
    title, ld, text = s.page_text(PAGE)
    assert title == "LM7805 Regulator (10pcs)"
    assert '"price": "5.90"' in ld
    assert "$5.90 In stock: 120" in text


def test_page_text_drops_nav_footer_and_code():
    _, _, text = s.page_text(PAGE)
    assert "Cart" not in text and "Gift cards" not in text and "color: red" not in text


def test_page_text_marks_struck_out_prices():
    _, _, text = s.page_text(PAGE)
    assert "[struck-out: $9.99]" in text


def test_page_text_is_capped_and_starts_near_heading():
    html = f"<body><p>{'banner ' * 2000}</p><h1>LM7805</h1><p>{'x' * 20000}</p></body>"
    _, _, text = s.page_text(html)
    assert len(text) == s.PAGE_CHARS
    assert "LM7805" in text  # heading is inside the window, not cut off by the banner


# ---------------------------------------------------------------- LLM JSON -> info
def test_to_info_full_reply():
    info = info_from({"cost": "5.90", "currency": "usd", "supplier": " Acme ",
                      "stock": 120, "pack_size": 10, "free_ship_min": 50,
                      "shipping_time": "2-3 days", "price_min": 5.9, "price_max": 7,
                      "price_breaks": [{"min_qty": 100, "price": 0.45}]})
    assert info["cost"] == 5.9 and info["currency"] == "USD" and info["cost_source"] == "llm"
    assert info["supplier"] == "Acme" and info["in_stock"] is True  # implied by stock
    assert info["pack_size"] == 10 and info["free_ship_min"] == 50
    assert info["price_range"] == [5.9, 7] and info["price_breaks"] == [(100, 0.45)]


@pytest.mark.parametrize("ship,expected", [(0, "Free"), (4.5, "4.50 USD"), (None, None),
                                           ("free", None), (-3, None)])
def test_to_info_shipping_cost(ship, expected):
    assert info_from({"currency": "USD", "shipping_cost": ship})["shipping_cost"] == expected


@pytest.mark.parametrize("cost", [0, "0.00", None, "call", True, -1])
def test_to_info_rejects_unusable_prices(cost):
    info = info_from({"cost": cost})
    assert info["cost"] is None and info["cost_source"] is None


def test_to_info_ignores_bad_types():
    info = info_from({"stock": "lots", "in_stock": "yes", "pack_size": 0, "supplier": 7,
                      "currency": 5, "price_breaks": [{"min_qty": 1, "price": 2},
                                                      "junk", {"min_qty": 10}],
                      "listing_page": "no"})
    assert info["stock"] is None and info["in_stock"] is None and info["pack_size"] == 1
    assert info["supplier"] is None and info["currency"] is None
    assert info["price_breaks"] == []  # qty-1 tier and incomplete tiers are dropped
    assert info["listing_page"] is False


def test_to_info_out_of_stock_wins_over_stock_count():
    assert info_from({"stock": 0, "in_stock": False})["in_stock"] is False


def test_listing_url_stays_a_listing_even_if_llm_disagrees():
    info = s._blank_info()
    info["listing_page"] = True
    assert s.to_info({"listing_page": False}, info, "llm")["listing_page"] is True


# ---------------------------------------------------------------- part_info
def test_part_info_sends_page_and_maps_reply(monkeypatch):
    monkeypatch.setattr(s, "_fetch_html", lambda url: PAGE)
    llm = FakeLLM({"cost": 5.9, "currency": "USD", "supplier": "Acme", "pack_size": 10})
    r = s.part_info(result(snippet="LM7805 from $5"), llm, "LM7805")
    _, prompt = llm.prompts[0]
    assert "Searching for: LM7805" in prompt and "JSON-LD:" in prompt
    assert "Page text:" in prompt and "LM7805 from $5" in prompt
    assert r["cost"] == 5.9 and r["cost_source"] == "llm" and r["supplier"] == "Acme"
    assert r["page_title"] == "LM7805 Regulator (10pcs)" and r["fetch_error"] is None


def test_failed_fetch_falls_back_to_snippet(monkeypatch):
    def boom(url):
        raise s.BackendError("HTTP 403")
    monkeypatch.setattr(s, "_fetch_html", boom)
    llm = FakeLLM({"cost": 1.25})
    r = s.part_info(result(snippet="LM7805 $1.25"), llm, "LM7805")
    assert "Page text" not in llm.prompts[0][1]
    assert r["cost"] == 1.25 and r["cost_source"] == "llm snippet"
    assert r["fetch_error"] == "HTTP 403" and r["supplier"] == "acme.com"


def test_parse_errors_stay_on_their_page(monkeypatch):
    monkeypatch.setattr(s, "_fetch_html", lambda url: PAGE)
    monkeypatch.setattr(s, "page_text", lambda html: 1 / 0)
    r = s.part_info(result(), FakeLLM({"cost": 2}), "LM7805")
    assert r["fetch_error"].startswith("parse error") and r["cost"] == 2


def test_llm_error_is_recorded_not_raised(monkeypatch):
    monkeypatch.setattr(s, "_fetch_html", lambda url: PAGE)
    r = s.part_info(result(), FakeLLM(error="HTTP 429"), "LM7805")
    assert r["cost"] is None and r["fetch_error"] == "llm: HTTP 429"


def test_listing_url_is_flagged(monkeypatch):
    monkeypatch.setattr(s, "_fetch_html", lambda url: PAGE)
    r = s.part_info(result("https://shop.com/search?q=lm7805"), FakeLLM(), "LM7805")
    assert r["listing_page"] is True


def test_snippets_info_maps_rows_by_index():
    rs = [result(f"https://site{i}.com/p", f"LM7805 #{i}") for i in range(3)]
    llm = FakeLLM({"results": [{"index": 2, "cost": 3}, {"index": 0, "cost": 1}, "junk"]})
    out = s.snippets_info(rs, llm, "LM7805")
    assert [r["cost"] for r in out] == [1, None, 3]
    assert out[0]["cost_source"] == "llm snippet" and out[1]["supplier"] == "site1.com"
    assert len(llm.prompts) == 1 and "[2] LM7805 #2" in llm.prompts[0][1]


# ---------------------------------------------------------------- qty + ranking
def test_apply_qty_uses_llm_pack_size():
    r = candidate(0, 5.90, qty=25, pack_size=10)
    assert r["packs_needed"] == 3 and r["subtotal"] == 17.7 and r["unit_cost"] == 0.59


def test_price_break_overrides_pack():
    r = candidate(0, 5.90, qty=150, pack_size=10, price_breaks=[(100, 0.45)])
    assert r["pack_size"] == 1 and r["subtotal"] == 67.5
    assert r["price_breaks"] == [{"min_qty": 100, "price": 0.45}]


def test_stock_counts_packs():
    assert candidate(0, 5.90, qty=25, pack_size=10, stock=3)["enough_stock"] is True
    assert candidate(0, 5.90, qty=25, pack_size=10, stock=2)["enough_stock"] is False


def test_apply_qty_frees_shipping_only_above_threshold():
    small = candidate(0, 0.89, shipping_cost=None, free_ship_min=100.0)
    assert small["shipping_cost"] is None and small["total"] is None
    big = candidate(0, 0.89, shipping_cost=None, free_ship_min=100.0, qty=200)
    assert big["shipping_cost"] == "Free" and big["total"] == big["subtotal"]


def test_snippet_prices_rank_below_page_prices():
    page, snip = candidate(1, 1.00), candidate(0, 1.00, cost_source="llm snippet")
    s.score_all([snip, page], "LM7805", 1)
    assert page["score"] < snip["score"]


# ---------------------------------------------------------------- LLM client
class FakePost:
    def __init__(self, status, body=None, headers=None, text=""):
        self.status_code, self._body = status, body
        self.headers, self.text = headers or {}, text

    def json(self):
        if self._body is None:
            raise ValueError("no json")
        return self._body


def chat(content):
    return FakePost(200, {"choices": [{"message": {"content": content}}]})


@pytest.fixture
def no_keys(monkeypatch):
    for env in ("GEMINI_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY", "LLM_MODEL"):
        monkeypatch.delenv(env, raising=False)
    monkeypatch.setattr(s, "_ollama_up", lambda: False)


@pytest.fixture
def llm(no_keys, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "k")
    monkeypatch.setattr(s.time, "sleep", lambda t: None)
    return s.LLM(rpm=60)


def replies(monkeypatch, *responses):
    calls, queue = [], list(responses)

    def post(url, json, timeout, headers):
        calls.append({"url": url, "body": json, "headers": headers})
        return queue.pop(0)
    monkeypatch.setattr(s.requests, "post", post)
    return calls


def test_no_provider_configured(no_keys):
    with pytest.raises(s.LLMError, match="GEMINI_API_KEY"):
        s.LLM()


def test_first_provider_with_a_key_wins(no_keys, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "a")
    monkeypatch.setenv("GROQ_API_KEY", "b")
    llm = s.LLM()
    assert llm.provider == "groq" and llm.key == "b"
    assert llm.url == "https://api.groq.com/openai/v1/chat/completions"


def test_ollama_used_when_running_and_no_keys(no_keys, monkeypatch):
    monkeypatch.setattr(s, "_ollama_up", lambda: True)
    assert s.LLM().provider == "ollama"


def test_model_override_order(no_keys, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    assert s.LLM().model == s.PROVIDERS["gemini"][2]
    monkeypatch.setenv("LLM_MODEL", "env-model")
    assert s.LLM().model == "env-model"
    assert s.LLM(model="arg-model").model == "arg-model"


def test_explicit_provider_needs_its_key(no_keys, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    with pytest.raises(s.LLMError, match="GROQ_API_KEY not set"):
        s.LLM("groq")


def test_request_shape(llm, monkeypatch):
    calls = replies(monkeypatch, chat('{"cost": 1}'))
    assert llm.json("sys", "user") == {"cost": 1}
    body = calls[0]["body"]
    assert body["temperature"] == 0 and body["response_format"] == {"type": "json_object"}
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    assert calls[0]["headers"]["Authorization"] == "Bearer k"


def test_code_fenced_json_is_accepted(llm, monkeypatch):
    replies(monkeypatch, chat('```json\n{"cost": 2.5}\n```'))
    assert llm.json("s", "u") == {"cost": 2.5}


def test_rate_limit_is_retried(llm, monkeypatch):
    calls = replies(monkeypatch, FakePost(429, headers={"Retry-After": "1"}),
                    FakePost(503), chat('{"ok": true}'))
    assert llm.json("s", "u") == {"ok": True} and len(calls) == 3


def test_non_json_reply_is_retried_then_fails(llm, monkeypatch):
    replies(monkeypatch, *[chat("sorry, I can't") for _ in range(5)])
    with pytest.raises(s.LLMError, match="not JSON"):
        llm.json("s", "u")


def test_client_error_fails_fast(llm, monkeypatch):
    calls = replies(monkeypatch, FakePost(400, text="model not found"))
    with pytest.raises(s.LLMError, match="HTTP 400: model not found"):
        llm.json("s", "u")
    assert len(calls) == 1


def test_requests_are_paced(no_keys, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "k")
    clock, slept = [100.0], []
    monkeypatch.setattr(s.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(s.time, "sleep", slept.append)
    llm = s.LLM(rpm=30)  # one request every 2s
    for _ in range(3):
        llm._wait_turn()
    assert slept == [0, 2, 4]
