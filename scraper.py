#!/usr/bin/env python3
"""
top5_search.py - Find the best 5 places to buy a part.

Searches ~30 results, scrapes each for price / supplier / stock / shipping,
then ranks them and returns the top 5 (see "ranking" section for the weights).

Search sources (first that works wins; no google.com scraping, so no CAPTCHAs):
  1. Brave Search API        (free tier, ~2k queries/mo)   env: BRAVE_API_KEY
  2. Serper.dev Google results (2,500 free, no card)       env: SERPER_API_KEY
  3. Google Programmable Search JSON API (100/day free)    env: GOOGLE_API_KEY, GOOGLE_CSE_ID
  4. DuckDuckGo HTML endpoint (no key; paginated politely)
If the main query gives fewer than --pool unique URLs, "buy <part>" and
"<part> in stock" variants top it up.

Each page is mined for cost, supplier, shipping time/cost, stock, pack size
and quantity price breaks: schema.org JSON-LD first, then meta/microdata
tags, then regex over page text and the search snippet. Sites that block
scripts (eBay, DigiKey, ...) fall back to the snippet.

Ranking (--rank best, default):
  score = order cost incl. shipping / cheapest order cost  x  penalties for
  out-of-stock or short stock, slow or unknown shipping, estimated shipping
  cost, low-confidence price (snippet / search-results page), listings that
  don't mention the query words, and suspiciously cheap outliers.
  1.00 = the cheapest option with no drawbacks. Lower is better.
  Listings without the part number or a price only fill in if too few remain,
  and at most --per-supplier (2) options come from one supplier.
--rank cheapest just sorts by order total (same filters).

Usage:
    pip install requests beautifulsoup4
    python scraper.py "LM7805 voltage regulator"
    python scraper.py "LM7805" -q 50                 # need 50 units
    fh --rank cheapest
    python scraper.py "LM7805" --pool 20 -n 5        # compare 50, show best 10
    python scraper.py "query" --json
    python scraper.py "query" --exact                # no "price"/variant queries
    python scraper.py "query" --no-fetch             # snippets only (fast)
"""
import argparse
import json
import os
import random
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from statistics import median
from urllib.parse import parse_qs, unquote, urlparse

import requests
from bs4 import BeautifulSoup

TIMEOUT = 15
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/128.0 Safari/537.36")


class BackendError(Exception):
    pass


# ---------------------------------------------------------------- backends
# Each backend pages through results until it has `n` (or runs out).
def brave(query, n):
    key = os.getenv("BRAVE_API_KEY")
    if not key:
        raise BackendError("BRAVE_API_KEY not set")
    out = []
    for page in range(10):  # Brave: count <= 20, offset is a page index (max 9)
        r = requests.get(
            "https://api.search.brave.com/res/v1/web/search",
            params={"q": query, "count": 20, "offset": page},
            headers={"X-Subscription-Token": key, "Accept": "application/json"},
            timeout=TIMEOUT,
        )
        if r.status_code != 200:
            if out:
                break
            raise BackendError(f"HTTP {r.status_code}")
        items = r.json().get("web", {}).get("results", [])
        out += [{"title": i.get("title", ""), "url": i.get("url", ""),
                 "snippet": BeautifulSoup(i.get("description", ""), "html.parser").get_text()}
                for i in items]
        if len(out) >= n or len(items) < 20:
            break
    return out[:n]


def serper(query, n):
    key = os.getenv("SERPER_API_KEY")
    if not key:
        raise BackendError("SERPER_API_KEY not set")
    out = []
    for page in range(1, min(-(-n // 10), 10) + 1):  # 10 per call = 1 credit each
        r = requests.post(
            "https://google.serper.dev/search",
            json={"q": query, "num": 10, "page": page},
            headers={"X-API-KEY": key, "Content-Type": "application/json"},
            timeout=TIMEOUT,
        )
        if r.status_code != 200:
            if out:
                break
            raise BackendError(f"HTTP {r.status_code}")
        items = r.json().get("organic", [])
        out += [{"title": i.get("title", ""), "url": i.get("link", ""),
                 "snippet": i.get("snippet", "")} for i in items]
        if len(items) < 10:
            break
    return out[:n]


def google_cse(query, n):
    key, cx = os.getenv("GOOGLE_API_KEY"), os.getenv("GOOGLE_CSE_ID")
    if not (key and cx):
        raise BackendError("GOOGLE_API_KEY / GOOGLE_CSE_ID not set")
    out = []
    for start in range(1, min(n, 100) + 1, 10):  # CSE: 10 per call, max 100 total
        r = requests.get(
            "https://www.googleapis.com/customsearch/v1",
            params={"key": key, "cx": cx, "q": query, "num": 10, "start": start},
            timeout=TIMEOUT,
        )
        if r.status_code != 200:
            if out:
                break
            raise BackendError(f"HTTP {r.status_code}")
        items = r.json().get("items", [])
        out += [{"title": i.get("title", ""), "url": i.get("link", ""),
                 "snippet": i.get("snippet", "")} for i in items]
        if len(items) < 10:
            break
    return out[:n]


def _clean_ddg_url(href):
    # DDG sometimes wraps links: //duckduckgo.com/l/?uddg=<encoded>
    if "duckduckgo.com/l/" in href:
        qs = parse_qs(urlparse(href).query)
        if "uddg" in qs:
            return unquote(qs["uddg"][0])
    return href


def _ddg_page(form, retries=2):
    """POST one DDG HTML results page; return (results, next-page form data or None)."""
    last = None
    for attempt in range(retries + 1):
        if attempt:
            # polite backoff; HTTP 202 is DDG's "slow down", so wait longer for it
            time.sleep(2 ** attempt * (3 if last == "HTTP 202" else 1) + random.random())
        try:
            r = requests.post(
                "https://html.duckduckgo.com/html/",
                data=form,
                headers={"User-Agent": UA, "Referer": "https://html.duckduckgo.com/"},
                timeout=TIMEOUT,
            )
        except requests.RequestException as e:
            last = str(e)
            continue
        if r.status_code != 200:
            last = f"HTTP {r.status_code}"
            continue
        soup = BeautifulSoup(r.text, "html.parser")
        # Challenge / anomaly page detection
        if soup.select_one(".anomaly-modal, #challenge-form") or "captcha" in r.text.lower():
            last = "challenge page returned"
            continue
        results = []
        for res in soup.select("div.result"):
            if "result--ad" in res.get("class", []):
                continue  # skip sponsored
            a = res.select_one("a.result__a")
            if not a:
                continue
            snip = res.select_one(".result__snippet")
            results.append({
                "title": a.get_text(strip=True),
                "url": _clean_ddg_url(a.get("href", "")),
                "snippet": snip.get_text(" ", strip=True) if snip else "",
            })
        nxt = None
        for f in soup.select("form"):
            data = {i["name"]: i.get("value", "") for i in f.select("input[name]")}
            if "s" in data and "vqd" in data:  # the "Next" button's hidden form
                nxt = data
        if results:
            return results, nxt
        last = "no results parsed"
    raise BackendError(last or "unknown error")


def duckduckgo(query, n):
    out, form = [], {"q": query}
    while form and len(out) < n:
        if out:
            time.sleep(1 + random.random())  # be gentle between pages
        try:
            page, form = _ddg_page(form)
        except BackendError:
            if out:
                break  # keep what we have
            raise
        out += page
    return out[:n]


BACKENDS = [("brave", brave), ("serper", serper), ("google_cse", google_cse), ("duckduckgo", duckduckgo)]


# ---------------------------------------------------------------- part info
CURRENCY_SYMBOLS = {"$": "USD", "€": "EUR", "£": "GBP", "¥": "JPY"}
# The lookahead rejects split-up display prices like "$ 5 . 99" (would read as $5).
MONEY = (r"(?:US\s?)?([$€£¥])\s?(\d{1,3}(?:,\d{3})+(?:\.\d{1,4})?|\d+(?:\.\d{1,4})?)"
         r"(?![\d,]|\s?\.\s?\d)")
MONEY_RE = re.compile(MONEY)
# Amounts that aren't the item's selling price: filters, free-shipping thresholds,
# struck-out list/was prices, discounts, shipping fees.
NOISE_BEFORE_RE = re.compile(
    r"(?:under|over|above|below|up to|\bon|orders?(?:\s(?:over|of|above))?|spend|save|"
    r"savings|off|coupon|was|list(?:\sprice)?|typical(?:\sprice)?|msrp|rrp|reg(?:ular)?|"
    r"compare\sat|retail|shipping|delivery|postage|credit|reward|gift\scard)\W{0,3}$", re.I)
NOISE_AFTER_RE = re.compile(r"^\s?(?:off|or\smore|\+|and\sup|minimum|shipping|delivery)\b",
                            re.I)
# Search / category pages list many different items, so we report the lowest price.
LISTING_URL_RE = re.compile(r"/s\?|/search|/sch/|/b/|[?&](?:q|k|_nkw|keywords?|search)=|"
                            r"/category|/c/|/browse", re.I)
FREE_SHIP_RE = re.compile(r"\bfree\s+(?:standard\s+|ground\s+|economy\s+|2-day\s+)?"
                          r"(?:shipping|delivery)\b", re.I)
SHIP_COST_RE = re.compile(r"(?:shipping|delivery|postage)(?:\s+(?:cost|fee|charge|rate))?"
                          r"[^$€£¥\n.]{0,25}" + MONEY, re.I)
SPAN = r"(\d+\s?(?:-|–|to)\s?\d+|\d+)\s?(business\s|working\s)?(hours?|days?|weeks?)"
SHIP_TIME_RES = [
    re.compile(r"\b(?:ships?|dispatch(?:es|ed)?|deliver(?:y|ed|s)?|arrives?|shipping)"
               r"\s(?:with)?in\s(?:about\s)?" + SPAN, re.I),
    re.compile(r"\blead\s?time\W{0,3}(?:of\s)?" + SPAN, re.I),
    re.compile(r"\b(?:arrives?|get it|delivered|delivery)\s(?:by\s)?"
               r"((?:mon|tue|wed|thu|fri|sat|sun)[a-z]*,?\s[a-z]{3,9}\.?\s\d{1,2})", re.I),
    re.compile(r"\b(ships\s(?:today|same\sday|next\sbusiness\sday)|same[-\s]day\sshipping)", re.I),
]
PACK_RES = [
    re.compile(r"\b(?:pack|set|lot|box|bag|case|package)\s(?:of\s)?(\d{1,5})\b", re.I),
    re.compile(r"\(?\b(\d{1,5})\s?[-\s]?(?:pcs|pc|pieces|pack|pk|count|ct|units?)\b\)?", re.I),
]
STOCK_RES = [
    re.compile(r"\b(\d{1,3}(?:,\d{3})+|\d+)\s(?:in\sstock|available|units?\sin\sstock)\b", re.I),
    re.compile(r"\b(?:in\sstock|stock|qty\savailable)\W{0,3}(\d{1,3}(?:,\d{3})+|\d+)\b", re.I),
    re.compile(r"\bonly\s(\d+)\sleft\b", re.I),
]
OUT_OF_STOCK_RE = re.compile(r"\b(?:out\sof\sstock|sold\sout|currently\sunavailable|backorder(?:ed)?)\b", re.I)


def _to_float(s):
    try:
        return float(str(s).replace(",", ""))
    except (TypeError, ValueError):
        return None


# JSON-LD keys that hold *other* products (carousels, accessories) on a product page.
RELATED_LD_KEYS = {"isRelatedTo", "isSimilarTo", "isAccessoryOrSparePartFor",
                   "isConsumableFor", "itemListElement", "review", "isVariantOf"}


def _iter_ld(node, skip=frozenset(), product=None):
    """Yield (dict, owning product name) for every dict inside a JSON-LD blob
    (handles @graph, lists, nesting); keys in `skip` aren't descended into."""
    if isinstance(node, dict):
        if _types(node) & {"Product", "ProductGroup"}:
            product = node.get("name") or product
        yield node, product
        for k, v in node.items():
            if k not in skip:
                yield from _iter_ld(v, skip, product)
    elif isinstance(node, list):
        for v in node:
            yield from _iter_ld(v, skip, product)


def _same_url(a, b):
    a, b = urlparse(a or ""), urlparse(b or "")
    return bool(a.netloc) and (a.netloc.removeprefix("www."), a.path.rstrip("/"), a.query) == \
        (b.netloc.removeprefix("www."), b.path.rstrip("/"), b.query)


def _types(d):
    t = d.get("@type", [])
    return {t} if isinstance(t, str) else set(t or [])


def _fmt_span(lo, hi, unit="days"):
    if lo is None and hi is None:
        return None
    if lo is None or lo == hi:
        return f"{hi} {unit}"
    return f"{lo}-{hi} {unit}" if hi is not None else f"{lo}+ {unit}"


def _ld_breaks(offer, info):
    """Collect quantity price breaks: Offer.eligibleQuantity or priceSpecification entries."""
    specs = offer.get("priceSpecification") or []
    specs = specs if isinstance(specs, list) else [specs]
    for s in [offer] + [s for s in specs if isinstance(s, dict)]:
        eq = s.get("eligibleQuantity")
        price = _to_float(s.get("price"))
        if isinstance(eq, dict) and price is not None:
            min_q = _to_float(eq.get("minValue") or eq.get("value"))
            if min_q:
                info["price_breaks"].append((int(min_q), price))


def _min_qty(offer):
    eq = offer.get("eligibleQuantity")
    return (_to_float(eq.get("minValue") or eq.get("value")) or 1) if isinstance(eq, dict) else 1


def _from_json_ld(soup, info, page_url="", title="", listing=False):
    """Price + details from schema.org JSON-LD. On a product page, only offers of the
    product in the page title count (not related-item carousels), and quantity-break
    offers (min qty > 1) are kept as price breaks rather than the base price."""
    offers = []  # (price, currency, offer url, product name, min qty)
    skip = frozenset() if listing else RELATED_LD_KEYS
    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(tag.string or "", strict=False)
        except ValueError:
            continue
        for d, product in _iter_ld(data, skip):
            types = _types(d)
            if types & {"Offer", "AggregateOffer"}:
                price = _to_float(d.get("lowPrice") or d.get("price"))
                if price:
                    offers.append((price, d.get("priceCurrency"), d.get("url"), product,
                                   _min_qty(d)))
                    high = _to_float(d.get("highPrice"))
                    if high and high != price:
                        offers.append((high, d.get("priceCurrency"), None, product, _min_qty(d)))
                _ld_breaks(d, info)
                avail = str(d.get("availability") or "")
                if avail and info["in_stock"] is None:
                    info["in_stock"] = avail.rsplit("/", 1)[-1] in ("InStock", "LimitedAvailability",
                                                                    "InStoreOnly", "OnlineOnly")
                inv = d.get("inventoryLevel")
                if isinstance(inv, dict) and info["stock"] is None:
                    stock = _to_float(inv.get("value"))
                    info["stock"] = int(stock) if stock is not None else None
                seller = d.get("seller")
                if isinstance(seller, dict) and seller.get("name") and not info["supplier"]:
                    info["supplier"] = seller["name"]
                ships = d.get("shippingDetails")
                for s in ships if isinstance(ships, list) else [ships]:
                    if not isinstance(s, dict):
                        continue
                    rate = s.get("shippingRate")
                    if isinstance(rate, dict) and info["shipping_cost"] is None:
                        v = _to_float(rate.get("value"))
                        if v is not None:
                            info["shipping_cost"] = "Free" if v == 0 else \
                                f"{v:.2f} {rate.get('currency', '')}".strip()
                    dt = s.get("deliveryTime")
                    if isinstance(dt, dict) and info["shipping_time"] is None:
                        lo = hi = 0
                        found = False
                        for part in ("handlingTime", "transitTime"):
                            p = dt.get(part)
                            if isinstance(p, dict):
                                found = True
                                lo += int(_to_float(p.get("minValue")) or 0)
                                hi += int(_to_float(p.get("maxValue") or p.get("minValue")) or 0)
                        if found and hi:
                            info["shipping_time"] = _fmt_span(lo, hi)
            if "Organization" in types and not info["supplier"] and d.get("name"):
                info["supplier"] = d["name"]
    if not offers:
        return
    if not listing and title:  # drop other products' offers
        t = _alnum(title)
        mine = [o for o in offers if o[3] and _alnum(o[3]) and
                (_alnum(o[3]) in t or t in _alnum(o[3]))]
        offers = mine or offers
    offers = [o for o in offers if o[4] <= 1] or offers  # qty-1 price, not bulk tiers
    prices = sorted({o[0] for o in offers})
    pick = next((o for o in offers if _same_url(o[2], page_url)), None)  # linked variant
    if pick is None:
        pick = min(offers)
    info["cost"], info["currency"] = pick[0], pick[1] or info["currency"]
    info["cost_source"] = "json-ld" if len(prices) == 1 or pick[2] else \
        f"json-ld (lowest of {len(prices)})"
    info["ld_prices"] = prices
    if len(prices) > 1:
        info["price_range"] = [prices[0], prices[-1]]
        info["price_count"] = len(prices)


def _from_meta(soup, info):
    def meta(*names):
        for n in names:
            m = soup.find("meta", attrs={"property": n}) or soup.find("meta", attrs={"name": n})
            if m and m.get("content"):
                return m["content"].strip()
        return None

    if info["cost"] is None:
        price = _to_float(meta("product:price:amount", "og:price:amount"))
        if price is None:
            vals = [_to_float(el.get("content") or re.sub(r"[^\d.,]", "", el.get_text()))
                    for el in soup.select("[itemprop=price]")]
            vals = [v for v in vals if v]
            price = vals[0] if vals else None
            if len(set(vals)) > 1:
                info["ld_prices"] = sorted(set(vals))  # cross-checked against the shown price
        if price is not None:
            info["cost"] = price
            cur = meta("product:price:currency", "og:price:currency")
            if not cur:
                el = soup.select_one("[itemprop=priceCurrency]")
                cur = el and (el.get("content") or el.get_text(strip=True))
            info["currency"] = cur or info["currency"]
            info["cost_source"] = "meta"
    if not info["supplier"]:
        info["supplier"] = meta("og:site_name", "application-name")


def _text_prices(text):
    """All plausible selling prices in text, in page order, as (value, symbol)."""
    out = []
    for m in MONEY_RE.finditer(text):
        before, after = text[max(0, m.start() - 30):m.start()], text[m.end():m.end() + 15]
        if NOISE_BEFORE_RE.search(before) or NOISE_AFTER_RE.search(after):
            continue
        v = _to_float(m.group(2))
        if v:
            out.append((v, m.group(1)))
    return out


def _from_text(text, info, source, listing=False):
    if info["cost"] is None:
        prices = _text_prices(text)
        if prices:
            # Product page: first real price is the item's. Listing page: lowest.
            v, sym = min(prices) if listing else prices[0]
            info["cost"] = v
            info["currency"] = CURRENCY_SYMBOLS.get(sym, sym)
            info["cost_source"] = source + (f" (lowest of {len(prices)})" if listing
                                            and len(prices) > 1 else "")
            vals = [p for p, _ in prices]
            if len(set(vals)) > 1:
                info["price_range"] = [min(vals), max(vals)]
                info["price_count"] = len(vals)
    if info["shipping_cost"] is None:
        if FREE_SHIP_RE.search(text):
            info["shipping_cost"] = "Free"
        else:
            m = SHIP_COST_RE.search(text)
            if m:
                info["shipping_cost"] = f"{m.group(1)}{m.group(2)}"
    if info["shipping_time"] is None:
        for rx in SHIP_TIME_RES:
            m = rx.search(text)
            if m:
                info["shipping_time"] = " ".join(g.strip() for g in m.groups() if g)
                break
    if info["stock"] is None:
        for rx in STOCK_RES:
            m = rx.search(text)
            if m:
                info["stock"] = int(m.group(1).replace(",", ""))
                break
    if info["in_stock"] is None:
        if OUT_OF_STOCK_RE.search(text):
            info["in_stock"] = False
        elif info["stock"] or re.search(r"\bin\sstock\b", text, re.I):
            info["in_stock"] = True


def _pack_size(title):
    """Units per listing, e.g. '(30pcs)' or 'Pack of 10'. Title-only: page text is too noisy."""
    for rx in PACK_RES:
        m = rx.search(title)
        if m and 1 < int(m.group(1)) <= 10000:
            return int(m.group(1))
    return 1


def apply_qty(r, qty):
    """Work out unit cost, packs to buy, order total and stock check for `qty` units."""
    breaks = sorted(set(r["price_breaks"]))
    r["price_breaks"] = [{"min_qty": q, "price": p} for q, p in breaks]
    # Price breaks are per-unit prices keyed by min quantity; use the best tier reached.
    tier = [p for q, p in breaks if q <= qty]
    pack = 1 if tier else _pack_size(r["title"])
    price = tier[-1] if tier else r["cost"]
    packs = -(-qty // pack)
    r.update(qty=qty, pack_size=pack, packs_needed=packs,
             unit_cost=None, subtotal=None, total=None, enough_stock=None)
    if r["stock"] is not None:
        # Distributors count stock in units; pack listings usually count packs.
        r["enough_stock"] = r["stock"] * pack >= qty
    elif r["in_stock"] is False:
        r["enough_stock"] = False
    if price is None:
        return r
    r["unit_cost"] = round(price / pack, 4)
    r["subtotal"] = round(price * packs, 2)
    ship = r["shipping_cost"]
    ship_val = 0.0 if ship == "Free" else _to_float(re.sub(r"[^\d.]", "", ship or "") or None)
    if ship_val is not None:
        r["total"] = round(r["subtotal"] + ship_val, 2)
    return r


def _blank_info():
    return {"cost": None, "currency": None, "cost_source": None, "supplier": None,
            "shipping_time": None, "shipping_cost": None, "stock": None, "in_stock": None,
            "price_breaks": [], "price_range": None, "price_count": 0,
            "listing_page": False, "fetch_error": None}


def part_info(result):
    """Fetch a result page and pull cost / supplier / shipping details."""
    domain = urlparse(result["url"]).netloc.lower().removeprefix("www.")
    info = _blank_info()
    listing = bool(LISTING_URL_RE.search(result["url"]))
    try:
        r = requests.get(result["url"], headers={"User-Agent": UA,
                         "Accept-Language": "en-US,en;q=0.9"}, timeout=TIMEOUT)
        if r.status_code != 200:
            raise BackendError(f"HTTP {r.status_code}")
        soup = BeautifulSoup(r.text, "html.parser")
        h1 = soup.find("h1")
        title = " ".join(h1.get_text(" ").split()) if h1 else ""
        _from_json_ld(soup, info, result["url"], title or result["title"], listing)
        _from_meta(soup, info)
        # Drop code and struck-out prices (list/was/strike-through) before reading text.
        for t in soup(["script", "style", "noscript", "s", "del", "strike"]):
            t.decompose()
        for t in soup.select('[class*="strike"], [class*="list-price"], [class*="was-price"], '
                             '[class*="old-price"], [class*="original-price"], [data-a-strike]'):
            t.decompose()
        text = " ".join(soup.get_text(" ").split())[:300_000]
        # The product's own price follows its <h1>; header/banner prices come before it.
        at = text.find(title) if title and not listing else -1
        main = text[at:] if at >= 0 else text
        shown = _text_prices(main[:3000])
        ld = info.pop("ld_prices", None)
        if shown and ld and info["cost"] != shown[0][0] and shown[0][0] in ld:
            info["cost"] = shown[0][0]  # the variant/price the page actually displays
            info["cost_source"] = info["cost_source"].split(" (")[0]
        _from_text(main, info, "page text", listing)
        if main is not text:
            _from_text(text, info, "page text", listing)  # banners: free shipping etc.
    except (BackendError, requests.RequestException) as e:
        info["fetch_error"] = str(e)
    _from_text(f"{result['title']} {result['snippet']}", info, "snippet", listing)
    info["listing_page"] = listing
    info["supplier"] = info["supplier"] or domain
    return {**result, **info}


def enrich(results):
    with ThreadPoolExecutor(max_workers=min(10, len(results) or 1)) as pool:
        return list(pool.map(part_info, results))


# ---------------------------------------------------------------- ranking
# Score = (effective order cost / cheapest effective cost) x penalty multipliers.
# 1.00 is the cheapest candidate with nothing counting against it; lower is better.
# Tune the weights here.
PENALTY = {
    "out_of_stock": 1.6,        # listed as out of stock / backordered
    "short_stock": 1.4,         # stock shown but below the quantity needed
    "stock_unknown": 1.03,
    "per_ship_day": 0.02,       # +2% per day of shipping beyond FAST_DAYS...
    "ship_day_cap": 0.40,       # ...capped at +40%
    "ship_time_unknown": 1.06,
    "ship_cost_unknown": 1.04,  # on top of adding the median known shipping cost
    "source": {"json-ld": 1.0, "meta": 1.0, "page text": 1.05, "snippet": 1.15},
    "listing_page": 1.12,       # price is the lowest of many listings, may not be this part
    "relevance": 0.6,           # x(1 + 0.6 * fraction of query words missing)
    "outlier": 1.35,            # unit cost < 25% of the median: likely wrong item/accessory
    "per_search_rank": 0.005,   # tiny tiebreak favouring higher search positions
}
FAST_DAYS = 2


def _alnum(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def relevance(query, r):
    """(fraction of query words found, part-number match). Part numbers = words with digits."""
    words = [w for w in re.findall(r"[a-z0-9][a-z0-9.\-/x]*", query.lower())
             if w not in {"price", "buy", "in", "stock", "for", "the", "a"}]
    hay = _alnum(f"{r['title']} {r['snippet']} {unquote(r['url'])}")
    found = [w for w in words if _alnum(w) and _alnum(w) in hay]
    keys = [w for w in words if re.search(r"\d", w)]
    key_hit = not keys or any(_alnum(k) in hay for k in keys)
    return (len(found) / len(words) if words else 1.0), key_hit


def ship_days(text, today=None):
    """Rough worst-case days until delivery from a shipping_time string."""
    if not text:
        return None
    t, today = text.lower(), today or date.today()
    if "today" in t or "same" in t:
        return 1
    if "next business day" in t:
        return 2
    m = re.search(r"(\d+)(?:\s?(?:-|–|to)\s?(\d+))?\s?(?:business\s|working\s)?(hour|day|week)", t)
    if m:
        n = int(m.group(2) or m.group(1))
        return {"hour": 1, "day": n, "week": 7 * n}[m.group(3)]
    m = re.search(r"([a-z]{3})[a-z]*\.?\s(\d{1,2})$", t)
    if m:
        try:
            d = datetime.strptime(f"{m.group(1)} {m.group(2)} {today.year}", "%b %d %Y").date()
        except ValueError:
            return None
        if d < today:
            d = d.replace(year=today.year + 1)
        return max((d - today).days, 1)
    return None


def _ship_value(ship):
    if ship == "Free":
        return 0.0
    return _to_float(re.sub(r"[^\d.]", "", ship or "") or None)


def score_all(results, query, qty):
    """Attach score + human-readable notes to every priced, relevant candidate."""
    priced = [r for r in results if r["subtotal"] is not None]
    known_ship = [v for v in (_ship_value(r["shipping_cost"]) for r in priced) if v is not None]
    ship_fill = median(known_ship) if known_ship else 0.0
    units = [r["unit_cost"] for r in priced if r["unit_cost"]]
    med_unit = median(units) if units else None
    main_cur = Counter(r["currency"] for r in priced).most_common(1)[0][0] if priced else None

    for r in results:
        r["score"], r["notes"] = None, []
        r["relevance"], key_hit = relevance(query, r)
        if r["subtotal"] is None:
            r["notes"].append("no price found")
            continue
        if not key_hit:
            r["notes"].append("part number not in listing")
        if r["currency"] and main_cur and r["currency"] != main_cur:
            r["notes"].append(f"priced in {r['currency']}, not {main_cur}")
        ship = _ship_value(r["shipping_cost"])
        r["effective_cost"] = round(r["subtotal"] + (ship if ship is not None else ship_fill), 2)

    ok = [r for r in results if r["subtotal"] is not None]
    # Baseline = cheapest *valid* option, so a wrong-part bargain doesn't skew everyone.
    valid = [r["effective_cost"] for r in ok if not r["notes"]] or \
        [r["effective_cost"] for r in ok]
    best = min(valid, default=0) or 0.01
    for r in ok:
        mult, notes = 1.0, r["notes"]
        if r["enough_stock"] is False and r["in_stock"] is False:
            mult *= PENALTY["out_of_stock"]; notes.append("out of stock")
        elif r["enough_stock"] is False:
            mult *= PENALTY["short_stock"]; notes.append(f"stock below {qty}")
        elif r["enough_stock"] is None and r["in_stock"] is None:
            mult *= PENALTY["stock_unknown"]
        days = ship_days(r["shipping_time"])
        r["ship_days"] = days
        if days is None:
            mult *= PENALTY["ship_time_unknown"]
        elif days > FAST_DAYS:
            mult *= 1 + min(PENALTY["per_ship_day"] * (days - FAST_DAYS), PENALTY["ship_day_cap"])
            if days >= 10:
                notes.append(f"slow: ~{days} days")
        if _ship_value(r["shipping_cost"]) is None:
            mult *= PENALTY["ship_cost_unknown"]; notes.append("shipping cost estimated")
        src = (r["cost_source"] or "snippet").split(" (")[0]
        mult *= PENALTY["source"].get(src, 1.1)
        if r["listing_page"]:
            mult *= PENALTY["listing_page"]; notes.append("search page: lowest listing")
        mult *= 1 + PENALTY["relevance"] * (1 - r["relevance"])
        if med_unit and len(units) >= 4 and r["unit_cost"] < 0.25 * med_unit:
            mult *= PENALTY["outlier"]; notes.append("far cheaper than others - verify item")
        mult *= 1 + PENALTY["per_search_rank"] * r["search_rank"]
        r["score"] = round(r["effective_cost"] / best * mult, 3)
    return results


def pick_top(results, top, mode="best", per_supplier=2):
    """Best `top` options. Hard filters first; loosened only if too few candidates pass."""
    def tier(r):
        if r["score"] is None:
            return 3  # unpriced
        if "part number not in listing" in r["notes"] or any(
                n.startswith("priced in") for n in r["notes"]):
            return 2
        if r["enough_stock"] is False:
            return 1
        return 0

    key = (lambda r: r["effective_cost"]) if mode == "cheapest" else (lambda r: r["score"])
    ordered = sorted(results, key=lambda r: (tier(r), key(r) if tier(r) < 3 else -r["relevance"],
                                             r["search_rank"]))
    picked, per = [], Counter()
    for r in ordered:  # first pass: cap listings per supplier for variety
        if per[r["supplier"]] < per_supplier:
            picked.append(r); per[r["supplier"]] += 1
        if len(picked) == top:
            break
    for r in ordered:  # fill up if the cap left gaps
        if len(picked) == top:
            break
        if r not in picked:
            picked.append(r)
    picked.sort(key=lambda r: ordered.index(r))
    return picked


# ---------------------------------------------------------------- main
def search(query, n=5, verbose=False, variants=()):
    """Top `n` unique URLs from the first working backend, topping up with `variants`."""
    errors = {}
    for name, fn in BACKENDS:
        try:
            res, seen = [], set()
            for q in (query, *variants):
                if len(res) >= n:
                    break
                try:
                    batch = fn(q, n)
                except BackendError:
                    if res:
                        continue  # extra variants are best-effort
                    raise
                for r in batch:
                    u = r["url"].split("#")[0].rstrip("/")
                    if u and u not in seen:
                        seen.add(u); res.append(r)
            if res:
                for i, r in enumerate(res[:n]):
                    r["search_rank"] = i
                return name, res[:n]
            errors[name] = "empty result"
        except (BackendError, requests.RequestException, ValueError) as e:
            errors[name] = str(e)
        if verbose:
            print(f"[{name}] skipped: {errors[name]}", file=sys.stderr)
    raise RuntimeError("All backends failed: " + json.dumps(errors, indent=2))


def main():
    p = argparse.ArgumentParser(
        description="Search many part listings, then rank the best N by cost/stock/shipping.")
    p.add_argument("query", nargs="+")
    p.add_argument("-n", "--top", type=int, default=5, help="options to return (default 5)")
    p.add_argument("--pool", type=int, default=30,
                   help="search results to scrape and compare (default 30)")
    p.add_argument("-q", "--qty", type=int, default=1,
                   help="units you need; used for pack math, price breaks and totals (default 1)")
    p.add_argument("--rank", choices=["best", "cheapest"], default="best",
                   help="best = cost weighted by stock/shipping/confidence (default); "
                        "cheapest = lowest order total only")
    p.add_argument("--per-supplier", type=int, default=2,
                   help="max options from one supplier before others get a turn (default 2)")
    p.add_argument("--json", action="store_true", help="output JSON")
    p.add_argument("--exact", action="store_true",
                   help="search the query as-is (default appends 'price' to favor listings)")
    p.add_argument("--no-fetch", action="store_true",
                   help="don't fetch result pages; only parse search snippets")
    p.add_argument("-v", "--verbose", action="store_true")
    a = p.parse_args()
    if a.qty < 1 or a.top < 1 or a.pool < a.top:
        p.error("need --qty >= 1, --top >= 1 and --pool >= --top")

    base = " ".join(a.query)
    query = base if a.exact or "price" in base.lower() else base + " price"
    variants = () if a.exact else (f"buy {base}", f"{base} in stock")
    try:
        source, results = search(query, a.pool, a.verbose, variants)
    except RuntimeError as e:
        msg = str(e)
        if "HTTP 202" in msg or "Connection reset" in msg or "challenge" in msg:
            msg += ("\n\nDuckDuckGo is rate-limiting this network (usually clears in "
                    "15-60 min).\nFor reliable 30-result searches, set a free Brave Search "
                    "API key (Serper needs no card):\n"
                    "  https://serper.dev  ->  export SERPER_API_KEY=...\n"
                    "  https://api.search.brave.com  ->  export BRAVE_API_KEY=...")
        sys.exit(msg)
    if a.verbose:
        print(f"[{source}] {len(results)} candidates, fetching...", file=sys.stderr)

    if a.no_fetch:
        for r in results:
            info = _blank_info()
            info["listing_page"] = bool(LISTING_URL_RE.search(r["url"]))
            _from_text(f"{r['title']} {r['snippet']}", info, "snippet", info["listing_page"])
            info["supplier"] = urlparse(r["url"]).netloc.lower().removeprefix("www.")
            r.update(info)
    else:
        results = enrich(results)
    results = [apply_qty(r, a.qty) for r in results]
    score_all(results, base, a.qty)
    top = pick_top(results, a.top, a.rank, a.per_supplier)
    stats = {"candidates": len(results),
             "fetched": sum(not r.get("fetch_error") for r in results),
             "priced": sum(r["subtotal"] is not None for r in results)}

    if a.json:
        print(json.dumps({"query": query, "qty": a.qty, "source": source, "rank": a.rank,
                          **stats, "results": top}, indent=2))
        return
    print(f'Best {len(top)} of {stats["candidates"]} results for "{base}"  x{a.qty}  '
          f'(via {source}; {stats["fetched"]} pages read, {stats["priced"]} with prices; '
          f'ranked by {a.rank})\n')

    def money(v, cur):
        return "n/a" if v is None else f"{v:,.2f} {cur or ''}".strip()

    for i, r in enumerate(top, 1):
        cur = r["currency"]
        cost = money(r["cost"], cur) + (f"  ({r['cost_source']})" if r["cost"] is not None else "")
        if r["pack_size"] > 1:
            cost += f"  per pack of {r['pack_size']}"
        stock = "n/a" if r["stock"] is None and r["in_stock"] is None else \
            (f"{r['stock']:,}" if r["stock"] is not None else
             "in stock" if r["in_stock"] else "out of stock")
        if r["enough_stock"] is False:
            stock += f"  (NOT enough for {a.qty})"
        score = "unranked (no price)" if r["score"] is None else \
            f"{r['score']:.2f}  (1.00 = cheapest, no drawbacks; search rank #{r['search_rank'] + 1})"
        print(f"{i}. {r['title']}\n   {r['url']}")
        print(f"   Score:         {score}")
        print(f"   Supplier:      {r['supplier']}")
        print(f"   Cost:          {cost}")
        if r["price_range"]:
            lo, hi = r["price_range"]
            what = "listings" if r["listing_page"] else "prices on page"
            print(f"   Price range:   {money(lo, cur)} - {money(hi, cur)}  "
                  f"({r['price_count']} {what})")
        if r["price_breaks"]:
            print("   Price breaks:  " + ", ".join(
                f"{b['min_qty']}+ @ {b['price']:g}" for b in r["price_breaks"]))
        print(f"   Unit cost:     {money(r['unit_cost'], cur)}")
        buy = f"{r['packs_needed']} pack(s) of {r['pack_size']}" if r["pack_size"] > 1 \
            else f"{r['packs_needed']} unit(s)"
        print(f"   Buy:           {buy} = {money(r['subtotal'], cur)}")
        print(f"   Stock:         {stock}")
        print(f"   Shipping time: {r['shipping_time'] or 'n/a'}")
        print(f"   Shipping cost: {r['shipping_cost'] or 'n/a'}")
        print(f"   Total:         {money(r['total'], cur)}"
              + ("" if r["total"] is not None or r["subtotal"] is None else
                 f"  (est. {money(r.get('effective_cost'), cur)} incl. typical shipping)"))
        if r["notes"]:
            print(f"   Notes:         {'; '.join(r['notes'])}")
        if r["fetch_error"] and a.verbose:
            print(f"   (page fetch failed: {r['fetch_error']})")
        print()


if __name__ == "__main__":
    main()