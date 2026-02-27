#!/usr/bin/env python3
"""CS2 Trade-Up ROI engine (single script)."""

from __future__ import annotations

import datetime as dt
import json
import math
import random
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import config

BASE_DIR = Path(__file__).resolve().parent
COLLECTIONS_FILE = BASE_DIR / "collections.json"
SKINS_FILE = BASE_DIR / "skins.json"
PRICES_FILE = BASE_DIR / "prices.json"
RESULTS_FILE = BASE_DIR / "results.json"

SKINPOCK_COLLECTIONS_URL = "https://skinpock.com/cs2/collections"
STEAM_PRICE_API = "https://steamcommunity.com/market/priceoverview/"

WEAR_RANGES = {"FN": (0.00, 0.07), "MW": (0.07, 0.15), "FT": (0.15, 0.38), "WW": (0.38, 0.45), "BS": (0.45, 1.00)}
WEAR_LONG = {"FN": "Factory New", "MW": "Minimal Wear", "FT": "Field-Tested", "WW": "Well-Worn", "BS": "Battle-Scarred"}
RARITY_ORDER = ["Consumer Grade", "Industrial Grade", "Mil-Spec", "Restricted", "Classified", "Covert", "Contraband"]


def _sleep_delay() -> None:
    time.sleep(max(0.0, float(config.REQUEST_DELAY_SECONDS)))


def _safe_get_text(url: str, params: dict[str, Any] | None = None) -> str:
    full = url
    if params:
        full = f"{url}?{urlencode(params)}"
    req = Request(full, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urlopen(req, timeout=20) as resp:
            text = resp.read().decode("utf-8", errors="ignore")
            _sleep_delay()
            return text
    except Exception as exc:
        print(f"[warn] GET failed {full}: {exc}")
        return ""


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def normalize_rarity(text: str) -> str:
    t = (text or "").strip()
    return {
        "Mil-Spec Grade": "Mil-Spec",
        "Mil Spec": "Mil-Spec",
        "Restricted Grade": "Restricted",
        "Classified Grade": "Classified",
        "Covert Grade": "Covert",
    }.get(t, t)


def strip_tags(html: str) -> str:
    return re.sub(r"<[^>]+>", " ", html)


def scrape_collections_and_skins() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    html = _safe_get_text(SKINPOCK_COLLECTIONS_URL)
    if not html:
        return read_json(COLLECTIONS_FILE, []), read_json(SKINS_FILE, [])

    collection_urls: dict[str, str] = {}
    for href, label in re.findall(r'<a[^>]+href="([^"]*collection[^"]*)"[^>]*>(.*?)</a>', html, flags=re.I | re.S):
        name = strip_tags(label).strip()
        if not name:
            continue
        url = href if href.startswith("http") else f"https://skinpock.com{href}"
        collection_urls[name] = url

    collections: list[dict[str, Any]] = []
    skins: list[dict[str, Any]] = []

    for fallback_name, url in sorted(collection_urls.items()):
        c_html = _safe_get_text(url)
        if not c_html:
            continue
        m = re.search(r"<h1[^>]*>(.*?)</h1>", c_html, flags=re.I | re.S)
        collection_name = strip_tags(m.group(1)).strip() if m else fallback_name
        collections.append({"name": collection_name, "url": url})

        rows = re.findall(r"<tr[^>]*>(.*?)</tr>|<div[^>]*class=\"[^\"]*(?:skin|card|row)[^\"]*\"[^>]*>(.*?)</div>", c_html, flags=re.I | re.S)
        for pair in rows:
            row = " ".join(pair)
            row_txt = strip_tags(row)
            anchors = re.findall(r'<a[^>]+href="([^"]*(?:skin|weapon)[^"]*)"[^>]*>(.*?)</a>', row, flags=re.I | re.S)
            if not anchors:
                continue
            skin_name = strip_tags(anchors[0][1]).strip()
            if not skin_name:
                continue
            rarity = ""
            for r in RARITY_ORDER:
                if r.lower() in row_txt.lower():
                    rarity = r
                    break
            nums = [float(x) for x in re.findall(r"\b(?:0(?:\.\d+)?|1(?:\.0+)?)\b", row_txt)]
            min_float, max_float = (min(nums), max(nums)) if len(nums) >= 2 else (0.0, 1.0)
            skins.append({"name": skin_name, "collection": collection_name, "rarity": normalize_rarity(rarity), "min_float": min_float, "max_float": max_float})

    dedup: dict[tuple[str, str], dict[str, Any]] = {}
    for s in skins:
        dedup[(s["collection"], s["name"])] = s
    collections = list({c["name"]: c for c in collections}.values())
    skins = list(dedup.values())

    write_json(COLLECTIONS_FILE, collections)
    write_json(SKINS_FILE, skins)
    print(f"[info] Scraped {len(collections)} collections and {len(skins)} skins")
    return collections, skins


def is_cache_fresh(prices_data: dict[str, Any]) -> bool:
    ts = prices_data.get("updated_at")
    if not ts:
        return False
    try:
        updated = dt.datetime.fromisoformat(ts)
    except ValueError:
        return False
    return (dt.datetime.utcnow() - updated).total_seconds() <= config.PRICE_CACHE_HOURS * 3600


def parse_price_string(value: str | None) -> float | None:
    if not value:
        return None
    cleaned = "".join(ch for ch in value if ch.isdigit() or ch in ".,")
    if not cleaned:
        return None
    if "," in cleaned and "." not in cleaned:
        cleaned = cleaned.replace(",", ".")
    cleaned = cleaned.replace(",", "")
    try:
        return float(cleaned)
    except ValueError:
        return None


def parse_volume(value: str | None) -> int:
    d = "".join(ch for ch in (value or "") if ch.isdigit())
    return int(d) if d else 0


def steam_price_for_market_hash(market_hash_name: str) -> tuple[float | None, int]:
    text = _safe_get_text(STEAM_PRICE_API, {"appid": 730, "currency": 1, "market_hash_name": market_hash_name})
    if not text:
        return None, 0
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None, 0
    if not data.get("success"):
        return None, 0
    price = parse_price_string(data.get("lowest_price") or data.get("median_price"))
    return price, parse_volume(data.get("volume"))


def fetch_prices(skins: list[dict[str, Any]]) -> dict[str, Any]:
    cache = read_json(PRICES_FILE, {})
    if cache and is_cache_fresh(cache):
        print("[info] Using fresh cached prices")
        return cache

    recs: dict[str, Any] = {}
    for skin in skins:
        recs.setdefault(skin["name"], {})
        for wear, wear_label in WEAR_LONG.items():
            mh = f"{skin['name']} ({wear_label})"
            price, listings = steam_price_for_market_hash(mh)
            recs[skin["name"]][wear] = {"market_hash_name": mh, "price": price, "listings": listings}

    out = {"updated_at": dt.datetime.utcnow().isoformat(), "skins": recs}
    write_json(PRICES_FILE, out)
    return out


def sample_float_for_wear(wear: str) -> float:
    lo, hi = WEAR_RANGES[wear]
    if wear == "FN":
        x = random.betavariate(2.0, 8.0)
    elif wear == "BS":
        x = random.betavariate(8.0, 2.0)
    else:
        x = random.random()
    return lo + (hi - lo) * x


def estimate_input_float_probability(skin_name: str, wear: str, prices: dict[str, Any], median_price: float) -> float:
    rec = prices.get("skins", {}).get(skin_name, {}).get(wear, {})
    listings = int(rec.get("listings") or 0)
    price = float(rec.get("price") or 0.0)
    listing_factor = 1.0 - math.exp(-listings / 35.0)
    density_factor = {"FN": 0.45, "MW": 0.72, "FT": 0.80, "WW": 0.77, "BS": 0.90}[wear]
    price_friction = 0.25 if price <= 0 else max(0.25, min(1.2, median_price / price if median_price > 0 else 1.0))
    return max(0.01, min(0.99, listing_factor * density_factor * price_friction))


def float_to_wear(value: float) -> str | None:
    for w, (lo, hi) in WEAR_RANGES.items():
        if lo <= value <= hi:
            return w
    return None


def liquidity_weight(outputs: list[dict[str, Any]], prices: dict[str, Any]) -> float:
    vals = []
    for out in outputs:
        s = prices.get("skins", {}).get(out["name"], {})
        for wear in WEAR_RANGES:
            vals.append(math.log1p(int((s.get(wear) or {}).get("listings") or 0)))
    if not vals:
        return 0.2
    score = (sum(vals) / len(vals)) / math.log1p(5000)
    return max(0.2, min(1.0, score))


def monte_carlo_output_success(input_wear: str, input_cost: float, outputs: list[dict[str, Any]], prices: dict[str, Any]) -> tuple[float, float, float, float, float]:
    success = 0
    profits: list[float] = []
    sales: list[float] = []
    avgs: list[float] = []
    for _ in range(config.MONTE_CARLO_SIMS):
        avg_input = sum(sample_float_for_wear(input_wear) for _ in range(10)) / 10.0
        avgs.append(avg_input)
        out = random.choice(outputs)
        o_min, o_max = float(out.get("min_float", 0.0)), float(out.get("max_float", 1.0))
        out_float = o_min + (o_max - o_min) * avg_input
        wear = float_to_wear(out_float)
        sale = 0.0
        if wear:
            rec = prices.get("skins", {}).get(out["name"], {}).get(wear, {})
            p = float(rec.get("price") or 0.0)
            if p > 0:
                sale = p * (1.0 - config.STEAM_TAX_PERCENT)
                success += 1
        sales.append(sale)
        profits.append(sale - input_cost)

    p_output = success / config.MONTE_CARLO_SIMS
    mean_profit = sum(profits) / len(profits) if profits else 0.0
    mu = mean_profit
    std_profit = (sum((p - mu) ** 2 for p in profits) / len(profits)) ** 0.5 if profits else 0.0
    mean_sale = sum(sales) / len(sales) if sales else 0.0
    avg_out_float = sum(avgs) / len(avgs) if avgs else 0.0
    return p_output, mean_profit, std_profit, mean_sale, avg_out_float


def analyze_tradeups(skins: list[dict[str, Any]], prices: dict[str, Any]) -> list[dict[str, Any]]:
    by_collection: dict[str, list[dict[str, Any]]] = {}
    for s in skins:
        by_collection.setdefault(s["collection"], []).append(s)

    all_prices: list[float] = []
    for sk in prices.get("skins", {}).values():
        for w in sk.values():
            p = w.get("price")
            if isinstance(p, (int, float)) and p > 0:
                all_prices.append(float(p))
    if all_prices:
        ap = sorted(all_prices)
        m = len(ap) // 2
        median_price = ap[m] if len(ap) % 2 else (ap[m - 1] + ap[m]) / 2
    else:
        median_price = 1.0

    results: list[dict[str, Any]] = []
    for collection, c_skins in by_collection.items():
        by_rarity: dict[str, list[dict[str, Any]]] = {}
        for s in c_skins:
            by_rarity.setdefault(s.get("rarity", ""), []).append(s)

        for i, rarity in enumerate(RARITY_ORDER[:-1]):
            outputs = by_rarity.get(RARITY_ORDER[i + 1], [])
            if not by_rarity.get(rarity) or not outputs:
                continue
            for inp in by_rarity[rarity]:
                for wear in WEAR_RANGES:
                    rec = prices.get("skins", {}).get(inp["name"], {}).get(wear, {})
                    in_price = float(rec.get("price") or 0.0)
                    if in_price <= 0:
                        continue
                    input_cost = in_price * 10.0
                    if input_cost > config.BUDGET_USD:
                        continue

                    p_single = estimate_input_float_probability(inp["name"], wear, prices, median_price)
                    p_inputs = p_single ** 10
                    p_output, mean_profit, std_profit, mean_sale, avg_float = monte_carlo_output_success(wear, input_cost, outputs, prices)
                    p_success = p_inputs * p_output
                    liq = liquidity_weight(outputs, prices)
                    ev = mean_sale * liq * p_success
                    profit = ev - input_cost

                    roi = mean_profit / input_cost if input_cost else -1.0
                    norm_std = std_profit / input_cost if input_cost else 0.0
                    risk_score = (roi * p_success) / (1.0 + norm_std)

                    if p_inputs < config.MIN_INPUT_AVAILABILITY_PROB or p_output < config.MIN_OUTPUT_SUCCESS_PROB:
                        continue
                    if p_success < config.MIN_OVERALL_TRADEUP_PROB or risk_score < config.MIN_RISK_ADJUSTED_SCORE:
                        continue

                    results.append({
                        "input_skin": inp["name"],
                        "collection": collection,
                        "input_wear": wear,
                        "input_cost": round(input_cost, 4),
                        "expected_value": round(ev, 4),
                        "profit": round(profit, 4),
                        "ROI_percent": round(roi * 100.0, 3),
                        "RiskAdjustedScore": round(risk_score, 6),
                        "liquidity_score": round(liq, 4),
                        "P_inputs_obtainable": round(p_inputs, 6),
                        "P_output_success": round(p_output, 6),
                        "P_tradeup_success": round(p_success, 6),
                        "profit_std_dev": round(std_profit, 4),
                        "average_output_float": round(avg_float, 6),
                        "output_count": len(outputs),
                        "single_output_probability": round(1.0 / len(outputs), 6),
                    })

    results.sort(key=lambda x: x["RiskAdjustedScore"], reverse=True)
    return results


def print_top_results(results: list[dict[str, Any]], n: int = 10) -> None:
    print("\n=== TOP TRADE-UPS (RiskAdjustedScore) ===")
    if not results:
        print("No trade-ups passed thresholds.")
        return
    for i, r in enumerate(results[:n], 1):
        print(f"{i:02d}. {r['input_skin']} [{r['input_wear']}] | {r['collection']} | Cost ${r['input_cost']:.2f} | EV ${r['expected_value']:.2f} | Profit ${r['profit']:.2f} | ROI {r['ROI_percent']:.2f}% | RAS {r['RiskAdjustedScore']:.4f}")


def run_cycle() -> None:
    for path, default in ((COLLECTIONS_FILE, []), (SKINS_FILE, []), (PRICES_FILE, {"updated_at": dt.datetime.utcnow().isoformat(), "skins": {}}), (RESULTS_FILE, {"generated_at": dt.datetime.utcnow().isoformat(), "result_count": 0, "tradeups": []})):
        if not path.exists():
            write_json(path, default)

    collections, skins = scrape_collections_and_skins()
    if not collections or not skins:
        print("[warn] Empty scrape result; using existing JSON if present.")
        collections = read_json(COLLECTIONS_FILE, [])
        skins = read_json(SKINS_FILE, [])

    prices = fetch_prices(skins)
    results = analyze_tradeups(skins, prices)
    payload = {
        "generated_at": dt.datetime.utcnow().isoformat(),
        "settings": {"budget_usd": config.BUDGET_USD, "steam_tax_percent": config.STEAM_TAX_PERCENT, "monte_carlo_sims": config.MONTE_CARLO_SIMS},
        "result_count": len(results),
        "tradeups": results,
    }
    write_json(RESULTS_FILE, payload)
    print_top_results(results)
    print(f"\n[info] Saved {len(results)} trade-ups to {RESULTS_FILE.name}")


def seconds_until_next_run(run_times: list[str]) -> float:
    now = dt.datetime.now()
    targets = []
    for rt in run_times:
        hh, mm = [int(x) for x in rt.split(":", 1)]
        t = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
        if t <= now:
            t += dt.timedelta(days=1)
        targets.append(t)
    return max(1.0, min((t - now).total_seconds() for t in targets))


def run_scheduler() -> None:
    mode = str(config.SCHEDULE_MODE).lower().strip()
    print(f"[info] Scheduler mode: {mode}")
    if mode == "once":
        run_cycle()
        return
    try:
        while True:
            run_cycle()
            if mode == "interval":
                sleep_s = max(1, int(config.INTERVAL_MINUTES * 60))
            elif mode == "times":
                sleep_s = int(seconds_until_next_run(config.RUN_TIMES))
            else:
                print(f"[warn] Unknown SCHEDULE_MODE={config.SCHEDULE_MODE!r}; exiting.")
                return
            print(f"[info] Sleeping {sleep_s}s...")
            time.sleep(sleep_s)
    except KeyboardInterrupt:
        print("\n[info] Ctrl+C received. Exiting cleanly.")


if __name__ == "__main__":
    run_scheduler()
