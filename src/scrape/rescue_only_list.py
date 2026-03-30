from __future__ import annotations

import json
import time
import hashlib
import re
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import pandas as pd
import requests

from src.scrape.extract_text import html_to_text

RESCUE_CSV = Path("output/rescue_list.csv")
WEB_DIR = Path("output/web")
CACHE_DIR = Path("cache/html")

WEB_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; ResearcherScraper/1.0)"}
TIMEOUT = 25
SLEEP_SECONDS = 0.7


def cache_key(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()


def fetch_html(url: str) -> tuple[str, int, str]:
    """
    Returns: (html, status_code, final_url)
    Uses a file cache keyed on the requested URL.
    """
    key = cache_key(url)
    cache_path = CACHE_DIR / f"{key}.html"
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8", errors="ignore"), 200, url

    resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT, allow_redirects=True)
    resp.raise_for_status()
    cache_path.write_text(resp.text, encoding="utf-8", errors="ignore")
    return resp.text, resp.status_code, resp.url


def build_record(name: str, url: str, text: str, status: str, http_status: Optional[int], final_url: str, error: Optional[str]) -> dict:
    return {
        "name": name,
        "url": url,
        "final_url": final_url,
        "status": status,          # ok | empty | blocked | skipped | error
        "http_status": http_status,
        "text": text,
        "snippet": (text or "")[:600],
        "error": error,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def find_existing_json_by_name(name: str) -> Optional[Path]:
    """
    Search output/web for a JSON with matching name field.
    If multiple match, return the first (good enough for now).
    """
    for p in WEB_DIR.glob("*.json"):
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if str(obj.get("name", "")).strip() == name:
            return p
    return None


def is_scrapable_target(url: str) -> bool:
    """
    Skip pages that are 'reachable' but not useful to scrape in this pipeline.
    - Google Sites that redirect to accounts.google.com login
    - LinkedIn root pages
    """
    u = (url or "").strip()
    if not u:
        return False

    host = (urlparse(u).netloc or "").lower()
    if "accounts.google.com" in host:
        return False
    if "linkedin.com" in host:
        return False
    return True


def main() -> None:
    if not RESCUE_CSV.exists():
        raise FileNotFoundError(f"Missing {RESCUE_CSV}. Save your pasted CSV there.")

    df = pd.read_csv(RESCUE_CSV)

    # Normalize columns (your header has "name", good)
    for col in ["name", "best_status", "best_final_or_error", "best_url"]:
        if col not in df.columns:
            df[col] = ""

    rescued_ok = 0
    updated = 0
    skipped = 0
    blocked = 0
    errors = 0

    for _, row in df.iterrows():
        name = str(row.get("name", "")).strip()
        best_status_raw = row.get("best_status", "")
        best_url = str(row.get("best_url", "")).strip()
        best_final = str(row.get("best_final_or_error", "")).strip()

        # Determine numeric status if possible
        try:
            best_status = int(best_status_raw)
        except Exception:
            best_status = None

        # Decide the URL to scrape:
        # - prefer best_final (it’s usually canonical after redirects)
        # - else best_url
        target_url = best_final if best_final else best_url

        if not name:
            continue

        # If 403, mark as blocked and don't attempt (you can choose to try later with special headers)
        if best_status == 403:
            p = find_existing_json_by_name(name)
            rec = build_record(
                name=name,
                url=target_url,
                text="",
                status="blocked",
                http_status=403,
                final_url=target_url,
                error="403 Forbidden (reachable but blocked to scraping)",
            )
            if p is None:
                # create a new file if none exists
                out_path = WEB_DIR / f"{re.sub(r'[^a-z0-9]+','-',name.lower()).strip('-')}-blocked.json"
                out_path.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
            else:
                p.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
            blocked += 1
            continue

        # Only try to scrape 2xx/3xx
        if best_status is None or not (200 <= best_status <= 399):
            skipped += 1
            continue

        # Skip non-scrapable targets (e.g., login redirects, linkedin)
        if not is_scrapable_target(target_url):
            p = find_existing_json_by_name(name)
            rec = build_record(
                name=name,
                url=target_url,
                text="",
                status="skipped",
                http_status=best_status,
                final_url=target_url,
                error="Skipped (login wall / non-scrapable domain)",
            )
            if p is None:
                out_path = WEB_DIR / f"{re.sub(r'[^a-z0-9]+','-',name.lower()).strip('-')}-skipped.json"
                out_path.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
            else:
                p.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
            skipped += 1
            continue

        # Scrape and update existing json if found
        existing = find_existing_json_by_name(name)
        try:
            html, sc, final_url = fetch_html(target_url)
            text = html_to_text(html)

            status = "ok" if text.strip() else "empty"
            rec = build_record(
                name=name,
                url=target_url,
                text=text,
                status=status,
                http_status=sc,
                final_url=final_url,
                error=None if status == "ok" else "Extracted 0 chars (likely JS-rendered or extractor removed content)",
            )

            if existing is not None:
                existing.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
                updated += 1
            else:
                out_path = WEB_DIR / f"{re.sub(r'[^a-z0-9]+','-',name.lower()).strip('-')}-rescued.json"
                out_path.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")

            if status == "ok":
                rescued_ok += 1

            print(f"[{status}] {name} -> {final_url} ({len(text.strip())} chars)")

        except Exception as e:
            rec = build_record(
                name=name,
                url=target_url,
                text="",
                status="error",
                http_status=None,
                final_url=target_url,
                error=str(e),
            )
            if existing is not None:
                existing.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
                updated += 1
            else:
                out_path = WEB_DIR / f"{re.sub(r'[^a-z0-9]+','-',name.lower()).strip('-')}-error.json"
                out_path.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")

            errors += 1
            print(f"[err] {name} -> {e}")

        time.sleep(SLEEP_SECONDS)

    print("\nSummary:")
    print(f"  ok rescued:      {rescued_ok}")
    print(f"  updated files:   {updated}")
    print(f"  blocked (403):   {blocked}")
    print(f"  skipped:         {skipped}")
    print(f"  errors:          {errors}")
    print(f"Outputs in: {WEB_DIR}/")


if __name__ == "__main__":
    main()
