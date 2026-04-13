from __future__ import annotations

import json
import re
import time
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd
import requests

from src.scrape.extract_text import html_to_text


CSV_PATH = Path("output/researchers_clean.csv")
OUT_DIR = Path("output/web")
CACHE_DIR = Path("cache/html")

OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; ResearcherScraper/1.0; +https://example.com)"
}

TIMEOUT = 20  # seconds
SLEEP_SECONDS = 0.6  # be polite


def is_nonempty(x: object) -> bool:
    if x is None:
        return False
    if isinstance(x, float) and pd.isna(x):
        return False
    s = str(x).strip()
    return s != "" and s.lower() not in {"nan", "none", "null"}


def safe_slug(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s[:80] if s else "unknown"


def cache_key(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()


def fetch_html(url: str) -> tuple[str, int]:
    key = cache_key(url)
    cache_path = CACHE_DIR / f"{key}.html"
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8", errors="ignore"), 200

    resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT, allow_redirects=True)
    resp.raise_for_status()
    cache_path.write_text(resp.text, encoding="utf-8", errors="ignore")
    return resp.text, resp.status_code



def build_output_record(name: str, url: str, text: str, error: Optional[str] = None, http_status: Optional[int] = None):
    status = "ok"
    if error:
        status = "error"
    elif len(text.strip()) == 0:
        status = "empty"

    return {
        "name": name,
        "url": url,
        "status": status,
        "http_status": http_status,
        "text": text,
        "snippet": text[:600],
        "error": error,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def main(limit: int = 20) -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Missing {CSV_PATH}. Run the clean step first.")

    df = pd.read_csv(CSV_PATH)

    # Ensure columns exist
    for col in ["Full Name", "Personal Website", "Web Bio Link"]:
        if col not in df.columns:
            df[col] = ""

    # Choose URL: Personal Website first, then Web Bio Link
    def pick_url(row) -> str:
        if is_nonempty(row.get("Personal Website", "")):
            return str(row["Personal Website"]).strip()
        if is_nonempty(row.get("Web Bio Link", "")):
            return str(row["Web Bio Link"]).strip()
        return ""

    df["seed_url"] = df.apply(pick_url, axis=1)
    df = df[df["seed_url"].apply(is_nonempty)].copy()
    df = df.head(limit)

    print(f"Scraping {len(df)} researcher homepages...")

    for idx, row in df.iterrows():
        name = str(row.get("Full Name", "")).strip()
        url = str(row.get("seed_url", "")).strip()
        rid = safe_slug(name) + "-" + str(idx)

        out_path = OUT_DIR / f"{rid}.json"

        # Skip if already scraped
        if out_path.exists():
            print(f"[skip] {name} -> {out_path.name}")
            continue

        try:
            html, status_code = fetch_html(url)
            text = html_to_text(html)
            rec = build_output_record(name=name, url=url, text=text, http_status=status_code)

            out_path.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[ok]   {name} ({len(text)} chars) -> {out_path.name}")

        except Exception as e:
            rec = build_output_record(name=name, url=url, text="", error=str(e), http_status=None)
            out_path.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[err]  {name} -> {e}")

        time.sleep(SLEEP_SECONDS)

    print(f"Done. Outputs in: {OUT_DIR}/  and cache in: {CACHE_DIR}/")


if __name__ == "__main__":
    # change limit here if you want
    main(limit=10_000_000_000)
