from __future__ import annotations
import json, time, hashlib
from pathlib import Path
import pandas as pd
import requests

from src.scrape.extract_text import html_to_text

RETRY_CSV = Path("output/web_retry_list.csv")
OUT_DIR = Path("output/web_retry")
CACHE_DIR = Path("cache/html")

OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; ResearcherScraper/1.0)"}
TIMEOUT = 25
SLEEP_SECONDS = 0.8

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

def main(limit: int | None = None) -> None:
    if not RETRY_CSV.exists():
        raise FileNotFoundError("Run make_retry_list.py first to create output/web_retry_list.csv")

    df = pd.read_csv(RETRY_CSV)
    if limit is not None:
        df = df.head(limit)

    print(f"Retrying {len(df)} failed/empty pages... outputs -> {OUT_DIR}/")

    for _, row in df.iterrows():
        name = str(row.get("name", "")).strip()
        url = str(row.get("url", "")).strip()
        file_name = str(row.get("file", "")).strip()
        out_path = OUT_DIR / file_name  # keep same filename

        # skip if already retried
        if out_path.exists():
            print(f"[skip] {name} (already retried)")
            continue

        try:
            html, sc = fetch_html(url)
            text = html_to_text(html)
            status = "ok" if text.strip() else "empty"
            rec = {
                "name": name,
                "url": url,
                "status": status,
                "http_status": sc,
                "text": text,
                "snippet": text[:600],
                "error": None,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            out_path.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[{status}] {name} ({len(text.strip())} chars) -> {out_path.name}")
        except Exception as e:
            rec = {
                "name": name,
                "url": url,
                "status": "error",
                "http_status": None,
                "text": "",
                "snippet": "",
                "error": str(e),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            out_path.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[err]  {name} -> {e}")

        time.sleep(SLEEP_SECONDS)

if __name__ == "__main__":
    main()
