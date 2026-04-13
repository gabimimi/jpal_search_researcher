from __future__ import annotations

import json
import re
import time
import hashlib
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests

from src.cv.extract_cv_text import (
    sniff_filetype,
    extract_pdf_text,
    extract_docx_text,
    extract_html_text,
)

CSV_PATH = Path("output/researchers_clean.csv")

OUT_DIR = Path("output/cv")
CACHE_DIR = Path("cache/cv")

OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; ResearcherCVFetcher/1.0)",
    "Accept": "*/*",
}
TIMEOUT = 30
SLEEP_SECONDS = 1


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


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


# ---------------------------
# Google Drive helpers
# ---------------------------

_DRIVE_FILE_RE = re.compile(r"drive\.google\.com/file/d/([^/]+)/", re.IGNORECASE)
_DRIVE_OPEN_RE = re.compile(r"drive\.google\.com/open\?id=([^&]+)", re.IGNORECASE)
_DRIVE_UC_ID_RE = re.compile(r"drive\.google\.com/uc\?.*?\bid=([^&]+)", re.IGNORECASE)


def drive_file_id(url: str) -> Optional[str]:
    u = url.strip()
    m = _DRIVE_FILE_RE.search(u)
    if m:
        return m.group(1)
    m = _DRIVE_OPEN_RE.search(u)
    if m:
        return m.group(1)
    m = _DRIVE_UC_ID_RE.search(u)
    if m:
        return m.group(1)
    return None


def drive_direct_download_url(file_id: str) -> str:
    return f"https://drive.google.com/uc?export=download&id={file_id}"


def maybe_expand_google_url_candidates(url: str) -> list[str]:
    """
    Given a CV URL, produce candidate URLs to try (Drive direct download, sites variants, etc.).
    """
    u = url.strip()
    candidates: list[str] = [u]

    fid = drive_file_id(u)
    if fid:
        candidates.insert(0, drive_direct_download_url(fid))

    if "sites.google.com" in u and "d=1" not in u:
        sep = "&" if "?" in u else "?"
        candidates.append(u + f"{sep}d=1")

    # de-dup
    seen: set[str] = set()
    out: list[str] = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


# ---------------------------
# HTML link extraction (key fix)
# ---------------------------

def _iter_hrefs(html: str) -> list[str]:
    # Grab href= values; good enough for lots of faculty pages
    hrefs = re.findall(r'href\s*=\s*"([^"]+)"', html, flags=re.IGNORECASE)
    hrefs += re.findall(r"href\s*=\s*'([^']+)'", html, flags=re.IGNORECASE)
    # Strip whitespace
    return [h.strip() for h in hrefs if h and h.strip()]


def _looks_like_cv_link(href: str) -> bool:
    h = href.lower()
    # direct files first
    if any(ext in h for ext in [".pdf", ".docx", ".doc"]):
        return True
    # common patterns
    return any(k in h for k in ["cv", "vita", "resume", "curriculum", "bio"])


def extract_best_cv_link_from_html(html: str, base_url: str) -> Optional[str]:
    """
    Pick the most likely CV download link from an HTML page, resolving relative URLs.
    Priority:
      1) Drive links
      2) PDFs/DOCX
      3) URLs containing cv/vita/resume/etc.
    """
    hrefs = _iter_hrefs(html)
    if not hrefs:
        return None

    # Resolve relative URLs
    resolved: list[str] = []
    for h in hrefs:
        # Skip javascript/mailto
        if h.lower().startswith(("javascript:", "mailto:", "#")):
            continue
        resolved.append(urljoin(base_url, h))

    # 1) Drive links
    for u in resolved:
        if "drive.google.com/file/d/" in u or "drive.google.com/open?id=" in u or "drive.google.com/uc?" in u:
            return u

    # 2) PDFs / DOCX
    for u in resolved:
        lu = u.lower()
        if ".pdf" in lu or ".docx" in lu or re.search(r"\.doc(\?|$)", lu):
            return u

    # 3) Any “cv-ish” links (including /cv or /vita pages)
    for u in resolved:
        if _looks_like_cv_link(u):
            return u

    return None


# ---------------------------
# Fetching with caching
# ---------------------------

def fetch_bytes_one(url: str) -> tuple[bytes, str | None, str]:
    """
    Returns: (content_bytes, content_type, final_url)
    Caches by url hash.
    """
    key = sha1(url)
    cache_path = CACHE_DIR / f"{key}.bin"
    meta_path = CACHE_DIR / f"{key}.json"

    if cache_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return cache_path.read_bytes(), meta.get("content_type"), meta.get("final_url", url)

    r = requests.get(url, headers=HEADERS, timeout=TIMEOUT, allow_redirects=True)
    r.raise_for_status()

    content = r.content
    content_type = r.headers.get("Content-Type")
    final_url = r.url

    cache_path.write_bytes(content)
    meta_path.write_text(
        json.dumps({"url": url, "final_url": final_url, "content_type": content_type}, indent=2),
        encoding="utf-8",
    )
    return content, content_type, final_url


def fetch_bytes(url: str) -> tuple[bytes, str | None, str]:
    """
    Tries multiple candidate URLs (Drive direct download, sites variants, etc.).
    Returns first successful fetch.
    """
    last_err: Optional[Exception] = None
    for candidate in maybe_expand_google_url_candidates(url):
        try:
            return fetch_bytes_one(candidate)
        except Exception as e:
            last_err = e
            continue
    assert last_err is not None
    raise last_err


# ---------------------------
# Output record
# ---------------------------

def build_record(
    name: str,
    cv_url: str,
    final_url: str,
    filetype: str,
    text: str,
    error: Optional[str] = None,
    content_type: Optional[str] = None,
) -> dict[str, Any]:
    status = "ok"
    if error:
        status = "error"
    elif len(text.strip()) == 0:
        status = "empty"

    return {
        "name": name,
        "cv_url": cv_url,
        "final_url": final_url,
        "content_type": content_type,
        "filetype": filetype,
        "status": status,  # ok | empty | error
        "text_len": len(text.strip()),
        "text": text,
        "snippet": text[:600],
        "error": error,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def main(limit: int | None = None) -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Missing {CSV_PATH}. Run the clean step first.")

    df = pd.read_csv(CSV_PATH)

    if "Full Name" not in df.columns:
        raise ValueError("CSV must include 'Full Name'")
    if "CV" not in df.columns:
        raise ValueError("CSV must include 'CV' column (from your Salesforce export)")

    df = df[df["CV"].apply(is_nonempty)].copy()
    if limit is not None:
        df = df.head(limit)

    print(f"Processing {len(df)} CVs...")

    for idx, row in df.iterrows():
        name = str(row["Full Name"]).strip()
        cv_url = str(row["CV"]).strip()

        out_name = f"{safe_slug(name)}-{idx}.json"
        out_path = OUT_DIR / out_name

        if out_path.exists():
            print(f"[skip] {name} -> {out_path.name}")
            continue

        if "@" in cv_url and not cv_url.startswith(("http://", "https://")):
            rec = build_record(
                name,
                cv_url,
                cv_url,
                "unknown",
                "",
                error="CV field looks like an email, not a URL",
            )
            out_path.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[err]  {name} -> CV is not a URL")
            continue

        try:
            data, content_type, final_url = fetch_bytes(cv_url)
            fb = data[:1024]
            ftype = sniff_filetype(final_url, content_type, fb)

            text = ""
            err: Optional[str] = None

            if ftype == "pdf":
                text = extract_pdf_text(data)

            elif ftype == "docx":
                text = extract_docx_text(data)

            elif ftype == "html":
                # Extract visible text (may be small even if page is legit)
                text = extract_html_text(data)

                # NEW: always try to find a better “CV” link from the raw HTML
                raw_html = data.decode("utf-8", errors="ignore")
                best = extract_best_cv_link_from_html(raw_html, base_url=final_url)

                # If we found a likely CV link, follow it and prefer its extracted text if it’s richer.
                if best:
                    try:
                        data2, ct2, final2 = fetch_bytes(best)
                        fb2 = data2[:1024]
                        ftype2 = sniff_filetype(final2, ct2, fb2)

                        text2 = ""
                        if ftype2 == "pdf":
                            text2 = extract_pdf_text(data2)
                        elif ftype2 == "docx":
                            text2 = extract_docx_text(data2)
                        elif ftype2 == "html":
                            text2 = extract_html_text(data2)

                        # Prefer the “more CV-like” result
                        if len(text2.strip()) > len(text.strip()):
                            text = text2
                            ftype = ftype2
                            content_type = ct2
                            final_url = final2
                    except Exception as e2:
                        # Keep the original page text; note why link-follow failed.
                        err = f"Found candidate CV link but fetch failed: {e2}"

                # If still tiny, mark as empty-ish with a useful note
                if len(text.strip()) < 800 and err is None:
                    err = "HTML text is small; page may link out to a PDF that isn't publicly accessible or is blocked."

            else:
                err = f"Unrecognized filetype: {ftype}"

            rec = build_record(
                name,
                cv_url,
                final_url,
                ftype,
                text,
                error=err,
                content_type=content_type,
            )
            out_path.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[{rec['status']}] {name} ({rec['filetype']}, {rec['text_len']} chars) -> {out_path.name}")

        except Exception as e:
            rec = build_record(name, cv_url, cv_url, "unknown", "", error=str(e))
            out_path.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[err]  {name} -> {e}")

        time.sleep(SLEEP_SECONDS)

    print(f"Done. Outputs in: {OUT_DIR}/  cache in: {CACHE_DIR}/")


if __name__ == "__main__":
    main(limit=None)