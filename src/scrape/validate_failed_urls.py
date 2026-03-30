from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import pandas as pd
import requests


RETRY_CSV = Path("output/web_retry_list.csv")
OUT_CSV = Path("output/url_validation_results.csv")

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; ResearcherScraper/1.0)"}
TIMEOUT = 15


def looks_like_email(s: str) -> bool:
    return bool(re.search(r"[^@\s]+@[^@\s]+\.[^@\s]+", s))


def normalize_url(u: str) -> str:
    u = (u or "").strip()
    u = u.replace(" ", "")
    return u


def candidates(url: str) -> list[str]:
    """
    Generate cheap alternative URLs worth trying.
    """
    url = normalize_url(url)
    if not url:
        return []

    # reject obvious junk
    low = url.lower()
    if low in {"notfound", "not found", "n/a", "na", "none", "null"}:
        return []
    if "doesn't seem to exist" in low or "salesforce" in low:
        return []
    if looks_like_email(url):
        return []

    # add scheme if missing
    if not re.match(r"^https?://", url):
        url = "https://" + url

    parsed = urlparse(url)

    # base root (scheme + netloc)
    root = urlunparse((parsed.scheme, parsed.netloc, "", "", "", ""))

    # common “bad path” fixes: try root + parent directories
    path_parts = [p for p in parsed.path.split("/") if p]
    parents = []
    for k in range(len(path_parts), 0, -1):
        parents.append("/" + "/".join(path_parts[: k - 1]))
    parents = list(dict.fromkeys(parents))  # unique, preserve order

    out: list[str] = []
    out.append(url)                 # original
    out.append(root)                # just the homepage
    for p in parents[:3]:
        out.append(root + p + "/")  # try parent dirs

    # http<->https
    if parsed.scheme == "https":
        out.append(url.replace("https://", "http://", 1))
        out.append(root.replace("https://", "http://", 1))
    else:
        out.append(url.replace("http://", "https://", 1))
        out.append(root.replace("http://", "https://", 1))

    # remove/add www
    host = parsed.netloc
    if host.startswith("www."):
        out.append(url.replace("://www.", "://", 1))
        out.append(root.replace("://www.", "://", 1))
    else:
        out.append(url.replace("://", "://www.", 1))
        out.append(root.replace("://", "://www.", 1))

    # de-dupe
    out2 = []
    seen = set()
    for x in out:
        x = x.strip()
        if x and x not in seen:
            out2.append(x)
            seen.add(x)
    return out2


def check(url: str) -> tuple[bool, int | None, str]:
    """
    Try HEAD then GET. Return (ok, status_code, final_url_or_error).
    ok is True for 2xx/3xx. 403 is treated as reachable-but-blocked.
    """
    try:
        r = requests.head(url, headers=HEADERS, timeout=TIMEOUT, allow_redirects=True)
        sc = r.status_code
        if 200 <= sc < 400:
            return True, sc, r.url
        if sc == 403:
            return True, sc, r.url  # reachable, but blocked to scraping
    except Exception:
        pass

    try:
        r = requests.get(url, headers=HEADERS, timeout=TIMEOUT, allow_redirects=True)
        sc = r.status_code
        if 200 <= sc < 400:
            return True, sc, r.url
        if sc == 403:
            return True, sc, r.url
        return False, sc, r.url
    except Exception as e:
        return False, None, str(e)


def main(limit: int | None = None) -> None:
    if not RETRY_CSV.exists():
        raise FileNotFoundError("Missing output/web_retry_list.csv. Run make_retry_list.py first.")

    df = pd.read_csv(RETRY_CSV)
    if limit is not None:
        df = df.head(limit)

    rows = []
    for _, row in df.iterrows():
        name = str(row.get("name", "")).strip()
        url = str(row.get("url", "")).strip()

        best_ok = False
        best_sc = None
        best_final = ""
        best_tested = ""

        tried = candidates(url)
        if not tried:
            rows.append({
                "name": name,
                "original_url": url,
                "best_url": "",
                "best_status": "",
                "best_final_or_error": "skipped (junk/email/empty)",
                "tried_count": 0,
            })
            continue

        for cand in tried[:10]:  # cap per person
            ok, sc, info = check(cand)
            if ok:
                best_ok = True
                best_sc = sc
                best_final = info
                best_tested = cand
                break

        rows.append({
            "name": name,
            "original_url": url,
            "best_url": best_tested if best_ok else "",
            "best_status": best_sc if best_ok else "",
            "best_final_or_error": best_final if best_ok else (info if 'info' in locals() else "unreachable"),
            "tried_count": len(tried),
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    print(f"Saved {OUT_CSV} (rows={len(out)})")

    # quick summary
    ok_count = (out["best_url"].astype(str).str.len() > 0).sum()
    print(f"Found working replacement or reachable(403) for: {ok_count}/{len(out)}")


if __name__ == "__main__":
    main()
