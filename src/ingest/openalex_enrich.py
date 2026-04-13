from __future__ import annotations

import csv
import hashlib
import json
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse, quote_plus

import pandas as pd
import requests

# ----------------------------
# Config
# ----------------------------

IN_CSV = Path("output/researchers_clean.csv")

OUT_DIR = Path("output/openalex")
AUTHORS_DIR = OUT_DIR / "authors"
WORKS_DIR = OUT_DIR / "works"
CACHE_DIR = Path("cache/openalex")

OUT_DIR.mkdir(parents=True, exist_ok=True)
AUTHORS_DIR.mkdir(parents=True, exist_ok=True)
WORKS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

BASE = "https://api.openalex.org"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; OpenAlexIngest/1.0; +https://example.com)"
}

TIMEOUT = 30

# Polite pacing to avoid 429s. Keep these conservative.
BASE_SLEEP = 0.6     # base delay before each request
JITTER = 0.4         # additional random jitter
MAX_429_RETRIES = 8  # how many times to retry on 429
MAX_NET_RETRIES = 4  # retries on transient network errors

PER_PAGE_AUTHORS = 10
PER_PAGE_WORKS = 200

# If you have an email, OpenAlex suggests adding it as mailto= for etiquette.
MAILTO = None  # e.g. "you@mit.edu"

# Use a session for connection reuse
SESSION = requests.Session()


# ----------------------------
# Helpers
# ----------------------------

def safe_slug(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s[:80] if s else "unknown"


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def domain_of(url: str) -> Optional[str]:
    if not url:
        return None
    try:
        u = url.strip()
        if not u.startswith(("http://", "https://")):
            u = "https://" + u
        p = urlparse(u)
        d = (p.netloc or "").lower()
        d = d.replace("www.", "")
        return d or None
    except Exception:
        return None


def is_nonempty(x: object) -> bool:
    if x is None:
        return False
    try:
        if isinstance(x, float) and pd.isna(x):
            return False
    except Exception:
        pass
    s = str(x).strip()
    return s != "" and s.lower() not in {"nan", "none", "null"}


def best_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    """Pick the first column that exists from candidates."""
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def looks_like_real_person_name(name: str) -> bool:
    """
    Filter out Salesforce footer / junk rows, test rows, etc.
    """
    n = norm(name)
    if not n:
        return False
    bad_substrings = [
        "confidential information",
        "do not distribute",
        "copyright",
        "salesforce.com",
        "all rights reserved",
        "test test",
        "fake first",
        "fake last",
    ]
    if any(b in n for b in bad_substrings):
        return False
    # require at least two tokens (first + last) and not too long
    tokens = n.split()
    if len(tokens) < 2:
        return False
    if len(n) > 90:
        return False
    return True


def cache_get(url: str) -> Optional[dict[str, Any]]:
    key = sha1(url)
    p = CACHE_DIR / f"{key}.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def cache_put(url: str, payload: dict[str, Any]) -> None:
    key = sha1(url)
    p = CACHE_DIR / f"{key}.json"
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _paced_sleep() -> None:
    time.sleep(BASE_SLEEP + random.random() * JITTER)


def _get_with_backoff(url: str, params: dict[str, Any]) -> dict[str, Any]:
    """
    GET with:
      - pacing to avoid 429
      - 429 handling using Retry-After when present
      - retries on transient network errors
    """
    # network retries wrapper
    for net_attempt in range(MAX_NET_RETRIES):
        try:
            # 429 retry loop
            for attempt in range(MAX_429_RETRIES):
                _paced_sleep()
                r = SESSION.get(url, headers=HEADERS, params=params, timeout=TIMEOUT)

                if r.status_code != 429:
                    r.raise_for_status()
                    return r.json()

                retry_after = r.headers.get("Retry-After")
                if retry_after is not None:
                    try:
                        wait = float(retry_after)
                    except ValueError:
                        wait = 10.0
                else:
                    wait = min(90.0, (2 ** attempt) + random.random() * 2.0)

                print(f"[429] sleeping {wait:.1f}s then retrying: {r.url}")
                time.sleep(wait)

            raise RuntimeError(f"OpenAlex 429 persisted after {MAX_429_RETRIES} retries: {url}")

        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            wait = min(60.0, (2 ** net_attempt) + random.random() * 2.0)
            print(f"[net] {e} -> sleeping {wait:.1f}s then retrying")
            time.sleep(wait)
            continue

    raise RuntimeError(f"Network failed after {MAX_NET_RETRIES} retries: {url}")


def openalex_get(path_or_url: str, params: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    if path_or_url.startswith("http"):
        url = path_or_url
    else:
        url = BASE + path_or_url

    # Add mailto parameter if set (recommended by OpenAlex)
    params = dict(params or {})
    if MAILTO and "mailto" not in params:
        params["mailto"] = MAILTO

    # Build final URL for caching key
    if params:
        qp = "&".join(f"{quote_plus(str(k))}={quote_plus(str(v))}" for k, v in sorted(params.items()))
        full = f"{url}?{qp}"
    else:
        full = url

    cached = cache_get(full)
    if cached is not None:
        return cached

    data = _get_with_backoff(url, params)
    cache_put(full, data)
    return data


def abstract_from_inverted_index(inv: Optional[dict[str, list[int]]]) -> str:
    """
    OpenAlex abstracts come as "abstract_inverted_index":
    word -> positions. Reconstruct into a string.
    """
    if not inv:
        return ""
    positions: dict[int, str] = {}
    for word, pos_list in inv.items():
        for p in pos_list:
            positions[p] = word
    if not positions:
        return ""
    return " ".join(positions[i] for i in sorted(positions.keys()))


# ----------------------------
# Matching logic
# ----------------------------

@dataclass
class PersonRow:
    name: str
    affiliation: Optional[str]
    website: Optional[str]
    website_domain: Optional[str]


def score_candidate(
    cand: dict[str, Any],
    person: PersonRow,
) -> tuple[int, list[str]]:
    """
    Heuristic scoring:
    - Name similarity (basic)
    - Institution match
    - ORCID presence is weak positive (signals real author record)
    """
    score = 0
    reasons: list[str] = []

    cand_name = norm(cand.get("display_name", ""))
    person_name = norm(person.name)

    # name overlap
    if cand_name == person_name:
        score += 40
        reasons.append("exact_name_match")
    else:
        # crude token overlap
        a = set(cand_name.split())
        b = set(person_name.split())
        inter = len(a & b)
        if inter >= max(2, min(len(a), len(b))):
            score += 25
            reasons.append("strong_name_token_overlap")
        elif inter >= 2:
            score += 15
            reasons.append("some_name_token_overlap")

    # affiliation match
    inst = cand.get("last_known_institution") or {}
    inst_name = norm(inst.get("display_name", ""))

    if person.affiliation:
        aff = norm(person.affiliation)
        if aff and inst_name and (aff in inst_name or inst_name in aff):
            score += 35
            reasons.append("institution_match")
        else:
            aff_tokens = set(aff.split())
            inst_tokens = set(inst_name.split())
            if len(aff_tokens & inst_tokens) >= 2 and inst_name:
                score += 15
                reasons.append("weak_institution_token_overlap")

    # ORCID present
    if is_nonempty(cand.get("orcid")):
        score += 5
        reasons.append("has_orcid")

    # Works count sanity
    wc = cand.get("works_count")
    if isinstance(wc, int):
        if wc >= 5:
            score += 5
            reasons.append("works_count>=5")
        if wc >= 50:
            score += 3
            reasons.append("works_count>=50")

    return score, reasons


def confidence_from_score(score: int) -> str:
    if score >= 65:
        return "high"
    if score >= 40:
        return "medium"
    return "low"


# ----------------------------
# Pipeline steps
# ----------------------------

def search_authors(name: str) -> list[dict[str, Any]]:
    data = openalex_get(
        "/authors",
        params={"search": name, "per-page": PER_PAGE_AUTHORS},
    )
    return data.get("results", []) or []


def fetch_all_works_for_author(author_id: str) -> list[dict[str, Any]]:
    """
    author_id can be:
      - 'https://openalex.org/A...'
      - or 'A...'
    We'll normalize to the OpenAlex short id.
    """
    short = author_id.strip()
    if short.startswith("https://openalex.org/"):
        short = short.rsplit("/", 1)[-1]

    works: list[dict[str, Any]] = []
    cursor = "*"
    while True:
        data = openalex_get(
            "/works",
            params={
                "filter": f"author.id:{short}",
                "per-page": PER_PAGE_WORKS,
                "cursor": cursor,
            },
        )
        batch = data.get("results", []) or []
        works.extend(batch)
        cursor = data.get("meta", {}).get("next_cursor")
        if not cursor:
            break
    return works


def simplify_work(w: dict[str, Any]) -> dict[str, Any]:
    host = w.get("primary_location") or {}
    source = (host.get("source") or {})
    ids = w.get("ids") or {}

    return {
        "openalex_id": w.get("id"),
        "doi": ids.get("doi"),
        "title": w.get("display_name"),
        "year": w.get("publication_year"),
        "type": w.get("type"),
        "cited_by_count": w.get("cited_by_count"),
        "venue": source.get("display_name"),
        "landing_page_url": (host.get("landing_page_url") or w.get("primary_location", {}).get("landing_page_url")),
        "abstract": abstract_from_inverted_index(w.get("abstract_inverted_index")),
        "authorships": [
            {
                "author_name": (a.get("author") or {}).get("display_name"),
                "author_openalex_id": (a.get("author") or {}).get("id"),
                "institutions": [i.get("display_name") for i in (a.get("institutions") or []) if i.get("display_name")],
            }
            for a in (w.get("authorships") or [])
        ],
    }


def _should_retry_existing(author_out: Path) -> bool:
    """
    Retry if previous status was error and it was due to 429 or network failure.
    Otherwise, skip.
    """
    try:
        prev = json.loads(author_out.read_text(encoding="utf-8"))
    except Exception:
        return True

    st = prev.get("status")
    if st != "error":
        return False

    err = str(prev.get("error", "")).lower()
    if "429" in err or "too many requests" in err or "nameResolutionError".lower() in err or "failed to resolve" in err:
        return True
    # for other errors, you might still want to retry, but default is no
    return False


def main() -> None:
    if not IN_CSV.exists():
        raise FileNotFoundError(f"Missing {IN_CSV}. You said you already created this in the clean step.")

    df = pd.read_csv(IN_CSV)

    # Try to find columns robustly
    name_col = best_col(df, ["Full Name", "Name", "full_name", "Researcher", "Contact Name"])
    aff_col = best_col(df, ["Institution", "Organization", "Affiliation", "University", "Primary Affiliation"])
    web_col = best_col(df, ["Website", "website", "Personal Website", "best_url", "Best URL", "Web", "Homepage"])

    if not name_col:
        raise ValueError(f"Could not find a name column. Columns available: {list(df.columns)}")

    summary_rows: list[dict[str, Any]] = []

    print(f"Reading {len(df)} rows from {IN_CSV}")
    print(f"Using columns: name={name_col!r}, affiliation={aff_col!r}, website={web_col!r}")
    print(f"Pacing: BASE_SLEEP={BASE_SLEEP}s JITTER={JITTER}s (with 429 backoff)")

    for i, row in df.iterrows():
        name = str(row.get(name_col, "")).strip()
        if not name:
            continue
        if not looks_like_real_person_name(name):
            # Skip obvious junk lines
            continue

        affiliation = None
        if aff_col and is_nonempty(row.get(aff_col)):
            affiliation = str(row.get(aff_col)).strip()

        website = None
        if web_col and is_nonempty(row.get(web_col)):
            website = str(row.get(web_col)).strip()

        person = PersonRow(
            name=name,
            affiliation=affiliation,
            website=website,
            website_domain=domain_of(website) if website else None,
        )

        slug = safe_slug(name)
        author_out = AUTHORS_DIR / f"{slug}.json"
        works_out = WORKS_DIR / f"{slug}.json"

        # Skip if already done, unless it was a retryable error
        if author_out.exists() and works_out.exists():
            if _should_retry_existing(author_out):
                print(f"[retry] {name}")
            else:
                print(f"[skip] {name}")
                try:
                    author_payload = json.loads(author_out.read_text(encoding="utf-8"))
                    summary_rows.append(author_payload.get("_summary_row", {"name": name, "status": "skipped"}))
                except Exception:
                    summary_rows.append({"name": name, "status": "skipped"})
                continue

        # 1) search candidates
        try:
            cands = search_authors(name)
        except Exception as e:
            payload = {
                "name": name,
                "status": "error",
                "error": f"author_search_failed: {e}",
                "candidates": [],
                "_summary_row": {"name": name, "status": "error", "error": str(e)},
            }
            author_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            works_out.write_text(json.dumps({"name": name, "works": [], "works_count": 0, "error": "author_search_failed"}, indent=2), encoding="utf-8")
            summary_rows.append(payload["_summary_row"])
            print(f"[err] {name} -> author search failed: {e}")
            continue

        # 2) score candidates
        scored: list[tuple[int, list[str], dict[str, Any]]] = []
        for c in cands:
            s, reasons = score_candidate(c, person)
            scored.append((s, reasons, c))
        scored.sort(key=lambda t: t[0], reverse=True)

        best = scored[0] if scored else None
        if not best or best[0] < 20:
            payload = {
                "name": name,
                "status": "no_match",
                "error": None,
                "person_input": {
                    "affiliation": affiliation,
                    "website": website,
                    "website_domain": person.website_domain,
                },
                "candidates": [
                    {
                        "score": s,
                        "reasons": reasons,
                        "openalex_id": c.get("id"),
                        "display_name": c.get("display_name"),
                        "orcid": c.get("orcid"),
                        "works_count": c.get("works_count"),
                        "last_known_institution": (c.get("last_known_institution") or {}).get("display_name"),
                    }
                    for (s, reasons, c) in scored
                ],
                "_summary_row": {
                    "name": name,
                    "status": "no_match",
                    "confidence": "low",
                    "openalex_id": None,
                    "display_name": None,
                    "institution": None,
                    "works_count": None,
                    "score": scored[0][0] if scored else 0,
                    "reasons": ";".join(scored[0][1]) if scored else "",
                },
            }
            author_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            works_out.write_text(json.dumps({"name": name, "works": [], "works_count": 0, "error": "no_match"}, indent=2), encoding="utf-8")
            summary_rows.append(payload["_summary_row"])
            print(f"[no]  {name} -> no good match (best score {payload['_summary_row']['score']})")
            continue

        best_score, best_reasons, cand = best
        conf = confidence_from_score(best_score)

        openalex_id = cand.get("id")
        inst = (cand.get("last_known_institution") or {}).get("display_name")
        wc = cand.get("works_count")

        author_payload = {
            "name": name,
            "status": "matched",
            "confidence": conf,
            "score": best_score,
            "reasons": best_reasons,
            "person_input": {
                "affiliation": affiliation,
                "website": website,
                "website_domain": person.website_domain,
            },
            "best_author": {
                "openalex_id": openalex_id,
                "display_name": cand.get("display_name"),
                "orcid": cand.get("orcid"),
                "works_count": wc,
                "cited_by_count": cand.get("cited_by_count"),
                "last_known_institution": inst,
            },
            "candidates": [
                {
                    "score": s,
                    "reasons": reasons,
                    "openalex_id": c.get("id"),
                    "display_name": c.get("display_name"),
                    "orcid": c.get("orcid"),
                    "works_count": c.get("works_count"),
                    "last_known_institution": (c.get("last_known_institution") or {}).get("display_name"),
                }
                for (s, reasons, c) in scored
            ],
        }

        # 3) fetch works for best author
        works_payload: dict[str, Any]
        try:
            works_raw = fetch_all_works_for_author(openalex_id)
            works_simple = [simplify_work(w) for w in works_raw]
            works_payload = {
                "name": name,
                "openalex_author_id": openalex_id,
                "works_count": len(works_simple),
                "works": works_simple,
            }
            works_out.write_text(json.dumps(works_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            author_payload["works_error"] = str(e)
            works_payload = {
                "name": name,
                "openalex_author_id": openalex_id,
                "works_count": 0,
                "error": str(e),
                "works": [],
            }
            works_out.write_text(json.dumps(works_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[warn] {name} matched but works fetch failed: {e}")

        # 4) persist author payload + summary row
        author_payload["_summary_row"] = {
            "name": name,
            "status": "matched",
            "confidence": conf,
            "openalex_id": openalex_id,
            "display_name": cand.get("display_name"),
            "institution": inst,
            "works_count": wc,
            "score": best_score,
            "reasons": ";".join(best_reasons),
            "downloaded_works": works_payload.get("works_count", 0),
            "works_error": works_payload.get("error"),
        }

        author_out.write_text(json.dumps(author_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        summary_rows.append(author_payload["_summary_row"])

        print(
            f"[ok] {name} -> {conf} ({best_score}) "
            f"OpenAlex={openalex_id} works={author_payload['_summary_row']['downloaded_works']}"
        )

    # 5) write summary CSV
    summary_path = OUT_DIR / "openalex_summary.csv"
    if summary_rows:
        keys = sorted({k for r in summary_rows for k in r.keys()})
        with summary_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in summary_rows:
                w.writerow(r)

    print(f"\nDone.")
    print(f"Authors saved in: {AUTHORS_DIR}/")
    print(f"Works saved in:   {WORKS_DIR}/")
    print(f"Summary CSV:      {summary_path}")


if __name__ == "__main__":
    main()