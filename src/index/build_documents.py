"""
Build retrieval documents/chunks from researcher profiles.

Reads:  output/profiles/*.json
Writes: output/index/documents.jsonl
        output/index/documents_summary.txt

Run:
    python3 -m src.index.build_documents
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

PROFILES_DIR = Path("output/profiles")
OUT_DIR = Path("output/index")
DOCUMENTS_JSONL = OUT_DIR / "documents.jsonl"
SUMMARY_TXT = OUT_DIR / "documents_summary.txt"

# Approximate target chunk size in characters (~4 chars/token → 300 tokens ≈ 1200 chars)
CHUNK_TARGET_CHARS = 1200
CHUNK_OVERLAP_CHARS = 200

# Only keep papers within this many years of today
WORKS_LAST_N_YEARS = 10


# ---------------------------------------------------------------------------
# Text chunking
# ---------------------------------------------------------------------------

def _split_into_chunks(text: str, target: int = CHUNK_TARGET_CHARS, overlap: int = CHUNK_OVERLAP_CHARS) -> List[str]:
    """Split text into overlapping chunks, preferring paragraph/sentence breaks."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= target:
        return [text]

    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + target, len(text))
        # Try to break at a paragraph boundary first
        segment = text[start:end]
        if end < len(text):
            # Look for a double-newline near the end
            split_pos = segment.rfind("\n\n")
            if split_pos == -1 or split_pos < target // 2:
                # Fall back to single newline
                split_pos = segment.rfind("\n")
            if split_pos == -1 or split_pos < target // 2:
                # Fall back to period + space
                split_pos = segment.rfind(". ")
            if split_pos != -1 and split_pos >= target // 2:
                segment = segment[: split_pos + 1]

        chunks.append(segment.strip())
        advance = max(len(segment) - overlap, 1)
        start += advance

    return [c for c in chunks if c]


# ---------------------------------------------------------------------------
# Profile field extraction
# ---------------------------------------------------------------------------

def _nonempty(v: Any) -> bool:
    if v is None:
        return False
    s = str(v).strip()
    return s not in {"", "nan", "none", "null", "NaN"}


def _profile_text(profile: dict) -> str:
    """Build a short synthesized text from spreadsheet_fields."""
    sf = profile.get("spreadsheet_fields") or {}
    parts: List[str] = []

    name = profile.get("name") or sf.get("Full Name") or ""
    if name:
        parts.append(f"Researcher: {name}")

    rt = sf.get("Researcher Type")
    if _nonempty(rt):
        parts.append(f"Researcher Type: {rt}")

    for field in [
        "Research Interests (open text)",
        "Sectors",
        "Sector/Initiative interest",
        "Initiatives",
        "Related Initiative(s)",
        "Regional Office Affiliation",
        "Regional interest",
        "Specific Country Interest",
        "Web Bio",
        "Publication Notes",
    ]:
        v = sf.get(field)
        if _nonempty(v):
            parts.append(f"{field}: {v}")

    # Extra sheet lists
    for field in ["initiatives", "offices"]:
        v = sf.get(field)
        if _nonempty(v):
            parts.append(f"{field}: {v}")

    # OpenAlex institution
    oa = profile.get("openalex_author") or {}
    if oa.get("institution"):
        parts.append(f"Institution: {oa['institution']}")

    return "\n".join(parts).strip()


def _build_profile_docs(profile: dict, slug: str, name: str) -> Iterator[dict]:
    text = _profile_text(profile)
    if not text:
        return
    sf = profile.get("spreadsheet_fields") or {}
    oa = profile.get("openalex_author") or {}
    yield {
        "doc_id": f"{slug}::profile",
        "researcher_slug": slug,
        "researcher_name": name,
        "doc_type": "profile",
        "year": None,
        "title": None,
        "text": text,
        "source": None,
        "metadata": {
            "institution": oa.get("institution"),
            "offices": sf.get("offices"),
            "initiatives": sf.get("Initiatives") or sf.get("initiatives"),
            "sectors": sf.get("Sectors"),
            "country": sf.get("Specific Country Interest"),
            "openalex_id": oa.get("openalex_id"),
        },
    }


def _build_web_docs(profile: dict, slug: str, name: str) -> Iterator[dict]:
    web = profile.get("website") or {}
    text = (web.get("text") or "").strip()
    if not text or web.get("status") not in {"ok"}:
        return
    url = web.get("url") or web.get("final_url")
    for i, chunk in enumerate(_split_into_chunks(text)):
        yield {
            "doc_id": f"{slug}::web::{i}",
            "researcher_slug": slug,
            "researcher_name": name,
            "doc_type": "website",
            "year": None,
            "title": None,
            "text": chunk,
            "source": url,
            "metadata": {},
        }


def _build_cv_docs(profile: dict, slug: str, name: str) -> Iterator[dict]:
    cv = profile.get("cv") or {}
    text = (cv.get("text") or "").strip()
    if not text or cv.get("status") not in {"ok"}:
        return
    url = cv.get("url") or cv.get("final_url")
    for i, chunk in enumerate(_split_into_chunks(text)):
        yield {
            "doc_id": f"{slug}::cv::{i}",
            "researcher_slug": slug,
            "researcher_name": name,
            "doc_type": "cv",
            "year": None,
            "title": None,
            "text": chunk,
            "source": url,
            "metadata": {},
        }


def _build_paper_docs(profile: dict, slug: str, name: str) -> Iterator[dict]:
    works_block = profile.get("openalex_works") or {}
    works = works_block.get("works") or []
    now_year = int(time.strftime("%Y"))
    min_year = now_year - WORKS_LAST_N_YEARS

    for w in works:
        year = w.get("year")
        if isinstance(year, int) and year < min_year:
            continue
        title = (w.get("title") or "").strip()
        abstract = (w.get("abstract") or "").strip()
        venue = (w.get("venue") or "").strip()

        parts = []
        if title:
            parts.append(title)
        if venue:
            parts.append(f"Published in: {venue}")
        if year:
            parts.append(f"Year: {year}")
        if abstract:
            parts.append(abstract)

        text = "\n".join(parts).strip()
        if not text:
            continue

        yield {
            "doc_id": f"{slug}::paper::{w.get('openalex_id', title[:40])}",
            "researcher_slug": slug,
            "researcher_name": name,
            "doc_type": "paper",
            "year": year,
            "title": title or None,
            "text": text,
            "source": w.get("landing_page_url") or w.get("doi"),
            "metadata": {
                "venue": venue or None,
                "cited_by_count": w.get("cited_by_count"),
                "openalex_id": w.get("openalex_id"),
            },
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    profile_paths = sorted(PROFILES_DIR.glob("*.json"))
    if not profile_paths:
        print(f"ERROR: No profiles found in {PROFILES_DIR}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(profile_paths)} profiles. Building documents...")

    counts: Dict[str, int] = {"profile": 0, "website": 0, "cv": 0, "paper": 0}
    total_chars: Dict[str, int] = {"profile": 0, "website": 0, "cv": 0, "paper": 0}
    errors = 0

    with DOCUMENTS_JSONL.open("w", encoding="utf-8") as fout:
        for i, path in enumerate(profile_paths):
            try:
                profile = json.loads(path.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"  SKIP {path.name}: {e}", file=sys.stderr)
                errors += 1
                continue

            slug = profile.get("slug") or path.stem
            name = profile.get("name") or slug

            generators = [
                _build_profile_docs(profile, slug, name),
                _build_web_docs(profile, slug, name),
                _build_cv_docs(profile, slug, name),
                _build_paper_docs(profile, slug, name),
            ]

            for gen in generators:
                for doc in gen:
                    fout.write(json.dumps(doc, ensure_ascii=False) + "\n")
                    dt = doc["doc_type"]
                    counts[dt] = counts.get(dt, 0) + 1
                    total_chars[dt] = total_chars.get(dt, 0) + len(doc["text"])

            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(profile_paths)} profiles processed...")

    total_docs = sum(counts.values())
    lines = [
        "DOCUMENT BUILD SUMMARY",
        f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Profiles processed: {len(profile_paths)} ({errors} errors)",
        f"Total documents: {total_docs}",
        "",
        "By doc_type:",
    ]
    for dt in ["profile", "website", "cv", "paper"]:
        n = counts.get(dt, 0)
        avg = (total_chars.get(dt, 0) // n) if n else 0
        lines.append(f"  {dt:<10} count={n:>6}  avg_chars={avg:>6}")

    lines.append("")
    lines.append(f"Output: {DOCUMENTS_JSONL}")

    summary = "\n".join(lines)
    SUMMARY_TXT.write_text(summary, encoding="utf-8")
    print(summary)
    print(f"\nDone. Documents written to: {DOCUMENTS_JSONL}")


if __name__ == "__main__":
    main()
