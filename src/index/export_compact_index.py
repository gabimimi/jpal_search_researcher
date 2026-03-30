"""
Export a compact per-researcher embedding index for the static web UI.

For each researcher, builds one vector from their chunks without letting huge
paper lists wash out website/profile text (common for very prolific authors).

Strategy:
  - Per (researcher, doc_type), keep a reservoir sample up to a type-specific cap.
  - Average embeddings within each type, then combine types with the same
    weights as src.index.search (paper > website > profile > cv).

Reads:  output/index/documents_with_embeddings.jsonl
        output/profiles/*.json
Writes: frontend/profiles_index.json
  Each researcher has "embedding" (all types) and optional "embedding_narrative"
  (website + profile + CV). Search uses max(similarity to either).

Run:
    python3 -m src.index.export_compact_index
"""
from __future__ import annotations

import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import AbstractSet, Any, Dict, List, Optional, Tuple

from src.index.search import KEYWORD_INDEX_FIELD, keyword_search_blob_from_profile

EMBEDDINGS_JSONL = Path("output/index/documents_with_embeddings.jsonl")
PROFILES_DIR = Path("output/profiles")
OUT_FILE = Path("frontend/profiles_index.json")
META_FILE = Path("output/index/embed_meta.json")

FLOAT_PRECISION = 5  # decimal places — enough for cosine similarity

# Max chunks kept per researcher per doc_type (reservoir sample if exceeded).
# Stops "mega-bibliography" authors from losing niche topics in the mean vector.
DOC_TYPE_CAPS: Dict[str, int] = {
    "paper": 48,
    "website": 10,
    "profile": 8,
    "cv": 8,
}
DEFAULT_TYPE_CAP = 12

# Match search.py DOC_TYPE_WEIGHTS for combining type-level means
DOC_TYPE_WEIGHTS: Dict[str, float] = {
    "paper": 1.0,
    "website": 0.85,
    "profile": 0.80,
    "cv": 0.75,
}
DEFAULT_TYPE_WEIGHT = 0.7

# Second vector: website + profile + CV only (no papers). Scoring uses max(full, narrative)
# so flagship topics on bios/sites aren’t lost to huge paper lists.
NARRATIVE_DOC_TYPES = frozenset({"website", "profile", "cv"})


def _norm(v: List[float]) -> float:
    s = sum(x * x for x in v)
    return math.sqrt(s) if s > 0 else 0.0


def _normalize(v: List[float]) -> List[float]:
    n = _norm(v)
    return [x / n for x in v] if n > 0 else v


def _mean_embeddings(vectors: List[List[float]]) -> List[float]:
    if not vectors:
        return []
    dim = len(vectors[0])
    acc = [0.0] * dim
    for v in vectors:
        for j, val in enumerate(v):
            acc[j] += val
    n = len(vectors)
    return [x / n for x in acc]


def _cap_for(doc_type: str) -> int:
    return DOC_TYPE_CAPS.get(doc_type, DEFAULT_TYPE_CAP)


def _reservoir_add(
    bucket: List[List[float]],
    emb: List[float],
    seen_count: int,
    cap: int,
) -> None:
    """Reservoir sample: keep at most `cap` vectors with uniform inclusion probability."""
    if cap <= 0:
        return
    if len(bucket) < cap:
        bucket.append(list(emb))
        return
    j = random.randint(1, seen_count)
    if j <= cap:
        bucket[random.randint(0, cap - 1)] = list(emb)


def _weighted_type_mean_for_slug(
    slug: str,
    buckets: Dict[Tuple[str, str], List[List[float]]],
    allowed_doc_types: Optional[AbstractSet[str]],
) -> Optional[List[float]]:
    """Weighted mean of per-type means; allowed_doc_types=None keeps all types."""
    combined: Optional[List[float]] = None
    wsum = 0.0
    dim: Optional[int] = None

    for dt in sorted({k[1] for k in buckets if k[0] == slug}):
        if allowed_doc_types is not None and dt not in allowed_doc_types:
            continue
        key = (slug, dt)
        vecs = buckets.get(key) or []
        if not vecs:
            continue
        mu = _mean_embeddings(vecs)
        if not mu:
            continue
        if dim is None:
            dim = len(mu)
        w = DOC_TYPE_WEIGHTS.get(dt, DEFAULT_TYPE_WEIGHT)
        if combined is None:
            combined = [w * x for x in mu]
        else:
            for j in range(dim):
                combined[j] += w * mu[j]
        wsum += w

    if combined is None or wsum <= 0:
        return None
    return [x / wsum for x in combined]


def load_profile_meta(profiles_dir: Path) -> Dict[str, Dict[str, Any]]:
    meta: Dict[str, Dict[str, Any]] = {}
    for p in profiles_dir.glob("*.json"):
        try:
            profile = json.loads(p.read_text(encoding="utf-8"))
            slug = profile.get("slug") or p.stem
            sf = profile.get("spreadsheet_fields") or {}
            oa = profile.get("openalex_author") or {}
            web = profile.get("website") or {}

            def v(field: str) -> Optional[str]:
                val = sf.get(field)
                if val and str(val).strip() not in {"", "nan", "None", "NaN"}:
                    return str(val).strip()
                return None

            kf = {k: val for k, val in {
                "Research Interests (open text)": v("Research Interests (open text)"),
                "Sectors": v("Sectors"),
                "Initiatives": v("Initiatives") or v("Related Initiative(s)"),
                "Regional Office Affiliation": v("Regional Office Affiliation"),
                "Specific Country Interest": v("Specific Country Interest"),
                "Web Bio": v("Web Bio"),
            }.items() if val}
            blob = keyword_search_blob_from_profile(profile)
            if blob.strip():
                kf[KEYWORD_INDEX_FIELD] = blob
            meta[slug] = {
                "name": profile.get("name") or slug,
                "website_url": web.get("url") or web.get("final_url"),
                "key_fields": kf,
                "institution": oa.get("institution"),
            }
        except Exception:
            continue
    return meta


def main() -> None:
    if not EMBEDDINGS_JSONL.exists():
        print(f"ERROR: {EMBEDDINGS_JSONL} not found. Run embed_documents first.", file=sys.stderr)
        sys.exit(1)

    # Read model from meta
    model = "text-embedding-3-small"
    if META_FILE.exists():
        model = json.loads(META_FILE.read_text()).get("model", model)

    random.seed(42)

    print(f"Loading embeddings from {EMBEDDINGS_JSONL}...")
    t0 = time.time()

    # (slug, doc_type) -> reservoir bucket + count for sampling
    buckets: Dict[Tuple[str, str], List[List[float]]] = defaultdict(list)
    type_counts: Dict[Tuple[str, str], int] = defaultdict(int)
    all_slugs: set = set()
    line_i = 0

    with EMBEDDINGS_JSONL.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            slug = doc.get("researcher_slug")
            emb = doc.get("embedding")
            if not slug or not emb:
                continue
            dt = str(doc.get("doc_type") or "unknown")
            key = (slug, dt)
            all_slugs.add(slug)
            type_counts[key] += 1
            _reservoir_add(buckets[key], emb, type_counts[key], _cap_for(dt))
            line_i += 1
            if line_i % 50000 == 0:
                print(f"  {line_i} chunks processed...", end="\r")

    total_kept = sum(len(v) for v in buckets.values())
    print(f"\nProcessed {line_i} chunks → {total_kept} kept in reservoirs for {len(all_slugs)} researchers in {time.time()-t0:.1f}s")

    print("Loading profile metadata...")
    profile_meta = load_profile_meta(PROFILES_DIR)

    print("Computing stratified embeddings (full + narrative max-pool source)…")
    researchers = []
    for slug in sorted(all_slugs):
        full_raw = _weighted_type_mean_for_slug(slug, buckets, None)
        if full_raw is None:
            continue

        norm_full = _normalize(full_raw)
        rounded = [round(x, FLOAT_PRECISION) for x in norm_full]

        narrative_raw = _weighted_type_mean_for_slug(slug, buckets, NARRATIVE_DOC_TYPES)
        narrative_rounded: Optional[List[float]] = None
        if narrative_raw is not None:
            norm_nar = _normalize(narrative_raw)
            narrative_rounded = [round(x, FLOAT_PRECISION) for x in norm_nar]
            # Skip duplicate when narrative ≡ full (paper-less profiles)
            if narrative_rounded == rounded:
                narrative_rounded = None

        meta = profile_meta.get(slug, {})
        row: Dict[str, Any] = {
            "slug": slug,
            "name": meta.get("name") or slug,
            "embedding": rounded,
            "website_url": meta.get("website_url"),
            "institution": meta.get("institution"),
            "key_fields": meta.get("key_fields", {}),
        }
        if narrative_rounded is not None:
            row["embedding_narrative"] = narrative_rounded
        researchers.append(row)

    payload = {
        "model": model,
        "generated": time.strftime("%Y-%m-%d"),
        "count": len(researchers),
        "researchers": researchers,
    }

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    OUT_FILE.write_text(raw, encoding="utf-8")

    size_mb = OUT_FILE.stat().st_size / 1_000_000
    print(f"Done. {len(researchers)} researchers written to {OUT_FILE} ({size_mb:.1f} MB)")
    print("Next: commit frontend/ to GitHub and deploy to GitHub Pages.")


if __name__ == "__main__":
    main()
