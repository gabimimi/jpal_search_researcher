"""
Export a compact per-researcher embedding index for the static web UI.

For each researcher, averages all their chunk embeddings into a single vector.
Loads key metadata from profiles.

Reads:  output/index/documents_with_embeddings.jsonl
        output/profiles/*.json
Writes: frontend/profiles_index.json

Run:
    python3 -m src.index.export_compact_index
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

EMBEDDINGS_JSONL = Path("output/index/documents_with_embeddings.jsonl")
PROFILES_DIR = Path("output/profiles")
OUT_FILE = Path("frontend/profiles_index.json")
META_FILE = Path("output/index/embed_meta.json")

FLOAT_PRECISION = 5  # decimal places — enough for cosine similarity


def _norm(v: List[float]) -> float:
    s = sum(x * x for x in v)
    return math.sqrt(s) if s > 0 else 0.0


def _normalize(v: List[float]) -> List[float]:
    n = _norm(v)
    return [x / n for x in v] if n > 0 else v


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

            meta[slug] = {
                "name": profile.get("name") or slug,
                "website_url": web.get("url") or web.get("final_url"),
                "key_fields": {k: val for k, val in {
                    "Research Interests (open text)": v("Research Interests (open text)"),
                    "Sectors": v("Sectors"),
                    "Initiatives": v("Initiatives") or v("Related Initiative(s)"),
                    "Regional Office Affiliation": v("Regional Office Affiliation"),
                    "Specific Country Interest": v("Specific Country Interest"),
                    "Web Bio": v("Web Bio"),
                }.items() if val},
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

    print(f"Loading embeddings from {EMBEDDINGS_JSONL}...")
    t0 = time.time()

    # Accumulate sum of embeddings per researcher
    sums: Dict[str, List[float]] = {}
    counts: Dict[str, int] = {}

    with EMBEDDINGS_JSONL.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            slug = doc.get("researcher_slug")
            emb = doc.get("embedding")
            if not slug or not emb:
                continue
            if slug not in sums:
                sums[slug] = [0.0] * len(emb)
                counts[slug] = 0
            for j, val in enumerate(emb):
                sums[slug][j] += val
            counts[slug] += 1
            if (i + 1) % 50000 == 0:
                print(f"  {i+1} chunks processed...", end="\r")

    print(f"\nLoaded {sum(counts.values())} chunks for {len(sums)} researchers in {time.time()-t0:.1f}s")

    print("Loading profile metadata...")
    profile_meta = load_profile_meta(PROFILES_DIR)

    print("Computing averaged + normalized embeddings...")
    researchers = []
    for slug, emb_sum in sorted(sums.items()):
        n = counts[slug]
        avg = [x / n for x in emb_sum]
        norm_emb = _normalize(avg)
        rounded = [round(x, FLOAT_PRECISION) for x in norm_emb]

        meta = profile_meta.get(slug, {})
        researchers.append({
            "slug": slug,
            "name": meta.get("name") or slug,
            "embedding": rounded,
            "website_url": meta.get("website_url"),
            "institution": meta.get("institution"),
            "key_fields": meta.get("key_fields", {}),
        })

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
