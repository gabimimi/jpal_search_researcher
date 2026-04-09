"""
Search researchers by a free-text query using vector similarity.

Usage:
    python3 -m src.index.search --query "K-12 education"
    python3 -m src.index.search --query "RCTs on vaccination" --top 20 --top-chunks 3
    python3 -m src.index.search --query "reproductive health" --filter-initiative "J-PAL"

By default, if frontend/profiles_index.json exists (from export_compact_index), that
compact index is used (~seconds, small memory). Use --chunk-index to score all
embedding chunks (slow: reads the full documents_with_embeddings.jsonl).

Environment:
    OPENAI_API_KEY  (required)
    Or create a `.env` file in the project root with:
        OPENAI_API_KEY=sk-...

Output: JSON list of {name, slug, score, evidence_snippets, key_fields}
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

EMBEDDINGS_JSONL = Path("output/index/documents_with_embeddings.jsonl")
META_FILE = Path("output/index/embed_meta.json")
PROFILES_DIR = Path("output/profiles")
COMPACT_INDEX_DEFAULT = Path("frontend/profiles_index.json")
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _load_dotenv_project_root() -> None:
    """Set os.environ from project .env if vars are unset or empty in the shell (no extra deps)."""
    path = _PROJECT_ROOT / ".env"
    if not path.is_file():
        return
    try:
        for raw in path.read_text(encoding="utf-8-sig").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            if key.lower().startswith("export "):
                key = key[7:].strip()
            val = val.strip().strip('"').strip("'")
            if key and val and (key not in os.environ or not str(os.environ.get(key, "")).strip()):
                os.environ[key] = val
    except OSError:
        pass


# Weights for aggregating chunk scores by doc_type
DOC_TYPE_WEIGHTS: Dict[str, float] = {
    "paper": 1.0,
    "website": 0.85,
    "profile": 0.80,
    "cv": 0.75,
}

# Keyword boost: weights reflect how precisely each field describes the researcher's work.
# Research interests and the full-text keyword blob are strongest; initiative fields are minor.
KEYWORD_BOOST_WEIGHTS: Dict[str, float] = {
    "Specific Country Interest":        0.12,
    "Research Interests (open text)":   0.10,
    "Website & publications (keyword index)": 0.10,
    "institution":                      0.08,
    "Sectors":                          0.06,
    "Regional Office Affiliation":      0.06,
    "Languages":                        0.05,
    "offices":                          0.05,
    "Regional interest":                0.04,
    "Web Bio":                          0.04,
    "Publication Notes":                0.03,
    "Initiatives":                      0.03,
    "initiatives":                      0.02,
    "Researcher Type":                  0.02,
    "Sector/Initiative interest":       0.02,
    "Related Initiative(s)":            0.01,
}
KEYWORD_INDEX_FIELD = "Website & publications (keyword index)"
KEYWORD_BOOST_CAP = 0.40

STOP_WORDS = {
    "a", "an", "the", "in", "on", "at", "of", "for", "to", "with",
    "and", "or", "is", "are", "was", "were", "be", "been", "by",
    "from", "that", "this", "it", "its", "i", "about", "into",
    "research", "study", "studies", "experiment", "experiments",
    "evidence", "impact", "effect", "effects", "using", "use",
}

DEFAULT_TOP_RESEARCHERS = 10
DEFAULT_TOP_CHUNKS = 5       # top matching chunks per researcher for evidence
DEFAULT_CANDIDATE_K = 500    # number of top chunks before aggregation


# ---------------------------------------------------------------------------
# Keyword boost helpers
# ---------------------------------------------------------------------------

def _infer_institution(profile: dict) -> str:
    """Derive primary institution from OpenAlex author or works data."""
    oa = profile.get("openalex_author") or {}
    explicit = oa.get("institution")
    if explicit and str(explicit).strip() not in ("", "None", "nan"):
        return str(explicit).strip()
    oa_id = oa.get("openalex_id", "")
    aname = (oa.get("display_name") or profile.get("name") or "").lower()
    works = (profile.get("openalex_works") or {}).get("works") or []
    counts: Dict[str, int] = {}
    for w in works:
        for a in w.get("authorships", []):
            matched = (oa_id and a.get("author_openalex_id") == oa_id) or \
                      (aname and a.get("author_name", "").lower() == aname)
            if matched:
                for inst in a.get("institutions", []):
                    if inst and str(inst).strip():
                        counts[str(inst).strip()] = counts.get(str(inst).strip(), 0) + 1
    if not counts:
        return ""
    return max(counts, key=counts.get)  # type: ignore[arg-type]


def keyword_search_blob_from_profile(profile: dict, max_chars: int = 25000) -> str:
    """Lowercase text for literal keyword matching (website, CV, paper titles+abstracts, institution)."""
    parts: List[str] = []

    inst = _infer_institution(profile)
    if inst:
        parts.append(inst)

    web = profile.get("website") or {}
    text = (web.get("text") or "").strip()
    if text:
        parts.append(text[:9000])

    cv = profile.get("cv") or {}
    cv_text = (cv.get("text") or "").strip()
    if cv_text and cv.get("status") == "ok":
        parts.append(cv_text[:3000])

    oa = profile.get("openalex_works") or {}
    works = oa.get("works") or []
    if isinstance(works, list):
        work_parts: List[str] = []
        for item in works[:50]:
            if not isinstance(item, dict):
                continue
            t = item.get("title")
            if t:
                work_parts.append(str(t))
            ab = (item.get("abstract") or "")[:300]
            if ab:
                work_parts.append(ab)
        if work_parts:
            parts.append("\n".join(work_parts))

    blob = " ".join(parts).lower()
    return blob[:max_chars] if blob else ""


def extract_query_terms(query: str) -> List[str]:
    """Return meaningful lowercase words from the query, excluding stop words."""
    words = re.findall(r"[a-zA-Z]{3,}", query.lower())
    out = [w for w in words if w not in STOP_WORDS]
    # “RCT” rarely appears literally; prose uses “randomized”.
    if "rct" in out:
        for x in ("randomized", "randomised"):
            if x not in out:
                out.append(x)
    return out


def load_profile_metadata(profiles_dir: Path) -> Dict[str, Dict[str, str]]:
    """Load key metadata fields for every researcher, keyed by slug."""
    sheet_fields = [f for f in KEYWORD_BOOST_WEIGHTS if f not in (KEYWORD_INDEX_FIELD, "institution")]
    meta: Dict[str, Dict[str, str]] = {}
    for p in profiles_dir.glob("*.json"):
        try:
            profile = json.loads(p.read_text(encoding="utf-8"))
            slug = profile.get("slug") or p.stem
            sf = profile.get("spreadsheet_fields") or {}
            row = {field: str(sf.get(field) or "").lower() for field in sheet_fields}
            inst = _infer_institution(profile)
            if inst:
                row["institution"] = inst.lower()
            blob = keyword_search_blob_from_profile(profile)
            if blob.strip():
                row[KEYWORD_INDEX_FIELD] = blob
            meta[slug] = row
        except Exception:
            continue
    return meta


def apply_keyword_boost(
    ranked: List[Dict[str, Any]],
    query_terms: List[str],
    profile_meta: Dict[str, Dict[str, str]],
) -> List[Dict[str, Any]]:
    """
    Add a keyword boost to each researcher's score based on how many
    query terms appear literally in their profile metadata fields.
    Re-sorts the list after applying boost.
    """
    if not query_terms:
        return ranked

    for r in ranked:
        slug = r["slug"]
        fields = profile_meta.get(slug, {})
        boost = 0.0
        matched: Dict[str, List[str]] = {}

        for field, field_weight in KEYWORD_BOOST_WEIGHTS.items():
            field_text = fields.get(field, "")
            if not field_text:
                continue
            hits = [t for t in query_terms if t in field_text]
            if hits:
                boost += field_weight * len(hits)
                matched[field] = hits

        boost = min(boost, KEYWORD_BOOST_CAP)
        r["score"] = round(r["score"] + boost, 4)
        r["keyword_boost"] = round(boost, 4)
        r["keyword_matches"] = matched

    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked


# ---------------------------------------------------------------------------
# Vector math
# ---------------------------------------------------------------------------

def _dot(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _norm(v: List[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def cosine_similarity(a: List[float], b: List[float]) -> float:
    na = _norm(a)
    nb = _norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return _dot(a, b) / (na * nb)


def _normalize_vec(v: List[float]) -> List[float]:
    n = _norm(v)
    return [x / n for x in v] if n > 0 else v


# ---------------------------------------------------------------------------
# Compact index (per-researcher averaged embedding — fast)
# ---------------------------------------------------------------------------

def load_compact_index(path: Path) -> List[Dict[str, Any]]:
    """Load researchers from export_compact_index output (profiles_index.json)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(data.get("researchers") or [])


def compact_rows_to_profile_meta(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, str]]:
    """Build keyword-boost field map from compact key_fields."""
    meta: Dict[str, Dict[str, str]] = {}
    for r in rows:
        slug = r.get("slug") or ""
        if not slug:
            continue
        kf = r.get("key_fields") or {}
        meta[slug] = {
            field: str(kf.get(field) or "").lower()
            for field in KEYWORD_BOOST_WEIGHTS
        }
    return meta


def search_compact(
    query_emb: List[float],
    rows: List[Dict[str, Any]],
    filter_initiative: Optional[str],
) -> List[Dict[str, Any]]:
    """
    Rank researchers by cosine similarity to query against stored unit vectors.
    Rows must be normalized embeddings (as produced by export_compact_index).
    """
    q = _normalize_vec(query_emb)
    needle = filter_initiative.lower() if filter_initiative else None

    ranked: List[Dict[str, Any]] = []
    for r in rows:
        slug = r.get("slug")
        emb = r.get("embedding")
        name = r.get("name") or slug
        if not slug or not emb:
            continue
        if needle:
            kf = r.get("key_fields") or {}
            init = str(kf.get("Initiatives") or "").lower()
            if needle not in init:
                continue
        # max(full embedding, narrative = website+profile+cv) matches export_compact_index + app.js
        score = _dot(q, emb)
        nar = r.get("embedding_narrative")
        if nar and len(nar) == len(emb):
            score = max(score, _dot(q, nar))
        kf = dict(r.get("key_fields") or {})
        if r.get("institution"):
            kf.setdefault("institution", r["institution"])
        ranked.append({
            "slug": slug,
            "name": name,
            "score": round(score, 4),
            "score_by_type": {"profile_avg": round(score, 4)},
            "_evidence_chunks": [],
            "_compact_key_fields": kf,
        })

    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked


# ---------------------------------------------------------------------------
# Load index
# ---------------------------------------------------------------------------

def load_index(path: Path) -> List[dict]:
    """Load all embedded documents into memory."""
    docs = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs


# ---------------------------------------------------------------------------
# Embed query
# ---------------------------------------------------------------------------

def embed_query(text: str, model: str, client) -> List[float]:
    for attempt in range(5):
        try:
            resp = client.embeddings.create(model=model, input=[text])
            return resp.data[0].embedding
        except Exception as e:
            if attempt >= 3:
                raise
            wait = 2 ** attempt
            print(f"  Retry {attempt+1} embedding query: {e}", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError("Query embedding failed")


# ---------------------------------------------------------------------------
# Retrieval + aggregation
# ---------------------------------------------------------------------------

def retrieve(
    query_emb: List[float],
    docs: List[dict],
    candidate_k: int,
    filter_fn=None,
) -> List[Tuple[float, dict]]:
    """Score all docs, apply optional filter, return top-k (score, doc) pairs."""
    scored = []
    for doc in docs:
        emb = doc.get("embedding")
        if not emb:
            continue
        if filter_fn and not filter_fn(doc):
            continue
        score = cosine_similarity(query_emb, emb)
        scored.append((score, doc))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:candidate_k]


def aggregate_by_researcher(
    scored_chunks: List[Tuple[float, dict]],
    top_chunks_per_researcher: int,
) -> List[Dict[str, Any]]:
    """
    Aggregate chunk scores into researcher scores.

    Strategy:
      - For each (researcher, doc_type) keep the max chunk score (weighted by doc_type).
      - Final researcher score = sum of those per-type max scores.
      - Also collect top evidence chunks.
    """
    from collections import defaultdict

    # per researcher: {doc_type: max_weighted_score}
    type_max: Dict[str, Dict[str, float]] = defaultdict(dict)
    # per researcher: list of (raw_score, doc) for top-k evidence
    evidence_chunks: Dict[str, List[Tuple[float, dict]]] = defaultdict(list)
    researcher_name: Dict[str, str] = {}

    for score, doc in scored_chunks:
        slug = doc["researcher_slug"]
        name = doc["researcher_name"]
        dt = doc["doc_type"]
        weight = DOC_TYPE_WEIGHTS.get(dt, 0.7)
        weighted = score * weight

        researcher_name[slug] = name
        if dt not in type_max[slug] or weighted > type_max[slug][dt]:
            type_max[slug][dt] = weighted

        evidence_chunks[slug].append((score, doc))

    results = []
    for slug, type_scores in type_max.items():
        total_score = sum(type_scores.values())

        # Pick top evidence chunks (diverse doc types first)
        chunks = sorted(evidence_chunks[slug], key=lambda x: x[0], reverse=True)
        seen_types: Dict[str, int] = {}
        top_evidence = []
        for raw_score, doc in chunks:
            dt = doc["doc_type"]
            seen_types[dt] = seen_types.get(dt, 0) + 1
            # Allow at most 2 chunks per doc_type in evidence
            if seen_types[dt] <= 2:
                top_evidence.append((raw_score, doc))
            if len(top_evidence) >= top_chunks_per_researcher:
                break

        results.append({
            "slug": slug,
            "name": researcher_name[slug],
            "score": round(total_score, 4),
            "score_by_type": {k: round(v, 4) for k, v in type_scores.items()},
            "_evidence_chunks": top_evidence,
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results


# ---------------------------------------------------------------------------
# Format output
# ---------------------------------------------------------------------------

def _snippet(text: str, max_chars: int = 300) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    # Try to cut at a sentence boundary
    cut = text[:max_chars]
    last_period = cut.rfind(". ")
    if last_period > max_chars // 2:
        return cut[: last_period + 1]
    return cut + "…"


def format_results(
    ranked: List[Dict[str, Any]],
    top_n: int,
    profiles_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    output = []
    for r in ranked[:top_n]:
        slug = r["slug"]

        # Load key_fields from profile if available (or from compact index row)
        key_fields: Dict[str, Any] = {}
        pre = r.get("_compact_key_fields")
        if pre:
            for field in [
                "Research Interests (open text)",
                "Sectors",
                "Initiatives",
                "Regional Office Affiliation",
                "Specific Country Interest",
                "Web Bio",
            ]:
                v = pre.get(field)
                if v and str(v).strip() not in {"", "nan", "None"}:
                    key_fields[field] = str(v).strip()
            if pre.get("institution"):
                key_fields["institution"] = pre["institution"]
        elif profiles_dir:
            ppath = profiles_dir / f"{slug}.json"
            if ppath.exists():
                try:
                    p = json.loads(ppath.read_text(encoding="utf-8"))
                    sf = p.get("spreadsheet_fields") or {}
                    for field in [
                        "Research Interests (open text)",
                        "Sectors",
                        "Initiatives",
                        "Regional Office Affiliation",
                        "Specific Country Interest",
                        "Web Bio",
                    ]:
                        v = sf.get(field)
                        if v and str(v).strip() not in {"", "nan", "None"}:
                            key_fields[field] = str(v).strip()
                    oa = p.get("openalex_author") or {}
                    if oa.get("institution"):
                        key_fields["institution"] = oa["institution"]
                except Exception:
                    pass

        # Build evidence snippets
        evidence: List[Dict[str, Any]] = []
        for raw_score, doc in r.get("_evidence_chunks", []):
            e: Dict[str, Any] = {
                "doc_type": doc["doc_type"],
                "score": round(raw_score, 4),
                "snippet": _snippet(doc["text"]),
            }
            if doc.get("title"):
                e["title"] = doc["title"]
            if doc.get("year"):
                e["year"] = doc["year"]
            if doc.get("source"):
                e["source"] = doc["source"]
            evidence.append(e)

        output.append({
            "rank": len(output) + 1,
            "name": r["name"],
            "slug": slug,
            "score": r["score"],
            "keyword_boost": r.get("keyword_boost", 0),
            "keyword_matches": r.get("keyword_matches", {}),
            "score_by_type": r["score_by_type"],
            "key_fields": key_fields,
            "evidence_snippets": evidence,
        })

    return output


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Search researchers by topic query")
    parser.add_argument("--query", "-q", required=True, help="Free-text research topic")
    parser.add_argument("--top", "-n", type=int, default=DEFAULT_TOP_RESEARCHERS,
                        help="Number of top researchers to return")
    parser.add_argument("--top-chunks", type=int, default=DEFAULT_TOP_CHUNKS,
                        help="Evidence chunks per researcher")
    parser.add_argument("--candidate-k", type=int, default=DEFAULT_CANDIDATE_K,
                        help="Top-K chunks retrieved before aggregation")
    parser.add_argument("--filter-initiative", default=None,
                        help="Filter docs to researchers with this initiative substring")
    parser.add_argument("--json", action="store_true", help="Output raw JSON (default: pretty-print)")
    parser.add_argument("--embeddings", default=str(EMBEDDINGS_JSONL),
                        help="Path to documents_with_embeddings.jsonl")
    parser.add_argument("--chunk-index", action="store_true",
                        help="Use full chunk embeddings file (slow, large). Default uses compact index when available.")
    parser.add_argument("--compact-index", default=str(COMPACT_INDEX_DEFAULT),
                        help="Path to profiles_index.json from export_compact_index")
    args = parser.parse_args()

    _load_dotenv_project_root()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print(
            "ERROR: Set OPENAI_API_KEY (e.g. export OPENAI_API_KEY=sk-... in your shell, "
            f"or add it to {_PROJECT_ROOT / '.env'}).",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: openai package not installed.", file=sys.stderr)
        sys.exit(1)

    compact_path = Path(args.compact_index)
    emb_path = Path(args.embeddings)
    use_compact = (not args.chunk_index) and compact_path.exists()

    if not use_compact and not emb_path.exists():
        print(
            "ERROR: No search index found. Either run `python3 -m src.index.export_compact_index` "
            f"(creates {compact_path}) or build {emb_path} and pass --chunk-index.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Determine embedding model from meta
    model = DEFAULT_MODEL if not META_FILE.exists() else json.loads(
        META_FILE.read_text(encoding="utf-8")
    ).get("model", DEFAULT_MODEL)

    client = OpenAI(api_key=api_key)

    print(f"Embedding query: '{args.query}'", file=sys.stderr)
    query_emb = embed_query(args.query, model, client)

    if use_compact:
        print(f"Loading compact index from {compact_path}...", file=sys.stderr)
        t0 = time.time()
        compact_rows = load_compact_index(compact_path)
        print(
            f"Loaded {len(compact_rows)} researchers in {time.time()-t0:.1f}s (compact mode)",
            file=sys.stderr,
        )
        ranked = search_compact(query_emb, compact_rows, args.filter_initiative)
        print("Applying keyword boost...", file=sys.stderr)
        query_terms = extract_query_terms(args.query)
        profile_meta = compact_rows_to_profile_meta(compact_rows)
        ranked = apply_keyword_boost(ranked, query_terms, profile_meta)
        results = format_results(ranked, args.top, None)
    else:
        print(f"Loading chunk index from {emb_path}...", file=sys.stderr)
        t0 = time.time()
        docs = load_index(emb_path)
        print(f"Loaded {len(docs)} chunks in {time.time()-t0:.1f}s", file=sys.stderr)

        filter_fn = None
        if args.filter_initiative:
            needle = args.filter_initiative.lower()
            def filter_fn(doc):
                meta = doc.get("metadata") or {}
                init = str(meta.get("initiatives") or "").lower()
                return needle in init

        print(f"Retrieving top {args.candidate_k} chunks...", file=sys.stderr)
        scored_chunks = retrieve(query_emb, docs, args.candidate_k, filter_fn)

        print("Aggregating by researcher...", file=sys.stderr)
        ranked = aggregate_by_researcher(scored_chunks, args.top_chunks)

        print("Applying keyword boost...", file=sys.stderr)
        query_terms = extract_query_terms(args.query)
        profile_meta = load_profile_metadata(PROFILES_DIR)
        ranked = apply_keyword_boost(ranked, query_terms, profile_meta)

        results = format_results(ranked, args.top, PROFILES_DIR)

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        _pretty_print(results, args.query)


def _pretty_print(results: List[Dict], query: str) -> None:
    print(f"\n{'='*70}")
    print(f"Query: {query}")
    print(f"Top {len(results)} researchers")
    print(f"{'='*70}\n")
    for r in results:
        boost_str = f"  +{r['keyword_boost']:.3f} keyword" if r.get("keyword_boost") else ""
        print(f"#{r['rank']}  {r['name']}  (score={r['score']:.4f}{boost_str})")
        if r.get("keyword_matches"):
            for field, terms in r["keyword_matches"].items():
                print(f"    ~ matched '{', '.join(terms)}' in [{field}]")
        if r.get("key_fields"):
            for k, v in r["key_fields"].items():
                print(f"    {k}: {v[:120]}")
        if r.get("evidence_snippets"):
            print("  Evidence:")
            for e in r["evidence_snippets"]:
                label = e["doc_type"]
                if e.get("title"):
                    label += f" | {e['title'][:80]}"
                if e.get("year"):
                    label += f" ({e['year']})"
                print(f"    [{label}]  score={e['score']:.4f}")
                print(f"    {e['snippet'][:200]}")
        print()


DEFAULT_MODEL = "text-embedding-3-small"

if __name__ == "__main__":
    main()
