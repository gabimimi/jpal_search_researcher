from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import pandas as pd


# =========================
# CONFIG
# =========================

SHEET_1_PATH = Path("output/researchers_clean.csv")
SHEET_2_PATH = Path("output/researchers_extra.csv")

CV_JSON_DIR = Path("output/cv")
WEB_JSON_DIR = Path("output/web")

# NEW: OpenAlex outputs
OPENALEX_AUTHORS_DIR = Path("output/openalex/authors")
OPENALEX_WORKS_DIR = Path("output/openalex/works")

OUT_DIR = Path("output/profiles")
SUMMARY_TXT = Path("output/profiles_summary.txt")

LIMIT: Optional[int] = None  # None for all

# Works filter
WORKS_LAST_N_YEARS = 10


# =========================
# Helpers
# =========================

def safe_slug(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s[:80] if s else "unknown"


def normalize_name(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s\-']", "", s)
    return s


def is_nonempty(x: object) -> bool:
    if x is None:
        return False
    if isinstance(x, float) and pd.isna(x):
        return False
    s = str(x).strip()
    return s != "" and s.lower() not in {"nan", "none", "null"}


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if path.suffix.lower() in {".csv"}:
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path.suffix} (use .csv or .xlsx)")


def find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    norm_map = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        if cand.lower().strip() in norm_map:
            return norm_map[cand.lower().strip()]
    return None


def load_json_dir(dir_path: Path) -> List[dict[str, Any]]:
    if not dir_path.exists():
        return []
    out: List[dict[str, Any]] = []
    for p in sorted(dir_path.glob("*.json")):
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            continue
    return out


def best_text_block(rec: Optional[dict[str, Any]]) -> dict[str, Any]:
    """
    Standardize CV/WEB records into a common shape.

    Handles:
      - CV schema (status/text_len/cv_url/final_url)
      - Web schema (url/text/snippet/error)
    """
    if not rec:
        return {
            "status": "missing",
            "text_len": 0,
            "text": "",
            "snippet": "",
            "url": None,
            "final_url": None,
            "filetype": None,
            "content_type": None,
            "error": None,
            "timestamp": None,
        }

    # pick a URL field that exists
    url = rec.get("cv_url") or rec.get("url") or rec.get("original_url")
    final_url = rec.get("final_url") or rec.get("best_final_or_error") or rec.get("best_url") or url

    text = rec.get("text") or ""
    snippet = rec.get("snippet") or text[:600]

    # infer status if missing
    status = rec.get("status")
    if not status:
        if rec.get("error"):
            status = "error"
        elif len(text.strip()) == 0:
            status = "empty"
        else:
            status = "ok"

    return {
        "status": status,                      # ok | empty | error | missing
        "text_len": int(rec.get("text_len") or len(text.strip())),
        "text": text,
        "snippet": snippet,
        "url": url,
        "final_url": final_url,
        "filetype": rec.get("filetype"),
        "content_type": rec.get("content_type"),
        "error": rec.get("error"),
        "timestamp": rec.get("timestamp"),
    }


def choose_key_fields(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    id_col = find_column(df, [
        "Id", "ID", "Salesforce ID", "SalesforceId", "Contact ID", "ContactId",
        "Researcher ID", "ResearcherId", "Record ID", "RecordId"
    ])
    name_col = find_column(df, [
        "Full Name", "Name", "Researcher Name", "Researcher", "Contact Name"
    ])
    return id_col, name_col


def merge_row_dicts(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    out = dict(a)
    for k, v in b.items():
        if k not in out:
            out[k] = v
        else:
            if (not is_nonempty(out[k])) and is_nonempty(v):
                out[k] = v
    return out


def load_openalex_by_slug(slug: str) -> Tuple[Optional[dict[str, Any]], Optional[dict[str, Any]]]:
    """
    Returns (author_json, works_json) if present.
    """
    a_path = OPENALEX_AUTHORS_DIR / f"{slug}.json"
    w_path = OPENALEX_WORKS_DIR / f"{slug}.json"
    author = None
    works = None
    if a_path.exists():
        try:
            author = json.loads(a_path.read_text(encoding="utf-8"))
        except Exception:
            author = None
    if w_path.exists():
        try:
            works = json.loads(w_path.read_text(encoding="utf-8"))
        except Exception:
            works = None
    return author, works


def filter_works_last_n_years(works_payload: Optional[dict[str, Any]], n_years: int) -> dict[str, Any]:
    """
    Keep only works in the last n years (based on publication year).
    """
    if not works_payload:
        return {"status": "missing", "works_count": 0, "works": []}

    works = works_payload.get("works") or []
    now_year = int(time.strftime("%Y"))
    min_year = now_year - n_years

    kept = []
    for w in works:
        y = w.get("year")
        if isinstance(y, int) and y >= min_year:
            kept.append(w)

    return {
        "status": "ok",
        "min_year": min_year,
        "works_count": len(kept),
        "works": kept,
    }


def openalex_block(author_payload: Optional[dict[str, Any]]) -> dict[str, Any]:
    if not author_payload:
        return {"status": "missing"}

    # author script uses status: matched | no_match | error
    st = author_payload.get("status", "unknown")
    if st == "matched":
        best = author_payload.get("best_author") or {}
        return {
            "status": "matched",
            "confidence": author_payload.get("confidence"),
            "score": author_payload.get("score"),
            "openalex_id": best.get("openalex_id"),
            "display_name": best.get("display_name"),
            "orcid": best.get("orcid"),
            "works_count": best.get("works_count"),
            "institution": best.get("last_known_institution"),
            "reasons": author_payload.get("reasons"),
        }
    if st == "no_match":
        return {"status": "no_match", "error": author_payload.get("error")}
    if st == "error":
        return {"status": "error", "error": author_payload.get("error")}
    return {"status": st, "error": author_payload.get("error")}


# =========================
# Main build
# =========================

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df1 = read_table(SHEET_1_PATH)
    df2 = read_table(SHEET_2_PATH)

    id1, name1 = choose_key_fields(df1)
    id2, name2 = choose_key_fields(df2)

    if name1 is None and id1 is None:
        raise ValueError(f"Could not find a name or id column in {SHEET_1_PATH}. Columns: {list(df1.columns)}")
    if name2 is None and id2 is None:
        raise ValueError(f"Could not find a name or id column in {SHEET_2_PATH}. Columns: {list(df2.columns)}")

    def index_table(df: pd.DataFrame, id_col: Optional[str], name_col: Optional[str]) -> Dict[str, dict[str, Any]]:
        idx: Dict[str, dict[str, Any]] = {}
        for _, row in df.iterrows():
            rowd = {k: (None if (isinstance(v, float) and pd.isna(v)) else v) for k, v in row.to_dict().items()}
            key = None
            if id_col and is_nonempty(rowd.get(id_col)):
                key = f"id:{str(rowd[id_col]).strip()}"
            elif name_col and is_nonempty(rowd.get(name_col)):
                key = f"name:{normalize_name(str(rowd[name_col]))}"
            if key:
                if key in idx:
                    idx[key] = merge_row_dicts(idx[key], rowd)
                else:
                    idx[key] = rowd
        return idx

    idx1 = index_table(df1, id1, name1)
    idx2 = index_table(df2, id2, name2)

    cv_recs = load_json_dir(CV_JSON_DIR)
    web_recs = load_json_dir(WEB_JSON_DIR)

    cv_by_name = {normalize_name(str(r.get("name", ""))): r for r in cv_recs if is_nonempty(r.get("name"))}
    web_by_name = {normalize_name(str(r.get("name", ""))): r for r in web_recs if is_nonempty(r.get("name"))}

    all_keys = sorted(set(idx1.keys()) | set(idx2.keys()))
    if LIMIT is not None:
        all_keys = all_keys[:LIMIT]

    stats = {
        "profiles_written": 0,
        "matched_cv": 0,
        "matched_web": 0,
        "matched_openalex_author": 0,
        "openalex_missing": 0,
        "openalex_no_match": 0,
        "openalex_error": 0,
        "cv_ok": 0, "cv_empty": 0, "cv_error": 0, "cv_missing": 0,
        "web_ok": 0, "web_empty": 0, "web_error": 0, "web_missing": 0,
    }

    for key in all_keys:
        row1 = idx1.get(key, {})
        row2 = idx2.get(key, {})
        merged_row = merge_row_dicts(row1, row2)

        name_val = None
        if name1 and is_nonempty(merged_row.get(name1)):
            name_val = str(merged_row[name1]).strip()
        elif name2 and is_nonempty(merged_row.get(name2)):
            name_val = str(merged_row[name2]).strip()
        else:
            for col in merged_row.keys():
                if "name" in str(col).lower() and is_nonempty(merged_row.get(col)):
                    name_val = str(merged_row[col]).strip()
                    break

        norm_name = normalize_name(name_val or "")
        slug = safe_slug(name_val or key.replace(":", "-"))

        cv_rec = cv_by_name.get(norm_name)
        web_rec = web_by_name.get(norm_name)

        cv_block = best_text_block(cv_rec)
        web_block = best_text_block(web_rec)

        if cv_rec:
            stats["matched_cv"] += 1
        if web_rec:
            stats["matched_web"] += 1

        if cv_block["status"] in {"ok", "empty", "error"}:
            stats[f"cv_{cv_block['status']}"] += 1
        else:
            stats["cv_missing"] += 1

        if web_block["status"] in {"ok", "empty", "error"}:
            stats[f"web_{web_block['status']}"] += 1
        else:
            stats["web_missing"] += 1

        # NEW: OpenAlex join by slug (same slug scheme as your openalex script)
        oa_author_json, oa_works_json = load_openalex_by_slug(slug)
        oa_author_block = openalex_block(oa_author_json)
        oa_works_block = filter_works_last_n_years(oa_works_json, WORKS_LAST_N_YEARS)

        if oa_author_block["status"] == "matched":
            stats["matched_openalex_author"] += 1
        elif oa_author_block["status"] == "no_match":
            stats["openalex_no_match"] += 1
        elif oa_author_block["status"] == "error":
            stats["openalex_error"] += 1
        else:
            stats["openalex_missing"] += 1

        # Combined text
        combined_parts = []
        if is_nonempty(name_val):
            combined_parts.append(f"Name: {name_val}")

        combined_parts.append("=== Spreadsheet Fields ===")
        for k, v in merged_row.items():
            if is_nonempty(v):
                combined_parts.append(f"{k}: {v}")

        if web_block["text_len"] > 0:
            combined_parts.append("=== Website Text ===")
            combined_parts.append(web_block["text"])

        if cv_block["text_len"] > 0:
            combined_parts.append("=== CV Text ===")
            combined_parts.append(cv_block["text"])

        # Add OpenAlex works titles/abstracts (lightweight, good for embedding)
        if oa_author_block.get("status") == "matched" and oa_works_block.get("works_count", 0) > 0:
            combined_parts.append("=== Publications (OpenAlex, last 10y) ===")
            for w in oa_works_block["works"][:50]:  # cap to avoid huge text blobs
                t = w.get("title") or ""
                y = w.get("year") or ""
                ab = (w.get("abstract") or "")[:800]
                combined_parts.append(f"- {t} ({y})")
                if ab:
                    combined_parts.append(ab)

        combined_text = "\n".join(combined_parts).strip()

        profile = {
            "key": key,
            "name": name_val,
            "slug": slug,
            "sources": {
                "sheet_1": {"path": str(SHEET_1_PATH), "id_col": id1, "name_col": name1},
                "sheet_2": {"path": str(SHEET_2_PATH), "id_col": id2, "name_col": name2},
                "cv_json_dir": str(CV_JSON_DIR),
                "web_json_dir": str(WEB_JSON_DIR),
                "openalex_authors_dir": str(OPENALEX_AUTHORS_DIR),
                "openalex_works_dir": str(OPENALEX_WORKS_DIR),
            },
            "spreadsheet_fields": merged_row,
            "website": web_block,
            "cv": cv_block,
            "openalex_author": oa_author_block,
            "openalex_works": oa_works_block,
            "combined_text_len": len(combined_text),
            "combined_text": combined_text,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        out_path = OUT_DIR / f"{slug}.json"
        out_path.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
        stats["profiles_written"] += 1

    # Summary
    lines = []
    lines.append("PROFILE BUILD SUMMARY")
    lines.append(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append(f"Sheet 1: {SHEET_1_PATH} (rows={len(df1)})")
    lines.append(f"Sheet 2: {SHEET_2_PATH} (rows={len(df2)})")
    lines.append(f"CV dir:   {CV_JSON_DIR} (files={len(list(CV_JSON_DIR.glob('*.json')))} if exists)")
    lines.append(f"Web dir:  {WEB_JSON_DIR} (files={len(list(WEB_JSON_DIR.glob('*.json')))} if exists)")
    lines.append(f"OA authors dir: {OPENALEX_AUTHORS_DIR} (files={len(list(OPENALEX_AUTHORS_DIR.glob('*.json')))} if exists)")
    lines.append(f"OA works dir:   {OPENALEX_WORKS_DIR} (files={len(list(OPENALEX_WORKS_DIR.glob('*.json')))} if exists)")
    lines.append("")
    lines.append(f"Profiles written: {stats['profiles_written']}")
    lines.append("")
    lines.append("CV matching:")
    lines.append(f"  matched: {stats['matched_cv']}")
    lines.append(f"  ok:      {stats.get('cv_ok', 0)}")
    lines.append(f"  empty:   {stats.get('cv_empty', 0)}")
    lines.append(f"  error:   {stats.get('cv_error', 0)}")
    lines.append(f"  missing: {stats.get('cv_missing', 0)}")
    lines.append("")
    lines.append("Website matching:")
    lines.append(f"  matched: {stats['matched_web']}")
    lines.append(f"  ok:      {stats.get('web_ok', 0)}")
    lines.append(f"  empty:   {stats.get('web_empty', 0)}")
    lines.append(f"  error:   {stats.get('web_error', 0)}")
    lines.append(f"  missing: {stats.get('web_missing', 0)}")
    lines.append("")
    lines.append("OpenAlex matching:")
    lines.append(f"  matched authors: {stats['matched_openalex_author']}")
    lines.append(f"  no_match:        {stats['openalex_no_match']}")
    lines.append(f"  error:           {stats['openalex_error']}")
    lines.append(f"  missing:         {stats['openalex_missing']}")
    lines.append("")

    SUMMARY_TXT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"Done. Profiles in: {OUT_DIR}/")
    print(f"Summary saved to: {SUMMARY_TXT}")


if __name__ == "__main__":
    main()