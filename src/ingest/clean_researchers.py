# src/ingest/clean_researchers.py
# Usage:
#   python src/ingest/clean_researchers.py data/Report-2026-02-19-11-25-38.xlsx
#
# Output:
#   output/researchers_clean.csv

from __future__ import annotations
import sys
import re
from pathlib import Path
import pandas as pd


KEEP_COLS = [
    "Full Name",
    "Researcher Type",
    "Personal Website",
    "CV",
    "Web Bio",
    "Web Bio Link",
    "Research Interests (open text)",
    "Regional Office Affiliation",
    "Regional interest",
    "Related Initiative(s)",
    "Initiatives",
    "Publication Notes",
    "Sectors",
    "Sector/Initiative interest",
    "Specific Country Interest",
]

# Omit from "missing columns" warning (only present after Salesforce fetch).
_OPTIONAL_KEEP_COLS = frozenset({"Researcher Type"})


def prepare_researchers_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter to known columns, normalize URLs/names, dedupe by Full Name.
    Shared by Excel ingest and Salesforce fetch.
    """
    existing = [c for c in KEEP_COLS if c in df.columns]
    missing = [c for c in KEEP_COLS if c not in df.columns]
    missing_warn = [c for c in missing if c not in _OPTIONAL_KEEP_COLS]
    if missing_warn:
        print(f"Warning: missing columns (skipped): {missing_warn}")

    df = df[existing].copy()

    if "Full Name" in df.columns:
        df["Full Name"] = df["Full Name"].apply(normalize_name)

    for url_col in ["Personal Website", "CV", "Web Bio Link"]:
        if url_col in df.columns:
            df[url_col] = df[url_col].apply(normalize_url)

    if "Full Name" in df.columns:
        df = df[df["Full Name"].astype(str).str.len() > 0].copy()

    useful_cols = [
        c for c in ["Personal Website", "CV", "Web Bio Link", "Web Bio", "Research Interests (open text)"]
        if c in df.columns
    ]

    def filled_count(row) -> int:
        cnt = 0
        for c in useful_cols:
            v = row.get(c, "")
            if isinstance(v, float) and pd.isna(v):
                continue
            if str(v).strip():
                cnt += 1
        return cnt

    df["_filled"] = df.apply(filled_count, axis=1)
    subset = ["Full Name"] if "Full Name" in df.columns else None
    if subset:
        df = df.sort_values(by=["_filled"], ascending=False)
        if "Researcher Type" in df.columns:
            merged_rows: list[pd.Series] = []
            for _, group in df.groupby("Full Name", sort=False):
                best = group.iloc[0].copy()
                types = sorted({str(x).strip() for x in group["Researcher Type"] if str(x).strip()})
                best["Researcher Type"] = "; ".join(types)
                merged_rows.append(best)
            df = pd.DataFrame(merged_rows)
        else:
            df = df.drop_duplicates(subset=subset, keep="first")
    df = df.drop(columns=["_filled"])
    return df.fillna("")


def normalize_url(x: object) -> str:
    """Normalize a URL-ish cell value into a clean URL string or empty string."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return ""

    # If it's multiple URLs or text, keep the first thing that looks like a URL
    # (common in messy exports).
    m = re.search(r"(https?://\S+|www\.\S+)", s)
    if m:
        s = m.group(1)

    # Remove trailing punctuation/parentheses/brackets that often stick to copied links
    s = s.strip().strip(").,;]}>\"'")

    # Add scheme if missing
    if s.startswith("www."):
        s = "https://" + s
    elif re.match(r"^[A-Za-z0-9.-]+\.[A-Za-z]{2,}(/.*)?$", s) and not s.startswith(("http://", "https://")):
        # looks like example.com/path
        s = "https://" + s

    return s


def normalize_name(x: object) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    s = str(x).strip()
    # Collapse internal whitespace
    s = re.sub(r"\s+", " ", s)
    return s


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python src/ingest/clean_researchers.py <path_to_excel>")
        sys.exit(1)

    in_path = Path(sys.argv[1]).expanduser().resolve()
    if not in_path.exists():
        raise FileNotFoundError(f"Excel file not found: {in_path}")

    out_dir = Path("output")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "researchers_clean.csv"

    df = pd.read_excel(in_path)
    df = prepare_researchers_dataframe(df)

    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}  (rows={len(df)})")


if __name__ == "__main__":
    main()
