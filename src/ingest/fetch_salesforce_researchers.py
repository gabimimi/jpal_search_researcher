"""
Pull researcher rows from Salesforce Tabular reports and write pipeline CSVs.

Fetches up to three reports:
  1. Affiliates      → merged into output/researchers_clean.csv
  2. Invited         → merged into output/researchers_clean.csv
  3. Initiatives     → output/researchers_extra.csv  (optional, replaces extra_sheet.py)

Environment (see salesforce_auth.py for OAuth fields):
  SALESFORCE_REPORT_AFFILIATES_ID    15- or 18-char report Id
  SALESFORCE_REPORT_INVITED_ID       15- or 18-char report Id

Optional:
  SALESFORCE_INITIATIVE_REPORT_ID    15- or 18-char report Id (initiative/office/end-date report)
  SALESFORCE_API_VERSION             default 59.0
  SALESFORCE_COLUMN_MAP              path to JSON object { "SF column label": "Pipeline column", ... }
                                     default: data/salesforce_column_map.json if that file exists

Run:
  python3 -m src.ingest.fetch_salesforce_researchers

Load .env from project root if present (same helper pattern as search.py).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd

from typing import Iterable, List

from src.ingest.clean_researchers import prepare_researchers_dataframe
from src.ingest.salesforce_auth import get_access_token
from src.ingest.salesforce_report import fetch_report_json, report_json_to_dataframe

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Common Salesforce report labels → names expected by clean_researchers.KEEP_COLS
_DEFAULT_SF_ALIASES: dict[str, str] = {
    "contact: full name": "Full Name",
    "contact name": "Full Name",
    "full name": "Full Name",
    "personal web site": "Personal Website",
    "personal website": "Personal Website",
    "web bio": "Web Bio",
    "web bio link": "Web Bio Link",
    "research interests (open text)": "Research Interests (open text)",
    "research interests": "Research Interests (open text)",
    "regional office affiliation": "Regional Office Affiliation",
    "regional interest": "Regional interest",
    "related initiative(s)": "Related Initiative(s)",
    "related initiatives": "Related Initiative(s)",
    "initiatives": "Initiatives",
    "publication notes": "Publication Notes",
    "sectors": "Sectors",
    "sector/initiative interest": "Sector/Initiative interest",
    "specific country interest": "Specific Country Interest",
    "cv": "CV",
    "cv link": "CV",
}


def _load_dotenv_project_root() -> None:
    path = _PROJECT_ROOT / ".env"
    if not path.is_file():
        return
    try:
        for raw in path.read_text(encoding="utf-8-sig").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            if key.lower().startswith("export "):
                key = key[7:].strip()
            val = val.strip().strip('"').strip("'")
            # Fill from .env when unset or empty in the shell (empty exports would block .env otherwise).
            if key and val and (key not in os.environ or not str(os.environ.get(key, "")).strip()):
                os.environ[key] = val
    except OSError:
        pass


def _apply_column_mapping(df: pd.DataFrame) -> pd.DataFrame:
    """Strip headers, apply JSON map file, then default SF aliases (lowercase keys)."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    map_path = os.environ.get("SALESFORCE_COLUMN_MAP", "").strip()
    if not map_path:
        default = _PROJECT_ROOT / "data" / "salesforce_column_map.json"
        if default.is_file():
            map_path = str(default)
    if map_path:
        p = Path(map_path).expanduser()
        if p.is_file():
            raw = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                df = df.rename(columns={str(k): str(v) for k, v in raw.items()})

    rename: dict[str, str] = {}
    for c in df.columns:
        low = c.lower()
        if low in _DEFAULT_SF_ALIASES and _DEFAULT_SF_ALIASES[low] not in df.columns:
            # avoid renaming if target already exists from a prior rename
            if c != _DEFAULT_SF_ALIASES[low]:
                rename[c] = _DEFAULT_SF_ALIASES[low]
    if rename:
        df = df.rename(columns=rename)
    return df


def _merge_duplicate_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """If concat produced duplicate labels (e.g. two 'Full Name'), coalesce row-wise."""
    if not df.columns.duplicated().any():
        return df
    out = pd.DataFrame(index=df.index)
    for col in pd.Index(df.columns).unique():
        block = df.loc[:, df.columns == col]
        if block.shape[1] == 1:
            out[col] = block.iloc[:, 0]
            continue
        acc = block.iloc[:, 0].astype(str).replace({"nan": "", "None": ""})
        for j in range(1, block.shape[1]):
            nxt = block.iloc[:, j].astype(str).replace({"nan": "", "None": ""})
            acc = acc.where(acc.str.strip().ne(""), nxt)
        out[col] = acc
    return out


_INITIATIVE_COL_ALIASES: dict[str, str] = {
    "contact: full name": "Full Name",
    "contact name": "Full Name",
    "full name": "Full Name",
    "nominee: full name": "Full Name",
    "initiative: initiative name": "Initiative: Initiative Name",
    "initiative name": "Initiative: Initiative Name",
    "office: office name": "Office: Office Name",
    "office name": "Office: Office Name",
    "end date": "End Date",
}


def _clean_tokens(values: Iterable[object]) -> List[str]:
    out: set[str] = set()
    for v in values:
        if v is None:
            continue
        if isinstance(v, float) and pd.isna(v):
            continue
        s = str(v).strip()
        if not s or s.lower() in {"nan", "none", "null"}:
            continue
        out.add(s)
    return sorted(out)


def _apply_initiative_column_mapping(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    rename: dict[str, str] = {}
    for c in df.columns:
        low = c.lower()
        if low in _INITIATIVE_COL_ALIASES and _INITIATIVE_COL_ALIASES[low] not in df.columns:
            if c != _INITIATIVE_COL_ALIASES[low]:
                rename[c] = _INITIATIVE_COL_ALIASES[low]
    if rename:
        df = df.rename(columns=rename)
    return df


def _process_initiative_report(
    instance: str, token_str: str, report_id: str, api_ver: str,
) -> None:
    """Fetch the initiative report and write output/researchers_extra.csv."""
    print(f"Fetching report (initiatives): {report_id} …")
    raw = fetch_report_json(instance, token_str, report_id, api_version=api_ver)
    df = report_json_to_dataframe(raw)
    df = _apply_initiative_column_mapping(df)
    print(f"  → {len(df)} rows, columns: {list(df.columns)}")

    required = {"Full Name", "Initiative: Initiative Name", "End Date"}
    optional = {"Office: Office Name"}
    missing_req = required - set(df.columns)
    missing_opt = optional - set(df.columns)
    if missing_req:
        print(
            f"Warning: initiative report missing expected columns: {sorted(missing_req)}. "
            f"Available: {list(df.columns)}. Writing what we have.",
            file=sys.stderr,
        )
    if missing_opt:
        print(f"  (optional columns not present: {sorted(missing_opt)})")

    if "Full Name" not in df.columns:
        print("ERROR: Initiative report has no 'Full Name' column after mapping.", file=sys.stderr)
        return

    df["Full Name"] = df["Full Name"].astype(str).str.strip()
    df = df[df["Full Name"].ne("") & df["Full Name"].str.lower().ne("nan")].copy()

    if "End Date" in df.columns:
        df["End Date"] = pd.to_datetime(df["End Date"], errors="coerce")

    agg_cols: dict[str, tuple[str, ...]] = {}
    if "Initiative: Initiative Name" in df.columns:
        agg_cols["initiatives"] = ("Initiative: Initiative Name", _clean_tokens)
    if "Office: Office Name" in df.columns:
        agg_cols["offices"] = ("Office: Office Name", _clean_tokens)
    if "End Date" in df.columns:
        agg_cols["latest_end_date"] = ("End Date", "max")

    if agg_cols:
        agg = df.groupby("Full Name", as_index=False).agg(**agg_cols)
    else:
        agg = df[["Full Name"]].drop_duplicates()

    out_path = Path("output") / "researchers_extra.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(out_path, index=False)
    print(f"Wrote: {out_path} (rows={len(agg)})")


def main() -> None:
    _load_dotenv_project_root()

    aff_id = os.environ.get("SALESFORCE_REPORT_AFFILIATES_ID", "").strip()
    inv_id = os.environ.get("SALESFORCE_REPORT_INVITED_ID", "").strip()
    init_id = os.environ.get("SALESFORCE_INITIATIVE_REPORT_ID", "").strip()
    if not aff_id or not inv_id:
        which = []
        if not aff_id:
            which.append("SALESFORCE_REPORT_AFFILIATES_ID")
        if not inv_id:
            which.append("SALESFORCE_REPORT_INVITED_ID")
        print(
            "ERROR: Missing: " + ", ".join(which) + ".\n"
            "  Set them in .env or export in your shell.\n"
            "  Report Id: open each report in Salesforce — the URL contains /00O... (15 or 18 characters).",
            file=sys.stderr,
        )
        sys.exit(1)

    api_ver = os.environ.get("SALESFORCE_API_VERSION", "59.0").strip()

    print("Authenticating…")
    access_token, instance = get_access_token()

    # --- Affiliates + Invited → researchers_clean.csv ---
    frames: list[pd.DataFrame] = []
    for label, rid in (("affiliates", aff_id), ("invited", inv_id)):
        print(f"Fetching report ({label}): {rid} …")
        raw = fetch_report_json(instance, access_token, rid, api_version=api_ver)
        df = report_json_to_dataframe(raw)
        df = _apply_column_mapping(df)
        df["Researcher Type"] = "J-PAL affiliate" if label == "affiliates" else "Invited researcher"
        print(f"  → {len(df)} rows, columns: {list(df.columns)[:8]}{'…' if len(df.columns) > 8 else ''}")
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)
    merged = _merge_duplicate_column_names(merged)
    print(f"Merged raw rows: {len(merged)}")
    if "Full Name" not in merged.columns:
        print(
            "ERROR: After column mapping there is no 'Full Name' column. "
            "Add data/salesforce_column_map.json (see data/salesforce_column_map.example.json).",
            file=sys.stderr,
        )
        sys.exit(1)

    cleaned = prepare_researchers_dataframe(merged)

    out_dir = Path("output")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "researchers_clean.csv"
    cleaned.to_csv(out_path, index=False)
    print(f"Wrote: {out_path} (rows={len(cleaned)})")

    # --- Initiative report → researchers_extra.csv ---
    if init_id:
        _process_initiative_report(instance, access_token, init_id, api_ver)
    else:
        print("Skipping initiative report (SALESFORCE_INITIATIVE_REPORT_ID not set).")
        print("  Run  python3 -m src.ingest.extra_sheet  to use local Excel instead.")

    print("\nDone. Next: python3 -m src.profile.build_profiles → make all")


if __name__ == "__main__":
    main()
