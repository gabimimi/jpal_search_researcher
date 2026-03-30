from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

# Update these paths if needed
XLS_PATH = Path("data/Report-2026-02-19-11-34-43.xlsx")
OUT_PATH = Path("output/researchers_extra.csv")


def _clean_tokens(values: Iterable[object]) -> List[str]:
    """
    Turn a column group into a sorted unique list of clean strings.
    Handles NaN/None/floats safely.
    """
    out: set[str] = set()
    for v in values:
        # drop NaN/None
        if v is None:
            continue
        if isinstance(v, float) and pd.isna(v):
            continue

        s = str(v).strip()
        if not s:
            continue

        low = s.lower()
        if low in {"nan", "none", "null"}:
            continue

        out.add(s)

    return sorted(out)


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not XLS_PATH.exists():
        raise FileNotFoundError(f"Missing {XLS_PATH} (expected in project root)")

    # read first sheet by default; if yours is explicitly "Report", keep it
    df = pd.read_excel(XLS_PATH, sheet_name=0)

    required = {"Full Name", "Initiative: Initiative Name", "Office: Office Name", "End Date"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {XLS_PATH.name}: {sorted(missing)}")

    # normalize name; drop rows without a usable name
    df["Full Name"] = df["Full Name"].astype(str).str.strip()
    df = df[df["Full Name"].ne("") & df["Full Name"].str.lower().ne("nan")].copy()

    # parse End Date safely
    df["End Date"] = pd.to_datetime(df["End Date"], errors="coerce")

    agg = (
        df.groupby("Full Name", as_index=False)
        .agg(
            initiatives=("Initiative: Initiative Name", _clean_tokens),
            offices=("Office: Office Name", _clean_tokens),
            latest_end_date=("End Date", "max"),
        )
    )

    agg.to_csv(OUT_PATH, index=False)
    print(f"Wrote: {OUT_PATH} rows={len(agg)}")


if __name__ == "__main__":
    main()
