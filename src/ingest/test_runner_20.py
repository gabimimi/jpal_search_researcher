# src/ingest/test_runner_20.py
# Usage:
#   python src/ingest/test_runner_20.py
#
# Assumes you already created:
#   output/researchers_clean.csv

from __future__ import annotations
from pathlib import Path
import pandas as pd


CSV_PATH = Path("output/researchers_clean.csv")


def is_nonempty(x: object) -> bool:
    if x is None:
        return False
    if isinstance(x, float) and pd.isna(x):
        return False
    s = str(x).strip()
    return s != "" and s.lower() not in {"nan", "none", "null"}


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Missing {CSV_PATH}. Run clean_researchers.py first to create it."
        )

    df = pd.read_csv(CSV_PATH)

    # Ensure expected columns exist (be forgiving)
    for col in ["Full Name", "Personal Website", "Web Bio Link", "CV"]:
        if col not in df.columns:
            df[col] = ""

    # Filter: must have either a personal website or web bio link
    has_site = df["Personal Website"].apply(is_nonempty)
    has_bio_link = df["Web Bio Link"].apply(is_nonempty)

    subset = df[has_site | has_bio_link].head(20).copy()

    print(f"Loaded rows: {len(df)}")
    print(f"Rows with website or bio link: {(has_site | has_bio_link).sum()}")
    print(f"Showing first {len(subset)}:\n")

    for i, row in subset.iterrows():
        name = str(row.get("Full Name", "")).strip()
        website_present = is_nonempty(row.get("Personal Website", ""))
        cv_present = is_nonempty(row.get("CV", ""))
        bio_link_present = is_nonempty(row.get("Web Bio Link", ""))

        print(
            f"- {name}\n"
            f"  website: {'YES' if website_present else 'NO'}\n"
            f"  web bio link: {'YES' if bio_link_present else 'NO'}\n"
            f"  cv: {'YES' if cv_present else 'NO'}\n"
        )


if __name__ == "__main__":
    main()
