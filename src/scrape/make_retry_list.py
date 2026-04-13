from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

WEB_DIR = Path("output/web")
OUT_REPORT = Path("output/web_scrape_report.csv")
OUT_RETRY = Path("output/web_retry_list.csv")

def main() -> None:
    rows = []
    for p in WEB_DIR.glob("*.json"):
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        rows.append({
            "file": p.name,
            "name": obj.get("name", ""),
            "url": obj.get("url", ""),
            "status": obj.get("status", "unknown"),
            "http_status": obj.get("http_status", ""),
            "text_len": len((obj.get("text") or "").strip()),
            "error": obj.get("error", ""),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        print("No JSON files found in output/web/")
        return

    df.to_csv(OUT_REPORT, index=False)

    # retry anything that is error or empty
    retry = df[(df["status"].isin(["error", "empty"])) | (df["text_len"] == 0)].copy()
    retry = retry.sort_values(["status", "http_status", "name"])
    retry.to_csv(OUT_RETRY, index=False)

    print("Saved:")
    print(" -", OUT_REPORT, f"(rows={len(df)})")
    print(" -", OUT_RETRY, f"(rows={len(retry)})")
    print("\nCounts:")
    print(df["status"].value_counts(dropna=False))

if __name__ == "__main__":
    main()
