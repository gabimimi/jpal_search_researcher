# save as: src/cv/summarize_cv_run.py
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

OUT_DIR = Path("output/cv")
SUMMARY_PATH = Path("output/cv_run_summary.txt")

def main() -> None:
    if not OUT_DIR.exists():
        raise FileNotFoundError(f"Missing directory: {OUT_DIR}")

    files = sorted(OUT_DIR.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No JSON files found in {OUT_DIR}")

    status_counts = Counter()
    filetype_counts = Counter()

    empty_text = 0
    zero_char = 0
    loading_cases = 0

    errors = []          # (name, cv_url, error)
    loading_examples = []  # (name, cv_url, text)

    total = 0
    for p in files:
        total += 1
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            status_counts["bad_json"] += 1
            errors.append((p.name, "", f"bad json: {e}"))
            continue

        status = str(rec.get("status", "unknown"))
        ftype = str(rec.get("filetype", "unknown"))

        status_counts[status] += 1
        filetype_counts[ftype] += 1

        text = rec.get("text") or ""
        text_stripped = str(text).strip()

        if text_stripped == "":
            empty_text += 1
            zero_char += 1

        # catch the Google Drive "Loading…" / "Loading..." / "Loading…"
        if text_stripped.lower().startswith("loading"):
            loading_cases += 1
            if len(loading_examples) < 20:
                loading_examples.append((rec.get("name", p.stem), rec.get("cv_url", ""), text_stripped[:200]))

        if status == "error":
            if len(errors) < 50:
                errors.append((rec.get("name", p.stem), rec.get("cv_url", ""), rec.get("error", "")))

    ok = status_counts.get("ok", 0)
    empty = status_counts.get("empty", 0)
    err = status_counts.get("error", 0)

    def pct(x: int) -> str:
        return f"{(100.0 * x / total):.1f}%" if total else "0.0%"

    lines = []
    lines.append("CV download/extract run summary")
    lines.append(f"Directory scanned: {OUT_DIR}")
    lines.append(f"Total JSON records: {total}")
    lines.append("")
    lines.append("Status counts:")
    lines.append(f"  ok:    {ok} ({pct(ok)})")
    lines.append(f"  empty: {empty} ({pct(empty)})")
    lines.append(f"  error: {err} ({pct(err)})")
    if status_counts.get("bad_json", 0):
        lines.append(f"  bad_json: {status_counts['bad_json']} ({pct(status_counts['bad_json'])})")

    lines.append("")
    lines.append("Text quality flags:")
    lines.append(f"  text_len == 0: {zero_char} ({pct(zero_char)})")
    lines.append(f"  'Loading…' pages: {loading_cases} ({pct(loading_cases)})")

    lines.append("")
    lines.append("Filetypes seen:")
    for k, v in filetype_counts.most_common():
        lines.append(f"  {k}: {v}")

    if loading_examples:
        lines.append("")
        lines.append("Examples of 'Loading…' cases (up to 20):")
        for name, url, snippet in loading_examples:
            lines.append(f"  - {name} | {url} | {snippet}")

    if errors:
        lines.append("")
        lines.append("Errors (up to 50):")
        for name, url, e in errors:
            lines.append(f"  - {name} | {url} | {e}")

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote: {SUMMARY_PATH}")

if __name__ == "__main__":
    main()