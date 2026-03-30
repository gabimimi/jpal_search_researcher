from __future__ import annotations

import json
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime


WEB_DIR = Path("output/web")
OUT_TXT = Path("output/web_scrape_summary.txt")


def main() -> None:
    if not WEB_DIR.exists():
        raise FileNotFoundError(f"Can't find {WEB_DIR}. Run your scraper first.")

    files = list(WEB_DIR.glob("*.json"))
    total = len(files)

    counts = Counter()
    http_counts = Counter()
    examples = defaultdict(list)

    for p in files:
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            counts["unreadable_json"] += 1
            if len(examples["unreadable_json"]) < 5:
                examples["unreadable_json"].append((p.name, str(e)))
            continue

        status = obj.get("status")
        text = obj.get("text") or ""
        text_len = len(str(text).strip())
        err = obj.get("error")
        http_status = obj.get("http_status")

        # If older JSONs don't have "status", infer it
        if not status:
            if err:
                status = "error"
            elif text_len == 0:
                status = "empty"
            else:
                status = "ok"

        # Normalize into buckets
        if status == "ok":
            counts["ok_nonempty"] += 1
            bucket = "ok_nonempty"
        elif status == "empty":
            counts["empty_0_chars"] += 1
            bucket = "empty_0_chars"
        elif status == "error":
            counts["error"] += 1
            bucket = "error"
        elif status == "blocked":
            counts["blocked_403"] += 1
            bucket = "blocked_403"
        elif status == "skipped":
            counts["skipped"] += 1
            bucket = "skipped"
        else:
            counts[f"status_{status}"] += 1
            bucket = f"status_{status}"

        if http_status is not None and http_status != "":
            http_counts[str(http_status)] += 1

        if len(examples[bucket]) < 5:
            examples[bucket].append((
                obj.get("name", ""),
                obj.get("url", ""),
                http_status,
                text_len,
                err
            ))

    good = counts.get("ok_nonempty", 0)
    zero = counts.get("empty_0_chars", 0)
    failed = counts.get("error", 0) + counts.get("unreadable_json", 0)

    lines = []
    lines.append("Web scrape status summary\n")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}\n")
    lines.append(f"Directory: {WEB_DIR.resolve()}\n")
    lines.append(f"Total JSON files: {total}\n\n")

    lines.append("Requested buckets:\n")
    lines.append(f"  Good (non-empty text): {good}\n")
    lines.append(f"  Empty (0 chars): {zero}\n")
    lines.append(f"  Failed (errors + unreadable): {failed}\n\n")

    lines.append("All counts:\n")
    for k, v in counts.most_common():
        lines.append(f"  - {k}: {v}\n")

    lines.append("\nHTTP status distribution (top 20):\n")
    for code, v in http_counts.most_common(20):
        lines.append(f"  - {code}: {v}\n")

    lines.append("\nExamples (up to 5 each bucket):\n")
    for bucket, exs in examples.items():
        lines.append(f"\n[{bucket}]\n")
        for ex in exs:
            if bucket == "unreadable_json":
                lines.append(f"  - file={ex[0]} error={ex[1]}\n")
            else:
                name, url, http_s, tl, err = ex
                lines.append(f"  - {name} | {url} | http={http_s} | text_len={tl} | err={err}\n")

    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    OUT_TXT.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote: {OUT_TXT}  (scanned {total} files)")


if __name__ == "__main__":
    main()
