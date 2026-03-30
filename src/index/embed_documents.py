"""
Embed documents using the OpenAI embeddings API.

Reads:  output/index/documents.jsonl
Writes: output/index/documents_with_embeddings.jsonl
        output/index/embed_checkpoint.txt  (last completed doc_id for resume)

Run:
    python3 -m src.index.embed_documents [--model text-embedding-3-small] [--batch 100]

Environment:
    OPENAI_API_KEY  (required)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List

DOCUMENTS_JSONL = Path("output/index/documents.jsonl")
OUT_JSONL = Path("output/index/documents_with_embeddings.jsonl")
CHECKPOINT_FILE = Path("output/index/embed_checkpoint.txt")
META_FILE = Path("output/index/embed_meta.json")

DEFAULT_MODEL = "text-embedding-3-small"
DEFAULT_BATCH = 100
MAX_CHARS_PER_TEXT = 8000   # truncate to avoid token limit overflows (~2000 tokens)


def _truncate(text: str, max_chars: int = MAX_CHARS_PER_TEXT) -> str:
    return text[:max_chars] if len(text) > max_chars else text


def _load_checkpoint() -> set:
    if CHECKPOINT_FILE.exists():
        return set(CHECKPOINT_FILE.read_text(encoding="utf-8").splitlines())
    return set()


def _save_checkpoint(done_ids: set) -> None:
    CHECKPOINT_FILE.write_text("\n".join(sorted(done_ids)), encoding="utf-8")


def _embed_batch(client, texts: List[str], model: str) -> List[List[float]]:
    """Call OpenAI embeddings API with retry on rate limit."""
    for attempt in range(6):
        try:
            resp = client.embeddings.create(model=model, input=texts)
            return [item.embedding for item in resp.data]
        except Exception as e:
            err_str = str(e)
            if "rate" in err_str.lower() or "429" in err_str:
                wait = 2 ** attempt
                print(f"  Rate limited, waiting {wait}s...", file=sys.stderr)
                time.sleep(wait)
            elif attempt >= 2:
                raise
            else:
                print(f"  Retry {attempt+1}: {e}", file=sys.stderr)
                time.sleep(2)
    raise RuntimeError("Embedding failed after retries")


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed documents with OpenAI API")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Embedding model name")
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH, help="Batch size")
    parser.add_argument("--resume", action="store_true", default=True, help="Resume from checkpoint (default: on)")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: Set OPENAI_API_KEY.", file=sys.stderr)
        sys.exit(1)

    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: openai package not installed. Run: pip install openai", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    if not DOCUMENTS_JSONL.exists():
        print(f"ERROR: {DOCUMENTS_JSONL} not found. Run build_documents first.", file=sys.stderr)
        sys.exit(1)

    # Load all docs
    docs = []
    with DOCUMENTS_JSONL.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))

    print(f"Loaded {len(docs)} documents from {DOCUMENTS_JSONL}")

    # Resume checkpoint
    done_ids: set = set()
    if args.resume and not OUT_JSONL.exists():
        args.resume = False  # can't resume if output doesn't exist
    if args.resume:
        done_ids = _load_checkpoint()
        print(f"Resuming: {len(done_ids)} already embedded.")

    # Open output file
    mode = "a" if args.resume and OUT_JSONL.exists() else "w"
    fout = OUT_JSONL.open(mode, encoding="utf-8")

    pending = [d for d in docs if d["doc_id"] not in done_ids]
    print(f"To embed: {len(pending)} documents (model={args.model}, batch={args.batch})")

    embedded = 0
    t0 = time.time()

    for i in range(0, len(pending), args.batch):
        batch = pending[i : i + args.batch]
        texts = [_truncate(d["text"]) for d in batch]

        try:
            embeddings = _embed_batch(client, texts, args.model)
        except Exception as e:
            print(f"\nERROR embedding batch {i//args.batch}: {e}", file=sys.stderr)
            print("Saving checkpoint and exiting. Re-run to resume.", file=sys.stderr)
            _save_checkpoint(done_ids)
            fout.close()
            sys.exit(1)

        for doc, emb in zip(batch, embeddings):
            out = dict(doc)
            out["embedding"] = emb
            out["embedding_model"] = args.model
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            done_ids.add(doc["doc_id"])

        embedded += len(batch)
        elapsed = time.time() - t0
        rate = embedded / elapsed if elapsed > 0 else 0
        remaining = len(pending) - embedded
        eta = remaining / rate if rate > 0 else 0
        print(f"  {embedded}/{len(pending)} embedded  ({rate:.1f}/s  ETA {eta:.0f}s)", end="\r")

        # Save checkpoint every 500 docs
        if embedded % 500 < args.batch:
            _save_checkpoint(done_ids)

    fout.close()
    _save_checkpoint(done_ids)

    # Write meta
    META_FILE.write_text(json.dumps({
        "model": args.model,
        "total_docs": len(docs),
        "embedded": len(done_ids),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, indent=2), encoding="utf-8")

    print(f"\nDone. {embedded} new embeddings written to {OUT_JSONL}")
    print(f"Total embedded: {len(done_ids)}/{len(docs)}")


if __name__ == "__main__":
    main()
