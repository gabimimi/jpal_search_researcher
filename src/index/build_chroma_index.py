"""
Build a ChromaDB HNSW index from the compact profiles index.

Run once after export_compact_index.py to enable fast approximate
nearest-neighbour search in search.py.

Usage:
    python3 -m src.index.build_chroma_index
    python3 -m src.index.build_chroma_index --compact-index frontend/profiles_index.json
                                             --chroma-dir output/chroma_index
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

COMPACT_INDEX_DEFAULT = Path("frontend/profiles_index.json")
CHROMA_INDEX_DEFAULT  = Path("output/chroma_index")
COLLECTION            = "researchers"
_BATCH                = 500


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ChromaDB index from compact profiles index")
    parser.add_argument("--compact-index", default=str(COMPACT_INDEX_DEFAULT),
                        help="Path to profiles_index.json (from export_compact_index)")
    parser.add_argument("--chroma-dir", default=str(CHROMA_INDEX_DEFAULT),
                        help="Directory where the ChromaDB files will be written")
    args = parser.parse_args()

    compact_path = Path(args.compact_index)
    chroma_dir   = Path(args.chroma_dir)

    if not compact_path.exists():
        print(
            f"ERROR: {compact_path} not found. Run export_compact_index first.",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        import chromadb
    except ImportError:
        print("ERROR: chromadb not installed. Run: pip install chromadb", file=sys.stderr)
        sys.exit(1)

    print(f"Loading compact index from {compact_path}…", file=sys.stderr)
    rows: list = json.loads(compact_path.read_text(encoding="utf-8")).get("researchers") or []
    print(f"Loaded {len(rows)} researchers", file=sys.stderr)

    chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_dir))

    try:
        client.delete_collection(COLLECTION)
        print(f"Deleted existing '{COLLECTION}' collection", file=sys.stderr)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    ids_buf: list[str]        = []
    embs_buf: list[list]      = []
    metas_buf: list[dict]     = []

    def _flush() -> None:
        if ids_buf:
            collection.upsert(ids=ids_buf, embeddings=embs_buf, metadatas=metas_buf)
            ids_buf.clear(); embs_buf.clear(); metas_buf.clear()

    n_full = n_nar = 0
    for r in rows:
        slug = r.get("slug")
        emb  = r.get("embedding")
        if not slug or not emb:
            continue

        # Build metadata — ChromaDB requires str/int/float/bool values only.
        # Omit the large keyword-blob field (handled separately during keyword boost).
        base: dict = {
            "slug": slug,
            "name": r.get("name") or slug,
        }
        if r.get("institution"):
            base["institution"] = str(r["institution"])[:512]
        kf = r.get("key_fields") or {}
        for field, val in kf.items():
            if field == "Website & publications (keyword index)":
                continue  # too large; not needed in chroma metadata
            sval = str(val).strip()
            if sval and sval not in ("nan", "None"):
                base[f"kf__{field}"] = sval[:512]

        # Full embedding
        meta_full = dict(base, emb_type="full")
        ids_buf.append(f"{slug}__full")
        embs_buf.append(emb)
        metas_buf.append(meta_full)
        n_full += 1

        # Narrative embedding (website + profile + cv), if available
        nar = r.get("embedding_narrative")
        if nar and len(nar) == len(emb):
            meta_nar = dict(base, emb_type="nar")
            ids_buf.append(f"{slug}__nar")
            embs_buf.append(nar)
            metas_buf.append(meta_nar)
            n_nar += 1

        if len(ids_buf) >= _BATCH:
            _flush()
            print(f"  Upserted {n_full + n_nar} entries so far…", file=sys.stderr)

    _flush()
    total = collection.count()
    print(
        f"Done — {total} entries in ChromaDB ({n_full} full, {n_nar} narrative).",
        file=sys.stderr,
    )
    print(f"Index saved to: {chroma_dir.resolve()}", file=sys.stderr)


if __name__ == "__main__":
    main()
