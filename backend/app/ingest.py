# ---------------------------------------------------------------------------
# CLI entrypoint:  python -m app.ingest
# Walks backend/data/pdfs/<track>/*.pdf, processes one PDF at a time
# (load -> chunk -> embed -> upsert), and rebuilds BM25 once at the end.
#
# Two things fixed vs. the original version:
#   1. No `all_chunks = []` accumulator. Each PDF is embedded and upserted
#      before the next one is even loaded, so memory use is bounded by one
#      document, not the whole corpus.
#   2. A hash manifest (manifest.json) skips PDFs that haven't changed since
#      last run, and re-indexes (delete + reprocess) only PDFs whose content
#      hash changed — so adding one new circular no longer re-embeds
#      thousands of existing chunks.
# ---------------------------------------------------------------------------
import hashlib
import json
from pathlib import Path

from . import chunker, config, indexer, pdf_loader


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 16), b""):
            h.update(block)
    return h.hexdigest()


def _load_manifest() -> dict:
    if config.MANIFEST_PATH.exists():
        return json.loads(config.MANIFEST_PATH.read_text(encoding="utf-8"))
    return {}


def _save_manifest(manifest: dict) -> None:
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    config.MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def run() -> None:
    config.PDF_DIR.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest()
    changed_any = False

    for track in config.TRACKS:
        track_dir = config.PDF_DIR / track
        if not track_dir.exists():
            track_dir.mkdir(parents=True, exist_ok=True)
            print(f"[skip] {track}: no folder yet (created empty {track_dir})")
            continue

        # rglob (not glob) — PDFs are often organized in year subfolders
        # (circulars/ANNEE 2018/, ANNEE 2019/, ...). A non-recursive glob
        # would silently find zero files in those subfolders, which then
        # makes every PDF in them look "deleted" in the stale-file cleanup
        # below and wipes them from the index — not what we want.
        pdfs = sorted(track_dir.rglob("*.pdf"))
        print(f"\n=== Track: {track} ({len(pdfs)} PDFs) ===")

        for pdf in pdfs:
            key = str(pdf.relative_to(config.PDF_DIR))
            file_hash = _file_hash(pdf)
            prev = manifest.get(key)

            if prev and prev["hash"] == file_hash:
                print(f"  = {pdf.name} unchanged, skipping")
                continue

            changed_any = True
            if prev:
                print(f"  ~ {pdf.name} changed, re-indexing")
                indexer.delete_by_source(key)
                indexer.remove_from_corpus(key)
            else:
                print(f"  + {pdf.name} new")

            try:
                pages = pdf_loader.load_pdf(str(pdf))
                chunks = chunker.chunk_document(pages, track, key)
                if not chunks:
                    print("    !! no text extracted, skipping")
                    continue
                indexer.upsert_chunks(chunks)       # embeds + writes, one PDF at a time
                indexer.append_to_corpus(chunks)
                manifest[key] = {"hash": file_hash, "track": track, "n_chunks": len(chunks)}
                _save_manifest(manifest)            # persist after each file — a crash
                                                     # mid-run doesn't lose prior progress
                print(f"    -> {len(chunks)} chunks indexed")
            except Exception as e:
                print(f"    !! failed: {e}")

    # PDFs that were deleted from disk since the last run: drop them too.
    existing_keys = {
        str(p.relative_to(config.PDF_DIR))
        for track in config.TRACKS
        for p in (config.PDF_DIR / track).rglob("*.pdf")
    }
    stale = [k for k in manifest if k not in existing_keys]
    for key in stale:
        print(f"  - {key} removed from disk, dropping from index")
        indexer.delete_by_source(key)
        indexer.remove_from_corpus(key)
        del manifest[key]
    if stale:
        _save_manifest(manifest)
        changed_any = True

    if changed_any:
        print("\nRebuilding BM25 sparse index from corpus.jsonl…")
        indexer.rebuild_bm25_from_corpus()
    else:
        print("\nNothing changed — BM25 index left untouched.")

    print("Done.")


if __name__ == "__main__":
    run()
