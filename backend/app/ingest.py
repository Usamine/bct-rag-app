# ---------------------------------------------------------------------------
# CLI entrypoint:  python -m app.ingest
# Walks backend/data/pdfs/<track>/*.pdf, chunks per track, builds indexes.
# ---------------------------------------------------------------------------
from pathlib import Path
from . import config, pdf_loader, chunker, indexer


def run() -> None:
    config.PDF_DIR.mkdir(parents=True, exist_ok=True)
    all_chunks = []

    for track in config.TRACKS:
        track_dir = config.PDF_DIR / track
        if not track_dir.exists():
            track_dir.mkdir(parents=True, exist_ok=True)
            print(f"[skip] {track}: no folder yet (created empty {track_dir})")
            continue

        pdfs = sorted(track_dir.glob("*.pdf"))
        print(f"\n=== Track: {track} ({len(pdfs)} PDFs) ===")
        for pdf in pdfs:
            print(f"  · {pdf.name}")
            try:
                pages = pdf_loader.load_pdf(str(pdf))
                chunks = chunker.chunk_document(pages, track, str(pdf.name))
                print(f"    → {len(chunks)} chunks")
                all_chunks.extend(chunks)
            except Exception as e:
                print(f"    !! failed: {e}")

    print(f"\nTotal chunks: {len(all_chunks)}")
    indexer.build_indexes(all_chunks)
    print("Done.")


if __name__ == "__main__":
    run()
