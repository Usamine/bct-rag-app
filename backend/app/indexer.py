# ---------------------------------------------------------------------------
# Build & persist the two indexes:
#   * ChromaDB collection — dense vectors from bge-m3
#   * BM25 (pickled)      — sparse keyword matching
#
# Two changes from the original version:
#   1. `upsert` instead of `add` + wipe-everything. Combined with the
#      deterministic ids from chunker.py, re-running ingest on an unchanged
#      PDF is now a cheap no-op instead of re-embedding the whole corpus.
#   2. A flat corpus.jsonl (text + metadata, no vectors) is kept alongside
#      Chroma. rank_bm25 has no incremental-update API — it must always be
#      rebuilt from scratch — but rebuilding from this small file is nearly
#      free, versus re-running the embedding model just to get BM25 back.
#
# IDs stay aligned between the two stores so retriever.py can merge results.
# ---------------------------------------------------------------------------
import json
import pickle
import re
from typing import Dict, Iterable, List

import chromadb
from rank_bm25 import BM25Okapi

from . import config, ollama_client

# --- tokenization (BM25) ---------------------------------------------------
# Simple unicode-aware tokenizer that keeps Arabic + Latin words.
_TOKEN_RE = re.compile(r"[\w\u0600-\u06FF]+", re.UNICODE)


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


# --- Chroma helpers --------------------------------------------------------
_client = None


def _get_collection():
    global _client
    if _client is None:
        # anonymized_telemetry=False silences Chroma's background telemetry
        # call, which throws a harmless but noisy
        # "capture() takes 1 positional argument but 3 were given" on some
        # chromadb versions. It's cosmetic (no data is lost either way) but
        # there's no reason to leave the warning spamming your ingest logs.
        _client = chromadb.PersistentClient(
            path=str(config.CHROMA_PATH),
            settings=chromadb.config.Settings(anonymized_telemetry=False),
        )
    # We supply our own embeddings, so no built-in embedding function needed.
    return _client.get_or_create_collection(name="bct", metadata={"hnsw:space": "cosine"})


def upsert_chunks(chunks: List[Dict], batch_size: int = None) -> None:
    """
    Embed + upsert one document's worth of chunks, in small batches.
    Called once per PDF from ingest.py — never with the whole corpus in
    memory at once (that was the original "memory bomb").

    Upsert means: an id that already exists gets its vector/text/metadata
    replaced in place; a new id gets inserted. Nothing is duplicated, and
    nothing else in the collection is touched.
    """
    if not chunks:
        return
    batch_size = batch_size or config.EMBED_BATCH
    coll = _get_collection()
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        # Embed the context-injected text, but store the original `text` as
        # the document body — that's what should come back at query time.
        vectors = ollama_client.embed_batch([c["embedding_text"] for c in batch])
        coll.upsert(
            ids=[c["id"] for c in batch],
            embeddings=vectors,
            documents=[c["text"] for c in batch],
            metadatas=[c["metadata"] for c in batch],
        )


def delete_by_source(source: str) -> None:
    """
    Drop every chunk belonging to one source file. Called before
    re-indexing a changed PDF (so stale chunks from the old version don't
    linger) and when a PDF is removed from disk entirely.
    """
    coll = _get_collection()
    try:
        coll.delete(where={"source": source})
    except Exception as e:
        print(f"  [warn] chroma delete_by_source({source}) failed: {e}")


# --- corpus.jsonl: the small text-only store BM25 rebuilds from ------------
def append_to_corpus(chunks: List[Dict]) -> None:
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.CORPUS_PATH, "a", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(
                {"id": c["id"], "text": c["text"], "metadata": c["metadata"]},
                ensure_ascii=False,
            ) + "\n")


def remove_from_corpus(source: str) -> None:
    """
    Rewrite corpus.jsonl excluding one source's rows.
    O(corpus size) — fine at BCT-circular scale (thousands of chunks).
    If this ever becomes the bottleneck, swap the flat file for sqlite
    and turn this into a single indexed DELETE.
    """
    if not config.CORPUS_PATH.exists():
        return
    kept = [row for row in _iter_corpus() if row["metadata"].get("source") != source]
    with open(config.CORPUS_PATH, "w", encoding="utf-8") as f:
        for row in kept:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _iter_corpus() -> Iterable[Dict]:
    if not config.CORPUS_PATH.exists():
        return
    with open(config.CORPUS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def rebuild_bm25_from_corpus() -> None:
    """
    Rebuild the sparse index from corpus.jsonl only — never by re-running
    the embedding model. Call once per ingest run, after all changed PDFs
    have been upserted (rank_bm25 has no incremental API, so this is always
    a full rebuild, just a cheap one).
    """
    ids, docs, metas, tokenized = [], [], [], []
    for row in _iter_corpus():
        ids.append(row["id"])
        docs.append(row["text"])
        metas.append(row["metadata"])
        tokenized.append(tokenize(row["text"]))

    if not tokenized:
        print("Corpus is empty — nothing to build BM25 from.")
        return

    bm25 = BM25Okapi(tokenized)
    payload = {"bm25": bm25, "ids": ids, "docs": docs, "metas": metas}
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.BM25_PATH, "wb") as f:
        pickle.dump(payload, f)
    print(f"BM25 saved → {config.BM25_PATH} ({len(ids)} chunks)")