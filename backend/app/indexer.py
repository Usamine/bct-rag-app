# ---------------------------------------------------------------------------
# Build & persist the two indexes:
#   * ChromaDB collection — dense vectors from bge-m3
#   * BM25 (pickled)      — sparse keyword matching
#
# We keep IDs aligned between the two so retriever.py can merge them.
# ---------------------------------------------------------------------------
import pickle
import re
from typing import List, Dict
import chromadb
from rank_bm25 import BM25Okapi
from . import config, ollama_client


# --- tokenization (BM25) ---------------------------------------------------
# Simple unicode-aware tokenizer that keeps Arabic + Latin words.
_TOKEN_RE = re.compile(r"[\w\u0600-\u06FF]+", re.UNICODE)


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


# --- Chroma helpers --------------------------------------------------------
def _get_collection():
    client = chromadb.PersistentClient(path=str(config.CHROMA_PATH))
    # We supply our own embeddings, so disable Chroma's embedding function.
    return client.get_or_create_collection(name="bct", metadata={"hnsw:space": "cosine"})


# --- public API ------------------------------------------------------------
def build_indexes(chunks: List[Dict]) -> None:
    """
    Embed all chunks and write both indexes to disk.
    `chunks` is the output of chunker.chunk_document for every PDF combined.
    """
    if not chunks:
        print("No chunks to index.")
        return

    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)

    # --- dense (Chroma) ----------------------------------------------------
    coll = _get_collection()
    # Wipe and rebuild for simplicity (incremental indexing can come later).
    try:
        coll.delete(where={"track": {"$ne": "__never__"}})
    except Exception:
        pass

    print(f"Embedding {len(chunks)} chunks with {config.EMBED_MODEL}…")
    BATCH = 16
    for i in range(0, len(chunks), BATCH):
        batch = chunks[i:i + BATCH]
        vectors = ollama_client.embed_batch([c["text"] for c in batch])
        coll.add(
            ids=[c["id"] for c in batch],
            embeddings=vectors,
            documents=[c["text"] for c in batch],
            metadatas=[c["metadata"] for c in batch],
        )
        print(f"  indexed {min(i + BATCH, len(chunks))}/{len(chunks)}")

    # --- sparse (BM25) -----------------------------------------------------
    tokenized = [tokenize(c["text"]) for c in chunks]
    bm25 = BM25Okapi(tokenized)
    payload = {
        "bm25": bm25,
        "ids": [c["id"] for c in chunks],
        "docs": [c["text"] for c in chunks],
        "metas": [c["metadata"] for c in chunks],
    }
    with open(config.BM25_PATH, "wb") as f:
        pickle.dump(payload, f)
    print(f"BM25 saved → {config.BM25_PATH}")
