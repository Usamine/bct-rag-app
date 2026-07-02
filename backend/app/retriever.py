# ---------------------------------------------------------------------------
# Hybrid retrieval: dense (Chroma) + sparse (BM25) fused with RRF.
# Loaded lazily so the FastAPI app starts fast.
# ---------------------------------------------------------------------------
import pickle
from typing import List, Dict, Tuple
import chromadb
from . import config, ollama_client, indexer

# Module-level caches (loaded once per process).
_bm25_payload = None
_collection = None


def _load_bm25():
    global _bm25_payload
    if _bm25_payload is None:
        if not config.BM25_PATH.exists():
            raise RuntimeError("BM25 index missing. Run `python -m app.ingest` first.")
        with open(config.BM25_PATH, "rb") as f:
            _bm25_payload = pickle.load(f)
    return _bm25_payload


def _load_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=str(config.CHROMA_PATH))
        _collection = client.get_or_create_collection(name="bct")
    return _collection


# --- individual retrievers -------------------------------------------------
def dense_search(query: str, k: int) -> List[Tuple[str, Dict]]:
    """Return [(chunk_id, {text, metadata, score}), …] ordered by similarity."""
    coll = _load_collection()
    qvec = ollama_client.embed(query)
    res = coll.query(query_embeddings=[qvec], n_results=k,
                     include=["documents", "metadatas", "distances"])
    out = []
    for cid, doc, meta, dist in zip(res["ids"][0], res["documents"][0],
                                    res["metadatas"][0], res["distances"][0]):
        score = 1 - dist
        
        # 💡 AMÉLIORATION : Seuil de sécurité. Si le score vectoriel est inférieur à 0.35, 
        # c'est du bruit hors-sujet (comme une recherche sur le mot 'Bonjour'). On l'ignore.
        if score < 0.35: 
            continue
            
        out.append((cid, {"text": doc, "metadata": meta, "score": score}))
    return out


def sparse_search(query: str, k: int) -> List[Tuple[str, Dict]]:
    """BM25 keyword search — great for IDs like 'Circulaire 2016-35'."""
    payload = _load_bm25()
    tokens = indexer.tokenize(query)
    scores = payload["bm25"].get_scores(tokens)
    # take top-k indices
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    out = []
    for i in top_idx:
        if scores[i] <= 0:
            continue
        out.append((payload["ids"][i], {
            "text": payload["docs"][i],
            "metadata": payload["metas"][i],
            "score": float(scores[i]),
        }))
    return out


# --- Reciprocal Rank Fusion -----------------------------------------------
def rrf_fuse(rankings: List[List[Tuple[str, Dict]]], k_const: int = config.RRF_K
             ) -> List[Tuple[str, Dict]]:
    """
    Standard RRF: score(d) = Σ 1/(k + rank_i(d)) across all rankings.
    Returns a single ranked list with the original payload attached.
    """
    scores: Dict[str, float] = {}
    payloads: Dict[str, Dict] = {}
    for ranking in rankings:
        for rank, (cid, payload) in enumerate(ranking, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k_const + rank)
            payloads.setdefault(cid, payload)
    fused = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [(cid, {**payloads[cid], "rrf": s}) for cid, s in fused]


def hybrid_search(query: str) -> List[Dict]:
    """Top-level retrieval entrypoint used by the API."""
    # 💡 AMÉLIORATION : Si la requête est ultra-courte (salutations, etc.), inutile de stresser la DB
    clean_query = query.strip().lower()
    if len(clean_query) < 4 or clean_query in ["bonjour", "hello", "salut", "مرحبا"]:
        return []

    dense = dense_search(query, config.DENSE_TOP_K)
    sparse = sparse_search(query, config.SPARSE_TOP_K)
    
    # Si les deux moteurs n'ont rien trouvé de pertinent sous le seuil
    if not dense and not sparse:
        return []
        
    fused = rrf_fuse([dense, sparse])[: config.FINAL_TOP_K]
    return [
        {
            "id": cid,
            "text": p["text"],
            "metadata": p["metadata"],
            "score": p["rrf"],
        }
        for cid, p in fused
    ]
