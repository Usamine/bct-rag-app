# ---------------------------------------------------------------------------
# Hybrid retrieval: dense (Chroma) + sparse (BM25) fused with RRF + Reranking.
# Loaded lazily so the FastAPI app starts fast.
# ---------------------------------------------------------------------------
import pickle
from typing import Dict, List, Optional, Tuple

from . import config, indexer, ollama_client

# Module-level caches (loaded once per process).
_bm25_payload = None
_reranker = None


def _load_bm25():
    global _bm25_payload
    if _bm25_payload is None:
        if not config.BM25_PATH.exists():
            raise RuntimeError("BM25 index missing. Run `python -m app.ingest` first.")
        with open(config.BM25_PATH, "rb") as f:
            _bm25_payload = pickle.load(f)
    return _bm25_payload


def _load_reranker():
    """Lazy load the Cross-Encoder reranker model to keep startup fast."""
    global _reranker
    if _reranker is None:
        try:
            from sentence_transformers import CrossEncoder
            # We use a fast, lightweight open-source reranker
            model_name = getattr(config, "RERANK_MODEL", "BAAI/bge-reranker-base")
            _reranker = CrossEncoder(model_name)
        except ImportError:
            raise RuntimeError(
                "La bibliothèque 'sentence-transformers' est manquante. "
                "Exécutez `pip install sentence-transformers` pour activer le Reranking."
            )
    return _reranker


# --- individual retrievers -------------------------------------------------
def dense_search(query: str, k: int, query_vec: Optional[List[float]] = None) -> List[Tuple[str, Dict]]:
    """
    Return [(chunk_id, {text, metadata, score}), …] ordered by similarity.

    `query_vec` lets a caller pass in an embedding it already computed
    elseer (e.g. main.py reusing the same vector it used for semantic
    routing), instead of this function silently re-embedding the same text
    and doubling the number of round trips to Ollama per request.
    """
    # get_collection() reuses indexer's client (same telemetry settings,
    # same on-disk store) instead of retriever.py spinning up its own
    # separate PersistentClient pointed at the same path.
    coll = indexer._get_collection()
    qvec = query_vec if query_vec is not None else ollama_client.embed(query)
    res = coll.query(query_embeddings=[qvec], n_results=k,
                     include=["documents", "metadatas", "distances"])
    out = []
    for cid, doc, meta, dist in zip(res["ids"][0], res["documents"][0],
                                    res["metadatas"][0], res["distances"][0]):
        score = 1 - dist

        # Safety threshold: below this cosine similarity, a "hit" is noise
        # (e.g. an off-topic query still returning the least-bad chunk).
        if score < config.DENSE_SCORE_THRESHOLD:
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
def rrf_fuse(rankings: List[List[Tuple[str, Dict]]], k_const: int = None) -> List[Tuple[str, Dict]]:
    """
    Standard RRF: score(d) = Σ 1/(k + rank_i(d)) across all rankings.
    Returns a single ranked list with the original payload attached.
    """
    k_const = k_const if k_const is not None else config.RRF_K
    scores: Dict[str, float] = {}
    payloads: Dict[str, Dict] = {}
    for ranking in rankings:
        for rank, (cid, payload) in enumerate(ranking, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k_const + rank)
            payloads.setdefault(cid, payload)
    fused = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [(cid, {**payloads[cid], "rrf": s}) for cid, s in fused]


def hybrid_search(query: str, query_vec: Optional[List[float]] = None) -> List[Dict]:
    """
    Top-level retrieval entrypoint used by the API.

    Uses Dense + Sparse search, fuses them via RRF, reranks the top candidates
    using a Cross-Encoder, and returns the absolute best chunks.
    """
    clean_query = query.strip().lower()
    if len(clean_query) < 4:
        return []

    # On demande un peu plus de chunks aux moteurs initiaux pour donner du choix au reranker
    dense = dense_search(query, config.DENSE_TOP_K, query_vec=query_vec)
    sparse = sparse_search(query, config.SPARSE_TOP_K)

    if not dense and not sparse:
        return []

    # 1. Fusion RRF globale
    fused = rrf_fuse([dense, sparse])

    # 2. Étape de Reranking (si activée dans la config)
    use_reranker = getattr(config, "USE_RERANKER", True)
    rerank_top_k = getattr(config, "RERANK_TOP_K", 25)

    if use_reranker and fused:
        # On ne garde que les 'X' meilleurs candidats du RRF pour les faire réévaluer par l'IA
        candidates_to_rerank = fused[:rerank_top_k]
        reranker = _load_reranker()

        # On prépare les paires (Question, Paragraphe)
        pairs = [(query, p["text"]) for _, p in candidates_to_rerank]
        
        # Calcul des nouveaux scores de pertinence absolue
        rerank_scores = reranker.predict(pairs)

        # On injecte le nouveau score dans chaque élément
        for idx, (cid, p) in enumerate(candidates_to_rerank):
            p["rerank_score"] = float(rerank_scores[idx])

        # Tri final basé UNIQUEMENT sur la pertinence sémantique réelle du reranker
        fused = sorted(candidates_to_rerank, key=lambda x: x[1]["rerank_score"], reverse=True)

    # 3. Découpage final selon la limite fixée pour le LLM
    fused = fused[: config.FINAL_TOP_K]

    return [
        {
            "id": cid,
            "text": p["text"],
            "metadata": p["metadata"],
            "score": p.get("rerank_score", p["rrf"]),
        }
        for cid, p in fused
    ]