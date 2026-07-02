# router.py
# ---------------------------------------------------------------------------
# Semantic Router using local bge-m3 embeddings. 
# Bypasses the RAG database for conversational queries or greetings.
# ---------------------------------------------------------------------------
from typing import Literal
from . import ollama_client

# Exemples de phrases types "Hors-Sujet" ou Salutations (Bilingue)
CHITCHAT_SAMPLES = [
    "bonjour", "hello", "salut", "مرحبا", "صباح الخير", "مساء الخير",
    "comment ça va", "how are you", "tu vas bien",
    "tu es qui", "who are you", "c'est quoi ton nom",
    "merci", "thanks", "شكرا", "au revoir", "bye", "à bientot"
]

_chitchat_embeddings = None

def _load_router_embeddings():
    """Lazily embeds the chitchat reference samples once."""
    global _chitchat_embeddings
    if _chitchat_embeddings is None:
        # Réutilisation de votre pipeline d'embedding existant
        _chitchat_embeddings = [ollama_client.embed(text) for text in CHITCHAT_SAMPLES]
    return _chitchat_embeddings


def _cosine_similarity(v1, v2) -> float:
    """Computes the dot product similarity between two vectors."""
    dot = sum(x * y for x, y in zip(v1, v2))
    mag1 = sum(x**2 for x in v1) ** 0.5
    mag2 = sum(x**2 for x in v2) ** 0.5
    if not mag1 or not mag2:
        return 0.0
    return dot / (mag1 * mag2)


def route_query(query: str, threshold: float = 0.40) -> Literal["chitchat", "rag"]:
    """
    Analyzes the query semantic closeness to standard greetings.
    Returns 'chitchat' to skip DB retrieval, or 'rag' for regulatory searches.
    """
    clean_q = query.strip().lower()
    
    # 1. Sécurité matérielle rapide (évite un appel d'embedding pour les mots simples)
    if clean_q in ["bonjour", "hello", "salut", "l", "hi", "مرحبا"]:
        return "chitchat"
        
    # 2. Comparaison sémantique vectorielle
    query_vec = ollama_client.embed(query)
    sample_vecs = _load_router_embeddings()
    
    max_score = 0.0
    for s_vec in sample_vecs:
        score = _cosine_similarity(query_vec, s_vec)
        if score > max_score:
            max_score = score
            
    # Si la phrase ressemble de près ou de loin à du bavardage, on déroute
    if max_score > threshold:
        return "chitchat"
        
    return "rag"