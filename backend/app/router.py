# ---------------------------------------------------------------------------
# Semantic Router using local bge-m3 embeddings.
# Bypasses the RAG database for conversational queries or greetings.
# ---------------------------------------------------------------------------
from typing import List, Literal, Optional, Tuple
import numpy as np

from . import config, ollama_client

# Exemples de phrases types "Hors-Sujet" ou Salutations (Multilingue)
CHITCHAT_SAMPLES = [
    # Français
    "bonjour", "hello", "salut", "bonsoir", "slt",
    "comment ca va", "comment allez vous", "ca va", "tu vas bien",
    "merci", "merci beaucoup", "un grand merci", "parfait merci",
    "au revoir", "bye", "a bientot", "bonne journée",
    "tu es qui", "c'est quoi ton nom", "qui es tu", "presentation",

    # Arabe & Arabizi
    "مرحبا", "أهلا", "صباح الخير", "مساء الخير", "السلام عليكم",
    "شكرا", "شكرا جزيلا", "يعطيك الصحة",
    "kayfa al hal", "labes", "chneya ahwalek", "cv", "aslema", "w rabi m3ak",
    "chkounek", "man anta", "من أنت",

    # Anglais
    "hi", "how are you", "how are u doing", "thanks", "thank you",
    "who are you", "what is your name", "what can you do"
]

_chitchat_embeddings = None


def _load_router_embeddings():
    """Lazily embeds the chitchat reference samples once per process."""
    global _chitchat_embeddings
    if _chitchat_embeddings is None:
        # embed_batch: one round trip for all samples instead of doing it one by one.
        _chitchat_embeddings = ollama_client.embed_batch(CHITCHAT_SAMPLES)
    return _chitchat_embeddings


def route_query(
    query: str,
    query_vec: Optional[List[float]] = None,
    threshold: float = None,
) -> Tuple[Literal["chitchat", "rag"], Optional[List[float]]]:
    
    threshold = threshold if threshold is not None else config.ROUTER_THRESHOLD
    clean_q = query.strip().lower()

    # 🛑 SÉCURITÉ ANTI-FAUX POSITIFS : 
    # Si la phrase est longue (ex: plus de 60 caractères), c'est forcément une vraie question.
    if len(clean_q) > 60:
        print(f"🧭 [ROUTER] Phrase longue ({len(clean_q)} chars) -> Route forcée : RAG")
        return "rag", query_vec

    # 1. Détection instantanée... (la suite de votre code reste identique)
    if clean_q in ["bonjour", "hello", "salut", "hi", "مرحبا", "kayfa al hal", "labes", "cv"]:
        print(f"🧭 [ROUTER] Exact Match -> Route: CHITCHAT")
        return "chitchat", query_vec

    # 2. Comparaison sémantique vectorielle via Numpy
    qvec = query_vec if query_vec is not None else ollama_client.embed(query)
    sample_vecs = _load_router_embeddings()

    q_arr = np.array(qvec)
    s_arr = np.array(sample_vecs)
    
    # Calcul rapide de la similarité cosinus matricielle
    dot_products = np.dot(s_arr, q_arr)
    q_norm = np.linalg.norm(q_arr)
    s_norms = np.linalg.norm(s_arr, axis=1)
    
    scores = dot_products / (q_norm * s_norms + 1e-8)
    max_score = float(np.max(scores))

    # 3. Décision basée sur le seuil
    classification = "chitchat" if max_score > threshold else "rag"
    
    print(f"🧭 [ROUTER] Score: {max_score:.3f} (Threshold: {threshold}) -> Route: {classification.upper()}")
    
    return classification, qvec