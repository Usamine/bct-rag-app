# app/ollama_client.py
# ---------------------------------------------------------------------------
import requests
from typing import List, Dict
from . import config
import json

_client = requests.Session()

def embed(text: str) -> List[float]:
    r = _client.post(f"{config.OLLAMA_URL}/api/embeddings", json={
        "model": config.EMBED_MODEL,
        "prompt": text
    })
    r.raise_for_status()
    return r.json()["embedding"]


def embed_batch(texts: List[str]) -> List[List[float]]:
    """
    Embed multiple texts in one round trip — this is what indexer.py calls
    during ingest, so a batch of chunks costs one HTTP call instead of one
    per chunk.

    Tries Ollama's batched endpoint first (`/api/embed`, which takes
    `input: list[str]` and returns `embeddings: list[list[float]]`). Falls
    back to calling `embed()` once per text if that endpoint isn't available
    (older Ollama versions only expose the single-prompt `/api/embeddings`).
    Order is preserved either way, which matters since indexer.py zips these
    vectors back up against `ids`/`metadatas` by position.
    """
    if not texts:
        return []
    try:
        r = _client.post(f"{config.OLLAMA_URL}/api/embed", json={
            "model": config.EMBED_MODEL,
            "input": texts,
        })
        r.raise_for_status()
        return r.json()["embeddings"]
    except Exception:
        return [embed(t) for t in texts]


def classify_intent(question: str) -> str:
    """Classifies the user query using Few-Shot Prompting."""
    
    r = _client.post(f"{config.OLLAMA_URL}/api/chat", json={
        "model": config.LLM_MODEL,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 5},
        "messages": [
            {"role": "system", "content": "You are a strict text classifier. You must output EXACTLY and ONLY the word 'CHAT' or 'BCT'. Do not output any other character or explanation."},
            # --- Début des exemples (Few-Shot) ---
            {"role": "user", "content": "bonjour"},
            {"role": "assistant", "content": "CHAT"},
            {"role": "user", "content": "what day is today"},
            {"role": "assistant", "content": "CHAT"},
            {"role": "user", "content": "kayfa al hal"},
            {"role": "assistant", "content": "CHAT"},
            {"role": "user", "content": "how are you doing"},
            {"role": "assistant", "content": "CHAT"},
            {"role": "user", "content": "quel est le statut de la banque centrale ?"},
            {"role": "assistant", "content": "BCT"},
            {"role": "user", "content": "قرار عدد 51"},
            {"role": "assistant", "content": "BCT"},
            # --- Fin des exemples ---
            # La vraie question de l'utilisateur :
            {"role": "user", "content": question}
        ],
    })
    
    r.raise_for_status()
    intent = r.json()["message"]["content"].strip().upper()
    
    # Petite astuce de debug : on affiche ce que le LLM a vraiment répondu dans votre terminal
    print(f"🧠 [CLASSIFIER] Question: '{question}' -> Intent: {intent}")
    
    return intent


def condense_question(question: str, history: List[Dict]) -> str:
    """
    Rewrites follow-up questions into standalone explicit queries using history.
    If the question is already standalone, returns it untouched.
    """
    if not history:
        return question

    # On garde les 4 derniers messages pour le contexte de reformulation
    history_str = ""
    for msg in history[-4:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_str += f"{role}: {msg['content']}\n"

    system_prompt = (
        "You are an expert query contextualizer. Your job is to analyze a conversation history and a new user question.\n"
        "If the new question is a follow-up (like 'Why?', 'And for article 39?', 'What about recidivism?'), rewrite it into a fully standalone, explicit question in the same language that includes all necessary context from the history.\n"
        "If the new question is ALREADY complete, standalone, and clear on its own, you MUST output it EXACTLY as it is, without changing a single word.\n\n"
        "Examples:\n"
        "History:\nUser: Quel est le montant de l'amende pour l'article 38?\nAssistant: 200 dinars par jour.\n"
        "New Question: Et si c'est une récidive?\n"
        "Output: Quelle est l'amende pour l'article 38 en cas de récidive?\n\n"
        "History:\nUser: Qu'est ce que la loi 2016?\nAssistant: C'est le statut de la BCT.\n"
        "New Question: La banque centrale peut-elle accorder un découvert à l'Etat?\n"
        "Output: La banque centrale peut-elle accorder un découvert à l'Etat tunisien?\n\n"
        "Rule: Output ONLY the final standalone question. No explanations, no markdown, no quotes."
    )

    r = _client.post(f"{config.OLLAMA_URL}/api/chat", json={
        "model": config.LLM_MODEL,
        "stream": False,
        "options": {"temperature": 0.0},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"HISTORY:\n{history_str}\nNEW QUESTION: {question}\nOutput:"},
        ],
    })
    r.raise_for_status()
    return r.json()["message"]["content"].strip()


def chat_raw_messages(messages: List[Dict]) -> str:
    r = _client.post(f"{config.OLLAMA_URL}/api/chat", json={
        "model": config.LLM_MODEL,
        "stream": False,
        "options": {"temperature": 0.1, "num_ctx": 4096},
        "messages": messages,
    })
    r.raise_for_status()
    return r.json()["message"]["content"]



def stream_chat_messages(messages: List[Dict]):
    """
    Appelle Ollama en mode STREAM et cède (yield) les morceaux de texte 
    au fur et à mesure de leur génération par le LLM.
    """
    r = _client.post(
        f"{config.OLLAMA_URL}/api/chat", 
        json={
            "model": config.LLM_MODEL,
            "stream": True,  # <-- Activer le streaming d'Ollama
            "options": {"temperature": 0.1, "num_ctx": 4096},
            "messages": messages,
        },
        stream=True # <-- Activer le streaming HTTP de la bibliothèque requests
    )
    r.raise_for_status()
    
    for line in r.iter_lines():
        if line:
            chunk = json.loads(line.decode("utf-8"))
            content = chunk.get("message", {}).get("content", "")
            if content:
                yield content