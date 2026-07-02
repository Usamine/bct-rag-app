# app/ollama_client.py
# ---------------------------------------------------------------------------
import requests
from typing import List, Dict
from . import config

_client = requests.Session()

def embed(text: str) -> List[float]:
    r = _client.post(f"http://localhost:11434/api/embeddings", json={
        "model": config.EMBED_MODEL,
        "prompt": text
    })
    r.raise_for_status()
    return r.json()["embedding"]


def classify_intent(question: str) -> str:
    """Classifies the user query into 'BCT' or 'CHAT'."""
    system_prompt = (
        "You are a strict query classifier. Analyze the user input.\n"
        "If the input is a greeting (bonjour, aslema, hi), polite small talk, a definition of your acronym (c'est quoi la BCT, abbreviation bct), "
        "or completely unrelated to Tunisian banking regulations, output exactly: CHAT\n"
        "If the input is a legitimate question about laws, circulars, or monetary regulations of the Banque Centrale de Tunisie, output exactly: BCT\n"
        "Rules: Output ONLY the word CHAT or BCT. No punctuation, no explanations."
    )
    
    r = _client.post("http://localhost:11434/api/chat", json={
        "model": config.LLM_MODEL,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 5},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Input: {question}\nClassification:"},
        ],
    })
    r.raise_for_status()
    return r.json()["message"]["content"].strip().upper()


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

    r = _client.post("http://localhost:11434/api/chat", json={
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
    r = _client.post("http://localhost:11434/api/chat", json={
        "model": config.LLM_MODEL,
        "stream": False,
        "options": {"temperature": 0.1, "num_ctx": 4096},
        "messages": messages,
    })
    r.raise_for_status()
    return r.json()["message"]["content"]