import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional  # <-- FIX 1 : Ajout de Optional ici
import re
from . import config, lang_utils, ollama_client, prompt, retriever, router

app = FastAPI(title="BCT RAG API", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Message(BaseModel):
    role: str
    content: str


class QueryRequest(BaseModel):
    question: str  # <-- FIX 2 : Remplacé 'query' par 'question' pour s'aligner avec Next.js
    history: Optional[List[Message]] = [] # <-- FIX 3 : Remplacé Dict par Message pour autoriser m.role et m.content


class Citation(BaseModel):
    id: str
    text: str
    metadata: dict
    score: float


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]


# --- canned bilingual replies -----------------------------------------------
_BCT_DEFINITION = {
    "fr": "BCT est l'abréviation de la **Banque Centrale de Tunisie**.",
    "ar": "BCT هي اختصار لـ **البنك المركزي التونسي**.",
}
_NO_CONTEXT_REFUSAL = {
    "fr": "Désolé, cette question sort du cadre de la réglementation de la BCT.",
    "ar": "عذراً، هذا السؤال خارج نطاق تشريعات البنك المركزي التونسي.",
}
_SERVICE_DOWN = {
    "fr": "Désolé, le service IA est momentanément indisponible. Merci de réessayer dans un instant.",
    "ar": "عذراً، الخدمة غير متوفرة حالياً. يرجى إعادة المحاولة après قليل.",
}

_DEFINITION_KEYWORDS = [
    "abréviation", "abriviation", "cest quoi la bct", "bct cest quoi", 
    "signifie bct", "stand for", "meaning of bct", "what is bct",
    "que veut dire bct", "ça veut dire quoi bct", "ca veut dire quoi bct"
]

def _lang(question: str) -> str:
    q_lower = question.lower()
    if "en arabe" in q_lower or "in arabic" in q_lower or "بالعربية" in q_lower:
        return "ar"
    if "en français" in q_lower or "in french" in q_lower or "بالفرنسية" in q_lower:
        return "fr"
    return "ar" if lang_utils.is_arabic(question) else "fr"


def _call_llm(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except requests.exceptions.RequestException as e:
        print(f"  [warn] Ollama call failed: {e}")
        return None


# app/main.py

@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(400, "Empty question")

    lang = _lang(question)
    clean_q = re.sub(r'[^\w\s]', '', question.lower()).strip()
    
    # Grâce au FIX 3, m.role et m.content fonctionnent maintenant parfaitement
    history_dicts = [{"role": m.role, "content": m.content} for m in req.history]
    history_dicts = history_dicts[-(config.MAX_HISTORY_TURNS * 2):]

    # =========================================================================
    # ÉTAPE 1 : Définition rapide de l'acronyme (Déterministe)
    # =========================================================================
    if any(w in clean_q for w in _DEFINITION_KEYWORDS):
        return {"answer": _BCT_DEFINITION[lang], "citations": []}

    # =========================================================================
    # ÉTAPE 2 : Routeur Sémantique Mathématique (Fiable à 100%)
    # =========================================================================
    q_embedding = _call_llm(ollama_client.embed, question)
    if q_embedding is None:
        raise HTTPException(503, _SERVICE_DOWN[lang])

    route, q_embedding = router.route_query(question, query_vec=q_embedding)

    if route == "chitchat":
        chat_prompt = (
            "Tu es l'assistant virtuel de la Banque Centrale de Tunisie. "
            "L'utilisateur te salue. "
            "RÈGLE ABSOLUE : Réponds UNIQUEMENT par une salutation polie (ex: 'Bonjour ! Comment puis-je vous aider avec la BCT ?'). "
            "NE PROPOSE JAMAIS d'aide pour des CV ou autre chose. "
            "Réponds dans la même langue que l'utilisateur."
        )
        messages = [{"role": "system", "content": chat_prompt}] + history_dicts + [
            {"role": "user", "content": question + "\n\n(Reminder: Answer nicely in French or Arabic based on the user's language. Do not use English.)"}
        ]
        answer = _call_llm(ollama_client.chat_raw_messages, messages)
        if answer is None:
            raise HTTPException(503, _SERVICE_DOWN[lang])
        return {"answer": answer, "citations": []}
        
    # ÉTAPE 3 : Réécriture de la question (Mémoire)
    # =========================================================================
    search_query = _call_llm(ollama_client.condense_question, question, history_dicts)
    if search_query is None:
        search_query = question 

    print(f"🔄 [MEMORY] Originale: '{question}' -> Reformulée: '{search_query}'")

    # --- SÉCURITÉ LIMITÉE AU DERNIER ÉCHANGE UNIQUEMENT ---
    # On ne prend que le tout dernier message s'il existe pour éviter de polluer le futur
    last_message = history_dicts[-1].get("content", "") if history_dicts else ""
    immediate_context = (last_message + " " + question).lower()
    
    print(f"📦 [CONTEXT DEBUG] Ce que la règle de sécurité analyse : '{immediate_context}'")

    # # La règle de traduction ne s'active QUE si l'utilisateur demande explicitement de l'arabe dans sa question ACTUELLE
    # is_translation = ("arab" in clean_q or "arabe" in clean_q) and ("bct" in immediate_context or "banque" in immediate_context or "bank" in immediate_context)
    # is_definition = any(w in clean_q for w in _DEFINITION_KEYWORDS)

    # if is_translation or is_definition:
    #     if "arab" in clean_q or "arabe" in clean_q:
    #         return {"answer": "البنك المركزي التونسي (Al-Bank Al-Markazi At-Tunisi)", "citations": []}
    #     return {"answer": _BCT_DEFINITION[lang], "citations": []}

    # =========================================================================
    # ÉTAPE 4 : Recherche hybride
    # =========================================================================
    chunks = retriever.hybrid_search(search_query, query_vec=q_embedding)

    if not chunks:
        return {"answer": _NO_CONTEXT_REFUSAL[lang], "citations": []}

    # =========================================================================
    # ÉTAPE 5 : Génération finale (RAG)
    # =========================================================================
    
    # On passe 'lang' en 3ème argument pour forcer l'arabe ou le français
    user_msg = prompt.build_user_prompt(question, chunks, lang)
    
    messages = [
        {"role": "system", "content": prompt.SYSTEM_PROMPT}
    ] + history_dicts + [
        {"role": "user", "content": user_msg}
    ]

    answer = _call_llm(ollama_client.chat_raw_messages, messages)
    if answer is None:
        raise HTTPException(503, _SERVICE_DOWN[lang])

    return {"answer": answer, "citations": chunks}