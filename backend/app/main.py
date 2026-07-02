# app/main.py
# ---------------------------------------------------------------------------
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
from . import retriever, prompt, ollama_client

app = FastAPI(title="BCT RAG API", version="1.0.0")

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
    question: str
    history: List[Message] = []


class Citation(BaseModel):
    id: str
    text: str
    metadata: dict
    score: float


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(400, "Empty question")

    clean_q = req.question.lower().strip()
    history_dicts = [{"role": m.role, "content": m.content} for m in req.history]

    # =========================================================================
    # ÉTAPE 1 : ROUTEUR DÉTERMINISTE (Évite les hallucinations de classification)
    # =========================================================================
    is_chitchat = any(w in clean_q for w in ["salut", "bonjour", "hello", "aslema", "ca va", "comment tu vas", "réveillé"])
    is_definition = any(w in clean_q for w in ["abréviation", "abriviation", "c'est quoi la bct", "bct c'est quoi", "signifie bct"])

    # BRANCHE CHAT : Gestion directe du bavardage et des définitions de l'acronyme
    if is_chitchat or is_definition:
        if is_definition:
            answer = "BCT est l'abréviation de la **Banque Centrale de Tunisie**."
        else:
            messages = [{"role": "system", "content": prompt.SYSTEM_PROMPT}] + history_dicts + [
                {"role": "user", "content": req.question + "\n\n(Reminder: Answer nicely in French or Arabic. Do not use English.)"}
            ]
            answer = ollama_client.chat_raw_messages(messages)
        return {"answer": answer, "citations": []}

    # =========================================================================
    # ÉTAPE 2 : FUSION INTELLIGENTE DE L'HISTORIQUE POUR LE RAG (Sans surcoût GPU)
    # =========================================================================
    search_query = req.question
    
    # Si c'est une question de suivi, on combine textuellement avec la dernière question de l'utilisateur
    context_words = ["pourquoi", "et si", "et pour", "récidive", "شرح", "العقوبة", "laquelle"]
    if history_dicts and any(w in clean_q for w in context_words):
        # Récupérer la toute dernière question posée par l'utilisateur dans l'historique
        last_user_q = next((m["content"] for m in reversed(history_dicts) if m["role"] == "user"), "")
        if last_user_q:
            # On fusionne les mots-clés (ex: "Quel est l'amende pour l'article 38 ?" + "Et si c'est une récidive ?")
            search_query = f"{last_user_q} {req.question}"

    # =========================================================================
    # ÉTAPE 3 : RECHERCHE HYBRIDE ET GÉNÉRATION FINALÉ
    # =========================================================================
    # On interroge ChromaDB/BM25 avec la requête enrichie en mots-clés
    chunks = retriever.hybrid_search(search_query)
    
    # On construit le prompt final contenant les documents
    user_msg = prompt.build_user_prompt(req.question, chunks)
    
    # On assemble l'historique de discussion pour que le LLM comprenne les pronoms (ex: "Pourquoi ?")
    messages = [
        {"role": "system", "content": prompt.SYSTEM_PROMPT}
    ] + history_dicts + [
        {"role": "user", "content": user_msg + "\n\n(Reminder: Answer strictly in French or Arabic. Do not use English.)"}
    ]
    
    answer = ollama_client.chat_raw_messages(messages)
    return {"answer": answer, "citations": chunks}