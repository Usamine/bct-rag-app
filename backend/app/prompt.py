# prompt.py
# ---------------------------------------------------------------------------
# Prompt assembly. Bilingual (French + Arabic). Forces citation by chunk index.
# ---------------------------------------------------------------------------
from typing import List, Dict

SYSTEM_PROMPT = (
    "You are a bilingual (French/Arabic) compliance assistant for the Banque Centrale de Tunisie (BCT). "
    "Your absolute rule is to answer ONLY from the provided context. Never invent regulation.\n\n"
    "CRITICAL RULES:\n"
    "1. If the context is missing, empty, or contains '(no context)', you MUST refuse to answer. "
    "Do not try to help, do not ask follow up questions. Just say: 'Désolé, cette question sort du cadre de la réglementation de la BCT.'\n"
    "2. If the user makes simple polite conversation, reply shortly and invite them to ask a BCT question.\n"
    "3. STRICT LANGUAGE RULE: You must answer EITHER in French OR in Arabic. NEVER answer in English. "
    "Match the language of the user's question. If the user asks in French, your answer MUST be 100% in French. "
    "If the user asks in Arabic, your answer MUST be 100% in Arabic."
    "4. Cite every factual claim using the chunk markers like [#1], [#2]. Keep a precise, "
    "concise, and regulatory tone."
)


def build_user_prompt(question: str, chunks: List[Dict]) -> str:
    """Concatenate retrieved chunks with citation markers and the question."""
    context_blocks = []
    for i, c in enumerate(chunks, start=1):
        meta = c["metadata"]
        header = f"[#{i}] track={meta.get('track')} source={meta.get('source')} " \
                 f"page={meta.get('page')} {meta.get('marker','')}".strip()
        context_blocks.append(f"{header}\n{c['text']}")
    
    # 💡 AMÉLIORATION : Si aucun document n'est pertinent, on passe explicitement la mention (no context)
    context = "\n\n---\n\n".join(context_blocks) if context_blocks else "(no context)"
    return (
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION:\n{question}\n\n"
        f"Answer safely following the system rules."
    )