# prompt.py
# ---------------------------------------------------------------------------
# Prompt assembly. Bilingual (French + Arabic). Forces citation by chunk index.
# ---------------------------------------------------------------------------
from typing import Dict, List

# prompt.py

SYSTEM_PROMPT = (
    "You are a professional compliance officer for the Banque Centrale de Tunisie (BCT).\n\n"
    "CRITICAL GROUND RULES:\n"
    "1. ANSWER ONLY FROM CONTEXT: Answer strictly using the provided context. If the answer is not in the context, you MUST refuse.\n"
    "2. NO HALLUCINATION / OUT-OF-SCOPE: If the user asks about an unrelated topic (e.g. CV, sports, generic requests), you must output EXACTLY and ONLY the official refusal phrase. Do not add helpful suggestions, do not add 'Par exemple'. SAY ABSOLUTELY NOTHING ELSE.\n"
    "3. CITATIONS: Support every factual statement by appending its source marker (e.g. [#1], [#2]).\n\n"
    "AMBIGUITY RULE (MULTIPLE ARTICLES):\n"
    "If the retrieved context contains multiple different rules with the same name (e.g., several 'Article 4') AND the user's question is too broad to know which one they mean:\n"
    "- Briefly summarize each version found in the context.\n"
    "- End your answer by asking the user to clarify (e.g., 'Lequel de ces articles recherchez-vous ?').\n"
    "- IMPORTANT: If the user has ALREADY clarified which one they want (e.g., 'le dernier', 'celui sur le prix'), DO NOT ask the question again. Just provide the answer."
)

def build_user_prompt(question: str, chunks: list, target_language: str) -> str:
    """Construit le prompt avec forçage de la langue."""
    context_blocks = []
    for i, c in enumerate(chunks, start=1):
        meta = c["metadata"]
        header = f"[#{i}] source={meta.get('source')} page={meta.get('page')}".strip()
        context_blocks.append(f"{header}\n{c['text']}")

    context = "\n\n---\n\n".join(context_blocks) if context_blocks else "(no context)"
    
    # FORÇAGE DE LA LANGUE ICI
    lang_instruction = "MUST answer strictly in FRENCH." if target_language == "fr" else "MUST answer strictly in ARABIC. Translate BCT concepts properly."
    
    return (
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION:\n{question}\n\n"
        f"Generate your response following the system rules. You {lang_instruction}"
    )