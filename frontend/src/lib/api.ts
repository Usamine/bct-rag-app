export type Citation = {
  id: string;
  text: string;
  score: number;
  metadata: {
    track: string;
    source: string;
    page: number;
    marker?: string;
  };
};

export type QueryResponse = {
  answer: string;
  citations: Citation[];
};

// ✅ NOUVEAU : On définit le type pour les messages de l'historique
export type HistoryMessage = {
  role: "user" | "assistant";
  content: string;
};

// ✅ MISE À JOUR : La fonction accepte maintenant l'historique en deuxième paramètre
export async function askQuestion(
  question: string,
  history: HistoryMessage[] = [] // Par défaut, un tableau vide s'il n'y a pas d'historique
): Promise<QueryResponse> {

  const r = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    // ✅ ON ENVOIE LES DEUX : La question actuelle ET l'historique
    body: JSON.stringify({ question, history }),
  });

  if (!r.ok) throw new Error((await r.json()).error || "Request failed");
  return r.json();
}