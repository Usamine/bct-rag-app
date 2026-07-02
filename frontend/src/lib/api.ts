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

export async function askQuestion(question: string): Promise<QueryResponse> {
  const r = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
  if (!r.ok) throw new Error((await r.json()).error || "Request failed");
  return r.json();
}
