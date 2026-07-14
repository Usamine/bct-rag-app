"use client";

import { useState, useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import { Send, Loader2, Info } from "lucide-react";
import { askQuestion, type Citation } from "@/lib/api";
import Citations from "./Citations";

type Msg = {
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
};

export default function Chat() {
  const [messages, setMessages] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [activeCitations, setActiveCitations] = useState<Citation[]>([]);
  const scrollRef = useRef<HTMLDivElement>(null);

  // Auto-scroll on new message
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, loading]);

  async function submit() {
    const q = input.trim();
    if (!q || loading) return;
    setInput("");

    // On extrait l'historique AVANT d'ajouter la nouvelle question
    const historyPayload = messages.map((m) => ({
      role: m.role,
      content: m.content,
    }));

    setMessages((m) => [...m, { role: "user", content: q }]);
    setLoading(true);

    try {
      // ✅ ON PASSE LA QUESTION ET L'HISTORIQUE !
      const res = await askQuestion(q, historyPayload);
      setMessages((m) => [
        ...m,
        { role: "assistant", content: res.answer, citations: res.citations },
      ]);
      setActiveCitations(res.citations);
    } catch (e: any) {
      setMessages((m) => [
        ...m,
        { role: "assistant", content: `**Error:** ${e.message}` },
      ]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex-1 grid grid-cols-1 lg:grid-cols-[1fr_380px] overflow-hidden bg-white">

      {/* --- Main Chat Column --- */}
      <section className="flex flex-col h-full overflow-hidden relative">

        {/* Scrollable Messages Area */}
        <div ref={scrollRef} className="flex-1 overflow-y-auto px-4 md:px-8 py-8 space-y-8 scroll-smooth">

          {/* Empty State */}
          {messages.length === 0 && (
            <div className="flex flex-col items-center justify-center h-full text-neutral-400">
              <p className="text-lg text-neutral-600 font-medium mb-2">
                Ask a question about BCT regulations — in French or Arabic.
              </p>
              <p className="text-sm italic text-neutral-400">
                e.g. « Que prévoit la Circulaire 2016-35 ? »
              </p>
            </div>
          )}

          {/* Messages */}
          {messages.map((m, i) => (
            <div
              key={i}
              className={`flex w-full max-w-4xl mx-auto ${m.role === "user" ? "justify-end" : "justify-start"}`}
            >
              {m.role === "user" ? (
                <div className="bg-gray-800 text-white rounded-2xl rounded-tr-sm px-5 py-3 max-w-[85%] md:max-w-[75%] shadow-sm text-[15px] leading-relaxed">
                  {m.content}
                </div>
              ) : (
                <div className="flex flex-col w-full max-w-[90%]">
                  <article
                    className="prose prose-sm md:prose-base prose-neutral max-w-none text-gray-800"
                    onClick={() => m.citations && setActiveCitations(m.citations)}
                  >
                    <ReactMarkdown>{m.content}</ReactMarkdown>
                  </article>

                  {m.citations && m.citations.length > 0 && (
                    <button
                      onClick={() => setActiveCitations(m.citations!)}
                      className="mt-3 flex items-center gap-1.5 text-xs font-medium text-neutral-500 hover:text-gray-800 transition-colors bg-neutral-100 self-start px-2.5 py-1 rounded-md cursor-pointer"
                    >
                      <Info className="w-3 h-3" />
                      {m.citations.length} Sources Used
                    </button>
                  )}
                </div>
              )}
            </div>
          ))}

          {/* Loading Indicator */}
          {loading && (
            <div className="flex max-w-4xl mx-auto items-center gap-3 text-neutral-500 text-sm py-2">
              <Loader2 className="w-5 h-5 animate-spin text-gray-400" />
              <span className="animate-pulse">Searching BCT corpus & generating...</span>
            </div>
          )}
        </div>

        {/* --- Composer (Input Area) --- */}
        <div className="shrink-0 bg-white pt-2 pb-6 px-4 md:px-8">
          <div className="max-w-3xl mx-auto relative flex items-end gap-2 bg-white border border-gray-200 shadow-sm rounded-2xl px-3 py-2 focus-within:ring-2 focus-within:ring-gray-200 focus-within:border-gray-300 transition-all">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  submit();
                }
              }}
              rows={1}
              dir="auto"
              placeholder="Posez votre question / اطرح سؤالك"
              className="flex-1 max-h-32 resize-none bg-transparent px-2 py-2 text-[15px] text-gray-800 placeholder-gray-400 focus:outline-none"
              style={{ minHeight: "44px" }}
            />
            <button
              onClick={submit}
              disabled={loading || !input.trim()}
              className="rounded-xl bg-gray-800 hover:bg-gray-700 text-white p-2.5 mb-0.5 disabled:opacity-30 disabled:hover:bg-gray-800 transition-colors flex-shrink-0"
              aria-label="Send message"
            >
              <Send className="w-4 h-4" />
            </button>
          </div>
          <div className="text-center mt-3 text-[11px] text-neutral-400">
            AI-generated responses may contain inaccuracies. Verify with official BCT documents.
          </div>
        </div>
      </section>

      {/* --- Citations Sidebar --- */}
      <aside className="hidden lg:flex flex-col border-l border-gray-100 bg-neutral-50 overflow-hidden">
        <Citations items={activeCitations} />
      </aside>
    </div>
  );
}