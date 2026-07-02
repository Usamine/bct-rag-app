import type { Citation } from "@/lib/api";
import { FileText, ChevronRight } from "lucide-react";

export default function Citations({ items }: { items: Citation[] }) {
  if (!items || items.length === 0) {
    return (
      <div className="flex items-center justify-center h-full p-6 text-sm text-neutral-400 text-center">
        Source chunks will appear here after each answer.
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      <div className="px-5 py-4 border-b border-gray-200 bg-white shadow-sm z-10 sticky top-0">
        <h2 className="text-xs font-bold uppercase tracking-wider text-gray-800 flex items-center gap-2">
          Retrieved Sources
          <span className="bg-gray-100 text-gray-600 py-0.5 px-2 rounded-full text-[10px]">
            {items.length}
          </span>
        </h2>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {items.map((c, i) => (
          <div
            key={c.id}
            className="rounded-xl border border-gray-200 bg-white p-4 shadow-sm hover:shadow-md transition-shadow"
          >
            <div className="flex items-start gap-2 mb-2">
              <FileText className="w-4 h-4 text-blue-600 mt-0.5 flex-shrink-0" />
              <div className="flex flex-wrap items-center gap-1.5 text-xs text-neutral-600 font-medium leading-tight">
                <span className="text-gray-900">[{i + 1}]</span>
                <ChevronRight className="w-3 h-3 text-gray-300" />
                <span className="uppercase text-[10px] tracking-wide text-blue-600 bg-blue-50 px-1.5 py-0.5 rounded">
                  {c.metadata.track}
                </span>
                <ChevronRight className="w-3 h-3 text-gray-300" />
                <span className="truncate max-w-[140px]" title={c.metadata.source}>
                  {c.metadata.source}
                </span>
                <span className="text-gray-400">· p.{c.metadata.page}</span>
              </div>
            </div>

            {c.metadata.marker && (
              <div className="text-[11px] text-gray-500 italic mb-2 bg-gray-50 px-2 py-1 rounded border border-gray-100 inline-block">
                {c.metadata.marker}
              </div>
            )}

            <p className="text-[13px] text-gray-700 leading-relaxed line-clamp-6 whitespace-pre-wrap">
              {c.text}
            </p>

            <div className="flex justify-end mt-3 border-t border-gray-50 pt-2">
              <span className="text-[10px] text-gray-400 font-mono" title="Reciprocal Rank Fusion Score">
                RRF Score: {c.score.toFixed(4)}
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}