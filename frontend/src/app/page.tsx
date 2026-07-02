import Image from "next/image";
import Chat from "@/components/Chat";

export default function Page() {
  return (
    <main className="h-screen flex flex-col bg-neutral-50 overflow-hidden">
      <header className="border-b bg-white px-6 py-4 flex items-center gap-4 shrink-0 shadow-sm">
        <Image
          src="/Central_Bank_of_Tunisia.png"
          alt="BCT Logo"
          width={120}
          height={40}
          className="h-10 w-auto object-contain"
          priority
        />
        <div>
          <h1 className="text-xl font-bold text-gray-800">BCT Compliance Assistant</h1>
          <p className="text-xs text-neutral-500 font-medium mt-0.5">
            Local, air-gapped · Ollama qwen2.5:3b · bge-m3 · Hybrid RRF
          </p>
        </div>
      </header>

      <Chat />
    </main>
  );
}