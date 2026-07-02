# BCT Bilingual Banking Regulation RAG

Air-gapped RAG Q&A system for Tunisian Central Bank regulations.

**Stack**
- **Frontend**: Next.js 14 (App Router) — chat UI + citations panel
- **Backend**: FastAPI (Python, procedural, no classes)
- **LLM**: Ollama (`qwen2.5:3b-instruct-q4_K_M`) — runs locally
- **Embeddings**: Ollama (`bge-m3`) — bilingual FR/AR
- **Vector store**: ChromaDB (local persistence)
- **Sparse retrieval**: BM25 (`rank_bm25`)
- **Fusion**: Reciprocal Rank Fusion (RRF)

## Hardware target
- 4 GB VRAM (qwen2.5:3b q4 fits in ~2.5 GB)
- 16 GB system RAM
- Local SSD

## Setup

### 1. Install Ollama and pull models
```bash
# https://ollama.com/download
ollama pull qwen2.5:3b-instruct-q4_K_M
ollama pull bge-m3
ollama serve  # runs on http://localhost:11434
```

### 2. Backend
```bash
cd backend
python -m venv .venv && source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt

# Drop your BCT PDFs in: backend/data/pdfs/<category>/*.pdf
#   categories: laws/, circulars/, rulebooks/, licensing/
python -m app.ingest          # one-shot ingestion (OCR + chunking + indexing)

uvicorn app.main:app --reload --port 8000
```

### 3. Frontend
```bash
cd frontend
npm install
npm run dev    # http://localhost:3000
```

## Architecture

```
User → Next.js chat UI
        │
        ▼  POST /api/chat  (proxy)
   FastAPI /query
        │
        ├── embed query  (Ollama bge-m3)
        ├── dense search (ChromaDB top-k)
        ├── sparse search (BM25 top-k)
        ├── RRF fuse → top 5 chunks
        └── prompt Ollama qwen2.5:3b → answer + cite chunk IDs
```

## Files

```
backend/
  app/
    config.py        # paths, model names, constants
    ollama_client.py # thin HTTP wrapper for Ollama
    pdf_loader.py    # PyMuPDF + pdfplumber + Tesseract fallback
    chunker.py       # per-track splitters (laws/circulars/rulebooks/licensing)
    indexer.py       # builds Chroma + BM25 indexes
    retriever.py     # hybrid search + RRF
    prompt.py        # bilingual system prompt + citation format
    main.py          # FastAPI app
    ingest.py        # CLI: scan PDFs → chunk → index
frontend/
  src/app/page.tsx                 # chat page
  src/app/api/chat/route.ts        # proxies to FastAPI
  src/components/Chat.tsx          # chat UI
  src/components/Citations.tsx     # source panel
  src/lib/api.ts                   # fetch helper
```
