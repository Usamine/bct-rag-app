# ---------------------------------------------------------------------------
# Central configuration. Everything tunable lives here.
# Keep this file dependency-free so any module can import it cheaply.
# ---------------------------------------------------------------------------
from pathlib import Path

# --- Paths ------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent          # backend/
DATA_DIR = BASE_DIR / "data"
PDF_DIR = DATA_DIR / "pdfs"                                # source PDFs live here
INDEX_DIR = DATA_DIR / "index"                             # Chroma + BM25 persistence
BM25_PATH = INDEX_DIR / "bm25.pkl"
CHROMA_PATH = INDEX_DIR / "chroma"

# NEW — backs the idempotent/incremental ingest pipeline:
#   CORPUS_PATH   flat text+metadata copy of every chunk (no vectors), so
#                 BM25 can be rebuilt from this alone instead of needing
#                 the embedding model re-run just to get the sparse index back.
#   MANIFEST_PATH per-PDF content hash, so `python -m app.ingest` skips
#                 files that haven't changed and only re-embeds files that have.
CORPUS_PATH = INDEX_DIR / "corpus.jsonl"
MANIFEST_PATH = INDEX_DIR / "manifest.json"

# Each subfolder of PDF_DIR is one "track" with its own chunking strategy.
# Added "notes" for the Note-aux-banques family (Note_2026_15_ar, Note_2026_43_fr, ...).
TRACKS = ("laws", "circulars", "rulebooks", "licensing", "notes")

# --- Ollama -----------------------------------------------------------------
OLLAMA_URL = "http://localhost:11434"
LLM_MODEL = "qwen2.5:7b-instruct-q4_K_M"      #qwen2.5:3b-instruct-q8_0//qwen2.5:3b-instruct-q4_K_M       # upgraded from q4_K_M: better answer quality,
                                                # ~3.5-4 GB VRAM/RAM (vs ~2.5 GB for q4_K_M —
                                                # drop back to q4_K_M if that's too tight)
EMBED_MODEL = "bge-m3"                         # bilingual FR/AR dense embeddings
EMBED_BATCH = 16                               # NEW — chunks per embed_batch() call during ingest

# --- Retrieval --------------------------------------------------------------
CHUNK_SIZE = 900            # characters; bge-m3 handles up to 8k tokens easily
CHUNK_OVERLAP = 150
DENSE_TOP_K = 20            # candidates from vector search
SPARSE_TOP_K = 20           # candidates from BM25
FINAL_TOP_K = 5      # chunks injected into LLM context
RRF_K = 60                  # standard RRF constant

# NEW — these were hardcoded magic numbers inside router.py / retriever.py.
# Centralizing them here means you can tune retrieval quality without
# hunting through two different files.
DENSE_SCORE_THRESHOLD = 0.35   # below this cosine similarity, a dense hit is noise — drop it
ROUTER_THRESHOLD = 0.70        # semantic-router similarity above which a query is "chitchat"
MAX_HISTORY_TURNS = 6          # cap on past user/assistant turns sent to the LLM as context



# --- Reranking Configuration ---
USE_RERANKER = True
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Modèle très performant, léger et open-source
RERANK_TOP_K = 25  # Le nombre de documents pré-sélectionnés par le RRF à réévaluer

# --- OCR --------------------------------------------------------------------
# Tesseract languages: French + Arabic. Install with:
#   sudo apt install tesseract-ocr-fra tesseract-ocr-ara
#
# NOTE: format changed from the old pytesseract-style "fra+ara" string to a
# list. Docling's TesseractCliOcrOptions(lang=...) takes a list of language
# codes, not a "+"-joined string — pass this straight through.
OCR_LANGS = ["fra", "ara"]
OCR_MIN_TEXT_RATIO = 0.05   # if extracted text < this ratio of page area, OCR
                            # (kept for reference/back-compat; Docling makes its
                            # own per-page OCR decision — see FORCE_FULL_PAGE_OCR
                            # below if you need to override that decision globally)
FORCE_FULL_PAGE_OCR = False # NEW — set True to force OCR on every page regardless
                            # of Docling's text-layer heuristic (useful if legacy
                            # scans have a garbled, non-empty native text layer)
