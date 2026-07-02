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

# Each subfolder of PDF_DIR is one "track" with its own chunking strategy.
TRACKS = ("laws", "circulars", "rulebooks", "licensing")

# --- Ollama -----------------------------------------------------------------
OLLAMA_URL = "http://localhost:11434"
LLM_MODEL = "qwen2.5:3b-instruct-q4_K_M"      # ~2.5 GB VRAM
EMBED_MODEL = "bge-m3"                         # bilingual FR/AR dense embeddings

# --- Retrieval --------------------------------------------------------------
CHUNK_SIZE = 900            # characters; bge-m3 handles up to 8k tokens easily
CHUNK_OVERLAP = 150
DENSE_TOP_K = 20            # candidates from vector search
SPARSE_TOP_K = 20           # candidates from BM25
FINAL_TOP_K = 5             # chunks injected into LLM context
RRF_K = 60                  # standard RRF constant

# --- OCR --------------------------------------------------------------------
# Tesseract languages: French + Arabic. Install with:
#   sudo apt install tesseract-ocr-fra tesseract-ocr-ara
OCR_LANGS = "fra+ara"
OCR_MIN_TEXT_RATIO = 0.05   # if extracted text < this ratio of page area, OCR
