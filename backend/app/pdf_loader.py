# ---------------------------------------------------------------------------
# PDF -> page-level text, via PyMuPDF (fitz) ONLY.
#
# Docling has been removed entirely due to repeated out-of-memory errors
# (std::bad_alloc) on large documents. This fallback relies purely on the
# native text layer of the PDFs.
# ---------------------------------------------------------------------------
import fitz  # PyMuPDF
from typing import Dict, List

def load_pdf(path: str) -> List[Dict]:
    """
    Convert one PDF and return page-level dicts:
        { "page": int, "text": str, "source": str }
        
    Uses pure PyMuPDF to extract native text. Extremely fast and memory
    efficient. Skips pages/documents that are purely scanned images.
    """
    pages = []
    try:
        # Open the PDF using PyMuPDF
        doc = fitz.open(path)
        
        # Iterate through every page
        for i, page in enumerate(doc, start=1):
            text = page.get_text("text") or ""
            pages.append({"page": i, "text": text.strip(), "source": path})
            
        doc.close()
        
    except Exception as e:
        print(f"  [error] Failed to read {path}: {e}")
        return [{"page": 1, "text": "", "source": path}]
        
    # Check if the entire document was basically blank (likely a scanned image)
    all_text = "".join([p["text"] for p in pages])
    if len(all_text.strip()) < 50:
        print(f"  [warn] {path}: No native text found (likely a scan). Skipping.")
        
    return pages