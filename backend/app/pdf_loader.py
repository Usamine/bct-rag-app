# ---------------------------------------------------------------------------
# PDF text extraction with OCR fallback.
# Strategy: PyMuPDF first (fast, accurate for born-digital PDFs).
# If a page returns too little text, fall back to Tesseract OCR
# (covers scanned BCT legacy laws/circulars).
# ---------------------------------------------------------------------------
import fitz                       # PyMuPDF
import pytesseract
from PIL import Image
import io
from typing import List, Dict
from . import config


def _page_needs_ocr(text: str, page_area_chars: int = 500) -> bool:
    """Heuristic: very short extracted text on a normal-sized page → scanned."""
    return len(text.strip()) < page_area_chars


def _ocr_page(page) -> str:
    """Render the page to an image and run Tesseract (fra+ara)."""
    pix = page.get_pixmap(dpi=300)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    return pytesseract.image_to_string(img, lang=config.OCR_LANGS)


def load_pdf(path: str) -> List[Dict]:
    """
    Open a PDF and return a list of page dicts:
        { "page": int, "text": str, "source": str }
    Pages with no readable text trigger an OCR pass automatically.
    """
    pages = []
    doc = fitz.open(path)
    for i, page in enumerate(doc, start=1):
        text = page.get_text("text") or ""
        if _page_needs_ocr(text):
            try:
                text = _ocr_page(page)
            except Exception as e:
                # OCR is best-effort; log and keep going.
                print(f"  [ocr-fail] {path} p.{i}: {e}")
                text = text or ""
        pages.append({"page": i, "text": text, "source": path})
    doc.close()
    return pages
