# ---------------------------------------------------------------------------
# Track-aware chunking.
# Each BCT document family has different structural cues. We dispatch on the
# `track` argument and fall back to a generic sliding-window splitter.
# Every chunk carries metadata used later for citation + scope filtering.
# ---------------------------------------------------------------------------
import re
import uuid
from typing import List, Dict
from . import config


# --- generic sliding window -------------------------------------------------
def _window_split(text: str, size: int, overlap: int) -> List[str]:
    """Greedy sliding window on characters. Tries to break on whitespace."""
    chunks, i, n = [], 0, len(text)
    while i < n:
        end = min(i + size, n)
        # snap to nearest whitespace within the last 80 chars to avoid mid-word cuts
        if end < n:
            j = text.rfind(" ", max(i + size - 80, i), end)
            if j != -1:
                end = j
        chunk = text[i:end].strip()
        if chunk:
            chunks.append(chunk)
        i = end - overlap if end - overlap > i else end
    return chunks


# --- laws: split on "Article N" --------------------------------------------
_ARTICLE_RE = re.compile(r"(?im)^\s*article\s+\d+[\.\-:]?", re.MULTILINE)


def _split_laws(pages: List[Dict]) -> List[Dict]:
    """Group pages by Article markers. Each article becomes one chunk."""
    full = "\n".join(p["text"] for p in pages)
    # Page offsets so we can map a char position back to a page number.
    offsets, cursor = [], 0
    for p in pages:
        offsets.append((cursor, p["page"]))
        cursor += len(p["text"]) + 1

    def page_at(pos: int) -> int:
        for start, pg in reversed(offsets):
            if pos >= start:
                return pg
        return 1

    matches = list(_ARTICLE_RE.finditer(full))
    out = []
    if not matches:
        return _generic_chunks(pages, track="laws")
    for idx, m in enumerate(matches):
        start = m.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(full)
        body = full[start:end].strip()
        if body:
            out.append({"text": body, "page": page_at(start),
                        "marker": m.group().strip()})
    return out


# --- circulars: split on "Circulaire" or numeric IDs like 2016-35 ----------
_CIRC_RE = re.compile(r"(?im)(circulaire\s+(?:n°|no\.?)\s*\d{4}[-/]\d+)")


def _split_circulars(pages: List[Dict]) -> List[Dict]:
    """One chunk per Circular ID block; window-split anything too long."""
    full = "\n".join(p["text"] for p in pages)
    matches = list(_CIRC_RE.finditer(full))
    if not matches:
        return _generic_chunks(pages, track="circulars")
    out = []
    for idx, m in enumerate(matches):
        start = m.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(full)
        body = full[start:end].strip()
        # window-split overlong circulars to keep chunks LLM-friendly
        for piece in _window_split(body, config.CHUNK_SIZE, config.CHUNK_OVERLAP):
            out.append({"text": piece, "marker": m.group().strip(), "page": 1})
    return out


# --- rulebooks: hierarchical headings (Chapitre/Section/Titre) -------------
_HEAD_RE = re.compile(r"(?im)^\s*(chapitre|section|titre|partie)\s+[ivxlcdm\d]+",
                      re.MULTILINE)


def _split_rulebooks(pages: List[Dict]) -> List[Dict]:
    full = "\n".join(p["text"] for p in pages)
    matches = list(_HEAD_RE.finditer(full))
    if not matches:
        return _generic_chunks(pages, track="rulebooks")
    out = []
    for idx, m in enumerate(matches):
        start = m.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(full)
        for piece in _window_split(full[start:end], config.CHUNK_SIZE, config.CHUNK_OVERLAP):
            out.append({"text": piece, "marker": m.group().strip(), "page": 1})
    return out


# --- licensing: keep whole document as one or few chunks -------------------
def _split_licensing(pages: List[Dict]) -> List[Dict]:
    """Licensing decisions are short and self-contained — keep coarse chunks."""
    return _generic_chunks(pages, track="licensing")


# --- generic fallback ------------------------------------------------------
def _generic_chunks(pages: List[Dict], track: str) -> List[Dict]:
    out = []
    for p in pages:
        for piece in _window_split(p["text"], config.CHUNK_SIZE, config.CHUNK_OVERLAP):
            out.append({"text": piece, "page": p["page"], "marker": ""})
    return out


# --- public entrypoint -----------------------------------------------------
_DISPATCH = {
    "laws": _split_laws,
    "circulars": _split_circulars,
    "rulebooks": _split_rulebooks,
    "licensing": _split_licensing,
}


def chunk_document(pages: List[Dict], track: str, source: str) -> List[Dict]:
    """Return chunks enriched with stable IDs + metadata for indexing."""
    splitter = _DISPATCH.get(track, lambda pgs: _generic_chunks(pgs, track))
    raw = splitter(pages)
    enriched = []
    for c in raw:
        if not c["text"].strip():
            continue
        enriched.append({
            "id": str(uuid.uuid4()),
            "text": c["text"],
            "metadata": {
                "track": track,
                "source": source,
                "page": c.get("page", 1),
                "marker": c.get("marker", ""),
            },
        })
    return enriched
