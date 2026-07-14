# ---------------------------------------------------------------------------
# Track-aware chunking.
# Each BCT document family has different structural cues. We dispatch on the
# `track` argument and fall back to a generic sliding-window splitter.
#
# Every chunk carries:
#   - a deterministic id (hash of source+track+marker+text), so re-running
#     ingest on an unchanged PDF produces the exact same ids and indexer.py
#     can `upsert` instead of blindly re-adding/duplicating,
#   - accurate page numbers even for multi-page circulars/rulebooks (all
#     splitters share the same offset->page mapping — previously only
#     _split_laws had this, everything else hardcoded page 1),
#   - an `embedding_text` that prepends the document name + structural
#     marker, so the embedding model sees "Circulaire 2026-05 — Article 2"
#     instead of a bare floating paragraph. `text` (used for citations and
#     what's shown to the user) is left untouched.
# ---------------------------------------------------------------------------
import hashlib
import re
from typing import Callable, Dict, List, Tuple

from . import config

# Human labels used only inside embedding_text, purely for context.
_TRACK_LABELS = {
    "laws": "Loi",
    "circulars": "Circulaire",
    "rulebooks": "Réglementation",
    "licensing": "Décision d'agrément",
    "notes": "Note aux banques",
}


# --- generic sliding window -------------------------------------------------
def _window_split_with_offsets(text: str, size: int, overlap: int) -> List[Tuple[str, int]]:
    """
    Greedy sliding window on characters, snapping to whitespace where
    possible. Returns (piece, start_offset_within_text) so callers can map
    each piece back to an absolute position in a larger string (and from
    there, to a page number) instead of losing that information.
    """
    chunks, i, n = [], 0, len(text)
    while i < n:
        end = min(i + size, n)
        if end < n:
            j = text.rfind(" ", max(i + size - 80, i), end)
            if j != -1:
                end = j
        piece = text[i:end]
        stripped = piece.strip()
        if stripped:
            lstrip_amount = len(piece) - len(piece.lstrip())
            chunks.append((stripped, i + lstrip_amount))
        i = end - overlap if end - overlap > i else end
    return chunks


def _window_split(text: str, size: int, overlap: int) -> List[str]:
    """Back-compat convenience wrapper when the offset isn't needed."""
    return [piece for piece, _ in _window_split_with_offsets(text, size, overlap)]


# --- shared page tracking ----------------------------------------------------
def _build_page_index(pages: List[Dict]) -> Tuple[str, Callable[[int], int]]:
    """
    Join page texts into one string for regex-splitting, and return a
    page_at(char_offset) -> page_number closure alongside it.

    Every splitter below (laws / circulars / rulebooks) uses this instead of
    hand-rolling its own offset math, and instead of hardcoding page=1.
    """
    offsets, cursor, parts = [], 0, []
    for p in pages:
        offsets.append((cursor, p["page"]))
        parts.append(p["text"])
        cursor += len(p["text"]) + 1  # +1 accounts for the "\n" joiner below

    full = "\n".join(parts)

    def page_at(pos: int) -> int:
        for start, pg in reversed(offsets):
            if pos >= start:
                return pg
        return pages[0]["page"] if pages else 1

    return full, page_at


# --- laws: split on "Article N" / "الفصل N" --------------------------------
# NOTE: Arabic ordinals are sometimes spelled out ("الفصل الأول") rather than
# numbered. The digit-only pattern below covers numbered articles reliably;
# spelled-out ordinals will fall through to the generic fallback for that
# document. Flag this to whoever owns the Arabic laws corpus if it matters.
_ARTICLE_RE = re.compile(r"(?im)^\s*(article|الفصل)\s+[\d\u0660-\u0669]+[\.\-:]?", re.MULTILINE)


def _split_laws(pages: List[Dict]) -> List[Dict]:
    """Group text by Article markers. Each article becomes one chunk."""
    full, page_at = _build_page_index(pages)
    matches = list(_ARTICLE_RE.finditer(full))
    if not matches:
        return _generic_chunks(pages, track="laws")

    out = []
    for idx, m in enumerate(matches):
        start = m.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(full)
        body = full[start:end].strip()
        if body:
            out.append({"text": body, "page": page_at(start), "marker": m.group().strip()})
    return out


# --- circulars / notes: split on document-number markers --------------------
# Bilingual, best-effort: covers "Circulaire N°2026-5" / "منشور ... عدد 6 لسنة 2026"
# and "Note ... N° 2026 - 43" / "مذكرة ... عدد 15 لسنة 2026".
# Real BCT headers put the label and the number on separate lines (see the
# uploaded Note_2026_43_fr / Note_2026_15_ar), so the gap between them must
# cross newlines — that's what [\s\S] buys over [^\n] here. Tune the bound /
# add phrasings if the wider corpus turns up other numbering styles.
_CIRC_RE = re.compile(
    r"(?is)(circulaire[\s\S]{0,80}?(?:n°|no\.?)\s*\d{4}\s*[-/]\s*\d+"
    r"|منشور[\s\S]{0,60}?عدد\s*\d+\s*لس[نا]ة\s*\d{4})"
)
_NOTE_RE = re.compile(
    r"(?is)(note[\s\S]{0,80}?(?:n°|no\.?)\s*\d{4}\s*[-\s]\s*\d+"
    r"|مذكرة[\s\S]{0,60}?عدد\s*\d+\s*لسنة\s*\d{4})"
)


def _split_by_marker(pages: List[Dict], marker_re: re.Pattern, track: str) -> List[Dict]:
    """Shared body for circulars and notes: one block per marker, window-split
    if the block is long, with page numbers carried through correctly."""
    full, page_at = _build_page_index(pages)
    matches = list(marker_re.finditer(full))
    if not matches:
        return _generic_chunks(pages, track=track)

    out = []
    for idx, m in enumerate(matches):
        start = m.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(full)
        body = full[start:end]
        for piece, rel_off in _window_split_with_offsets(body, config.CHUNK_SIZE, config.CHUNK_OVERLAP):
            out.append({
                "text": piece,
                "marker": m.group().strip(),
                "page": page_at(start + rel_off),
            })
    return out


def _split_circulars(pages: List[Dict]) -> List[Dict]:
    return _split_by_marker(pages, _CIRC_RE, "circulars")


def _split_notes(pages: List[Dict]) -> List[Dict]:
    return _split_by_marker(pages, _NOTE_RE, "notes")


# --- rulebooks: hierarchical headings (Chapitre/Section/Titre/Partie) -------
_HEAD_RE = re.compile(
    r"(?im)^\s*(chapitre|section|titre|partie|باب|فصل|قسم|عنوان)\s+[ivxlcdm\d\u0660-\u0669]+",
    re.MULTILINE,
)


def _split_rulebooks(pages: List[Dict]) -> List[Dict]:
    full, page_at = _build_page_index(pages)
    matches = list(_HEAD_RE.finditer(full))
    if not matches:
        return _generic_chunks(pages, track="rulebooks")

    out = []
    for idx, m in enumerate(matches):
        start = m.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(full)
        body = full[start:end]
        for piece, rel_off in _window_split_with_offsets(body, config.CHUNK_SIZE, config.CHUNK_OVERLAP):
            out.append({
                "text": piece,
                "marker": m.group().strip(),
                "page": page_at(start + rel_off),
            })
    return out


# --- licensing: short, self-contained decisions — keep coarse chunks --------
def _split_licensing(pages: List[Dict]) -> List[Dict]:
    return _generic_chunks(pages, track="licensing")


# --- generic fallback ------------------------------------------------------
def _generic_chunks(pages: List[Dict], track: str) -> List[Dict]:
    out = []
    for p in pages:
        for piece in _window_split(p["text"], config.CHUNK_SIZE, config.CHUNK_OVERLAP):
            out.append({"text": piece, "page": p["page"], "marker": ""})
    return out


# --- deterministic IDs + context injection ----------------------------------
def _stable_id(source: str, track: str, marker: str, text: str) -> str:
    """
    Hash-based id instead of uuid4. Same input (same PDF, same chunk
    boundaries) always yields the same id, which is what lets indexer.py
    `upsert` safely — re-running ingest on an untouched PDF is then a no-op
    instead of duplicating or re-embedding everything.
    """
    h = hashlib.sha256()
    for part in (source, track, marker, text):
        h.update(part.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def _make_embedding_text(source: str, track: str, marker: str, text: str) -> str:
    """
    Prepend document identity + structural marker so the embedding model
    (and BM25 tokenizer) see context instead of a bare floating paragraph —
    e.g. "[Circulaire — Cir_2026_05_fr.pdf — circulaire n°2026-5]\n<text>".
    The original `text` field is left untouched for display/citation.
    """
    label = _TRACK_LABELS.get(track, track)
    header_bits = [label, source]
    if marker:
        header_bits.append(marker)
    header = " — ".join(header_bits)
    return f"[{header}]\n{text}"


# --- public entrypoint -----------------------------------------------------
_DISPATCH = {
    "laws": _split_laws,
    "circulars": _split_circulars,
    "rulebooks": _split_rulebooks,
    "licensing": _split_licensing,
    "notes": _split_notes,
}


def chunk_document(pages: List[Dict], track: str, source: str) -> List[Dict]:
    """
    Return chunks enriched with a stable id, the original text, a
    context-injected embedding_text, and citation metadata.
    """
    splitter = _DISPATCH.get(track, lambda pgs: _generic_chunks(pgs, track))
    raw = splitter(pages)

    enriched = []
    for c in raw:
        text = c["text"].strip()
        if not text:
            continue
        marker = c.get("marker", "")
        enriched.append({
            "id": _stable_id(source, track, marker, text),
            "text": text,
            "embedding_text": _make_embedding_text(source, track, marker, text),
            "metadata": {
                "track": track,
                "source": source,
                "page": c.get("page", 1),
                "marker": marker,
            },
        })
    return enriched
