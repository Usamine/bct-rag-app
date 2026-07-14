# ---------------------------------------------------------------------------
# One tiny, dependency-free heuristic, shared wherever code needs to pick
# between a French and an Arabic canned string (greetings, refusals, etc).
# A full language-detection library would be overkill for a binary FR/AR
# decision — presence of Arabic-script characters is enough here.
# ---------------------------------------------------------------------------
import re

_ARABIC_RE = re.compile(r"[\u0600-\u06FF]")


def is_arabic(text: str) -> bool:
    """True if `text` contains any Arabic-script character."""
    return bool(_ARABIC_RE.search(text or ""))
