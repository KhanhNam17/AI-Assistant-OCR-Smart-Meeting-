"""
core/text_utils.py — conservative post-OCR text cleanup.

Exposes two functions:

* :func:`dehyphenate` — rejoins words split across an OCR line break,
  e.g. ``"chuyển- động"`` → ``"chuyểnđộng"``.

* :func:`reclassify_block_type` — overrides a block's IR ``type`` using
  the OCR'd text itself (``Chương .../Điều \\d+.``), instead of trusting
  the layout model's label unconditionally.

``dehyphenate`` is intentionally cautious: it only joins when the segment
**after** the hyphen starts with a lowercase letter (a Vietnamese tonal
vowel counts as lowercase under Unicode). Compound terms like
``"đa-văn-hóa"`` and proper hyphens in numeric ranges (``"2024-2025"``)
are left alone.

When in doubt, both functions return the input unchanged.
"""

from __future__ import annotations

import logging
import re
from typing import Final

_log = logging.getLogger(__name__)


# Pattern explanation:
#   (\w+)     — preceding word (\w under re.UNICODE includes Vietnamese)
#   -         — literal hyphen
#   \s+       — one or more whitespace chars (the OCR concatenator inserts
#               a space between line crops, so newlines are already
#               normalised to spaces by the time we see them here)
#   ([a-zà-ÿ]) — first char of the continuation MUST be lowercase Latin
#               (including Vietnamese tonal vowels covered by à-ÿ +
#               extended below). Uppercase implies a new word, not a
#               continuation, so we leave it alone.
#   (\w*)     — rest of the continuation
_HYPHENATION_RE: Final[re.Pattern] = re.compile(
    r"(\w+)-\s+([a-zà-ÿăâđêôơưạ-ỹ])(\w*)",
    flags=re.UNICODE,
)


def dehyphenate(text: str) -> str:
    """
    Join words broken across an OCR line break.

    Returns the cleaned text. When the heuristic doesn't apply, returns
    the input unchanged.

    Examples
    --------
    >>> dehyphenate("chuyển- động đồng bộ")
    'chuyểnđộng đồng bộ'

    >>> dehyphenate("đa-văn-hóa")   # tight compound, no whitespace
    'đa-văn-hóa'

    >>> dehyphenate("Việt-Mỹ")      # second word capitalised: leave alone
    'Việt-Mỹ'

    >>> dehyphenate("2024-2025")    # numeric range
    '2024-2025'
    """
    if not text or "-" not in text:
        return text

    joined_count = 0

    def _join(m: re.Match) -> str:
        nonlocal joined_count
        joined_count += 1
        return m.group(1) + m.group(2) + m.group(3)

    out = _HYPHENATION_RE.sub(_join, text)
    if joined_count:
        _log.debug("dehyphenate: joined %d split words", joined_count)
    return out


# ── Structural reclassification (Chương / Điều) ─────────────────────────────
#
# Why this exists
# ----------------
# YOLO / Docling label confidence is not reliable for Vietnamese legal
# documents: a "Điều N." heading is visually just a bold line, so on a
# noisy scan it commonly gets labelled "text" / "plain text" instead of
# "title" / "section_header". When that happens the heading silently
# degrades into a regular paragraph downstream (no bold, no Word Heading
# style, no ToC entry).
#
# This function is a cheap, deterministic safety net: run it on the
# block's OCR'd ``original_text`` *after* OCR, and let the pattern match
# override whatever the layout model guessed. It never demotes a type —
# only promotes "text"-like blocks up to a heading when the wording is
# unambiguous.
#
# Deliberately tolerant of common OCR confusions:
#   * "Điều" is frequently misread as "Diều" (Đ → D) — both accepted.
#   * Roman numerals after "Chương" (I, II, III, IV, ...) or Arabic
#     digits (rare, but seen on some scans) are both accepted.

_CHUONG_RE: Final[re.Pattern] = re.compile(
    r"^\s*Ch[uư]ơng\s+[IVXLCDM\d]+", re.IGNORECASE,
)
_DIEU_RE: Final[re.Pattern] = re.compile(
    r"^\s*(?:Điều|Diều)\s+\d+\.", re.IGNORECASE,
)

# Types the layout model can plausibly mislabel a heading as. Anything
# already classified as one of these is left untouched — we only want
# to *upgrade* misclassified body text, never downgrade a caption or
# already-correct heading.
_RECLASSIFIABLE_TYPES: Final[frozenset[str]] = frozenset({
    "text", "list_item", "caption", "page_header", "page_footer",
})


def reclassify_block_type(current_type: str, text: str) -> str:
    """
    Return the IR block type, overriding ``current_type`` when ``text``
    unambiguously looks like a "Chương ..." or "Điều N." heading.

    Call this right after OCR, once ``original_text`` is known — e.g. in
    ``build_ir`` (document_pipeline.py) or ``_item_to_block``
    (docling_pipeline.py) — and use the returned value as ``IRBlock.type``
    instead of the raw layout-model label.

    Only overrides when ``current_type`` is one of :data:`_RECLASSIFIABLE_TYPES`
    (typically "text"). Blocks the layout model already tagged as
    "table", "figure", etc. are left alone — a table cell that happens to
    start with "Điều" should not be turned into a page heading.

    Examples
    --------
    >>> reclassify_block_type("text", "Điều 5. Tuyên truyền, phổ biến ...")
    'section_header'

    >>> reclassify_block_type("text", "Diều 10. Quy tắc chung")  # OCR'd "Đ"→"D"
    'section_header'

    >>> reclassify_block_type("text", "Chương II QUY TẮC GIAO THÔNG ĐƯỜNG BỘ")
    'title'

    >>> reclassify_block_type("table", "Điều 5 nói về ...")  # already a table cell
    'table'

    >>> reclassify_block_type("text", "1. Người tham gia giao thông ...")
    'text'
    """
    if current_type not in _RECLASSIFIABLE_TYPES or not text:
        return current_type

    stripped = text.strip()
    if _CHUONG_RE.match(stripped):
        return "title"
    if _DIEU_RE.match(stripped):
        return "section_header"
    return current_type


__all__ = ["dehyphenate", "reclassify_block_type"]