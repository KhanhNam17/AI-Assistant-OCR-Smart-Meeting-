"""
Filter out hallucinated phrases and foreign scripts (Chinese, Thai, Lao) from FunASR transcripts.
"""
from __future__ import annotations
import re

_CHINESE_RE = re.compile(r"[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]")
_THAI_RE = re.compile(r"[\u0E00-\u0E7F]")
_LAO_RE = re.compile(r"[\u0E80-\u0EFF]")
_BLOCKED_SCRIPT_RE = re.compile(r"[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF\u0E00-\u0E7F\u0E80-\u0EFF]")

_HALLUCINATION_PHRASES = [
    r"thank you for watching",
    r"thanks for watching",
    r"cảm ơn các bạn đã theo dõi",
    r"cảm ơn các bạn đã xem",
    r"chúc các bạn một ngày vui vẻ",
    r"hẹn gặp lại các bạn",
    r"subtitles by",
    r"amara\.org",
]
_HALLUCINATION_RE = re.compile("|".join(_HALLUCINATION_PHRASES), re.IGNORECASE)


def _blocked_script_ratio(text: str) -> float:
    visible = [c for c in text if not c.isspace()]
    if not visible:
        return 0.0
    blocked = [c for c in visible if _BLOCKED_SCRIPT_RE.search(c)]
    return len(blocked) / len(visible)


def looks_like_blocked_language(text: str, drop_threshold: float = 0.2) -> bool:
    return _blocked_script_ratio(text) >= drop_threshold


def clean_asr_text(text: str, drop_threshold: float = 0.2) -> str:
    """
    Clean ASR text by dropping hallucinated phrases and segments dominated by foreign scripts.
    Returns empty string if the segment should be discarded.
    """
    if not text:
        return ""
    text = text.strip()
    if not text:
        return ""

    if _HALLUCINATION_RE.search(text):
        text = _HALLUCINATION_RE.sub("", text).strip()

    if looks_like_blocked_language(text, drop_threshold):
        return ""

    return text
