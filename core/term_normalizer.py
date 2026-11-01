"""
core/term_normalizer.py — Chuẩn hoá thuật ngữ kỹ thuật/số liệu trong transcript.

Bối cảnh:
    ASR offline (Zipformer/Whisper) thường nhận dạng ĐÚNG về mặt ngữ âm nhưng
    ghi sai chính tả các thuật ngữ an ninh mạng và số liệu kỹ thuật, vì các
    chuỗi này hiếm/không xuất hiện trong dữ liệu huấn luyện:
        "hai phẩy năm tép"                    → "2,5 TB"
        "cv hai không hai sáu một hai ba bốn" → "CVE-2026-1234"
        "as hai năm sáu"                      → "AES-256"
        "no more ramson"                      → "No More Ransom"
        "dốc bít" / "lốc bít"                 → "LockBit"
        "qr bay chuẩn" / "iap chuẩn"          → "IRP chuẩn"

    Module này chạy SAU punctuation_restorer.restore() trong bước finalize.

Hai nhóm chuẩn hoá:
    1. Số liệu (số đọc, thập phân, đơn vị, mã CVE/AES) — xử lý bằng quy tắc
       trong file này vì cần TÍNH TOÁN (ghép số, format), không chỉ thay thế.
    2. Thuật ngữ/tên riêng (LockBit, VPN, IRP, ...) — nạp từ
       core/data/domain_terms.json. Để THÊM THUẬT NGỮ MỚI, chỉ cần sửa file
       JSON đó (xem hướng dẫn format trong file), KHÔNG cần sửa code này.

Hook mở rộng:
    `llm_fixup` — callable tuỳ chọn, được gọi SAU các bước trên với text đã
    chuẩn hoá một phần, để xử lý thêm các thuật ngữ chưa có trong lexicon
    (vd: mã CVE mới, tên sản phẩm chưa khai báo). Mặc định không bật.
"""
from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from typing import Callable, Optional

_DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "domains")
_DOMAIN_TERMS_PATH = os.path.join(_DATA_DIR, "domain_terms.json")


# ──────────────────────────────────────────────────────────────────────────
# 1) Số đọc tiếng Việt → chữ số
# ──────────────────────────────────────────────────────────────────────────
_DIGIT_WORDS = {
    "không": "0",
    "một": "1",
    "hai": "2",
    "ba": "3",
    "bốn": "4",
    "tư": "4",
    "năm": "5",
    "lăm": "5",
    "sáu": "6",
    "bảy": "7",
    "bẩy": "7",
    "tám": "8",
    "chín": "9",
}
_DIGIT_WORD_RE = "|".join(sorted(_DIGIT_WORDS, key=len, reverse=True))


def _digit_seq_to_str(seq_text: str) -> str:
    """'hai không hai sáu' -> '2026'"""
    return "".join(_DIGIT_WORDS[w] for w in seq_text.lower().split())


# "<digit> phẩy <digit>" -> "<digit>,<digit>"  (vd: "hai phẩy năm" -> "2,5")
_DECIMAL_RE = re.compile(
    rf"\b({_DIGIT_WORD_RE})\s+phẩy\s+({_DIGIT_WORD_RE})\b",
    flags=re.IGNORECASE | re.UNICODE,
)


def _normalize_decimals(text: str) -> str:
    def _repl(m: "re.Match[str]") -> str:
        return f"{_DIGIT_WORDS[m.group(1).lower()]},{_DIGIT_WORDS[m.group(2).lower()]}"

    return _DECIMAL_RE.sub(_repl, text)


# ──────────────────────────────────────────────────────────────────────────
# 2) Đơn vị dung lượng: "<số> tép/ti bi/tê bê" -> "<số> TB", v.v.
# ──────────────────────────────────────────────────────────────────────────
_UNIT_MAP = {
    "tép": "TB",
    "ti bi": "TB",
    "tê bê": "TB",
    "ghi ga": "GB",
    "ghi ga bai": "GB",
    "mê ga": "MB",
    "mê ga bai": "MB",
}
_UNIT_RE = re.compile(
    r"(\d+(?:,\d+)?)\s+("
    + "|".join(re.escape(u) for u in sorted(_UNIT_MAP, key=len, reverse=True))
    + r")\b",
    flags=re.IGNORECASE | re.UNICODE,
)


def _normalize_units(text: str) -> str:
    def _repl(m: "re.Match[str]") -> str:
        return f"{m.group(1)} {_UNIT_MAP[m.group(2).lower()]}"

    return _UNIT_RE.sub(_repl, text)


# ──────────────────────────────────────────────────────────────────────────
# 3) Mã CVE: "cv <8 số đọc>" -> "CVE-YYYY-NNNN"
# ──────────────────────────────────────────────────────────────────────────
_CVE_RE = re.compile(
    rf"\bcv\s+((?:(?:{_DIGIT_WORD_RE})\s+){{7}}(?:{_DIGIT_WORD_RE}))\b",
    flags=re.IGNORECASE | re.UNICODE,
)


def _normalize_cve(text: str) -> str:
    def _repl(m: "re.Match[str]") -> str:
        digits = _digit_seq_to_str(m.group(1))
        return f"CVE-{digits[:4]}-{digits[4:]}"

    return _CVE_RE.sub(_repl, text)


# ──────────────────────────────────────────────────────────────────────────
# 4) AES: "as/a e s <3 số đọc>" -> "AES-NNN"
# ──────────────────────────────────────────────────────────────────────────
_AES_RE = re.compile(
    rf"\b(?:as|a\s*s|a\s*e\s*s)\s+((?:(?:{_DIGIT_WORD_RE})\s+){{2}}(?:{_DIGIT_WORD_RE}))\b",
    flags=re.IGNORECASE | re.UNICODE,
)


def _normalize_aes(text: str) -> str:
    def _repl(m: "re.Match[str]") -> str:
        return f"AES-{_digit_seq_to_str(m.group(1))}"

    return _AES_RE.sub(_repl, text)


# ──────────────────────────────────────────────────────────────────────────
# 5) Lexicon thuật ngữ/tên riêng — nạp từ core/data/domain_terms.json
# ──────────────────────────────────────────────────────────────────────────
def _phrase_to_pattern(phrase: str) -> str:
    """
    'lốc bít' -> r'\\blốc\\s*bít\\b'

    Mỗi từ trong cụm được escape riêng và nối bằng \\s* (cho phép ASR tách/
    dính âm tiết khác nhau), bọc \\b ở 2 đầu để khớp nguyên cụm.
    """
    words = phrase.strip().split()
    escaped = r"\s*".join(re.escape(w) for w in words)
    return rf"\b{escaped}\b"


@lru_cache(maxsize=10) # Tăng cache để lưu được nhiều chủ đề cùng lúc
def _load_domain_terms(domain_file: str) -> list[tuple["re.Pattern[str]", str]]:
    """Đọc và biên dịch regex dựa trên tên file được chọn từ UI."""
    if not domain_file or domain_file == "Không dùng":
        return []

    filepath = os.path.join(_DATA_DIR, domain_file)
    compiled: list[tuple["re.Pattern[str]", str]] = []
    
    if not os.path.isfile(filepath):
        return compiled

    with open(filepath, "r", encoding="utf-8") as f:
        entries = json.load(f)

    for entry in entries:
        canonical = entry["canonical"]
        for variant in entry.get("variants", []):
            pattern = variant[3:] if variant.startswith("re:") else _phrase_to_pattern(variant)
            compiled.append((re.compile(pattern, flags=re.IGNORECASE | re.UNICODE), canonical))

    return compiled

def reload_domain_terms() -> None:
    _load_domain_terms.cache_clear()


def list_domain_terms() -> list[tuple[str, str]]:
    """Trả về [(pattern_str, canonical), ...] — phục vụ debug/kiểm tra lexicon."""
    return [(pat.pattern, canonical) for pat, canonical in _load_domain_terms()]


def load_raw_domain_terms() -> list[dict]:
    """
    Đọc domain_terms.json dạng thô (chưa biên dịch regex), dùng cho UI quản lý
    thuật ngữ (trang Cài đặt): [{"canonical": "...", "variants": [...]}, ...]
    """
    if not os.path.isfile(_DOMAIN_TERMS_PATH):
        return []
    with open(_DOMAIN_TERMS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_raw_domain_terms(entries: list[dict]) -> None:
    """Ghi `entries` ra domain_terms.json và xoá cache để áp dụng ngay (hot-reload)."""
    os.makedirs(_DATA_DIR, exist_ok=True)
    with open(_DOMAIN_TERMS_PATH, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
    reload_domain_terms()


# ──────────────────────────────────────────────────────────────────────────
# 6) Số đọc ghép tiếng Việt → số nguyên (10–31 cho ngày, 1–12 cho tháng)
# ──────────────────────────────────────────────────────────────────────────
_TENS = {
    "mười": 10, "hai mươi": 20, "ba mươi": 30,
}
_UNITS_NUM = {
    "một": 1, "mốt": 1, "hai": 2, "ba": 3, "bốn": 4, "tư": 4,
    "năm": 5, "lăm": 5, "sáu": 6, "bảy": 7, "bẩy": 7,
    "tám": 8, "chín": 9,
}
# "mùng X" = ngày 1–9 đầu tháng
_MUNG_RE = re.compile(
    r"\bmùng\s+(một|hai|ba|bốn|tư|năm|lăm|sáu|bảy|bẩy|tám|chín)\b",
    flags=re.IGNORECASE | re.UNICODE,
)

_COLLOQUIAL_NUMS = {
    "hai mốt": 21, "hai hai": 22, "hai ba": 23, "hai tư": 24, "hai lăm": 25,
    "hai sáu": 26, "hai bảy": 27, "hai bẩy": 27, "hai tám": 28, "hai chín": 29,
    "ba mốt": 31
}

def _read_vn_number(text: str) -> int | None:
    """
    Chuyển cụm số đọc tiếng Việt thành int.
    Hỗ trợ: 1–9 (đơn vị), mười, mười một … mười chín,
             hai mươi … hai mươi chín, ba mươi, ba mươi mốt và các số đọc tắt.
    """
    text = text.strip().lower()

    if text in _COLLOQUIAL_NUMS:
        return _COLLOQUIAL_NUMS[text]

    # mùng X đã được xử lý riêng bên ngoài
    for tens_str, tens_val in sorted(_TENS.items(), key=lambda x: -len(x[0])):
        if text.startswith(tens_str):
            rest = text[len(tens_str):].strip()
            if not rest:
                return tens_val
            unit = _UNITS_NUM.get(rest)
            if unit:
                return tens_val + unit
            return None

    # đơn vị 1–9
    return _UNITS_NUM.get(text)


# ──────────────────────────────────────────────────────────────────────────
# 7) Ngày tháng năm & giờ phút
# ──────────────────────────────────────────────────────────────────────────

# Các từ số đọc dùng trong ngày/tháng (đơn và ghép)
_NUM_WORDS = (
    r"mùng\s+\w+"                          # mùng hai, mùng chín
    r"|ba\s+mươi\s+mốt|ba\s+mốt"           # 31, 30
    r"|ba\s+mươi"
    r"|hai\s+mươi\s+(?:mốt|hai|ba|bốn|tư|năm|lăm|sáu|bảy|bẩy|tám|chín)"  # 21–29
    r"|hai\s+mươi"                         # 20
    r"|hai\s+(?:mốt|hai|ba|tư|lăm|sáu|bảy|bẩy|tám|chín)"                 # 21-29 đọc tắt (hai tư, hai lăm)
    r"|mười\s+(?:một|hai|ba|bốn|tư|năm|lăm|sáu|bảy|bẩy|tám|chín)"        # 11–19
    r"|mười"                               # 10
    r"|(?:một|hai|ba|bốn|tư|năm|lăm|sáu|bảy|bẩy|tám|chín)"               # 1–9
)

# Năm 4 chữ số đọc: "hai không hai sáu" → 2026
_YEAR_WORDS = (
    rf"(?:(?:{_DIGIT_WORD_RE})\s+){{3}}(?:{_DIGIT_WORD_RE})"
)

# Pattern: [ngày [mai]] <N> tháng <M> [năm <YYYY>]
_DATE_RE = re.compile(
    rf"\b(ngày\s+(?:mai\s+)?)({_NUM_WORDS})\s+tháng\s+({_NUM_WORDS})"
    rf"(?:\s+năm\s+({_YEAR_WORDS}))?",
    flags=re.IGNORECASE | re.UNICODE,
)

_DATE_PATCH_RE = re.compile(
    rf"\b(ngày\s+(?:mai\s+)?\d{{1,2}}/\d{{2}})\s+({_YEAR_WORDS})\b",
    flags=re.IGNORECASE | re.UNICODE,
)

def _patch_partial_dates(text: str) -> str:
    def _repl(m: re.Match) -> str:
        year_str = _digit_seq_to_str(m.group(2))
        return f"{m.group(1)}/{year_str}"
    return _DATE_PATCH_RE.sub(_repl, text)

# Pattern: giờ — "mười bảy giờ" hoặc "mười bảy giờ ba mươi phút"
_TIME_RE = re.compile(
    rf"\b({_NUM_WORDS})\s+giờ(?:\s+({_NUM_WORDS})\s+phút)?",
    flags=re.IGNORECASE | re.UNICODE,
)

def _parse_day_month(text: str) -> int | None:
    """Xử lý cả 'mùng hai' lẫn số ghép thông thường."""
    m = _MUNG_RE.match(text.strip())
    if m:
        return _UNITS_NUM.get(m.group(1).lower())
    return _read_vn_number(text)

def _normalize_datetime(text: str) -> str:
    # Giờ phút trước (để tránh "mười bảy" trong ngày bị nuốt)
    def _repl_time(m: re.Match) -> str:
        h = _read_vn_number(m.group(1))
        if h is None:
            return m.group(0)
        minute_text = m.group(2)
        mi = _read_vn_number(minute_text) if minute_text else 0
        if mi is None:
            mi = 0
        return f"{h:02d}:{mi:02d}"

    text = _TIME_RE.sub(_repl_time, text)

    # Ngày tháng [năm]
    def _repl_date(m: re.Match) -> str:
        prefix = m.group(1) # Lấy chữ "ngày " hoặc "ngày mai "
        day = _parse_day_month(m.group(2))
        month = _parse_day_month(m.group(3))
        year_text = m.group(4)

        if day is None or month is None:
            return m.group(0)

        prefix_clean = " ".join(prefix.strip().split()) # Chuẩn hóa khoảng trắng: "ngày" hoặc "ngày mai"

        if year_text:
            year_str = _digit_seq_to_str(year_text)
            return f"{prefix_clean} {day:02d}/{month:02d}/{year_str}"
        else:
            return f"{prefix_clean} {day:02d}/{month:02d}"

    text = _DATE_RE.sub(_repl_date, text)
    text = _patch_partial_dates(text)
    return text

# ──────────────────────────────────────────────────────────────────────────
# Hàm chính
# ──────────────────────────────────────────────────────────────────────────
def normalize_terms(text: str, domain_file: str = None, llm_fixup: Optional[Callable[[str], str]] = None) -> str:
    if not text or not text.strip():
        return text

    text = _normalize_decimals(text)
    text = _normalize_cve(text)
    text = _normalize_aes(text)
    text = _normalize_units(text)
    text = _normalize_datetime(text)

    # Nếu có chọn chủ đề từ UI, nạp file JSON đó và quét
    if domain_file and domain_file != "Không dùng":
        for pattern, canonical in _load_domain_terms(domain_file):
            text = pattern.sub(canonical, text)

    if llm_fixup is not None:
        text = llm_fixup(text)

    return text


# ──────────────────────────────────────────────────────────────────────────
# TEST
# ──────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    samples = [
        "ổ cứng có dung lượng hai phẩy năm tép bị mã hoá",
        "lỗ hổng cv hai không hai sáu một hai ba bốn đã được vá",
        "dữ liệu được mã hoá bằng as hai năm sáu",
        "hãy truy cập no more ramson để giải mã",
        "đây là cuộc tấn công của dốc bít",
        "công ty chưa có qr bay chuẩn để xử lý sự cố",
        "đơn vị đã triển khai vi pi en và ai đi ét để giám sát",
    ]
    for s in samples:
        print("IN :", s)
        print("OUT:", normalize_terms(s))
        print()
