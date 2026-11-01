# core/punctuation_restorer.py
#
# Mục đích: Khôi phục dấu câu và chuẩn hóa viết hoa cho transcript.
# Giải pháp mới: 
#   1. Ép viết thường toàn bộ văn bản (triệt tiêu lỗi viết hoa ngẫu hứng của Whisper).
#   2. Chạy Rule-based (bù đắp dấu phẩy, chấm nếu Acoustic Punctuation bỏ sót).
#   3. Viết hoa chuẩn chỉ ở đầu câu (sau dấu chấm, phẩy).
#   4. Dùng Dictionary để khôi phục lại tên riêng, thuật ngữ (Domain-specific).

import re

# ────────────────────────────────────────────────────────────────────────────
# TỪ ĐIỂN TÊN RIÊNG & THUẬT NGỮ (DOMAIN-SPECIFIC)
# ────────────────────────────────────────────────────────────────────────────
_PROPER_NOUNS = [
    # Tên địa danh / Tổ chức
    "hà nam", "đồng văn", "nga", "ukraine", "hà lan", "thiên ân",
    
    # Tên người (đại biểu, đối tượng)
    "hoàng anh tuấn", "lê thị hồng", "phạm đức", "tuấn", "hồng", "trang", "khương", "dương", "đức"
]

_ACRONYMS = [
    # Thuật ngữ IT / An ninh mạng
    "cve", "aes", "rsa", "vpn", "ids", "ips", "irp", 
    "tb", "gb", "lockbit", "ransomware", "backdoor",
    "stt", "tts", "asr", "ai", "vnpt", "fpt", "ui"
]

# ────────────────────────────────────────────────────────────────────────────
# HẰNG SỐ — TỪ ĐIỂN TỪ KHOÁ TIẾNG VIỆT (DỰ PHÒNG CHO RULE-BASED)
# ────────────────────────────────────────────────────────────────────────────
_QUESTION_STARTERS = (
    r"(?:tại sao|vì sao|tại vì sao|lý do gì|lý do nào)"
    r"|(?:như thế nào|thế nào|ra sao|thế ra|vậy thì)"
    r"|(?:ở đâu|đâu|khi nào|bao giờ|ai|cái gì|bao nhiêu|đúng không|phải không)"
)

_CLAUSE_CONJUNCTIONS = [
    "nhưng mà", "nhưng", "tuy nhiên", "tuy vậy", "thế nhưng",
    "vì vậy", "vì thế", "do đó", "cho nên",
    "bởi vì", "bởi thế", "bởi vậy",
    "mặc dù", "dù vậy", "dù sao",
    "thế mà", "vậy mà", "đồng thời", "hơn nữa", "ngoài ra"
]

_SENTENCE_STARTERS = [
    "thưa", "vâng", "dạ", "đúng rồi", "đúng vậy",
    "thật ra", "thực ra", "theo tôi", "tôi nghĩ",
    "ví dụ", "chẳng hạn", "đầu tiên", "cuối cùng",
    "một là", "hai là", "ba là", "bốn là", 
    "thứ nhất", "thứ hai", "thứ ba", "thứ tư",
    "tiếp theo", "báo cáo", "kết luận"
]


# ────────────────────────────────────────────────────────────────────────────
# XỬ LÝ LOGIC (RULE-BASED & CAPITALIZATION)
# ────────────────────────────────────────────────────────────────────────────
def _normalize_spaces(text: str) -> str:
    """Chuẩn hóa khoảng trắng thừa."""
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def _fix_existing_punctuation(text: str) -> str:
    """Sửa dấu câu đã có nhưng đặt sai vị trí."""
    text = re.sub(r"\s+([,\.?!;:])", r"\1", text)
    text = re.sub(r"([,\.?!;:])([^\s\d\"\'\.])", r"\1 \2", text) # Thêm \. để tránh phá hỏng dấu ...
    text = re.sub(r"\.\s*,", ".", text)
    text = re.sub(r"\.{2,}", ".", text)
    return text

def _capitalize_sentences(text: str) -> str:
    """Viết hoa chữ đầu câu sau dấu chấm, chấm hỏi, chấm than."""
    if not text:
        return text
    
    # Viết hoa chữ đầu tiên của đoạn
    text = text[0].upper() + text[1:]

    # Viết hoa sau các dấu kết câu
    def _cap(m):
        return m.group(1) + " " + m.group(2).upper()

    text = re.sub(r"([\.?!])\s+([a-zàáâãèéêìíòóôõùúýăđơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỷỹ])", 
                  _cap, text)
    return text

def _add_question_marks(text: str) -> str:
    """Thêm '?' vào cuối câu hỏi nếu đang thiếu."""
    sentences = re.split(r"(?<=[\.?!])\s+", text)
    result = []
    question_re = re.compile(r"^(?:" + _QUESTION_STARTERS + r")\b", flags=re.IGNORECASE | re.UNICODE)

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        if (question_re.match(sent) and not sent.endswith("?") and len(sent.split()) >= 4):
            sent = sent.rstrip(".") + "?"
        result.append(sent)

    return " ".join(result)

def _add_commas_before_conjunctions(text: str) -> str:
    """Thêm dấu phẩy trước liên từ nối mệnh đề nếu chưa có."""
    for conj in _CLAUSE_CONJUNCTIONS:
        pattern = r"(\w[\w\s]{8,}?)(?<![,\.?!])\s+(" + re.escape(conj) + r"\b)"
        text = re.sub(pattern, r"\1, \2", text, flags=re.IGNORECASE | re.UNICODE)
    return text

def _split_run_on_sentences(text: str) -> str:
    """Tách các câu dài bị dính liền nhau dựa trên từ khóa bắt đầu."""
    for starter in _SENTENCE_STARTERS:
        pattern = r"(\w(?:[^\.?!\n]){20,}?)\s+(?<![\.?!,])(" + re.escape(starter) + r"\b)"
        text = re.sub(pattern, 
                      lambda m: m.group(1).rstrip() + ". " + m.group(2).capitalize(), 
                      text, flags=re.IGNORECASE | re.UNICODE)
    return text

def _capitalize_proper_nouns(text: str) -> str:
    """Khôi phục lại tên riêng và thuật ngữ sau khi đã bị ép viết thường toàn bộ."""
    # Viết hoa chữ cái đầu cho tên riêng (vd: hà nam -> Hà Nam)
    for noun in _PROPER_NOUNS:
        pattern = re.compile(r'\b' + re.escape(noun) + r'\b', re.IGNORECASE | re.UNICODE)
        text = pattern.sub(lambda m: m.group().title(), text)
        
    # Viết hoa toàn bộ cho thuật ngữ viết tắt (vd: cve -> CVE)
    for acr in _ACRONYMS:
        pattern = re.compile(r'\b' + re.escape(acr) + r'\b', re.IGNORECASE | re.UNICODE)
        text = pattern.sub(lambda m: m.group().upper(), text)
        
    # Xóa lỗi lặp từ ảo giác của Whisper (vd: "cà phê cà phê" -> "cà phê")
    text = re.sub(r"\b(\w{3,})\s+\1\b", r"\1", text, flags=re.IGNORECASE | re.UNICODE)
    
    return text


# ────────────────────────────────────────────────────────────────────────────
# HÀM CHÍNH: restore()
# ────────────────────────────────────────────────────────────────────────────
def restore(text: str) -> str:
    """
    Hàm chính khôi phục dấu câu cho transcript.
    """
    if not text or not text.strip():
        return text

    # 1. ÉP VIẾT THƯỜNG TOÀN BỘ (Sát thủ diệt lỗi viết hoa vô tội vạ của Whisper)
    text = text.lower()

    # 2. Xóa khoảng trắng thừa
    text = _normalize_spaces(text)

    # 3. Chạy màng lọc Rule-based (Bù đắp cho Acoustic Punctuation nếu cần)
    text = _add_commas_before_conjunctions(text)
    text = _add_question_marks(text)
    text = _split_run_on_sentences(text)
    text = _fix_existing_punctuation(text)

    # 4. Viết hoa đầu câu (dựa trên dấu chấm, phẩy, hỏi chấm)
    text = _capitalize_sentences(text)

    # 5. Trả lại tên riêng & thuật ngữ bằng Dictionary
    text = _capitalize_proper_nouns(text)

    return _normalize_spaces(text)

# ────────────────────────────────────────────────────────────────────────────
# TEST ĐỘC LẬP
# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_text = "TẠI SAO lại là CVE hai nghìn Không trăm hai sáu Lỗ hổng này ở trung tâm Dữ Liệu tỉnh hà nam đã Bị Khai thác bởi ransomware lockbit"
    print("INPUT :", test_text)
    print("OUTPUT:", restore(test_text))