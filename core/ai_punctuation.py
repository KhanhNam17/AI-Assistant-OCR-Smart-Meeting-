# core/ai_punctuation.py
import os
import re
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForTokenClassification

class AIPunctuationRestorer:
    def __init__(self, model_path="models/xlm-roberta-capu"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"⏳ Đang nạp AI Punctuation OFFLINE từ: {model_path} lên {self.device}...")
        
        # 1. Nạp Tokenizer (Lấy online 1 lần base model để có đủ file cấu hình phân tách từ)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        except Exception:
            print("⚠️ Thiếu file Tokenizer offline, đang lấy cấu trúc xlm-roberta-base dự phòng...")
            self.tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
            
        # 2. PHẪU THUẬT CONFIG: Sửa lỗi Size Mismatch
        config = AutoConfig.from_pretrained(model_path, local_files_only=True)
        
        # Ép buộc kích thước khung xương khớp 100% với file pytorch_model.bin
        config.vocab_size = 250003
        config.max_position_embeddings = 514
        config.type_vocab_size = 1
        config.num_labels = 15
        
        # Tự động đọc danh sách nhãn từ thư mục vocabulary của bạn
        labels_path = os.path.join(model_path, "vocabulary", "labels.txt")
        if os.path.exists(labels_path):
            with open(labels_path, "r", encoding="utf-8") as f:
                labels = [line.strip() for line in f if line.strip()]
            config.id2label = {i: lbl for i, lbl in enumerate(labels)}
            config.label2id = {lbl: i for i, lbl in enumerate(labels)}
        else:
            print("⚠️ Không tìm thấy file labels.txt trong thư mục vocabulary!")

        # 3. Nạp Model với Config đã được sửa
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_path,
            config=config,
            local_files_only=True,
            trust_remote_code=True
        ).to(self.device)
        
        self.model.eval()
        self.id2label = config.id2label

    def restore(self, text: str) -> str:
        if not text or not text.strip():
            return text
            
        text = text.lower()
        words = text.split()
        if not words:
            return text
            
        encoding = self.tokenizer(words, is_split_into_words=True, return_tensors="pt", truncation=True, max_length=512)
        # Lưu word_ids TRƯỚC khi chuyển sang dict (BatchEncoding mới có method này)
        word_ids = encoding.word_ids()
        inputs = {k: v.to(self.device) for k, v in encoding.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1).squeeze().tolist()
            
        if not isinstance(predictions, list):
            predictions = [predictions]
        restored_words = []
        prev_word_idx = None
        
        for i, word_idx in enumerate(word_ids):
            if word_idx is None or word_idx == prev_word_idx:
                continue
            
            orig_word = words[word_idx]
            # Bảo vệ fallback nếu id không tồn tại
            label = self.id2label.get(predictions[i], "O").upper()
            
            if "UPPER" in label or "CAP" in label:
                orig_word = orig_word.capitalize()
            elif "ALL_CAPS" in label:
                orig_word = orig_word.upper()
                
            if "COMMA" in label or "PHAY" in label or "," in label:
                orig_word += ","
            elif "PERIOD" in label or "CHAM" in label or "." in label:
                orig_word += "."
            elif "QUESTION" in label or "HOI" in label or "?" in label:
                orig_word += "?"
                
            restored_words.append(orig_word)
            prev_word_idx = word_idx
            
        return " ".join(restored_words)


# ────────────────────────────────────────────────────────────────────────────
# BỘ LỌC ĐỊNH DẠNG (POST-PROCESSING)
# ────────────────────────────────────────────────────────────────────────────
def _polish_formatting(text: str) -> str:
    """Hàm gọt giũa lại chính tả sau khi AI đã điền dấu câu"""
    if not text:
        return text
        
    # 1. Ép viết hoa chữ cái ĐẦU TIÊN của toàn bộ đoạn văn
    text = text[0].upper() + text[1:]

    # 2. DỌN DẸP LỖI SAI NGỮ NGHĨA VÀ DẤU THỪA (PHẢI CHẠY TRƯỚC KHI VIẾT HOA)
    text = re.sub(r'\s+,', ',', text)           # Xóa khoảng trắng trước dấu phẩy
    text = re.sub(r',\s*,', ',', text)          # Xóa dấu phẩy lặp (,,)
    
    # Dọn dẹp dấu chấm vô lý bị AI chèn trước chữ "báo cáo" (Không phân biệt hoa thường)
    text = re.sub(r'(và phải|lập|và)\.\s+báo cáo', r'\1 báo cáo', text, flags=re.IGNORECASE)
    text = text.replace("ba, mốt, tháng", "ba mốt tháng")

    # Xóa dấu phẩy ảo giác chen giữa các con số/năm (vd: năm, hai, không hai, sáu -> năm hai không hai sáu)
    _num_words = r'(năm|hai|không|một|ba|bốn|năm|sáu|bảy|tám|chín|mười|mốt)'
    # Chạy lặp 2 lần để gỡ sạch các dấu phẩy dính liền nhau (như "năm, hai, không, hai")
    text = re.sub(fr'{_num_words},\s*{_num_words}', r'\1 \2', text, flags=re.IGNORECASE)
    text = re.sub(fr'{_num_words},\s*{_num_words}', r'\1 \2', text, flags=re.IGNORECASE)

    # 3. ÉP VIẾT HOA SAU DẤU CÂU KẾT THÚC (CHẠY CUỐI CÙNG)
    def _cap_after_punct(m):
        return m.group(1) + m.group(2).upper()
    
    # Regex tìm dấu chấm/hỏi/than, theo sau là khoảng trắng, rồi đến một chữ cái tiếng Việt
    text = re.sub(r'([.?!]\s+)([a-zàáâãèéêìíòóôõùúýăđơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỷỹ])', 
                  _cap_after_punct, text)

    return text

# Thiết lập Singleton
_ai_restorer = None

def restore_text(text: str) -> str:
    """Hàm gọi chính thức từ file giao diện điều khiển"""
    global _ai_restorer
    if _ai_restorer is None:
        _ai_restorer = AIPunctuationRestorer()
        
    # Bước 1: Cho AI xử lý ngữ nghĩa và dấu câu
    ai_output = _ai_restorer.restore(text)
    
    # Bước 2: Cho Code dọn dẹp định dạng sau cùng
    return _polish_formatting(ai_output)