"""
text_metrics.py
===============

Module tính toán các chỉ số độ chính xác nội dung văn bản (CER, WER).
Bao gồm bộ tiền xử lý chuẩn hóa Unicode chuyên dụng cho tiếng Việt.
"""

import unicodedata
import jiwer
import json
from pathlib import Path

def normalize_text(text: str, lower: bool = False) -> str:
    """
    Tiền xử lý và chuẩn hóa chuỗi văn bản.
    Đưa các ký tự tiếng Việt dạng tổ hợp (NFD) về dạng dựng sẵn (NFC).
    """
    if not isinstance(text, str):
        return ""
        
    # 1. Chuẩn hóa Unicode về NFC (cực kỳ quan trọng cho tiếng Việt)
    norm_text = unicodedata.normalize('NFC', text)
    
    # 2. Xóa khoảng trắng thừa (trailing/leading và multiple spaces)
    norm_text = " ".join(norm_text.split())
    
    # 3. Đưa về chữ thường nếu cần
    if lower:
        norm_text = norm_text.lower()
        
    return norm_text

def calculate_cer(ground_truth: str, prediction: str, ignore_case: bool = False) -> float:
    """
    Tính toán Character Error Rate (CER).
    """
    gt_clean = normalize_text(ground_truth, lower=ignore_case)
    pred_clean = normalize_text(prediction, lower=ignore_case)
    
    if len(gt_clean) == 0:
        return 1.0 if len(pred_clean) > 0 else 0.0
        
    # Tách từng chữ cái bằng khoảng trắng để jiwer tính CER
    gt_chars = " ".join(list(gt_clean))
    pred_chars = " ".join(list(pred_clean))
    
    return jiwer.wer(gt_chars, pred_chars)

def calculate_wer(ground_truth: str, prediction: str, ignore_case: bool = True) -> float:
    """
    Tính toán Word Error Rate (WER).
    """
    gt_clean = normalize_text(ground_truth, lower=ignore_case)
    pred_clean = normalize_text(prediction, lower=ignore_case)
    
    if len(gt_clean) == 0:
        return 1.0 if len(pred_clean) > 0 else 0.0
        
    return jiwer.wer(gt_clean, pred_clean)


# ==============================================================================
# PHẦN TEST VỚI DỮ LIỆU THẬT
# ==============================================================================
def extract_gt_text_from_json(json_path: Path) -> str:
    """Hàm phụ trợ để nối các block text từ file JSON Ground Truth."""
    if not json_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file GT: {json_path}")
        
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    full_text = ""
    for page in data.get("pages", []):
        # Sắp xếp các block theo thứ tự order
        sorted_blocks = sorted(page.get("blocks", []), key=lambda x: x.get("order", 0))
        for block in sorted_blocks:
            # Chỉ lấy nội dung từ các block text hoặc bảng biểu
            if block.get("content_kind") in ["text", "table_html"]:
                full_text += block.get("content", "") + "\n\n"
                
    return full_text.strip()

if __name__ == "__main__":
    print("=== TEST TEXT METRICS VỚI FILE THỰC TẾ ===")
    
    # Cấu hình đường dẫn (Điều chỉnh lại nếu thư mục của bạn khác)
    # Giả định script đang chạy từ gốc thư mục dự án
    base_dir = Path(__file__).resolve().parent.parent.parent
    gt_json_path = base_dir / "dataset" / "ground_truth" / "vanban1_gt.json"
    pred_md_path = base_dir / "dataset" / "ground_truth" / "vanban1_ocr.md"
    
    try:
        # 1. Nạp và xử lý Ground Truth
        print(f"Đang đọc Ground Truth từ: {gt_json_path.name}...")
        gt_text = extract_gt_text_from_json(gt_json_path)
        
        # 2. Nạp Prediction
        print(f"Đang đọc Prediction từ: {pred_md_path.name}...")
        if not pred_md_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file OCR: {pred_md_path}")
            
        with open(pred_md_path, 'r', encoding='utf-8') as f:
            pred_text = f.read().strip()
            
        # 3. Chạy tính toán
        print("\nĐang tính toán CER và WER...")
        cer_score = calculate_cer(gt_text, pred_text)
        wer_score = calculate_wer(gt_text, pred_text)
        
        print("-" * 40)
        print(f"Kết quả Benchmark cho tài liệu 'vanban1':")
        print(f"-> CER (Tỷ lệ lỗi ký tự): {cer_score:.2%} ({cer_score:.4f})")
        print(f"-> WER (Tỷ lệ lỗi từ):    {wer_score:.2%} ({wer_score:.4f})")
        print("-" * 40)
        
    except Exception as e:
        print(f"\n❌ Lỗi trong quá trình test: {str(e)}")
        print("Vui lòng kiểm tra lại đường dẫn file trong phần Test.")