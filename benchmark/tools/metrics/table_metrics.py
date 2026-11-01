"""
table_metrics.py
================

Module tính toán các chỉ số độ chính xác bảng biểu (TEDS, TEDS-Struct).
Sử dụng thư viện table_recognition_metric (SWHL).
"""

import re
import json
from pathlib import Path

try:
    from table_recognition_metric import TEDS
except ImportError:
    TEDS = None

def _ensure_body_wrapper(html_str: str) -> str:
    """
    Thu vien table_recognition_metric (SWHL) doi hoi chuoi HTML sau khi
    lxml.html.fromstring() parse xong phai co duong dan xpath 'body/table'
    (xem main.py cua thu vien: `xpath_ele = "body/table"`). Neu chuoi truyen
    vao la <table>...</table> TRAN (khong co <html><body> bao ngoai), lxml
    tra ve chinh the <table> lam root -- KHONG co the <body> nao trong cay
    -- nen xpath luon rong va ham __call__ roi vao nhanh `return 0.0` bat
    ke 2 bang giong nhau toi dau.

    day chinh la nguyen nhan gay ket qua "TEDS = 0.00%" cho toan bo 6/6
    bang trong lan chay truoc, du sai lech thuc te giua GT va prediction
    la rat nho (chi vai loi chinh ta, khong co loi row/col-span).
    """
    s = (html_str or "").strip()
    if not s:
        return s
    if "<body" in s.lower():
        return s
    return f"<html><body>{s}</body></html>"


def calculate_teds(gt_html: str, pred_html: str) -> float:
    """
    Tính toán TEDS (Tree-Edit-Distance-based Similarity) bao gồm cả Cấu trúc và Nội dung.
    """
    if TEDS is None:
        raise ImportError("Vui lòng cài đặt thư viện: pip install table_recognition_metric")

    teds_evaluator = TEDS()
    # Chu ky that cua thu vien la __call__(self, pred, gt) -- pred TRUOC,
    # gt SAU (nguoc voi thu tu tham so trong ham nay). Ve mat toan hoc TEDS
    # doi xung (EditDist(A,B) == EditDist(B,A)) nen dao nguoc khong lam sai
    # ket qua, nhung van sua cho dung ngu nghia / tranh nham lan sau nay.
    return teds_evaluator(
        pred=_ensure_body_wrapper(pred_html),
        gt=_ensure_body_wrapper(gt_html),
    )


def calculate_teds_struct(gt_html: str, pred_html: str) -> float:
    """
    Tính toán TEDS-Struct (Chỉ đánh giá Cấu trúc hàng/cột, bỏ qua nội dung text).
    """
    if TEDS is None:
        raise ImportError("Vui lòng cài đặt thư viện: pip install table_recognition_metric")

    teds_evaluator = TEDS(structure_only=True)
    return teds_evaluator(
        pred=_ensure_body_wrapper(pred_html),
        gt=_ensure_body_wrapper(gt_html),
    )


# ==============================================================================
# PHẦN TEST VỚI DỮ LIỆU THẬT
# ==============================================================================
def extract_gt_tables(json_path: Path) -> list:
    """Trích xuất danh sách các mã HTML bảng biểu từ file Ground Truth JSON."""
    if not json_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file GT: {json_path}")
        
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    tables = []
    for page in data.get("pages", []):
        sorted_blocks = sorted(page.get("blocks", []), key=lambda x: x.get("order", 0))
        for block in sorted_blocks:
            if block.get("content_kind") == "table_html":
                tables.append(block.get("content", ""))
    return tables

def extract_pred_tables(md_text: str) -> list:
    """Sử dụng Regex để tìm tất cả các thẻ <table>...</table> trong file Markdown OCR."""
    pattern = re.compile(r"<table.*?>.*?</table>", re.IGNORECASE | re.DOTALL)
    return pattern.findall(md_text)

if __name__ == "__main__":
    print("=== TEST TABLE METRICS VỚI FILE THỰC TẾ ===")
    
    # Cấu hình đường dẫn
    base_dir = Path(__file__).resolve().parent.parent.parent
    gt_json_path = base_dir / "dataset" / "ground_truth" / "vanban1_gt.json"
    pred_md_path = base_dir / "dataset" / "ground_truth" / "vanban1_ocr.md"
    
    try:
        # 1. Trích xuất bảng từ Ground Truth
        print(f"Đang đọc Bảng biểu từ: {gt_json_path.name}...")
        gt_tables = extract_gt_tables(gt_json_path)
        print(f"-> Tìm thấy {len(gt_tables)} bảng trong Ground Truth.")
        
        # 2. Trích xuất bảng từ Prediction
        print(f"Đang đọc Bảng biểu từ: {pred_md_path.name}...")
        with open(pred_md_path, 'r', encoding='utf-8') as f:
            pred_text = f.read()
        pred_tables = extract_pred_tables(pred_text)
        print(f"-> Tìm thấy {len(pred_tables)} bảng trong Prediction.")
        
        # 3. Chạy tính toán (so sánh theo cặp)
        if len(gt_tables) > 0 and len(pred_tables) > 0:
            print("\nĐang tính toán TEDS và TEDS-Struct...")
            
            total_teds = 0.0
            total_teds_struct = 0.0
            
            pairs_to_evaluate = min(len(gt_tables), len(pred_tables))
            
            for i in range(pairs_to_evaluate):
                score = calculate_teds(gt_tables[i], pred_tables[i])
                score_struct = calculate_teds_struct(gt_tables[i], pred_tables[i])
                
                total_teds += score
                total_teds_struct += score_struct
                print(f"  - Bảng {i+1}: TEDS = {score:.2%} | TEDS-Struct = {score_struct:.2%}")
            
            avg_teds = total_teds / pairs_to_evaluate
            avg_teds_struct = total_teds_struct / pairs_to_evaluate
            
            print("-" * 50)
            print(f"KẾT QUẢ TRUNG BÌNH CHO {pairs_to_evaluate} BẢNG ĐỐI CHIẾU:")
            print(f"-> TEDS (Tổng thể):  {avg_teds:.2%} ({avg_teds:.4f})")
            print(f"-> TEDS-Struct:      {avg_teds_struct:.2%} ({avg_teds_struct:.4f})")
            print("-" * 50)
            
            if len(gt_tables) != len(pred_tables):
                print(f"⚠️ Cảnh báo: Số lượng bảng không khớp (GT: {len(gt_tables)}, Pred: {len(pred_tables)}).")
                
    except Exception as e:
        print(f"\n❌ Lỗi trong quá trình test: {str(e)}")