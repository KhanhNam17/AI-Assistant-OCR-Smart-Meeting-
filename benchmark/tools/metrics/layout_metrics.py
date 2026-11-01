"""
layout_metrics.py
=================

Module tính toán các chỉ số về Bố cục:
1. Content Loss Rate (Tỷ lệ mất nội dung do OCR bỏ sót)
2. Reading Order NED (Độ chính xác của luồng đọc)

SỬA (so với bản gốc): get_sequences_for_test() dùng str.find() để so khớp
TUYỆT ĐỐI content của GT trong prediction -- chỉ cần lệch 1 ký tự (lỗi
chính tả bình thường, đã được đo riêng ở text_metrics.py) là block đó
"biến mất" hoàn toàn khỏi pred_seq, bị tính nhầm thành "đảo thứ tự đọc"
trong khi thực chất là lỗi nội dung. Ví dụ thực tế: GT "2. HÌNH ẢNH SẢN
PHẨM" không tìm thấy vì prediction OCR ra "2. HỆNH ẢNH SẢN PHẨM" (sai 1
ký tự) -- block này ở ĐÚNG vị trí, không hề bị đảo lộn.

Sửa bằng rapidfuzz.fuzz.QRatio: cho phép sai lệch nhỏ (threshold) vẫn
được coi là "tìm thấy", chỉ khi không có đoạn nào đủ giống (dưới
threshold) mới tính là thật sự mất/đảo vị trí.
"""

import json
from pathlib import Path

try:
    import editdistance
except ImportError:
    editdistance = None

try:
    from rapidfuzz import fuzz
except ImportError:
    fuzz = None

# Ngưỡng độ tương đồng (0-100). GT/prediction được coi là "cùng 1 block"
# khi điểm fuzzy-match >= ngưỡng này. 60 đủ chịu được vài lỗi chính tả/
# khác biệt xuống dòng nhỏ, nhưng vẫn từ chối khi nội dung thực sự khác
# nhau (vd 2 đoạn văn khác chủ đề).
_FUZZY_MATCH_THRESHOLD = 60


def calculate_content_loss(gt_data: dict) -> float:
    """
    Tính tỷ lệ Content Loss bằng cách đếm cờ 'added_missing' trong Ground Truth.

    KHÔNG đổi hàm này -- đã đúng từ bản gốc, xác nhận qua test thực tế
    cho kết quả đúng (7.69% = 2/26, khớp với 2 block bị OCR end2end bỏ
    sót hoàn toàn đã phát hiện thủ công ở bước tạo GT).
    """
    total_blocks = 0
    missing_blocks = 0

    for page in gt_data.get("pages", []):
        for block in page.get("blocks", []):
            if block.get("type") == "image" and not block.get("content"):
                continue
            total_blocks += 1
            if block.get("added_missing", False):
                missing_blocks += 1

    if total_blocks == 0:
        return 0.0
    return missing_blocks / total_blocks


def calculate_reading_order_ned(gt_sequence: list, pred_sequence: list) -> float:
    """
    Tính Normalized Edit Distance (NED) cho mạch đọc.
    gt_sequence: List các ID block chuẩn (vd: [1, 2, 3, 4])
    pred_sequence: List các ID block do OCR sinh ra, ĐÃ được fuzzy-match
        (vd: [1, 3, 2] -- block 2 và 3 bị đảo chỗ nhau)
    """
    if editdistance is None:
        raise ImportError("Vui lòng cài đặt thư viện: pip install editdistance")

    if len(gt_sequence) == 0 and len(pred_sequence) == 0:
        return 1.0

    dist = editdistance.eval(gt_sequence, pred_sequence)
    max_len = max(len(gt_sequence), len(pred_sequence))

    if max_len == 0:
        return 1.0
    return 1.0 - (dist / max_len)


# ==============================================================================
# PHẦN TEST VỚI DỮ LIỆU THẬT
# ==============================================================================

def _split_pred_into_candidates(pred_md_text: str) -> list[dict]:
    """
    Tách prediction markdown thành danh sách các 'đoạn ứng viên' để fuzzy-
    match, thay vì tìm chuỗi con trên toàn bộ văn bản (cách cũ dễ fuzzy-
    match nhầm lung tung, vì 1 đoạn ngắn có thể "gần giống" nhiều chỗ).

    Tách theo: khối <table>...</table> (giữ nguyên 1 khối), và các đoạn
    văn còn lại (tách theo dòng trống hoặc trước mỗi heading '#').
    """
    import re

    candidates: list[dict] = []
    table_re = re.compile(r"<table.*?</table>", re.IGNORECASE | re.DOTALL)

    last_end = 0
    for m in table_re.finditer(pred_md_text):
        text_chunk = pred_md_text[last_end:m.start()]
        for para in re.split(r"\n\s*\n|\n(?=#)", text_chunk):
            para = para.strip()
            if para:
                candidates.append({"pos": last_end + text_chunk.find(para), "text": para})
        candidates.append({"pos": m.start(), "text": m.group(0)})
        last_end = m.end()

    tail = pred_md_text[last_end:]
    for para in re.split(r"\n\s*\n|\n(?=#)", tail):
        para = para.strip()
        if para:
            candidates.append({"pos": last_end + tail.find(para), "text": para})

    candidates.sort(key=lambda c: c["pos"])
    return candidates


def get_sequences_for_test(gt_data: dict, pred_md_text: str):
    """
    Lấy chuỗi ID chuẩn (GT) và chuỗi ID ước lượng từ Prediction bằng FUZZY
    MATCHING (thay vì str.find() tuyệt đối ở bản gốc).

    Với mỗi block GT, tìm đoạn ứng viên trong prediction có điểm tương
    đồng (rapidfuzz) CAO NHẤT. Nếu điểm đó >= _FUZZY_MATCH_THRESHOLD, coi
    là "tìm thấy" và lấy vị trí của đoạn ứng viên đó để xếp vào pred_seq
    theo đúng thứ tự xuất hiện trong prediction. Nếu không có đoạn nào đủ
    ngưỡng, block đó được coi là thật sự vắng mặt khỏi mạch đọc (khác với
    "sai chính tả nhẹ" vẫn còn nhận diện được).
    """
    if fuzz is None:
        raise ImportError("Vui lòng cài đặt thư viện: pip install rapidfuzz")

    gt_blocks = []
    block_id = 1
    for page in gt_data.get("pages", []):
        sorted_blocks = sorted(page.get("blocks", []), key=lambda x: x.get("order", 0))
        for b in sorted_blocks:
            if b.get("type") == "image" or b.get("added_missing", False):
                continue
            gt_blocks.append({
                "id": block_id,
                "content": b.get("content", "").strip()[:80],
            })
            block_id += 1

    gt_seq = [b["id"] for b in gt_blocks]

    candidates = _split_pred_into_candidates(pred_md_text)

    pred_positions = []
    for b in gt_blocks:
        if not b["content"]:
            continue
        best_score = -1.0
        best_pos = None
        for cand in candidates:
            score = fuzz.QRatio(b["content"], cand["text"][: max(80, len(b["content"]))])
            if score > best_score:
                best_score = score
                best_pos = cand["pos"]
        if best_score >= _FUZZY_MATCH_THRESHOLD:
            pred_positions.append({"id": b["id"], "pos": best_pos, "score": best_score})

    pred_positions.sort(key=lambda x: x["pos"])
    pred_seq = [p["id"] for p in pred_positions]

    return gt_seq, pred_seq


if __name__ == "__main__":
    print("=== TEST LAYOUT METRICS VỚI FILE THỰC TẾ (fuzzy match) ===")

    base_dir = Path(__file__).resolve().parent.parent.parent
    gt_json_path = base_dir / "dataset" / "ground_truth" / "vanban1_gt.json"
    pred_md_path = base_dir / "dataset" / "ground_truth" / "vanban1_ocr.md"

    try:
        with open(gt_json_path, "r", encoding="utf-8") as f:
            gt_data = json.load(f)

        with open(pred_md_path, "r", encoding="utf-8") as f:
            pred_text = f.read()

        loss_rate = calculate_content_loss(gt_data)

        gt_seq, pred_seq = get_sequences_for_test(gt_data, pred_text)
        ned_score = calculate_reading_order_ned(gt_seq, pred_seq)

        print("Đang phân tích mạch đọc...")
        print(f"-> Chuỗi chuẩn (GT):   {gt_seq}")
        print(f"-> Chuỗi OCR (Pred): {pred_seq}")
        missing_ids = sorted(set(gt_seq) - set(pred_seq))
        print(f"-> Block KHÔNG tìm thấy (dù fuzzy, threshold={_FUZZY_MATCH_THRESHOLD}): {missing_ids}")

        print("\n" + "-" * 50)
        print("KẾT QUẢ BENCHMARK BỐ CỤC CHO 'vanban1':")
        print(f"-> Content Loss Rate (Tỷ lệ bỏ sót): {loss_rate:.2%} ({loss_rate:.4f})")
        print(f"-> Reading Order NED (Mạch đọc):     {ned_score:.2%} ({ned_score:.4f})")
        print("-" * 50)

    except Exception as e:
        print(f"\n❌ Lỗi trong quá trình test: {str(e)}")
        raise