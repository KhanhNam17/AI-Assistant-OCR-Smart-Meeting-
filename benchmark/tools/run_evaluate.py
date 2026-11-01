"""
run_evaluate.py
===============

Script tổng hợp chạy toàn bộ pipeline Benchmark OCR.
Tự động nạp Ground Truth và Prediction, gọi các hàm từ 3 module metrics,
và xuất ra báo cáo chi tiết cho tài liệu.
"""

import csv
import json
import re
import traceback
from datetime import datetime
from pathlib import Path

try:
    from rapidfuzz import fuzz
except ImportError:
    fuzz = None

# Import các hàm tính toán từ 3 module
from metrics.text_metrics import extract_gt_text_from_json, calculate_cer, calculate_wer
from metrics.table_metrics import extract_gt_tables, extract_pred_tables, calculate_teds, calculate_teds_struct
from metrics.layout_metrics import calculate_content_loss, get_sequences_for_test, calculate_reading_order_ned

# Ngưỡng độ tương đồng (0-100) để coi 1 bảng GT và 1 bảng Pred là "cùng một
# bảng" khi ghép cặp theo nội dung. Cùng logic với _FUZZY_MATCH_THRESHOLD
# trong layout_metrics.py.
_TABLE_MATCH_THRESHOLD = 50


def _table_text(html: str) -> str:
    """Lấy text thô từ HTML bảng (bỏ tag) để so khớp fuzzy, không cần parser đầy đủ."""
    text = re.sub(r"<[^>]+>", " ", html or "")
    return " ".join(text.split())


def _pair_tables(gt_tables: list, pred_tables: list, threshold: int = _TABLE_MATCH_THRESHOLD):
    """
    Ghép cặp bảng GT-Pred theo NỘI DUNG (fuzzy match), thay vì theo index
    (min(len(gt), len(pred)) rồi zip tuần tự như bản cũ).

    Vấn đề của cách ghép theo index: nếu OCR bỏ sót hoặc chèn thừa 1 bảng
    ở giữa tài liệu (không phải bảng cuối), mọi bảng phía sau vị trí đó sẽ
    bị lệch chỉ số -- so sánh nhầm bảng A với bảng B dù số lượng bảng tình
    cờ bằng nhau, khiến TEDS trung bình sai lệch mà không có cảnh báo nào.

    Cách làm: tính điểm tương đồng cho MỌI cặp (i, j) có thể, ghép tham lam
    theo điểm giảm dần, mỗi bảng GT và mỗi bảng Pred chỉ được dùng tối đa 1
    lần. Bảng GT nào không tìm được bảng Pred nào đạt ngưỡng sẽ bị coi là
    "mất bảng" và KHÔNG được tính vào điểm trung bình (thay vì bị lặng lẽ
    bỏ qua như cách min() cũ).

    Trả về:
        matched: list các tuple (gt_idx, pred_idx, score)
        unmatched_gt: list các gt_idx không ghép được (bảng bị OCR bỏ sót)
        unmatched_pred: list các pred_idx không ghép được (bảng OCR bịa thêm/nhận nhầm)
    """
    if fuzz is None:
        raise ImportError("Vui lòng cài đặt thư viện: pip install rapidfuzz")

    gt_texts = [_table_text(t) for t in gt_tables]
    pred_texts = [_table_text(t) for t in pred_tables]

    scored_pairs = []
    for i, gt_t in enumerate(gt_texts):
        if not gt_t:
            continue
        for j, pred_t in enumerate(pred_texts):
            if not pred_t:
                continue
            score = fuzz.QRatio(gt_t[:500], pred_t[:500])
            scored_pairs.append((score, i, j))

    # Ghép tham lam: điểm cao nhất trước, mỗi bên chỉ dùng được 1 lần
    scored_pairs.sort(key=lambda x: x[0], reverse=True)
    used_gt, used_pred = set(), set()
    matched = []
    for score, i, j in scored_pairs:
        if i in used_gt or j in used_pred:
            continue
        if score < threshold:
            break  # đã sort giảm dần nên phần còn lại chắc chắn cũng dưới ngưỡng
        matched.append((i, j, score))
        used_gt.add(i)
        used_pred.add(j)

    unmatched_gt = [i for i in range(len(gt_tables)) if i not in used_gt]
    unmatched_pred = [j for j in range(len(pred_tables)) if j not in used_pred]
    return matched, unmatched_gt, unmatched_pred


# Thứ tự cột cố định cho file CSV kết quả -- giữ cố định để các lần chạy
# sau vẫn ghi đúng cột dù logic tính toán nội bộ có thay đổi.
_CSV_FIELDNAMES = [
    "timestamp",
    "document",
    "cer",
    "wer",
    "content_loss",
    "reading_order_ned",
    "teds",
    "teds_struct",
    "tables_matched",
    "gt_tables_total",
    "pred_tables_total",
    "tables_missing_in_pred",   # index các bảng GT không tìm được trong Pred
    "tables_extra_in_pred",     # index các bảng Pred không khớp bảng GT nào
]


def save_results_to_csv(doc_name: str, results: dict, csv_path: Path) -> None:
    """
    Lưu kết quả benchmark của 1 tài liệu thành 1 dòng trong file CSV tổng,
    theo đúng định dạng results/baseline_ocr_scores.csv: mỗi lần chạy thêm
    (append) 1 dòng mới, không ghi đè các dòng cũ -- để tích lũy lịch sử
    benchmark qua nhiều tài liệu / nhiều lần chạy (vd so sánh trước-sau khi
    đổi pipeline OCR).

    Header chỉ được ghi 1 lần khi file chưa tồn tại hoặc đang rỗng.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "document": doc_name,
        "cer": round(results["text"]["cer"], 4),
        "wer": round(results["text"]["wer"], 4),
        "content_loss": round(results["layout"]["content_loss"], 4),
        "reading_order_ned": round(results["layout"]["reading_order_ned"], 4),
        "teds": round(results["table"]["teds"], 4),
        "teds_struct": round(results["table"]["teds_struct"], 4),
        "tables_matched": results["table"]["tables_found"],
        "gt_tables_total": results["table"]["gt_tables_total"],
        "pred_tables_total": results["table"]["pred_tables_total"],
        "tables_missing_in_pred": ";".join(str(i) for i in results["table"]["unmatched_gt"]),
        "tables_extra_in_pred": ";".join(str(i) for i in results["table"]["unmatched_pred"]),
    }

    file_exists = csv_path.exists() and csv_path.stat().st_size > 0
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"💾 Đã lưu kết quả vào: {csv_path}")

def evaluate_document(gt_json_path: Path, pred_md_path: Path, csv_path: Path = None):
    """
    Thực thi đánh giá toàn diện cho 1 cặp tài liệu GT - Pred.

    csv_path: nếu được truyền, kết quả sẽ được append (thêm dòng) vào file
        CSV này sau khi in báo cáo -- ví dụ results/baseline_ocr_scores.csv.
        Nếu None, chỉ in báo cáo ra console như trước, không lưu file.
    """
    doc_name = gt_json_path.stem.replace('_gt', '')
    print(f"🚀 BẮT ĐẦU ĐÁNH GIÁ TÀI LIỆU: {doc_name}")
    print("=" * 60)
    
    # ---------------------------------------------------------
    # 1. Nạp dữ liệu chung
    # ---------------------------------------------------------
    with open(gt_json_path, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)
        
    with open(pred_md_path, 'r', encoding='utf-8') as f:
        pred_md_text = f.read()

    results = {}

    # ---------------------------------------------------------
    # 2. Đánh giá Nội dung (Text Metrics)
    # ---------------------------------------------------------
    print("[1/3] Đang phân tích Nội dung văn bản (CER, WER)...")
    gt_text_full = extract_gt_text_from_json(gt_json_path)
    
    cer_score = calculate_cer(gt_text_full, pred_md_text)
    wer_score = calculate_wer(gt_text_full, pred_md_text)
    
    results['text'] = {
        'cer': cer_score,
        'wer': wer_score
    }

    # ---------------------------------------------------------
    # 3. Đánh giá Bố cục & Mạch đọc (Layout Metrics)
    # ---------------------------------------------------------
    print("[2/3] Đang phân tích Bố cục & Luồng đọc (Loss, NED)...")
    loss_rate = calculate_content_loss(gt_data)
    
    gt_seq, pred_seq = get_sequences_for_test(gt_data, pred_md_text)
    ned_score = calculate_reading_order_ned(gt_seq, pred_seq)
    
    results['layout'] = {
        'content_loss': loss_rate,
        'reading_order_ned': ned_score
    }

    # ---------------------------------------------------------
    # 4. Đánh giá Bảng biểu (Table Metrics)
    # ---------------------------------------------------------
    print("[3/3] Đang phân tích Cấu trúc Bảng biểu (TEDS)...")
    gt_tables = extract_gt_tables(gt_json_path)
    pred_tables = extract_pred_tables(pred_md_text)

    avg_teds = 0.0
    avg_teds_struct = 0.0

    # Ghép cặp theo NỘI DUNG (fuzzy match), không theo index -- xem docstring
    # của _pair_tables() để biết lý do cách cũ (min(len,len) rồi zip) có thể
    # so sánh nhầm cặp khi 1 bảng ở giữa tài liệu bị OCR bỏ sót/nhận thừa.
    matched, unmatched_gt, unmatched_pred = _pair_tables(gt_tables, pred_tables)
    pairs_to_evaluate = len(matched)

    if pairs_to_evaluate > 0:
        total_teds = 0.0
        total_teds_struct = 0.0

        for gt_idx, pred_idx, _score in matched:
            total_teds += calculate_teds(gt_tables[gt_idx], pred_tables[pred_idx])
            total_teds_struct += calculate_teds_struct(gt_tables[gt_idx], pred_tables[pred_idx])

        avg_teds = total_teds / pairs_to_evaluate
        avg_teds_struct = total_teds_struct / pairs_to_evaluate

    results['table'] = {
        'teds': avg_teds,
        'teds_struct': avg_teds_struct,
        'tables_found': pairs_to_evaluate,
        'gt_tables_total': len(gt_tables),
        'pred_tables_total': len(pred_tables),
        'unmatched_gt': unmatched_gt,      # bảng GT không tìm được bảng Pred tương ứng (bị OCR bỏ sót)
        'unmatched_pred': unmatched_pred,  # bảng Pred không khớp bảng GT nào (OCR nhận nhầm/thừa)
    }

    # ---------------------------------------------------------
    # 5. Xuất Báo Cáo Tổng Hợp
    # ---------------------------------------------------------
    print("\n" + "📊 BẢNG KẾT QUẢ BENCHMARK TỔNG HỢP".center(60))
    print("-" * 60)
    print(f"{'Nhóm Chỉ Số':<25} | {'Metric':<20} | {'Điểm Số':<10}")
    print("-" * 60)
    
    # Text
    print(f"{'1. NỘI DUNG (TEXT)':<25} | {'CER (Lỗi ký tự) ↓':<20} | {results['text']['cer']:.2%}")
    print(f"{'':<25} | {'WER (Lỗi từ) ↓':<20} | {results['text']['wer']:.2%}")
    print("-" * 60)
    
    # Layout
    print(f"{'2. BỐ CỤC (LAYOUT)':<25} | {'Content Loss ↓':<20} | {results['layout']['content_loss']:.2%}")
    print(f"{'':<25} | {'Reading Order NED ↑':<20} | {results['layout']['reading_order_ned']:.2%}")
    print("-" * 60)
    
    # Table
    tbl = results['table']
    if tbl['tables_found'] > 0:
        print(f"{'3. BẢNG BIỂU (TABLE)':<25} | {'TEDS Tổng thể ↑':<20} | {tbl['teds']:.2%}")
        print(f"{' (Trên '+str(tbl['tables_found'])+' bảng)':<24} | {'TEDS Cấu trúc ↑':<20} | {tbl['teds_struct']:.2%}")
    else:
        print(f"{'3. BẢNG BIỂU (TABLE)':<25} | {'Không tìm thấy bảng trong tài liệu này':<33}")

    if tbl['unmatched_gt'] or tbl['unmatched_pred']:
        print("-" * 60)
        print(f"⚠️  GT có {tbl['gt_tables_total']} bảng, Pred có {tbl['pred_tables_total']} bảng "
              f"-- {tbl['tables_found']} cặp ghép được theo nội dung.")
        if tbl['unmatched_gt']:
            print(f"   -> Bảng GT KHÔNG tìm được trong Pred (OCR bỏ sót): index {tbl['unmatched_gt']}")
        if tbl['unmatched_pred']:
            print(f"   -> Bảng Pred KHÔNG khớp bảng GT nào (OCR nhận thừa/nhầm): index {tbl['unmatched_pred']}")

    print("=" * 60)

    if csv_path is not None:
        save_results_to_csv(doc_name, results, csv_path)

    return results

if __name__ == "__main__":
    # Cấu hình đường dẫn test
    base_dir = Path(__file__).resolve().parent.parent
    gt_path = base_dir / "dataset" / "ground_truth" / "vanban1_gt.json"
    pred_path = base_dir / "dataset" / "ground_truth" / "vanban1_ocr.md"
    csv_path = base_dir / "results" / "baseline_ocr_scores.csv"

    try:
        evaluate_document(gt_path, pred_path, csv_path=csv_path)
    except Exception as e:
        print(f"❌ Lỗi thực thi: {e}")
        traceback.print_exc()