"""
extract_pdf_gt.py
=================

Script trích xuất tọa độ Bounding Box (bbox) và văn bản từ PDF điện tử.
Tự động xuất ra định dạng JSON tương thích với hệ thống Benchmark OCR.
"""

import fitz  # PyMuPDF
import json
from pathlib import Path

def extract_blocks_to_gt_format(pdf_path: Path, output_json_path: Path):
    """
    Đọc file PDF, lấy tọa độ từng khối văn bản (block) và lưu thành file JSON.
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file PDF: {pdf_path}")

    # Mở file PDF
    doc = fitz.open(pdf_path)
    gt_data = {"pages": []}

    print(f"Đang xử lý PDF: {pdf_path.name} ({len(doc)} trang)...")

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        
        # get_text("blocks") trả về danh sách các tuple có dạng:
        # (x0, y0, x1, y1, "nội dung text", block_no, block_type)
        blocks = page.get_text("blocks")
        
        page_data = {
            "page_num": page_num + 1,
            "n_blocks": 0,
            "blocks": []
        }
        
        order_counter = 1
        
        # Sắp xếp các block theo trục Y từ trên xuống dưới (để ước lượng mạch đọc cơ bản)
        blocks.sort(key=lambda b: (b[1], b[0]))

        for b in blocks:
            x0, y0, x1, y1, text, block_no, block_type = b
            
            # Làm sạch text (bỏ khoảng trắng và dấu xuống dòng thừa)
            text_clean = text.strip()
            if not text_clean:
                continue
                
            # block_type == 0 là văn bản (text), 1 là hình ảnh (image)
            # PyMuPDF không tự phân biệt được bảng, nên mặc định gán là text hoặc image
            content_kind = "text" if block_type == 0 else "image"
            
            # Khởi tạo object block theo đúng chuẩn Schema
            block_info = {
                "order": order_counter,
                "type": content_kind,
                # Làm tròn tọa độ tới 1 chữ số thập phân cho gọn file
                "bbox": [round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)],
                "content_kind": content_kind,
                "content": text_clean,
                "verified": False,       # Mặc định False để chờ con người duyệt lại
                "added_missing": False   # Vì máy móc ra được nên chắc chắn không phải bị bỏ sót
            }
            
            page_data["blocks"].append(block_info)
            order_counter += 1
            
        # Cập nhật tổng số block của trang
        page_data["n_blocks"] = len(page_data["blocks"])
        gt_data["pages"].append(page_data)

    # Lưu ra file JSON
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(gt_data, f, ensure_ascii=False, indent=4)
        
    print(f"✅ Đã xuất bản nháp Ground Truth thành công!")
    print(f"-> Lưu tại: {output_json_path}")

if __name__ == "__main__":
    # Cấu hình đường dẫn (sửa lại theo đường dẫn thực tế của bạn)
    # Ví dụ: lấy file vanban1.pdf tạo ra bản nháp vanban1_gt_draft.json
    base_dir = Path(__file__).resolve().parent
    input_pdf = base_dir / "vanban1.pdf" 
    output_json = base_dir / "vanban1_gt_draft.json"
    
    try:
        # Nếu chưa có file vanban1.pdf, tạo một file pdf rỗng để tránh lỗi sập script khi test
        if not input_pdf.exists():
            doc = fitz.open()
            page = doc.new_page()
            page.insert_text((50, 50), "Tập đoàn Bưu chính Viễn thông Việt Nam", fontsize=12)
            page.insert_text((50, 100), "Cộng hòa Xã hội Chủ nghĩa Việt Nam", fontsize=12)
            doc.save(input_pdf)
            
        extract_blocks_to_gt_format(input_pdf, output_json)
    except Exception as e:
        print(f"❌ Lỗi: {e}")