"""
core/table_engines/mineru_engine.py
=====================================

Table engine chạy MinerU (qua CLI, không phải AtomModelSingleton — API nội bộ
đó đã đổi tên giữa các version như đã gặp lỗi trước đây) trên crop bảng, parse
HTML trả về thành list[TableRow] theo đúng contract của TableReconstructorBase.

Giới hạn cần biết trước khi dùng:
    - MinerU trả về HTML bảng đã có SẴN text bên trong từng ô (từ OCR nội bộ
      của chính nó, PaddleOCR-based). Engine này KHÔNG re-OCR lại bằng
      self.ocr_engine (VietOCR/EasyOCR) của bạn — vì MinerU không trả kèm bbox
      per-cell trong middle.json (chỉ có 1 khối HTML cho cả bảng), nên không
      có toạ độ để crop lại từng ô mà OCR riêng.
    - Do đó bbox của mọi TableCell trả về là (0,0,0,0) placeholder — nếu pipeline
      của bạn có bước downstream dựa vào bbox thật của cell (vd
      ocr_cell_grid/infer_rowspans_by_border), bước đó sẽ không có tác dụng với
      engine này vì đã có text sẵn và không có bbox để làm việc.
    - Text lấy nguyên từ MinerU — chất lượng chính tả phụ thuộc OCR nội bộ của
      MinerU (đã ghi nhận có lỗi dấu tiếng Việt ở một số bản chạy trước đó).
      Nếu cần độ chính xác chính tả cao hơn, cân nhắc hậu xử lý text bằng LLM
      polish riêng (đã bàn ở phần đầu dự án) thay vì OCR lại.
    - Mỗi lần gọi tốn 1 subprocess `mineru` (load lại model) — chậm hơn hẳn
      RapidTable/img2table. Cân nhắc dùng engine này có chọn lọc (vd chỉ khi
      engine mặc định trả kết quả rỗng/nghi ngờ) thay vì mặc định cho mọi bảng.

Cài đặt:
    pip install -U "mineru[pipeline]"
    pip install beautifulsoup4   # đã có sẵn trong ocr_env (bs4 4.14.3)
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from bs4 import BeautifulSoup

from table_engines.base import TableCell, TableReconstructorBase, TableRow

_log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Gọi MinerU CLI trên 1 ảnh crop, trả về HTML bảng thô
# ══════════════════════════════════════════════════════════════════════════════

def _find_table_html(block: dict) -> Optional[str]:
    """Đệ quy tìm content HTML của span type='table' trong 1 block middle.json.

    Lưu ý: bản MinerU đang dùng lưu HTML bảng ở key "html" trong span (không
    phải "content" như code gốc trước đây), nên ưu tiên "html" và fallback
    sang "content" để tương thích ngược với các version khác có thể dùng key
    khác nhau."""
    for line in block.get("lines", []):
        for span in line.get("spans", []):
            if span.get("type") == "table":
                html = span.get("html") or span.get("content")
                if html:
                    return html
    for sub in block.get("blocks", []):
        found = _find_table_html(sub)
        if found:
            return found
    return None


def _run_mineru_on_crop(
    crop_path: Path,
    mineru_backend: str = "pipeline",
    keep_debug_files: bool = False,
) -> str:
    """Chạy `mineru` CLI trên 1 ảnh, trả về HTML bảng ("" nếu không tìm thấy)."""
    import os
    import sys
    import json
    import shutil
    import tempfile
    import subprocess
    
    # 1. Tự động truy vết ĐƯỜNG DẪN TUYỆT ĐỐI của tool trong Conda env
    env_dir = Path(sys.executable).parent
    
    if os.name == 'nt':  # Trên Windows
        mineru_exe = env_dir / "Scripts" / "mineru.exe"
        if not mineru_exe.exists():
            mineru_exe = env_dir / "Scripts" / "magic-pdf.exe" # Fallback tên cũ
    else:                # Trên Linux/Mac
        mineru_exe = env_dir / "bin" / "mineru"
        if not mineru_exe.exists():
            mineru_exe = env_dir / "bin" / "magic-pdf"
            
    # Nếu vẫn không thấy, dùng shutil.which hoặc chuỗi mặc định
    exe_str = str(mineru_exe) if mineru_exe.exists() else (shutil.which("mineru") or "mineru")
    
    tmp_out = Path(tempfile.mkdtemp(prefix="mineru_table_"))
    try:
        cmd = [exe_str, "-p", str(crop_path), "-o", str(tmp_out), "-b", mineru_backend]
        
        # 2. Gọi tiến trình con
        is_windows = (os.name == 'nt')
        result = subprocess.run(cmd, capture_output=True, text=True, shell=is_windows)

        if result.returncode != 0:
            _log.warning(f"mineru CLI lỗi cho {crop_path.name}: {result.stderr[-500:]}")
            return ""

        candidates = list(tmp_out.rglob("*_middle.json"))
        if not candidates:
            _log.warning(f"Không tìm thấy middle.json cho {crop_path.name}")
            return ""

        with open(candidates[0], "r", encoding="utf-8") as f:
            middle = json.load(f)

        html_parts = []
        seen_block_types: set[str] = set()
        for page_info in middle.get("pdf_info", []):
            for block in page_info.get("para_blocks", []):
                seen_block_types.add(block.get("type", "?"))
                # Không lọc theo block.get("type") == "table" nữa: schema thực tế
                # có thể trả top-level type là "table_body" (hoặc "table" bọc
                # ngoài "table_body"/"table_caption"/"table_footnote" tuỳ version).
                # _find_table_html tự đệ quy và chỉ lấy đúng span type=="table"
                # nên gọi cho MỌI block là an toàn và không phụ thuộc schema.
                html = _find_table_html(block)
                if html:
                    html_parts.append(html)

        if not html_parts:
            _log.debug(
                f"Không tìm thấy HTML bảng nào trong {candidates[0].name}. "
                f"Các block type gặp phải: {sorted(seen_block_types)}"
            )

        return "\n".join(html_parts)

    finally:
        if keep_debug_files:
            _log.info(f"Giữ lại output MinerU tại: {tmp_out}")
        else:
            shutil.rmtree(tmp_out, ignore_errors=True)


# ══════════════════════════════════════════════════════════════════════════════
# Parse HTML (rowspan/colspan) -> list[TableRow] dense grid
# ══════════════════════════════════════════════════════════════════════════════

def _html_table_to_rows(html: str) -> list[TableRow]:
    """
    Chuyển 1 bảng HTML (rowspan/colspan chuẩn) thành dense grid list[TableRow],
    theo đúng convention của base.py: ô neo (anchor) giữ row_span/col_span thật
    và text; các ô bị gộp (absorbed) có row_span=0, col_span=0, text="".
    """
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    if table is None:
        return []

    trs = table.find_all("tr")
    # pending[col_idx] = (số dòng còn phải "chiếm chỗ", ) — do rowspan từ hàng trên
    pending: dict[int, int] = {}
    rows_out: list[TableRow] = []

    for r_idx, tr in enumerate(trs):
        cells_in_row: list[TableCell] = []
        col_idx = 0
        cell_tags = tr.find_all(["td", "th"])
        tag_iter = iter(cell_tags)

        while True:
            if pending.get(col_idx, 0) > 0:
                cells_in_row.append(TableCell(
                    row=r_idx, col=col_idx, row_span=0, col_span=0, text="",
                ))
                pending[col_idx] -= 1
                if pending[col_idx] <= 0:
                    del pending[col_idx]
                col_idx += 1
                continue

            try:
                tag = next(tag_iter)
            except StopIteration:
                break

            colspan = int(tag.get("colspan", 1) or 1)
            rowspan = int(tag.get("rowspan", 1) or 1)
            text = tag.get_text(strip=True)

            cells_in_row.append(TableCell(
                row=r_idx, col=col_idx,
                row_span=rowspan, col_span=colspan,
                bbox=(0.0, 0.0, 0.0, 0.0),  # MinerU không trả bbox per-cell
                text=text,
            ))

            # Các cột bị colspan chiếm thêm trong CÙNG hàng này -> ô rỗng absorbed
            for extra in range(1, colspan):
                cells_in_row.append(TableCell(
                    row=r_idx, col=col_idx + extra, row_span=0, col_span=0, text="",
                ))

            # Nếu có rowspan > 1, đánh dấu các cột (kể cả cột do colspan chiếm)
            # phải bị "pending" ở các hàng tiếp theo.
            if rowspan > 1:
                for extra in range(0, colspan):
                    pending[col_idx + extra] = rowspan - 1

            col_idx += colspan

        rows_out.append(TableRow(cells=cells_in_row))

    return rows_out


# ══════════════════════════════════════════════════════════════════════════════
# Engine class
# ══════════════════════════════════════════════════════════════════════════════

class MineruTableReconstructor(TableReconstructorBase):
    """Table engine dùng MinerU CLI (layout+table structure), text lấy từ chính
    OCR nội bộ của MinerU. Trả [] (fall-through) nếu MinerU không detect được
    bảng nào trong crop — pipeline sẽ tự chuyển sang engine kế tiếp theo logic
    hiện có của bạn."""

    name = "mineru"

    def __init__(self, ocr_engine=None, mineru_backend: str = "pipeline",
                 keep_debug_files: bool = False):
        super().__init__(ocr_engine)
        self.mineru_backend = mineru_backend
        self.keep_debug_files = keep_debug_files

    def reconstruct(self, blk) -> list[TableRow]:
        if blk.crop is None:
            return []

        with tempfile.TemporaryDirectory(prefix="mineru_crop_") as tmp_dir:
            crop_path = Path(tmp_dir) / "table_crop.jpg"
            blk.crop.convert("RGB").save(crop_path)

            html = _run_mineru_on_crop(
                crop_path,
                mineru_backend=self.mineru_backend,
                keep_debug_files=self.keep_debug_files,
            )

        if not html:
            _log.info("MinerU không trích xuất được bảng nào trong crop này — "
                       "fall through sang engine kế tiếp.")
            return []

        rows = _html_table_to_rows(html)
        if not rows:
            _log.warning("MinerU trả về HTML nhưng parse ra 0 hàng — kiểm tra "
                          "format HTML trả về (có thể khác cấu trúc mong đợi).")
        return rows


__all__ = ["MineruTableReconstructor"]