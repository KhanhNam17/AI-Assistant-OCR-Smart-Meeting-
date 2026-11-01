"""
core/grounding_docx_builder.py
===============================
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from core.document_pipeline import DocumentPage, IRBlock, OCRLine, TableCell, TableRow

_log = logging.getLogger(__name__)

_BLOCK_RE = re.compile(
    r"<\s*\|?\s*/?\s*ref\s*\|?\s*>\s*([a-zA-Z_]+)\s*<\s*\|?\s*/?\s*ref\s*\|?\s*>"
    r"\s*<\s*\|?\s*/?\s*det\s*\|?\s*>\s*"
    r"\[\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]\]\s*"
    r"<\s*\|?\s*/?\s*det\s*\|?\s*>",
    re.IGNORECASE,
)

_LABEL_TYPE_MAP: dict[str, str] = {
    "title":        "title",
    "sub_title":    "section_header",
    "figure_title": "caption",
    "text":         "text",
    "table":        "table",
    "image":        "figure",
    "picture":      "figure",
    "figure":       "figure",
    "caption":      "caption",
    "footer":       "page_footer",
    "header":       "page_header",
}

_MD_PREFIX_RE = re.compile(r"^\s*#{1,6}\s*")

# ── Cỡ chữ DOCX xuất ra ──────────────────────────────────────────────────────
# Chỉnh các số Pt(...) dưới đây nếu muốn font to/nhỏ hơn. Văn bản hành chính /
# Niên giám Thống kê gốc thường dùng Times New Roman 13-14pt cho thân bài,
# 14-16pt bold cho tiêu đề — trước đây code không set size cho đoạn văn
# thường nên nó rơi về mặc định Calibri 11pt của python-docx, nhìn nhỏ hơn
# hẳn bản gốc.
_FONT_NAME        = "Times New Roman"
_FONT_SIZE_BODY   = 13   # đoạn văn thường (blk.type == "text")
_FONT_SIZE_TITLE  = 15   # title / header (in đậm)
_FONT_SIZE_TABLE  = 11   # nội dung ô bảng
_FONT_SIZE_CAPTION = 11  # chú thích ảnh / placeholder hình ảnh


def _bbox_iou(a: tuple, b: tuple) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0: return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _map_bbox(
    x1: float, y1: float, x2: float, y2: float,
    *, page_w: int, page_h: int,
    bbox_mode: str = "auto",       # Chuyển mặc định thành "auto"
    model_canvas: int = 1024,     
) -> tuple[float, float, float, float]:
    """Quy đổi bbox từ hệ toạ độ model dùng sang toạ độ pixel thật của ảnh trang."""
    if bbox_mode == "raw":
        return x1, y1, x2, y2

    # Tự động tính toán scale và padding bất chấp tỷ lệ ảnh ngang hay dọc
    if bbox_mode == "auto":
        max_side = max(page_w, page_h)
        scale = max_side / float(model_canvas)
        
        # Tính phần viền bị đệm (nếu ảnh không vuông)
        pad_x = (max_side - page_w) / 2.0
        pad_y = (max_side - page_h) / 2.0
        
        real_x1 = (x1 * scale) - pad_x
        real_y1 = (y1 * scale) - pad_y
        real_x2 = (x2 * scale) - pad_x
        real_y2 = (y2 * scale) - pad_y
        return real_x1, real_y1, real_x2, real_y2

    r = model_canvas / float(max(page_w, page_h))

    if bbox_mode == "long_side":
        return x1 / r, y1 / r, x2 / r, y2 / r

    if bbox_mode == "padded_square":
        resized_w, resized_h = page_w * r, page_h * r
        pad_x = (model_canvas - resized_w) / 2.0
        pad_y = (model_canvas - resized_h) / 2.0
        return (x1 - pad_x) / r, (y1 - pad_y) / r, (x2 - pad_x) / r, (y2 - pad_y) / r

    raise ValueError(f"bbox_mode không hợp lệ: {bbox_mode!r}")


def _filter_spurious_image_blocks(
    raw_records: list[dict],
    *, page_w: int, page_h: int,
    min_area_frac: float = 0.004,
    max_area_frac: float = 0.85,
    dup_iou_thresh: float = 0.35,
) -> list[dict]:
    """Loại bỏ block label 'image'/'picture'/'figure' khi: (1) diện tích
    bất thường (quá nhỏ = mảnh vụn, quá lớn = model detect nhầm cả trang),
    hoặc (2) đè lên vùng text/table đã OCR đúng ở block khác — dấu hiệu
    model gán trùng nhãn image lên nội dung đã đọc được rồi."""
    text_like = [r for r in raw_records if r["label"] not in ("image", "picture", "figure")]
    kept = []
    for r in raw_records:
        if r["label"] not in ("image", "picture", "figure"):
            kept.append(r); continue
        x1, y1, x2, y2 = r["bbox"]
        area_frac = ((x2 - x1) * (y2 - y1)) / float(max(1, page_w * page_h))
        if area_frac < min_area_frac or area_frac > max_area_frac:
            _log.info(f"  bỏ block image bbox={r['bbox']} (area_frac={area_frac:.4f} bất thường)")
            continue
        overlap = max((_bbox_iou(r["bbox"], t["bbox"]) for t in text_like), default=0.0)
        if overlap > dup_iou_thresh:
            _log.info(f"  bỏ block image bbox={r['bbox']} (trùng {overlap:.2f} IoU với text/table)")
            continue
        kept.append(r)
    return kept


def _parse_page_blocks(
    raw_page_text: str,
    *,
    page_size: tuple[int, int],
    bbox_mode: str,
    model_canvas: int,
    page_image_path: Path | None,
    figures_out_dir: Path,
    page_idx: int,
    stem: str,
) -> list[IRBlock]:
    matches = list(_BLOCK_RE.finditer(raw_page_text))
    if not matches:
        _log.warning(f"grounding_docx_builder: trang {page_idx + 1} không tìm thấy thẻ <|ref|>/<|det|>.")
        return []

    page_w, page_h = page_size

    # ── Pha 1: parse toàn bộ block (chưa crop ảnh) ──────────────────────
    raw_records: list[dict] = []
    for i, m in enumerate(matches):
        label_raw = m.group(1).strip().lower()
        x1, y1, x2, y2 = (int(m.group(k)) for k in (2, 3, 4, 5))
        x1, y1, x2, y2 = _map_bbox(
            x1, y1, x2, y2, page_w=page_w, page_h=page_h,
            bbox_mode=bbox_mode, model_canvas=model_canvas,
        )
        x1 = max(0, min(int(round(x1)), page_w))
        x2 = max(0, min(int(round(x2)), page_w))
        y1 = max(0, min(int(round(y1)), page_h))
        y2 = max(0, min(int(round(y2)), page_h))

        content_start = m.end()
        content_end = matches[i + 1].start() if i + 1 < len(matches) else len(raw_page_text)
        content = raw_page_text[content_start:content_end].strip().strip("$").strip()

        raw_records.append({"label": label_raw, "bbox": (x1, y1, x2, y2), "content": content})

    # ── Pha 2: lọc block image giả (trùng nội dung / bất thường) ────────
    raw_records = _filter_spurious_image_blocks(raw_records, page_w=page_w, page_h=page_h)

    # ── Pha 3: chỉ giờ mới mở ảnh trang + crop những block còn lại ──────
    page_img = None
    if page_image_path is not None and page_image_path.exists():
        try:
            from PIL import Image as _PIL
            page_img = _PIL.open(str(page_image_path)).convert("RGB")
        except Exception as e:
            _log.warning(f"Không mở được ảnh trang {page_image_path}: {e}")

    blocks: list[IRBlock] = []
    next_id = 0
    for rec in raw_records:
        label_raw = rec["label"]
        x1, y1, x2, y2 = rec["bbox"]
        content = rec["content"]
        ir_type = _LABEL_TYPE_MAP.get(label_raw, "text")

        if ir_type == "table":
            rows = _html_or_plain_table_to_rows(content)
            if not rows: continue
            blocks.append(IRBlock(
                id=next_id, type="table", bbox=(x1, y1, x2, y2),
                table_rows=rows, original_text=content
            ))
            next_id += 1
            continue

        if ir_type == "figure":
            fig_path = ""
            if page_img is not None and (x2 - x1) >= 2 and (y2 - y1) >= 2:
                try:
                    # Bổ sung 30 pixels padding để tránh lẹm con dấu/chữ ký
                    padding = 30
                    crop_x1 = max(0, int(x1 - padding))
                    crop_y1 = max(0, int(y1 - padding))
                    crop_x2 = min(page_w, int(x2 + padding))
                    crop_y2 = min(page_h, int(y2 + padding))

                    crop = page_img.crop((crop_x1, crop_y1, crop_x2, crop_y2))
                    figures_out_dir.mkdir(parents=True, exist_ok=True)
                    out_file = figures_out_dir / f"{stem}_p{page_idx:04d}_b{next_id:03d}.png"
                    crop.save(str(out_file))
                    fig_path = str(out_file)
                except Exception as e:
                    _log.warning(f"Crop figure thất bại (block {next_id}): {e}")
            blocks.append(IRBlock(
                id=next_id, type="figure", bbox=(x1, y1, x2, y2),
                original_text=_clean_text(content), figure_path=fig_path,
            ))
            next_id += 1
            continue

        text = _clean_text(content)
        if not text: continue
        blocks.append(IRBlock(
            id=next_id, type=ir_type, bbox=(x1, y1, x2, y2),
            ocr_lines=[OCRLine(text=text, confidence=0.99)],
            original_text=text,
        ))
        next_id += 1

    return blocks

def _clean_text(content: str) -> str:
    lines = [ln for ln in content.splitlines()]
    cleaned = []
    for ln in lines:
        ln = _MD_PREFIX_RE.sub("", ln).rstrip()
        cleaned.append(ln)
    while cleaned and not cleaned[0].strip(): cleaned.pop(0)
    while cleaned and not cleaned[-1].strip(): cleaned.pop()
    return "\n".join(cleaned).strip()

def _html_or_plain_table_to_rows(content: str) -> list[TableRow]:
    if "<table" in content.lower(): return _html_table_to_rows(content)
    return _plain_text_table_to_rows(content)

def _html_table_to_rows(html: str) -> list[TableRow]:
    try: from bs4 import BeautifulSoup
    except ImportError: return []
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    if table is None: return []

    trs = table.find_all("tr")
    pending: dict[int, int] = {}
    rows_out: list[TableRow] = []

    for r_idx, tr in enumerate(trs):
        cells_in_row: list[TableCell] = []
        col_idx = 0
        tag_iter = iter(tr.find_all(["td", "th"]))

        while True:
            if pending.get(col_idx, 0) > 0:
                cells_in_row.append(TableCell(row=r_idx, col=col_idx, row_span=0, col_span=0, text=""))
                pending[col_idx] -= 1
                if pending[col_idx] <= 0: del pending[col_idx]
                col_idx += 1
                continue
            try: tag = next(tag_iter)
            except StopIteration: break

            colspan = int(tag.get("colspan", 1) or 1)
            rowspan = int(tag.get("rowspan", 1) or 1)
            text = tag.get_text(strip=True)

            cells_in_row.append(TableCell(row=r_idx, col=col_idx, row_span=rowspan, col_span=colspan, bbox=(0.0, 0.0, 0.0, 0.0), text=text))
            for extra in range(1, colspan):
                cells_in_row.append(TableCell(row=r_idx, col=col_idx + extra, row_span=0, col_span=0, text=""))
            if rowspan > 1:
                for extra in range(0, colspan): pending[col_idx + extra] = rowspan - 1
            col_idx += colspan
        rows_out.append(TableRow(cells=cells_in_row))
    return rows_out

def _plain_text_table_to_rows(content: str) -> list[TableRow]:
    """Tích hợp thuật toán gộp ô liền mạch từ test_docx_builder"""
    lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
    if not lines: return []
    
    parsed_rows = []
    for line in lines:
        cells = [c.strip() for c in line.split("|")]
        # Xóa cột trống đầu/cuối nếu có do dấu pipe "|" ở 2 mép
        if cells and cells[0] == '' and len(cells) > 1: cells.pop(0)
        if cells and cells[-1] == '': cells.pop()
        
        # Nếu cột đầu bị rỗng (tức là dòng thụt vào) -> Nối vào hàng trên
        if parsed_rows and cells and cells[0] == '':
            for c_idx, cell_text in enumerate(cells):
                if cell_text:
                    if c_idx < len(parsed_rows[-1]):
                        parsed_rows[-1][c_idx] += '\n' + cell_text
                    else:
                        parsed_rows[-1].append(cell_text)
        else:
            parsed_rows.append(cells)
            
    rows_out: list[TableRow] = []
    for r_idx, row_data in enumerate(parsed_rows):
        cells_out = [TableCell(row=r_idx, col=c_idx, text=txt, row_span=1, col_span=1) for c_idx, txt in enumerate(row_data)]
        rows_out.append(TableRow(cells=cells_out))
    return rows_out

def _render_smart_docx(page: DocumentPage, doc):
    """Render DOCX với tính năng Tự động Căn lề Nội suy theo tọa độ DeepSeek"""
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Pt, Inches
    import re

    # Hàm lọc ký tự điều khiển ẩn gây crash MS Word
    def safe_text(txt):
        if not txt:
            return ""
        return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', str(txt))

    for blk in sorted(page.blocks, key=lambda b: b.reading_order):
        page_w = page.width_px if page.width_px > 0 else 1000
        x1, y1, x2, y2 = blk.bbox if blk.bbox and len(blk.bbox) == 4 else (0, 0, 0, 0)
        
        # Chuẩn hóa tọa độ về thang 1000 để đánh giá
        norm_x1 = (x1 / page_w) * 1000
        norm_x2 = (x2 / page_w) * 1000

        # Logic Căn lề
        alignment = WD_ALIGN_PARAGRAPH.LEFT
        if norm_x1 > 450:
            alignment = WD_ALIGN_PARAGRAPH.RIGHT
        elif 'title' in blk.type.lower() or 'header' in blk.type.lower() or (norm_x1 > 200 and norm_x2 < 800):
            alignment = WD_ALIGN_PARAGRAPH.CENTER
        elif norm_x1 < 200 and norm_x2 > 800:
            alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

        if blk.type == "table" and blk.table_rows:
            max_cols = max((len(r.cells) for r in blk.table_rows), default=1)
            # Khống chế số cột tối đa theo giới hạn của MS Word (63 cột)
            max_cols = min(max_cols, 63)
            
            tbl = doc.add_table(rows=len(blk.table_rows), cols=max_cols)
            tbl.style = "Table Grid"
            for r_idx, row in enumerate(blk.table_rows):
                for c_idx, cell in enumerate(row.cells):
                    if c_idx < max_cols:
                        doc_cell = tbl.cell(r_idx, c_idx)
                        # Dùng safe_text để làm sạch nội dung trong bảng
                        doc_cell.text = safe_text(cell.text)
                        for paragraph in doc_cell.paragraphs:
                            for run in paragraph.runs:
                                run.font.size = Pt(_FONT_SIZE_TABLE)
                                run.font.name = _FONT_NAME
                                if r_idx == 0: run.bold = True
            doc.add_paragraph()
            
        elif blk.type == "figure":
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            if blk.figure_path and Path(blk.figure_path).exists():
                try:
                    from PIL import Image as _PIL
                    with _PIL.open(blk.figure_path) as _im:
                        crop_w_px, _crop_h_px = _im.size
                        
                    # Ràng buộc kích thước tối thiểu (0.5 inch) để chống crash MS Word
                    target_w_in = max(0.5, min(6.0, crop_w_px / 150.0))
                    p.add_run().add_picture(blk.figure_path, width=Inches(target_w_in))
                except Exception as e:
                    _log.warning(f"Nhúng ảnh thất bại ({blk.figure_path}): {e}")
                    r = p.add_run("[KHU VỰC HÌNH ẢNH / CHỮ KÝ]")
                    r.font.size = Pt(_FONT_SIZE_CAPTION)
                    r.font.name = _FONT_NAME
            else:
                r = p.add_run("[KHU VỰC HÌNH ẢNH / CHỮ KÝ]")
                r.font.size = Pt(_FONT_SIZE_CAPTION)
                r.font.name = _FONT_NAME
                
        else:
            # Dùng safe_text để làm sạch văn bản thông thường
            p = doc.add_paragraph(safe_text(blk.original_text))
            p.alignment = alignment
            is_heading = 'title' in blk.type.lower() or 'header' in blk.type.lower()
            for run in p.runs:
                run.font.name = _FONT_NAME
                if is_heading:
                    run.bold = True
                    run.font.size = Pt(_FONT_SIZE_TITLE)
                else:
                    run.font.size = Pt(_FONT_SIZE_BODY)

def build_docx_from_grounding(
    raw_md_path: str | Path,
    pages_dir: str | Path,
    stem: str,
    output_docx: str | Path,
    *,
    figures_out_dir: str | Path | None = None,
    bbox_mode: str = "auto",       # "raw" | "long_side" | "padded_square"
                                   # — dùng debug_draw_bboxes() để xác định
                                   # đúng mode TRƯỚC khi tin file DOCX kết quả.
    model_canvas: int = 1024,     # khớp base_size trong deepseek_ocr_engine.py
) -> Path:
    raw_md_path = Path(raw_md_path)
    pages_dir   = Path(pages_dir)
    output_docx = Path(output_docx)
    figures_out_dir = Path(figures_out_dir) if figures_out_dir else output_docx.parent / "figures_final"

    if not raw_md_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file {raw_md_path}.")

    raw_text = raw_md_path.read_text(encoding="utf-8")
    page_chunks = raw_text.split("\n\n---\n\n")

    from PIL import Image as _PIL
    ir_pages: list[DocumentPage] = []
    
    for page_idx, chunk in enumerate(page_chunks):
        page_img_path = pages_dir / f"{stem}_p{page_idx:04d}.png"
        page_w, page_h = 1000, 1000
        if page_img_path.exists():
            with _PIL.open(str(page_img_path)) as im:
                page_w, page_h = im.size

        blocks = _parse_page_blocks(
            chunk, page_size=(page_w, page_h),
            bbox_mode=bbox_mode, model_canvas=model_canvas,
            page_image_path=page_img_path if page_img_path.exists() else None,
            figures_out_dir=figures_out_dir, page_idx=page_idx, stem=stem,
        )
        if not blocks: continue

        blocks.sort(key=lambda b: (b.bbox[1] if len(b.bbox) >= 2 else 0.0, b.bbox[0] if len(b.bbox) >= 2 else 0.0))
        for i, b in enumerate(blocks): b.reading_order = i

        ir_pages.append(DocumentPage(
            page_index=page_idx, width_px=float(page_w), height_px=float(page_h),
            source_path=str(page_img_path), blocks=blocks,
        ))

    from docx import Document
    from docx.shared import Pt, Inches
    from docx.enum.text import WD_BREAK_TYPE
    from docx.oxml.ns import qn as _qn

    doc = Document()
    _normal_style = doc.styles["Normal"]
    _normal_style.font.name = _FONT_NAME
    _normal_style.font.size = Pt(_FONT_SIZE_BODY)
    for section in doc.sections:
        section.top_margin = section.bottom_margin = Inches(0.75)
        section.left_margin = section.right_margin = Inches(0.75)
    body = doc.element.body
    for elem in list(body):
        if elem.tag != _qn("w:sectPr"): body.remove(elem)

    for i, page in enumerate(ir_pages):
        if i > 0:
            para = doc.add_paragraph()
            para.add_run().add_break(WD_BREAK_TYPE.PAGE)
        # Sử dụng hàm dựng layout thông minh vừa xây dựng
        _render_smart_docx(page, doc)

    output_docx.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(output_docx))
    _log.info(f"grounding_docx_builder: đã lưu {output_docx}")
    return output_docx

def debug_draw_bboxes(
    raw_md_path: str | Path,
    pages_dir: str | Path,
    stem: str,
    page_idx: int,
    output_png: str | Path,
    *,
    bbox_mode: str = "raw",
    model_canvas: int = 1024,
) -> Path:
    """Vẽ toàn bộ bbox parse được của MỘT trang lên đúng ảnh trang gốc để
    kiểm tra bằng mắt. Chạy với lần lượt bbox_mode="raw" / "long_side" /
    "padded_square" — mode nào cho khung đỏ khớp đúng vị trí chữ/ảnh/con
    dấu trên trang thì dùng mode đó cho build_docx_from_grounding()."""
    from PIL import Image as _PIL, ImageDraw as _ImageDraw

    raw_text = Path(raw_md_path).read_text(encoding="utf-8")
    page_chunks = raw_text.split("\n\n---\n\n")
    if page_idx >= len(page_chunks):
        raise IndexError(f"page_idx={page_idx} vượt quá số trang ({len(page_chunks)}).")

    page_img_path = Path(pages_dir) / f"{stem}_p{page_idx:04d}.png"
    if not page_img_path.exists():
        raise FileNotFoundError(f"Không tìm thấy ảnh trang: {page_img_path}")

    img = _PIL.open(str(page_img_path)).convert("RGB")
    draw = _ImageDraw.Draw(img)

    blocks = _parse_page_blocks(
        page_chunks[page_idx],
        page_size=img.size, bbox_mode=bbox_mode, model_canvas=model_canvas,
        page_image_path=None, figures_out_dir=Path("/tmp"),
        page_idx=page_idx, stem=stem,
    )
    for b in blocks:
        if len(b.bbox) == 4:
            draw.rectangle(list(b.bbox), outline="red", width=3)
            draw.text((b.bbox[0] + 2, max(0, b.bbox[1] - 14)), f"#{b.id} {b.type}", fill="red")

    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(output_png))
    return output_png


__all__ = ["build_docx_from_grounding", "debug_draw_bboxes"]