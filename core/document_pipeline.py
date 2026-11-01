"""
document_pipeline.py  —  Clean Production Pipeline (Markdown-first)
====================================================================

Architecture (PP-Structure style, Markdown intermediate):
    Input PDF/Image
        ↓
    Stage 1 · Rasterise          (PyMuPDF → PNG per page, handle rotation)
        ↓
    Stage 2 · Layout Detection   (YOLO DocLayNet → LayoutBlock list)
        ↓
    Stage 3 · OCR per block      (EasyOCR NPU, recognition-only per crop)
        ↓
    Stage 4 · Table Structure    (TATR → cell grid with border detection)
        ↓
    Stage 5 · IR Build           (unified DocumentPage per page)
        ↓
    Stage 6 · Translation        (Qwen3-4B, one call per block/cell)
        ↓
    Stage 7 · Markdown Build     (IR → clean Markdown string per page)
        ↓
    Stage 8 · DOCX Export        (Markdown → python-docx, no vMerge complexity)
        ↓
    Final translated Markdown + DOCX + PDF

Why Markdown as intermediate
─────────────────────────────
• Tables become | col | col | — no vMerge XML, no python-docx merge bugs
• Text blocks become clean paragraphs
• Translation operates on clean plain text, not garbled OCR blobs
• The markdown is readable and debuggable at every step
• Follows what Docling and PP-Structure recovery_to_doc.py actually produce

Key design rules (from REFACTOR.md)
──────────────────────────────────────
• Each PDF page rasterised via PyMuPDF (handles rotation automatically)
• OCR uses recognize_region() only — YOLO already found bboxes
• TATR with per-class thresholds (rows/cols at 0.07)
• Row-span detection via horizontal border-line check per column
• NO drawing on the source image ever
• NPU: OCR recognition, translation
• CPU: layout, table structure, markdown build, docx export

Usage
─────
    python document_pipeline.py <pdf_or_image> [vi->en|en->vi] [output_dir]
"""

from __future__ import annotations

import io
import logging
import os
import sys
import time
from collections import Counter
from copy import copy as _copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


_log = logging.getLogger(__name__)


# Make sibling modules importable whether the CLI is run as
# ``python document_pipeline.py`` (from core/) or
# ``python core/document_pipeline.py`` (from project root).
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
_PARENT_DIR = _THIS_DIR.parent
if str(_PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(_PARENT_DIR))
from md_ocr_translator import translate_ocr_markdown


# ══════════════════════════════════════════════════════════════════════════════
# § A  —  DATA TYPES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class OCRLine:
    text:       str
    confidence: float
    polygon:    list[tuple[float, float]] = field(default_factory=list)


# TableCell / TableRow live in core.table_engines.base — re-export so the
# rest of this module (and external importers) keep working unchanged.
from table_engines.base import TableCell, TableRow  # noqa: E402


@dataclass
class IRBlock:
    """One detected region on the page."""
    id:             int
    type:           str
    bbox:           tuple
    reading_order:  int            = 0
    ocr_lines:      list[OCRLine]  = field(default_factory=list)
    table_rows:     list[TableRow] = field(default_factory=list)
    original_text:  str            = ""
    translated_text: str           = ""
    skip:           bool           = False
    figure_path:    str            = ""   # filesystem path to the saved crop
                                          # (set during build_ir for figure
                                          # blocks; empty for everything else)


@dataclass
class DocumentPage:
    page_index:  int
    width_px:    float
    height_px:   float
    source_path: str
    blocks:      list[IRBlock] = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════════════
# § B  —  RASTERISER
# ══════════════════════════════════════════════════════════════════════════════

def rasterise_page(
    source_path: str | Path,
    page_index:  int = 0,
    dpi:         int = 150,
    output_dir:  str | Path = ".",
) -> Path:
    """
    Rasterise one PDF page to PNG. PyMuPDF Matrix applies /Rotate automatically,
    so output is always upright regardless of PDF scan rotation.
    Returns source path unchanged for image inputs.
    """
    import fitz

    source_path = Path(source_path)
    if source_path.suffix.lower() not in (".pdf",):
        return source_path

    doc  = fitz.open(str(source_path))
    page = doc[page_index]
    mat  = fitz.Matrix(dpi / 72.0, dpi / 72.0)
    pix  = page.get_pixmap(matrix=mat, alpha=False)
    doc.close()

    out_path = Path(output_dir) / f"_page_{page_index:04d}.png"
    pix.save(str(out_path))
    _log.debug(f"Rasterised page {page_index}: {pix.width}x{pix.height}px")
    return out_path


# ══════════════════════════════════════════════════════════════════════════════
# § C  —  LAYOUT DETECTION
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class LayoutBlock:
    id:    int
    label: str
    bbox:  tuple
    crop:  Any = None


_NON_TEXT_LABELS = {
    # Ultralytics DocLayNet
    "figure", "image", "Figure", "Image", "Picture", "picture",
    # DocLayout-YOLO DocStructBench
    "isolate_formula",  # rendered as figure since we can't translate math
}
_TABLE_LABELS    = {"table", "Table"}
_BOLD_LABELS     = {"title", "Title", "section_header", "Section-header",
                    "page-header", "Page-header"}
_ITALIC_LABELS   = {"caption", "Caption", "figure_caption",
                    "table_caption", "table_footnote",
                    "formula_caption"}
# DocLayout-YOLO "abandon" class = page furniture (headers, footers, page
# numbers, watermarks). Skip during OCR so it doesn't pollute the body.
_ABANDON_LABELS  = {"abandon"}


class LayoutDetector:
    """
    Layout detector backed by either Ultralytics YOLO (DocLayNet weights) or
    OpenDataLab's DocLayout-YOLO (DocStructBench / DocSynth weights). Returns
    LayoutBlocks sorted in reading order.

    Backend is picked automatically from the weight filename when
    ``backend='auto'`` (default): any path containing ``doclayout`` resolves
    to the doclayout-yolo package; otherwise we use ultralytics. Override
    with ``backend='ultralytics'`` or ``backend='doclayout'`` to force.

    Confidence threshold defaults to 0.25 for ultralytics weights and 0.20
    for DocLayout-YOLO — values used in the upstream demos.
    """

    def __init__(
        self,
        model_path: str,
        backend:    str   = "auto",
        conf_thresh: float | None = None,
        imgsz:      int   = 1024,
    ):
        self.model_path = model_path
        self.imgsz      = imgsz
        if backend == "auto":
            backend = ("doclayout"
                       if "doclayout" in str(model_path).lower()
                       else "ultralytics")
        if backend not in ("ultralytics", "doclayout"):
            raise ValueError(
                f"backend must be ultralytics|doclayout|auto, got {backend!r}"
            )
        self.backend     = backend
        self.conf_thresh = (
            conf_thresh if conf_thresh is not None
            else (0.20 if backend == "doclayout" else 0.25)
        )
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        if self.backend == "doclayout":
            try:
                from doclayout_yolo import YOLOv10
            except ImportError as e:
                raise RuntimeError(
                    "DocLayout-YOLO backend requires the `doclayout-yolo` "
                    "package. Install with:  pip install doclayout-yolo"
                ) from e
            self._model = YOLOv10(self.model_path)
            _log.info(f"DocLayout-YOLO loaded: {self.model_path}")
        else:
            from ultralytics import YOLO
            self._model = YOLO(self.model_path)
            _log.info(f"Ultralytics YOLO loaded: {self.model_path}")

    def detect(self, img_path: str | Path) -> list[LayoutBlock]:
        self._load()
        from PIL import Image as _PIL
        import numpy as np

        img = _PIL.open(str(img_path)).convert("RGB")
        if self.backend == "doclayout":
            # DocLayout-YOLO uses the same .predict() API but expects a
            # larger imgsz (1024) and a lower default conf threshold.
            results = self._model.predict(
                np.array(img),
                imgsz=self.imgsz,
                conf=self.conf_thresh,
                verbose=False,
            )[0]
        else:
            results = self._model(np.array(img), verbose=False)[0]

        # Pull boxes + confidences out of the model output so we can rank
        # overlapping boxes during cross-class NMS below.
        raw_boxes: list[tuple[float, float, float, float, str, float]] = []
        for box, cls, conf in zip(
            results.boxes.xyxy, results.boxes.cls, results.boxes.conf
        ):
            x0, y0, x1, y1 = map(float, box)
            label = results.names[int(cls)]
            raw_boxes.append((x0, y0, x1, y1, label, float(conf)))

        # Cross-class NMS. DocLayout-YOLO runs NMS *within* each class, so a
        # heading region can simultaneously fire a "title" and a "plain text"
        # box (we saw this on page 1 of vanban-1-10 — "HÌNH ẢNH SẢN PHẨM"
        # rendered twice, once as a heading and once as a paragraph). This
        # pass keeps the higher-priority label when two boxes overlap.
        raw_boxes = _cross_class_nms(raw_boxes, iou_thresh=0.55)

        blocks: list[LayoutBlock] = []
        for i, (x0, y0, x1, y1, label, _conf) in enumerate(raw_boxes):
            crop = img.crop((int(x0), int(y0), int(x1), int(y1)))
            blocks.append(LayoutBlock(id=i, label=label,
                                      bbox=(x0, y0, x1, y1), crop=crop))

        blocks.sort(key=lambda b: (round(b.bbox[1] / 20) * 20, b.bbox[0]))
        for idx, b in enumerate(blocks):
            b.id = idx

        blocks = _split_wide_blocks(blocks, float(img.width))
        for idx, b in enumerate(blocks):
            b.id = idx

        _log.info(
            f"Layout: {len(blocks)} blocks — "
            + ", ".join(f"{k}={v}" for k, v in
                        sorted(Counter(b.label for b in blocks).items()))
        )
        return blocks


# Priority used by _cross_class_nms when two overlapping boxes have very
# similar confidence — higher number wins. Tables and figures dominate
# (we don't want them re-classified as paragraphs), then captions, then
# titles/headers, then body text. Page furniture (abandon) loses everything.
_LABEL_PRIORITY: dict[str, int] = {
    "table":        100,  "Table": 100,
    "figure":        90,  "Figure": 90, "Picture": 90, "picture": 90,
    "image":         90,  "Image": 90,
    "isolate_formula": 85,
    "table_caption": 70,  "figure_caption": 70,
    "table_footnote":70,  "formula_caption": 70,
    "caption":       70,  "Caption": 70,
    "title":         60,  "Title": 60,
    "section_header":50,  "Section-header": 50,
    "page-header":   40,  "Page-header": 40,
    "plain text":    30,  "plain_text": 30, "text": 30, "Text": 30,
    "list_item":     30,  "formula": 30,
    "page-footer":   20,  "Page-footer": 20,
    "abandon":        0,
}


def _bbox_iou(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    """Standard axis-aligned IoU. Returns 0 on degenerate boxes."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0); iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1); iy1 = min(ay1, by1)
    iw = max(0.0, ix1 - ix0); ih = max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    a_area = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    b_area = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = a_area + b_area - inter
    return inter / union if union > 0 else 0.0


def _cross_class_nms(
    raw_boxes: list[tuple[float, float, float, float, str, float]],
    iou_thresh: float = 0.55,
) -> list[tuple[float, float, float, float, str, float]]:
    """
    Suppress boxes that overlap heavily with a higher-priority box of a
    different class. YOLO's per-class NMS leaves cases where the same
    text region wins both a 'title' and a 'plain text' box; we keep the
    higher-priority class (see ``_LABEL_PRIORITY``) and drop the other.

    Confidence is used as a tiebreaker within the same priority bucket so
    a slightly weaker duplicate of the same class still gets removed.
    """
    if not raw_boxes:
        return raw_boxes

    # Sort by (priority desc, confidence desc) so the first kept box always
    # outranks subsequent overlaps.
    def _rank(b):
        _, _, _, _, lbl, conf = b
        return (_LABEL_PRIORITY.get(lbl, 30), conf)

    ordered = sorted(raw_boxes, key=_rank, reverse=True)
    kept: list[tuple[float, float, float, float, str, float]] = []
    for box in ordered:
        bb = (box[0], box[1], box[2], box[3])
        suppress = False
        for k in kept:
            if _bbox_iou(bb, (k[0], k[1], k[2], k[3])) >= iou_thresh:
                suppress = True
                break
        if not suppress:
            kept.append(box)
    if len(kept) < len(raw_boxes):
        _log.debug(
            f"_cross_class_nms: {len(raw_boxes)} -> {len(kept)} boxes "
            f"(dropped {len(raw_boxes) - len(kept)} cross-class overlaps)"
        )
    return kept


def _has_vertical_gap(
    blk: LayoutBlock,
    band_frac: float = 0.20,
    ink_thresh: int = 180,
    max_band_ink_ratio: float = 0.02,
) -> bool:
    """
    True iff the middle ``band_frac`` (default 20%) of the block's crop
    contains essentially no ink (< 2% of pixels are dark). This is the
    real signal of a two-column header/footer; a single-column heading
    that just happens to be wide has ink across the whole width and
    returns False, so we don't bisect it.
    """
    if blk.crop is None:
        return False
    try:
        arr = np.array(blk.crop.convert("L"), dtype=np.uint8)
    except Exception:
        return False
    if arr.size == 0:
        return False
    h, w = arr.shape
    if w < 40:
        return False
    band_w = max(8, int(w * band_frac))
    cx = w // 2
    band = arr[:, max(0, cx - band_w // 2): min(w, cx + band_w // 2)]
    if band.size == 0:
        return False
    ink_ratio = float((band < ink_thresh).sum()) / float(band.size)
    return ink_ratio < max_band_ink_ratio


def _split_wide_blocks(
    blocks: list[LayoutBlock],
    page_width: float,
    split_labels: set | None = None,
    threshold: float = 0.55,
) -> list[LayoutBlock]:
    """
    Split *true* two-column header/footer blocks at the page midpoint.

    Previously this also split ``title`` blocks, which bisected wide
    single-column salutations like ``Kính gửi: Tập đoàn Bưu chính Viễn
    thông Việt Nam`` into a garbled left/right pair. Titles are now left
    alone — only page-header / page-footer get the split, and only when
    we see a clear vertical whitespace band in the middle of the block
    (gap detector), which is the actual marker of two-column layout.
    """
    if split_labels is None:
        split_labels = {"page-header", "Page-header", "page-footer",
                        "Page-footer"}
    result  = []
    next_id = max((b.id for b in blocks), default=0) + 1
    mid     = page_width / 2.0

    for blk in blocks:
        bx0, by0, bx1, by1 = blk.bbox
        if blk.label not in split_labels or (bx1 - bx0) < page_width * threshold:
            result.append(blk)
            continue

        # Gap detector: only split when there's actually a vertical
        # whitespace band near the middle of the block. A single-column
        # wide title with no gap should stay whole.
        if not _has_vertical_gap(blk):
            result.append(blk)
            continue

        left      = _copy(blk)
        left.bbox = (bx0, by0, min(bx1, mid - 4), by1)
        if blk.crop:
            lw = max(1, int(min(bx1, mid - 4) - bx0))
            left.crop = blk.crop.crop((0, 0, min(lw, blk.crop.width), blk.crop.height))

        right      = _copy(blk)
        right.id   = next_id
        next_id   += 1
        right.bbox = (max(bx0, mid + 4), by0, bx1, by1)
        if blk.crop:
            lw = max(1, int(min(bx1, mid - 4) - bx0))
            right.crop = blk.crop.crop((min(lw, blk.crop.width), 0,
                                        blk.crop.width, blk.crop.height))
        result.extend([left, right])
    return result


# ══════════════════════════════════════════════════════════════════════════════
# § D  —  OCR RUNNER
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class BlockOCRResult:
    block: LayoutBlock
    lines: list[OCRLine]


class BlockOCRRunner:
    """
    Run EasyOCR NPU recognize_region() per layout block.
    NEVER calls recognize() — YOLO already found the bboxes.

    ``ocr_figures``: when True, figure / Picture / Image blocks are
    *also* OCR'd. Useful for Vietnamese government letters where the
    title is rendered into the header banner image (e.g. "TẬP ĐOÀN
    BƯU CHÍNH VIỄN THÔNG"). The extracted text becomes the figure
    block's ``original_text`` and is rendered above the embedded
    image in DOCX / Markdown.
    """

    def __init__(self, ocr_engine, *, ocr_figures: bool = True):
        self.engine      = ocr_engine
        self.ocr_figures = ocr_figures

    def process_blocks(self, blocks: list[LayoutBlock]) -> list[BlockOCRResult]:
        # Lazy import: keeps document_pipeline importable when text_utils
        # is missing (older deployments). Failure here just disables the
        # post-OCR cleanup, not the OCR itself.
        try:
            from text_utils import dehyphenate
        except ImportError:
            def dehyphenate(s: str) -> str:
                return s

        # Default skip set: tables get their own per-cell OCR path,
        # abandon = page furniture we don't want in the body. Figures
        # are skipped *unless* the caller asked for figure-text
        # extraction (titles burnt into banner images).
        skip = _TABLE_LABELS | _ABANDON_LABELS
        if not self.ocr_figures:
            skip = skip | _NON_TEXT_LABELS
        results = []

        for blk in blocks:
            if blk.label in skip or blk.crop is None:
                continue
            is_figure = blk.label in _NON_TEXT_LABELS
            try:
                # Multi-line figure regions (header banners, e.g. national
                # emblem + 3-line institution name) need each visible line
                # preserved. Body text and table-cell paths join with " "
                # because in-flow text should stay as one paragraph.
                if is_figure:
                    try:
                        text, conf = self.engine.recognize_region(
                            blk.crop, line_separator="\n",
                        )
                    except TypeError:
                        # Older engines without the kwarg.
                        text, conf = self.engine.recognize_region(blk.crop)
                else:
                    text, conf = self.engine.recognize_region(blk.crop)
            except Exception as e:
                _log.warning(f"OCR block {blk.id}: {e}")
                text, conf = "", 0.0

            # Rejoin words broken across OCR line splits. Conservative
            # heuristic — only joins when the continuation starts with a
            # lowercase letter (Vietnamese tonal vowels included). Skip
            # dehyphenation for figure blocks so their explicit line
            # breaks survive.
            if not is_figure:
                text = dehyphenate(text)

            lines = []
            if text.strip():
                for raw in text.split("\n"):
                    s = raw.strip()
                    if s:
                        lines.append(OCRLine(text=s, confidence=conf))
            results.append(BlockOCRResult(block=blk, lines=lines))

        _log.info(f"OCR: {sum(len(r.lines) for r in results)} lines "
                  f"across {len(results)} blocks")
        return results


# ══════════════════════════════════════════════════════════════════════════════
# § E  —  TABLE RECONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════
#
# TATR with per-class thresholds + border-line span detection.
#
# The border-line approach (replacing the failed pixel-content approach):
#   For each column independently, at each row boundary y-position,
#   check whether a real horizontal border line exists in that column's x-range.
#   If no border → the rows share a merged cell → absorb into anchor row.
#
# This correctly identifies that col 1 (dense spec text) has borders at every
# row boundary, while col 0 (device name) and col 2 (date) have none.

# Re-exported from core.table_engines.base for backwards compatibility with
# any caller that did `from document_pipeline import _nms, _tatr_iou, ...`.
# (TATR_THRESH and the LegacyTATRCPUReconstructor alias were removed when
# the TATR engines were retired in favour of img2table.)
from table_engines.base import (  # noqa: E402
    tatr_iou as _tatr_iou,
    nms as _nms,
    infer_rowspans_by_border as _infer_rowspans_by_border,
)
from table_engines import build_reconstructor  # noqa: E402


# ══════════════════════════════════════════════════════════════════════════════
# § F  —  IR BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def _normalise_label(label: str) -> str:
    mapping = {
        # ── Ultralytics / DocLayNet labels ───────────────────────────────
        "title":"title",         "Title":"title",
        "text":"text",           "Text":"text",
        "section_header":"section_header",
        "Section-header":"section_header",
        "page-header":"page_header",  "Page-header":"page_header",
        "page-footer":"page_footer",  "Page-footer":"page_footer",
        "caption":"caption",     "Caption":"caption",
        "table":"table",         "Table":"table",
        "figure":"figure",       "Figure":"figure",
        "image":"figure",        "Image":"figure",
        "Picture":"figure",      "picture":"figure",
        "list_item":"list_item", "formula":"formula",
        # ── DocLayout-YOLO DocStructBench labels ─────────────────────────
        # title, table, figure already covered above.
        "plain text":      "text",
        "plain_text":      "text",
        "abandon":         "page_footer",      # treated as page chrome
        "figure_caption":  "caption",
        "table_caption":   "caption",
        "table_footnote":  "caption",
        "isolate_formula": "figure",           # render as image, can't translate
        "formula_caption": "caption",
    }
    return mapping.get(label, "text")


def _is_garbled(lines: list[OCRLine]) -> bool:
    """True for blocks that are clearly unreadable (stamp, signature, noise)."""
    import unicodedata
    if not lines:
        return True
    confs = [l.confidence for l in lines if l.confidence > 0]
    if not confs or sum(confs) / len(confs) < 0.25:
        return True
    text    = " ".join(l.text for l in lines)
    letters = sum(1 for c in text if unicodedata.category(c)[0] in ('L', 'N'))
    if letters < 3:
        return True
    garbage = sum(1 for c in text
                  if unicodedata.category(c) in ('Cc', 'Cf', 'Co', 'Cs')
                  or (unicodedata.category(c) == 'Po' and c in '<>{}[]|\\^~`'))
    return len(text) > 0 and garbage / len(text) > 0.50


def _table_fill_ratio(rows: list[TableRow]) -> float:
    """Fraction of anchor cells that contain non-trivial text. Used as
    a quality signal for detecting mis-classified Table blocks (TOC,
    outline pages, etc.)."""
    total = 0
    filled = 0
    for row in rows:
        for cell in row.cells:
            if cell.row_span == 0 or cell.col_span == 0:
                continue
            total += 1
            if (cell.text or "").strip():
                filled += 1
    return filled / total if total > 0 else 0.0


def _looks_like_misclassified_table(rows: list[TableRow]) -> bool:
    """
    Heuristic to detect "this isn't really a table" — typically TOC
    pages where YOLO confused the dotted-leader columns with table
    rules. Returns True to demote the block to paragraph rendering.

    Calibrated against false positives we hit earlier: short Vietnamese
    spec tables are commonly ``header + 1 data row`` with one big text
    cell on the right and empty middle cells — that's still a valid
    table and must NOT be demoted. We protect those by keeping all
    ``n_rows ≤ 2`` tables that have any content, and only demoting
    larger grids when the fill is genuinely poor or the TOC pattern
    matches.

    Signals — ALL must point to "not a real table":

    1. Tables with ≤2 rows are always kept (header + body pattern).
    2. Larger tables (≥3 rows) with very low fill (<20%) are demoted.
    3. ≥3 cells matching the TOC dotted-leader+page-number signature
       (``... 12``) demote regardless of size.
    """
    if not rows:
        return True

    fill   = _table_fill_ratio(rows)
    n_rows = len(rows)

    # Tiny tables: keep unless completely empty. Header+body is the
    # default shape for Vietnamese government spec tables and the rule
    # we had here was eating legitimate ones.
    if n_rows <= 2:
        return fill == 0.0

    # Larger grids with almost no content are misclassifications.
    if fill < 0.20:
        return True

    # TOC dotted-leader signature: ``<title>...<digits>``. Needs ≥3
    # matching cells across the grid to fire; one or two false hits
    # aren't enough.
    import re
    toc_like = 0
    for row in rows:
        for cell in row.cells:
            if cell.row_span == 0 or cell.col_span == 0:
                continue
            txt = (cell.text or "").strip()
            if not txt or len(txt) < 12:
                continue
            if re.search(r"[.\s]{3,}\d{1,4}\s*$", txt):
                toc_like += 1
    if toc_like >= 3:
        return True

    return False


def _assign_reading_order(blocks: list[IRBlock]) -> list[IRBlock]:
    """Two-column aware reading order (LayoutParser column_sort approach)."""
    if not blocks:
        return blocks

    page_x0 = min(b.bbox[0] for b in blocks)
    page_x1 = max(b.bbox[2] for b in blocks)
    mid     = (page_x0 + page_x1) / 2.0

    lefts  = [b for b in blocks if b.bbox[2] <= mid * 1.1]
    rights = [b for b in blocks if b.bbox[0] >= mid * 0.9]
    two_col = len(lefts) >= 2 and len(rights) >= 2

    if two_col:
        col_l = sorted([b for b in blocks if b.bbox[0] < mid],
                       key=lambda b: b.bbox[1])
        col_r = sorted([b for b in blocks if b.bbox[0] >= mid],
                       key=lambda b: b.bbox[1])
        merged = []
        li = ri = 0
        while li < len(col_l) and ri < len(col_r):
            if col_l[li].bbox[1] <= col_r[ri].bbox[1]:
                merged.append(col_l[li]); li += 1
            else:
                merged.append(col_r[ri]); ri += 1
        merged.extend(col_l[li:])
        merged.extend(col_r[ri:])
        ordered = merged
    else:
        ordered = sorted(blocks, key=lambda b: (b.bbox[1], b.bbox[0]))

    for idx, b in enumerate(ordered):
        b.reading_order = idx
    return ordered


def build_ir(
    img_path:       str | Path,
    layout_blocks:  list[LayoutBlock],
    ocr_results:    list[BlockOCRResult],
    table_rows_map: dict[int, list[TableRow]],
    page_index:     int = 0,
    figure_dir:     str | Path | None = None,
) -> DocumentPage:
    """Assemble DocumentPage IR from all upstream stages.

    When ``figure_dir`` is provided, figure-type blocks have their YOLO crop
    saved as a PNG inside that directory and the path is recorded on the
    IR block so downstream exporters can embed the actual image instead of
    a placeholder.
    """
    from PIL import Image as _PIL

    # Lazy + defensive: text_utils is dependency-free but we don't want a
    # missing module to break IR building. Falls back to a no-op.
    try:
        from text_utils import reclassify_block_type
    except ImportError:
        def reclassify_block_type(current_type: str, _text: str) -> str:
            return current_type

    with _PIL.open(str(img_path)) as im:
        w, h = im.size

    ocr_map    = {r.block.id: r for r in ocr_results}
    ir_blocks: list[IRBlock] = []

    fig_dir_path: Path | None = None
    if figure_dir is not None:
        fig_dir_path = Path(figure_dir)
        fig_dir_path.mkdir(parents=True, exist_ok=True)

    for blk in layout_blocks:
        is_table   = blk.label in _TABLE_LABELS
        is_figure  = blk.label in _NON_TEXT_LABELS
        ocr_lines  = ocr_map[blk.id].lines if blk.id in ocr_map else []

        if ocr_lines:
            # Figure blocks: keep one visible line per OCR line so the
            # banner / header rendering preserves vertical layout.
            # Everything else: join with " " — those are inline-flow
            # paragraphs where the OCR's per-line split is a side
            # effect of CRAFT detection, not intentional.
            if is_figure:
                original_text = "\n".join(l.text for l in ocr_lines)
            else:
                original_text = " ".join(l.text for l in ocr_lines)
        elif is_table:
            rows = table_rows_map.get(blk.id, [])
            original_text = " | ".join(
                c.text for row in rows for c in row.cells
                if c.text.strip() and c.row_span != 0
            )
        else:
            original_text = ""

        # Skip from text-output rendering only when the block is garbled
        # (stamp / signature / noise) on a non-table, non-figure block.
        # Previously "abandon"-labelled blocks (page furniture per the
        # layout model) were also force-skipped — that silently dropped
        # legitimate running headers/footers (e.g. "CÔNG BÁO/Số .../Ngày
        # ..." + page number) whenever the model tagged them "abandon".
        # They're now kept so the exported DOCX/Markdown preserves the
        # source document's own header/footer lines in place. Figures
        # are kept too so OCR text caught inside a banner / logo / header
        # image survives as the figure's caption; the figure-rendering
        # branch in page_to_markdown and table_docx_writer emits both
        # the image and the text.
        skip = (not is_table and not is_figure and _is_garbled(ocr_lines))

        # Layout-model label is a guess — text that unambiguously reads
        # as "Chương .../Điều N." gets promoted to a real heading even
        # if YOLO mislabelled it as plain body text.
        ir_type = _normalise_label(blk.label)
        if not is_table and not is_figure:
            ir_type = reclassify_block_type(ir_type, original_text)

        # Save figure crop to disk so the exporters can embed it. We keep
        # the original YOLO bounding-box crop (not a re-render) to preserve
        # any vector elements / line art exactly as the layout model saw them.
        figure_path = ""
        if is_figure and fig_dir_path is not None and blk.crop is not None:
            fig_file = fig_dir_path / f"p{page_index:03d}_b{blk.id:03d}.png"
            try:
                blk.crop.save(str(fig_file))
                figure_path = str(fig_file)
            except Exception as e:
                _log.warning(f"figure save failed for block {blk.id}: {e}")

        ir_blocks.append(IRBlock(
            id            = blk.id,
            type          = ir_type,
            bbox          = blk.bbox,
            ocr_lines     = ocr_lines,
            table_rows    = table_rows_map.get(blk.id, []),
            original_text = original_text,
            skip          = skip,
            figure_path   = figure_path,
        ))

    ir_blocks = _assign_reading_order(ir_blocks)
    _log.info(f"IR page {page_index}: {len(ir_blocks)} blocks "
              f"({sum(1 for b in ir_blocks if b.type=='table')} tables, "
              f"{sum(1 for b in ir_blocks if b.skip)} skipped)")
    return DocumentPage(page_index=page_index,
                        width_px=float(w), height_px=float(h),
                        source_path=str(img_path), blocks=ir_blocks)


# ══════════════════════════════════════════════════════════════════════════════
# § G  —  TRANSLATION ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class TranslationEngine:
    """
    Translate DocumentPage blocks one item at a time.
    One API call per block — reliable with 4B models.
    Includes two-stage OCR fix pass before translation.

    Progress reporting
    ------------------
    The "Translating N items" and per-item counter lines are emitted
    with ``print()`` (not ``_log.info``) on purpose: the UI's progress
    parser reads subprocess stdout and only sees lines without the
    ``[INFO] __main__ — `` prefix. Logging-style lines still go to
    the on-disk run log via the FileHandler.
    """

    def __init__(self, direction: str = "vi->en"):
        self.direction = direction

    def translate_page(
        self, page: DocumentPage,
        *,
        page_label: str | None = None,
    ) -> DocumentPage:
        from translator import translate_text, _call_llm

        src      = self.direction.split("->")[0]
        src_lang = {"vi": "Vietnamese", "en": "English"}.get(src, src)

        items: list[tuple] = []
        for blk in page.blocks:
            if blk.skip:
                continue
            if blk.type == "table":
                for row in blk.table_rows:
                    for cell in row.cells:
                        if cell.row_span != 0 and cell.text.strip():
                            items.append(("cell", cell))
            else:
                if blk.original_text.strip():
                    items.append(("block", blk))

        if not items:
            return page

        # Emit to stdout (not _log.info) so the UI's progress parser
        # picks them up — the UI reads subprocess.stdout and skips
        # logger-formatted "[INFO] __main__ —" prefixed lines.
        if page_label:
            print(f"  {page_label} …")
        print(f"  Translating {len(items)} items ({self.direction})")
        _log.info(f"Translating {len(items)} items ({self.direction})")

        for i, item in enumerate(items):
            if i % 10 == 0:
                print(f"  {i}/{len(items)}", flush=True)
                _log.info(f"  {i}/{len(items)}")

            if item[0] == "block":
                blk   = item[1]
                fixed = self._fix_ocr(blk.original_text, src_lang, _call_llm)
                blk.translated_text = translate_text(fixed, self.direction)
            else:
                cell  = item[1]
                fixed = self._fix_ocr(cell.text, src_lang, _call_llm)
                cell.translated = translate_text(fixed, self.direction)

        # Build table block-level translated_text from cells
        for blk in page.blocks:
            if blk.type == "table" and blk.table_rows:
                blk.translated_text = "\n".join(
                    " | ".join(c.translated or c.text for c in row.cells
                               if c.row_span != 0)
                    for row in blk.table_rows
                )

        return page

    def _fix_ocr(self, text: str, src_lang: str, _call_llm) -> str:
        import re, sys
        from pathlib import Path as _P

        try:
            sys.path.insert(0, str(_P(__file__).parent))
            from easyocr_qairt_engine import clean_ocr_text
            text = clean_ocr_text(text)
        except Exception:
            pass

        noise = len(re.findall(r'(?:^1 |\b1 (?=[A-Z])|[%\[\]\{\}\|<>])', text))
        if noise == 0 or len(text) < 8:
            return text

        system = (
            f"You are an OCR post-processor for {src_lang} documents. "
            "Fix OCR errors (spurious '1 ', wrong letters, garbled words). "
            f"Return ONLY the fixed {src_lang} text. Do not translate."
        )
        result = _call_llm(f"Fix OCR errors:\n{text}", system)
        if result and 0.4 * len(text) < len(result) < 2.5 * len(text):
            return result
        return text


# ══════════════════════════════════════════════════════════════════════════════
# § H  —  MARKDOWN BUILDER
# ══════════════════════════════════════════════════════════════════════════════
#
# IR → Markdown string per page.
# This is the key architectural shift: instead of writing python-docx XML
# directly (which requires vMerge, coordinate math, etc.), we build clean
# Markdown first. Tables become pipe-delimited rows — no merge complexity.
#
# PP-Structure recovery_to_doc.py does the same: iterate blocks in reading
# order, emit structured text output per block type.

def page_to_markdown(page: DocumentPage, use_translated: bool = True) -> str:
    """
    Convert one DocumentPage IR to a Markdown string.

        title / section_header / page_header  →  ## Heading
        text / list_item / formula            →  Paragraph
        caption                               →  *italic*
        page_footer                           →  _italic_
        table (with rows)                     →  | pipe table |
        figure                                →  [Figure]
        skip=True (non-table)                 →  omitted
    """
    lines: list[str] = []

    for blk in sorted(page.blocks, key=lambda b: b.reading_order):

        # ── Figure ────────────────────────────────────────────────────────────
        if blk.type == "figure":
            # If the OCR runner extracted text from inside the figure
            # (titles burnt into banner images, e.g. "TẬP ĐOÀN BƯU CHÍNH
            # VIỄN THÔNG"), emit it above the image so it survives into
            # the translated output. Translated text wins over original
            # if available; fall back to original OCR otherwise.
            if use_translated:
                fig_text = (blk.translated_text or "").strip() \
                           or (blk.original_text or "").strip()
            else:
                fig_text = (blk.original_text or "").strip()
            if fig_text:
                # Force visible line breaks by emitting each detected
                # OCR line as its own Markdown paragraph. Markdown
                # collapses single newlines inside a paragraph, so we
                # use a blank line between to guarantee separation.
                fig_paragraphs = [
                    p.strip() for p in fig_text.split("\n") if p.strip()
                ]
                lines.append("\n" + "\n\n".join(fig_paragraphs) + "\n")
            if blk.figure_path:
                # Emit a real Markdown image reference. The DOCX exporter
                # recognises this syntax and embeds the file; Markdown viewers
                # (GitHub, VS Code, etc.) render it inline.
                lines.append(f"\n![Figure]({blk.figure_path})\n")
            elif not fig_text:
                lines.append("\n[Figure]\n")
            continue

        # ── Skip garbled non-table blocks ─────────────────────────────────────
        if blk.skip:
            continue

        # ── Table ─────────────────────────────────────────────────────────────
        if blk.type == "table" and blk.table_rows:
            md_table = _table_to_markdown(blk.table_rows, use_translated)
            if md_table:
                lines.append("")
                lines.append(md_table)
                lines.append("")
            continue

        # ── Text blocks ───────────────────────────────────────────────────────
        # Graceful fallback: if asked for the translated string and it's
        # missing (LLM server down, expansion-guard rejected the response,
        # or just empty), use the original OCR text so the document isn't
        # blank. Mirrors the same fallback already in table_docx_writer.
        if use_translated:
            text = (blk.translated_text or "").strip() \
                or (blk.original_text or "").strip()
        else:
            text = (blk.original_text or "").strip()
        if not text:
            continue

        if blk.type in ("title", "section_header", "page_header"):
            lines.append(f"\n## {text}\n")
        elif blk.type == "caption":
            lines.append(f"\n*{text}*\n")
        elif blk.type == "page_footer":
            lines.append(f"\n_{text}_\n")
        else:
            lines.append(f"\n{text}\n")

    return "\n".join(lines)


def _table_to_markdown(rows: list[TableRow], use_translated: bool) -> str:
    """
    Convert TableRow/TableCell grid to a Markdown pipe table.

    Absorbed cells (row_span=0) render as empty cells in their row.
    Anchor cells with row_span>1 carry their content in the first row only
    — Markdown pipe tables don't support cell merging, which is fine.
    The content is correct; rendering software can optionally merge later.
    """
    if not rows:
        return ""

    num_cols = max(len(r.cells) for r in rows)
    if num_cols == 0:
        return ""

    md_rows: list[str] = []

    for r_idx, row in enumerate(rows):
        cells_text: list[str] = []
        for c_idx in range(num_cols):
            if c_idx < len(row.cells):
                cell = row.cells[c_idx]
                if cell.row_span == 0:
                    cells_text.append("")
                else:
                    # Fall back to original OCR text when translation is
                    # missing so the table renders something rather than
                    # a row of empty pipes.
                    if use_translated:
                        t = (cell.translated or "").strip() \
                            or (cell.text or "").strip()
                    else:
                        t = (cell.text or "").strip()
                    # Escape pipes (would break the row), and convert
                    # in-cell line breaks to <br> — GitHub-flavored
                    # Markdown renders <br> inside table cells as a real
                    # line break. A bare \n would split the row instead.
                    t = t.replace("|", "\\|").replace("\n", "<br>")
                    cells_text.append(t)
            else:
                cells_text.append("")

        md_rows.append("| " + " | ".join(cells_text) + " |")

        if r_idx == 0:
            md_rows.append("|" + "|".join(["---"] * num_cols) + "|")

    return "\n".join(md_rows)


# ══════════════════════════════════════════════════════════════════════════════
# § I  —  DOCX EXPORTER
# ══════════════════════════════════════════════════════════════════════════════
#
# Markdown → DOCX using python-docx.
# Simple and correct: no vMerge XML, no coordinate math.
# Tables: pipe rows → Word table rows, first row bold.

# Match a Markdown image reference on a line on its own:
#   ![alt text](path/to/image.png)
# We use named groups so the embed code is readable.
import re as _re
_MD_IMAGE_RE = _re.compile(r"^!\[(?P<alt>[^\]]*)\]\((?P<path>[^)]+)\)\s*$")


def markdown_to_docx(
    markdown_pages: list[str],
    output_path:    str | Path,
) -> Path:
    """
    Convert a list of per-page Markdown strings to a single DOCX file.
    Hỗ trợ cả Markdown Table (|...|) và HTML Table (<table>) từ DeepSeek-OCR-2.
    """
    from docx import Document
    from docx.shared import Pt, Inches
    from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK_TYPE
    from docx.oxml.ns import qn as _qn
    from docx.oxml import OxmlElement as _OXE
    from bs4 import BeautifulSoup

    doc = Document()
    for section in doc.sections:
        section.top_margin    = Inches(0.75)
        section.bottom_margin = Inches(0.75)
        section.left_margin   = Inches(0.75)
        section.right_margin  = Inches(0.75)

    body = doc.element.body
    for elem in list(body):
        if elem.tag != _qn('w:sectPr'):
            body.remove(elem)

    text_width_in = 8.5 - 0.75 * 2   # 7 inches

    for page_idx, md in enumerate(markdown_pages):
        if page_idx > 0:
            para = doc.add_paragraph()
            para.add_run().add_break(WD_BREAK_TYPE.PAGE)

        md_lines = md.split("\n")
        n        = len(md_lines)
        i        = 0

        while i < n:
            line = md_lines[i]
            stripped = line.strip()

            # ── 1. Heading ────────────────────────────────────────────────────
            if line.startswith("## "):
                text = line[3:].strip()
                if text:
                    p   = doc.add_paragraph()
                    run = p.add_run(text)
                    run.bold      = True
                    run.font.size = Pt(11)
                    p.paragraph_format.space_before = Pt(4)
                    p.paragraph_format.space_after  = Pt(2)
                i += 1
                continue

            # ── 2. HTML Table (DeepSeek End-to-End) ───────────────────────────
            if stripped.startswith("<table"):
                html_lines = [line]
                # Thu thập toàn bộ khối HTML cho đến khi gặp thẻ đóng
                while i + 1 < n and not html_lines[-1].strip().endswith("</table>"):
                    i += 1
                    html_lines.append(md_lines[i])
                
                html_str = "\n".join(html_lines)
                soup = BeautifulSoup(html_str, 'html.parser')
                trs = soup.find_all('tr')
                
                if not trs:
                    i += 1
                    continue
                
                # Tính toán kích thước lưới thực tế (tính cả colspan)
                max_cols = 0
                for tr in trs:
                    cols = sum(int(td.get('colspan', 1)) for td in tr.find_all(['td', 'th']))
                    max_cols = max(max_cols, cols)
                
                tbl = doc.add_table(rows=len(trs), cols=max_cols)
                tbl.style = "Table Grid"
                
                tbl_pr = tbl._tblPr
                tbl_w  = _OXE("w:tblW")
                tbl_w.set(_qn("w:w"), str(int(text_width_in * 1440)))
                tbl_w.set(_qn("w:type"), "dxa")
                tbl_pr.append(tbl_w)
                
                occupied = set() # Ma trận theo dõi các ô đã bị gộp
                
                for r_idx, tr in enumerate(trs):
                    c_idx = 0
                    for td in tr.find_all(['td', 'th']):
                        # Bỏ qua các ô đã bị chiếm dụng bởi rowspan/colspan trước đó
                        while (r_idx, c_idx) in occupied:
                            c_idx += 1
                        
                        rowspan = int(td.get('rowspan', 1))
                        colspan = int(td.get('colspan', 1))
                        
                        start_cell = tbl.cell(r_idx, c_idx)
                        
                        # Xử lý gộp ô (Merge)
                        if rowspan > 1 or colspan > 1:
                            end_cell = tbl.cell(r_idx + rowspan - 1, c_idx + colspan - 1)
                            start_cell.merge(end_cell)
                            for r in range(r_idx, r_idx + rowspan):
                                for c in range(c_idx, c_idx + colspan):
                                    occupied.add((r, c))
                        else:
                            occupied.add((r_idx, c_idx))
                        
                        # Điền văn bản
                        cell_text = td.get_text(separator="\n", strip=True)
                        for p in list(start_cell.paragraphs):
                            p._element.getparent().remove(p._element)
                        p = start_cell.add_paragraph(cell_text)
                        p.paragraph_format.space_before = Pt(0)
                        p.paragraph_format.space_after  = Pt(0)
                        for run in p.runs:
                            if r_idx == 0 or td.name == 'th':
                                run.bold = True
                            run.font.size = Pt(9)
                        
                        c_idx += colspan
                
                doc.add_paragraph()
                i += 1
                continue

            # ── 3. Markdown Pipe Table (Qwen / MinerU) ────────────────────────
            if line.startswith("|"):
                tbl_lines = []
                while i < n and md_lines[i].startswith("|"):
                    tbl_lines.append(md_lines[i])
                    i += 1

                data_lines = [l for l in tbl_lines if not all(c in "|-: \t" for c in l.strip())]
                if not data_lines:
                    continue

                rows_data = []
                for tl in data_lines:
                    cells = [c.strip() for c in tl.strip().strip("|").split("|")]
                    rows_data.append(cells)

                if not rows_data:
                    continue

                num_cols = max(len(r) for r in rows_data)
                tbl      = doc.add_table(rows=len(rows_data), cols=num_cols)
                tbl.style = "Table Grid"

                tbl_pr = tbl._tblPr
                tbl_w  = _OXE("w:tblW")
                tbl_w.set(_qn("w:w"), str(int(text_width_in * 1440)))
                tbl_w.set(_qn("w:type"), "dxa")
                tbl_pr.append(tbl_w)

                for r_idx, row_data in enumerate(rows_data):
                    for c_idx in range(num_cols):
                        ct   = row_data[c_idx] if c_idx < len(row_data) else ""
                        ct   = ct.replace("\\|", "|").replace("<br>", "\n")
                        wcell = tbl.cell(r_idx, c_idx)
                        for p in list(wcell.paragraphs):
                            p._element.getparent().remove(p._element)
                        p   = wcell.add_paragraph(ct)
                        p.paragraph_format.space_before = Pt(0)
                        p.paragraph_format.space_after  = Pt(0)
                        for run in p.runs:
                            if r_idx == 0:
                                run.bold = True
                            else:
                                run.font.size = Pt(9)

                doc.add_paragraph()
                continue

            # ── 4. Figure ─────────────────────────────────────────────────────
            if stripped == "[Figure]":
                p = doc.add_paragraph("[Figure]")
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                i += 1
                continue
            
            _img_match = _MD_IMAGE_RE.match(stripped)
            if _img_match:
                img_path = _img_match.group("path")
                if img_path and Path(img_path).exists():
                    p = doc.add_paragraph()
                    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    try:
                        p.add_run().add_picture(img_path, width=Inches(6.0))
                    except Exception as e:
                        p.add_run("[Figure]")
                else:
                    p = doc.add_paragraph("[Figure]")
                    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                i += 1
                continue

            # ── 5. Italic lines (*text* or _text_) ────────────────────────────
            if ((stripped.startswith("*") and stripped.endswith("*"))
                    or (stripped.startswith("_") and stripped.endswith("_"))):
                text = stripped[1:-1].strip()
                if text:
                    p   = doc.add_paragraph()
                    run = p.add_run(text)
                    run.italic    = True
                    run.font.size = Pt(9)
                i += 1
                continue

            # ── 6. Regular paragraph ──────────────────────────────────────────
            if stripped:
                p   = doc.add_paragraph()
                run = p.add_run(stripped)
                run.font.size = Pt(10)
                p.paragraph_format.space_before = Pt(2)
                p.paragraph_format.space_after  = Pt(2)
            i += 1

    output_path = Path(output_path)
    doc.save(str(output_path))
    return output_path


def docx_to_pdf(docx_path: Path, output_dir: Path) -> Path:
    """Convert DOCX → PDF via docx2pdf (Word) or LibreOffice."""
    out = output_dir / f"{docx_path.stem}.pdf"
    try:
        from docx2pdf import convert
        convert(str(docx_path), str(out))
        if out.exists() and out.stat().st_size > 0:
            return out
    except Exception:
        pass
    try:
        import subprocess
        r = subprocess.run(
            ["soffice", "--headless", "--convert-to", "pdf",
             "--outdir", str(output_dir), str(docx_path)],
            capture_output=True, timeout=60)
        if r.returncode == 0 and out.exists():
            return out
    except Exception:
        pass
    _log.warning("PDF conversion unavailable — returning DOCX")
    return docx_path


# ══════════════════════════════════════════════════════════════════════════════
# § J  —  PIPELINE ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# § J0 —  DEBUG DUMP (stage-by-stage artefacts)
# ══════════════════════════════════════════════════════════════════════════════

def _debug_dump_page(
    dump_dir,
    pg:              int,
    img_path,
    blocks:          list,
    ocr_results:     list,
    table_rows_map:  dict,
    page_ir,
) -> None:
    """
    Save per-stage artefacts for one page under ``<dump_dir>/page_<NN>/``:

        page.png                  copy of rasterised page
        layout.png                page with bbox + label overlay
        layout.json               raw bbox list
        ocr.json                  per-block OCR text + confidence
        tables/                   per-table cropped PNG + IR JSON
        ir.json                   full DocumentPage IR (reading order, spans)

    These let us inspect what each stage produced without re-running the
    pipeline.
    """
    import json
    import shutil
    from PIL import Image, ImageDraw, ImageFont

    pdir = Path(dump_dir) / f"page_{pg:02d}"
    pdir.mkdir(parents=True, exist_ok=True)

    # 1. raw page PNG
    try:
        shutil.copyfile(str(img_path), str(pdir / "page.png"))
    except Exception:
        pass

    # 2. layout overlay
    try:
        page_img = Image.open(str(img_path)).convert("RGB")
        canvas   = page_img.copy()
        draw     = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except Exception:
            font = ImageFont.load_default()
        palette = {
            "table":   "#d62728",
            "Table":   "#d62728",
            "figure":  "#9467bd",
            "Figure":  "#9467bd",
            "picture": "#9467bd",
            "Picture": "#9467bd",
        }
        for b in blocks:
            x0, y0, x1, y1 = b.bbox
            colour = palette.get(b.label, "#1f77b4")
            draw.rectangle([(x0, y0), (x1, y1)], outline=colour, width=3)
            label = f"#{b.id} {b.label}"
            draw.rectangle(
                [(x0, max(0, y0 - 16)), (x0 + 8 * len(label), y0)],
                fill=colour,
            )
            draw.text((x0 + 2, max(0, y0 - 16)), label,
                      fill="white", font=font)
        canvas.save(str(pdir / "layout.png"))
    except Exception as e:
        _log.warning(f"debug-dump layout overlay failed: {e}")

    # 3. layout json
    (pdir / "layout.json").write_text(
        json.dumps(
            [{"id": b.id, "label": b.label, "bbox": list(b.bbox)}
             for b in blocks],
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # 4. ocr json
    (pdir / "ocr.json").write_text(
        json.dumps(
            [
                {
                    "block_id": r.block.id,
                    "label":    r.block.label,
                    "lines":    [
                        {"text": l.text, "confidence": l.confidence}
                        for l in r.lines
                    ],
                }
                for r in ocr_results
            ],
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # 5. per-table crops + IR + coverage stats
    if table_rows_map:
        tdir = pdir / "tables"
        tdir.mkdir(exist_ok=True)
        block_by_id = {b.id: b for b in blocks}
        page_table_stats: list[dict] = []
        for tb_id, rows in table_rows_map.items():
            blk = block_by_id.get(tb_id)
            if blk is None:
                continue
            try:
                if blk.crop is not None:
                    blk.crop.save(str(tdir / f"table_{tb_id:02d}.png"))
            except Exception:
                pass
            (tdir / f"table_{tb_id:02d}.json").write_text(
                json.dumps(
                    [
                        [
                            {
                                "row":      c.row,
                                "col":      c.col,
                                "row_span": c.row_span,
                                "col_span": c.col_span,
                                "bbox":     list(c.bbox),
                                "text":     c.text,
                            }
                            for c in row.cells
                        ]
                        for row in rows
                    ],
                    indent=2, ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            # Per-table coverage stats — useful for diagnosing
            # img2table misses and cell-OCR garbage. An anchor cell
            # is one with row_span > 0 and col_span > 0; mergee
            # cells (span 0) are intentionally skipped from the
            # denominator.
            n_rows        = len(rows)
            anchor_cells  = [c for r in rows for c in r.cells
                             if c.row_span and c.col_span]
            n_anchors     = len(anchor_cells)
            n_text        = sum(1 for c in anchor_cells if c.text.strip())
            n_short_text  = sum(1 for c in anchor_cells
                                if 0 < len(c.text.strip()) < 3)
            n_long_garbage = sum(
                1 for c in anchor_cells
                if len(c.text) > 80 and " " not in c.text
            )
            page_table_stats.append({
                "table_id":              tb_id,
                "rows":                  n_rows,
                "anchor_cells":          n_anchors,
                "cells_with_text":       n_text,
                "text_coverage_pct":     round(
                    100.0 * n_text / n_anchors, 1
                ) if n_anchors else 0.0,
                "very_short_cells":      n_short_text,
                "suspect_garbage_cells": n_long_garbage,
            })

        if page_table_stats:
            (tdir / "_stats.json").write_text(
                json.dumps(page_table_stats, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

    # 6. full IR
    (pdir / "ir.json").write_text(
        json.dumps(
            {
                "page_index":    page_ir.page_index,
                "width_px":      page_ir.width_px,
                "height_px":     page_ir.height_px,
                "blocks": [
                    {
                        "id":             b.id,
                        "type":           b.type,
                        "bbox":           list(b.bbox),
                        "reading_order":  b.reading_order,
                        "skip":           b.skip,
                        "original_text":  b.original_text,
                        "n_table_rows":   len(b.table_rows),
                    }
                    for b in page_ir.blocks
                ],
            },
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"        debug-dump → {pdir}")


class DocumentTranslationPipeline:
    """
    Full end-to-end pipeline: PDF/Image → translated Markdown + DOCX + PDF.

    Flow per page:
        Rasterise → Layout → OCR → TATR → IR Build → Translation
    Then all pages together:
        Markdown Build → DOCX Export → PDF Convert
    """

    def __init__(
        self,
        yolo_model:           str,
        qairt_sdk:            str,
        model_dir:            str,
        lang:                 str               = "vi",
        direction:            str               = "vi->en",
        output_dir:           str | Path        = "output",
        dpi:                  int               = 150,
        table_engine:         str               = "img2table",
        translate:            bool              = True,
        debug_dump:           str | os.PathLike | None = None,
        ocr_engine:           str               = "vietocr",
        vietocr_weights:      str | os.PathLike | None = None,
        vietocr_variant:      str               = "vgg_transformer",
        vietocr_beam:         bool              = False,
        layout_backend:       str               = "auto",
        layout_conf:          float | None      = None,
        layout_imgsz:         int               = 1024,
        ocr_figures:          bool              = True,
        engine_mode:          str               = "blocks",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # "blocks" (mặc định)  = pipeline hiện có: YOLO → OCR per-block → TATR
        #                        → IR → dịch → markdown/docx. KHÔNG đổi gì.
        # "end2end"            = bỏ qua toàn bộ YOLO/table-structure/IR, đưa
        #                        thẳng ảnh CẢ TRANG vào DeepSeek-OCR-2 với
        #                        prompt <|grounding|>, model tự làm layout +
        #                        OCR + dựng bảng trong 1 lần gọi. Chưa hỗ trợ
        #                        dịch (không có IR block để dịch từng phần).
        self._engine_mode = (engine_mode or "blocks").lower()
        if self._engine_mode not in ("blocks", "end2end"):
            raise ValueError(
                f"engine_mode phải là 'blocks' hoặc 'end2end', nhận: {engine_mode!r}"
            )

        self._yolo_path  = yolo_model
        self._qairt_sdk  = qairt_sdk
        self._model_dir  = model_dir
        self._lang       = lang
        self._direction  = direction
        self._dpi        = dpi
        self._table_engine     = (table_engine or "img2table").lower()
        self._translate        = bool(translate)
        self._debug_dump_dir   = Path(debug_dump) if debug_dump else None
        if self._debug_dump_dir is not None:
            self._debug_dump_dir.mkdir(parents=True, exist_ok=True)
        self._ocr_engine_name  = (ocr_engine or "vietocr").lower()
        if self._engine_mode == "end2end" and self._ocr_engine_name != "deepseek":
            _log.warning(
                f"engine_mode='end2end' chỉ hỗ trợ DeepSeek-OCR-2 (model duy nhất "
                f"có ocr_full_page_markdown) — bỏ qua ocr_engine={ocr_engine!r} "
                f"đã truyền, tự chuyển sang 'deepseek'."
            )
            self._ocr_engine_name = "deepseek"
        self._vietocr_weights = (
            str(vietocr_weights) if vietocr_weights is not None else None
        )
        self._vietocr_variant = vietocr_variant
        self._vietocr_beam    = bool(vietocr_beam)
        self._layout_backend  = layout_backend
        self._layout_conf     = layout_conf
        self._layout_imgsz    = layout_imgsz
        self._ocr_figures     = bool(ocr_figures)

        self._detector   = None
        self._ocr_engine = None
        self._ocr_runner = None
        self._table_rec  = None
        self._translator = TranslationEngine(direction)

    def _init(self):
        if self._engine_mode == "end2end":
            # Chế độ end2end chỉ cần đúng 1 engine — không cần LayoutDetector,
            # table_rec, hay bất kỳ OCR engine nào khác của luồng "blocks".
            if self._ocr_engine is None:
                from deepseek_ocr_engine import DeepSeekOCREngine
                self._ocr_engine = DeepSeekOCREngine()
                _log.info("[end2end] OCR engine: DeepSeek-OCR-2 (grounding, full-page)")
            return

        if self._detector is None:
            self._detector = LayoutDetector(
                self._yolo_path,
                backend     = self._layout_backend,
                conf_thresh = self._layout_conf,
                imgsz       = self._layout_imgsz,
            )

        # ── Device-usage banner ─────────────────────────────────────────────
        # Make it loud which stages run on NPU vs CPU.
        #
        # NOTE: the EasyOCR CRAFT path on NPU goes through the QAIRT subprocess
        # (qnn-net-run.exe), NOT onnxruntime's QNN EP. The QAIRT SDK is detected
        # by checking whether its CLI binary exists on disk.
        _qairt_bin   = Path(self._qairt_sdk) / "bin" / "x86_64-windows-msvc" / "qnn-net-run.exe"
        _qairt_ready = _qairt_bin.exists()

        _layout_backend_actual = (
            "doclayout" if (self._layout_backend == "auto"
                            and "doclayout" in str(self._yolo_path).lower())
            else self._layout_backend if self._layout_backend != "auto"
            else "ultralytics"
        )
        _yolo_dev = f"CPU (PyTorch / {_layout_backend_actual})"

        if self._ocr_engine_name == "easyocr":
            _det_dev = ("NPU (QAIRT CRAFT)" if _qairt_ready
                        else "❌ NPU UNAVAILABLE — qnn-net-run.exe missing, "
                             "will fail when called")
            _rec_dev = ("NPU (QAIRT EasyOCR rec — lossy on Vietnamese tones)"
                        if _qairt_ready else "❌ NPU UNAVAILABLE")
        elif self._ocr_engine_name == "vietocr":
            _det_dev = ("NPU (QAIRT CRAFT)" if _qairt_ready
                        else "❌ NPU UNAVAILABLE — qnn-net-run.exe missing, "
                             "will fall back to VietOCR projection splitter")
            _rec_dev = "CPU (VietOCR PyTorch FP32)"
        elif self._ocr_engine_name == "tesseract":
            _det_dev = "CPU (Tesseract internal)"
            _rec_dev = "CPU (Tesseract vie+eng)"
        elif self._ocr_engine_name == "easyocr-cpu":
            _det_dev = "CPU (EasyOCR CRAFT)"
            _rec_dev = "CPU (EasyOCR recognizer)"
        else:
            _det_dev = "CPU"
            _rec_dev = "CPU"

        _log.info("─" * 60)
        _log.info(f"  DEVICE PLAN  (QAIRT SDK ready: {_qairt_ready})")
        _log.info(f"    Layout    : {_yolo_dev}")
        _log.info(f"    Text det  : {_det_dev}")
        _log.info(f"    Text rec  : {_rec_dev}")
        if self._translate:
            _log.info(
                "    Translation: LLM (pipelined — runs in parallel with OCR)"
            )
        _log.info("─" * 60)

        if self._ocr_engine is None:
            if self._ocr_engine_name == "easyocr":
                from easyocr_qairt_engine import QAIRTEasyOCREngine
                self._ocr_engine = QAIRTEasyOCREngine(
                    workspace_dir  = str(Path(".").resolve()),
                    qairt_sdk_path = self._qairt_sdk,
                    model_dir      = self._model_dir,
                    lang           = self._lang,
                )
                _log.info("OCR engine: QAIRTEasyOCR (CRAFT + rec, full NPU)")
            elif self._ocr_engine_name == "vietocr":
                # Hybrid: EasyOCR CRAFT detector on NPU + VietOCR recognizer
                # on CPU. The "vietocr" name is kept because the recognizer
                # is what drives output quality and what the user picks for.
                from hybrid_easyviet_engine import HybridEasyVietEngine
                self._ocr_engine = HybridEasyVietEngine(
                    workspace_dir   = str(Path(".").resolve()),
                    qairt_sdk_path  = self._qairt_sdk,
                    model_dir       = self._model_dir,
                    lang            = self._lang,
                    vietocr_weights = self._vietocr_weights,
                    vietocr_variant = self._vietocr_variant,
                    vietocr_beam    = self._vietocr_beam,
                )
                _log.info(
                    f"OCR engine: Hybrid EasyOCR-CRAFT (NPU) "
                    f"+ VietOCR-{self._vietocr_variant} (CPU)"
                )
            elif self._ocr_engine_name == "tesseract":
                # Tesseract: pure CPU, no NPU. Acts as both detector and
                # recognizer. Useful as a fallback when QAIRT isn't
                # installed and as a comparison baseline against the
                # hybrid path. Vietnamese accuracy is generally lower
                # than VietOCR but failure modes differ — sometimes
                # captures cells the hybrid misses, and vice versa.
                from tesseract_engine import TesseractEngine
                self._ocr_engine = TesseractEngine(lang=self._lang)
                _log.info("OCR engine: Tesseract (CPU)")
            elif self._ocr_engine_name == "easyocr-cpu":
                # Pure CPU EasyOCR. The standalone Python library
                # (not the QAIRT NPU build). Runs on any machine, no
                # QAIRT SDK or model files needed. Exposes
                # ``recognize_with_boxes`` which the image-table
                # reconstructor uses for borderless tables.
                from easyocr_cpu_engine import EasyOCRCPUEngine
                self._ocr_engine = EasyOCRCPUEngine(lang=self._lang)
                _log.info("OCR engine: EasyOCR (pure CPU)")
            elif self._ocr_engine_name == "deepseek":
                from deepseek_ocr_engine import DeepSeekOCREngine
                # Khởi tạo engine, truyền tên môi trường conda của bạn
                self._ocr_engine = DeepSeekOCREngine()
                _log.info("OCR engine: DeepSeek-OCR-2 (Subprocess / CPU)")
            else:
                raise ValueError(
                    f"Unknown ocr_engine: {self._ocr_engine_name!r} "
                    "(expected easyocr|easyocr-cpu|vietocr|tesseract)"
                )
            self._ocr_runner = BlockOCRRunner(
                self._ocr_engine,
                ocr_figures=self._ocr_figures,
            )

        if self._table_rec is None:
            self._table_rec = build_reconstructor(
                self._table_engine,
                ocr_engine=self._ocr_engine,
                cfg={"lang": self._lang},
            )
            _log.info(
                f"Table engine: {self._table_engine} "
                f"({type(self._table_rec).__name__})"
            )

    def run(self, source_path: str | Path) -> dict[str, Path]:
        """
        Process source PDF/image.
        Returns: {"md": Path, "docx": Path, "pdf": Path}
        """
        import fitz

        source_path = Path(source_path)
        self._init()

        if self._engine_mode == "end2end":
            return self._run_end2end(source_path)

        suffix    = source_path.suffix.lower()
        num_pages = len(fitz.open(str(source_path))) if suffix == ".pdf" else 1

        t_start = time.perf_counter()
        timings: dict[str, float] = {}

        print(f"\nSource: {source_path.name}  ({num_pages} pages)")

        all_pages_ir:  list[DocumentPage] = []
        temp_imgs:     list[Path]         = []

        # ── Parallel translation pipeline ───────────────────────────────────
        # Translation is network/LLM-bound; OCR is CPU/NPU-bound. Running
        # them sequentially leaves either the LLM idle (during OCR) or
        # the CPU idle (during translation). Spawn a single-worker thread
        # pool so each finished page starts translating immediately while
        # OCR keeps chewing through the next page. Concurrency=1 is enough
        # to overlap with OCR; bumping it doesn't help with single-server
        # Ollama because Ollama serialises requests anyway.
        import concurrent.futures as _cf
        translate_pool = (
            _cf.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="translate"
            )
            if self._translate else None
        )
        translate_futures: list[tuple[int, _cf.Future]] = []
        translate_started_at = None  # wall-clock when the first job dispatched

        for pg in range(num_pages):
            sep = "─" * 56
            print(f"\n{sep}\n  PAGE {pg+1}/{num_pages}\n{sep}")

            # 1. Rasterise
            t0       = time.perf_counter()
            img_path = rasterise_page(source_path, pg, self._dpi, self.output_dir)
            if img_path.name.startswith("_page_"):
                temp_imgs.append(img_path)
            timings[f"p{pg+1}.raster"] = time.perf_counter() - t0
            print(f"  [1/4] Rasterised: {img_path.name}")

            # 2. Layout detection
            t0     = time.perf_counter()
            blocks = self._detector.detect(img_path)
            timings[f"p{pg+1}.layout"] = time.perf_counter() - t0
            cnt    = Counter(b.label for b in blocks)
            print(f"  [2/4] Layout: {len(blocks)} blocks — "
                  + ", ".join(f"{k}={v}" for k, v in sorted(cnt.items())))

            # 2b. Table salvage — img2table catches tables YOLO mistagged as
            #     Figure/Picture/Text and synthesises blocks for tables YOLO
            #     missed entirely. Only runs when the active engine is
            #     img2table-based (otherwise we'd be paying for a CPU pass
            #     no one will use).
            if self._table_engine == "img2table" and blocks:
                try:
                    from PIL import Image as _PIL
                    from table_salvage import salvage_tables
                    page_pil = _PIL.open(str(img_path)).convert("RGB")
                    t0 = time.perf_counter()
                    blocks = salvage_tables(
                        page_pil, blocks, block_cls=LayoutBlock,
                    )
                    timings[f"p{pg+1}.salvage"] = time.perf_counter() - t0
                except Exception as e:
                    _log.warning(f"table_salvage failed: {e}")

            # 3. OCR
            t0          = time.perf_counter()
            ocr_results = self._ocr_runner.process_blocks(blocks)
            timings[f"p{pg+1}.ocr"] = time.perf_counter() - t0
            print(f"  [3/4] OCR: {sum(len(r.lines) for r in ocr_results)} lines")

            # 4. Table reconstruction
            t0             = time.perf_counter()
            table_rows_map: dict[int, list[TableRow]] = {}
            failed_tables: list = []
            tblocks = [b for b in blocks if b.label in _TABLE_LABELS]
            if self._table_rec and tblocks:
                for tb in tblocks:
                    rows = self._table_rec.reconstruct(tb)
                    if rows and not _looks_like_misclassified_table(rows):
                        table_rows_map[tb.id] = rows
                    elif rows:
                        # The reconstruction succeeded but the result is
                        # the unmistakable signature of a TOC / outline
                        # page that YOLO mis-labelled as Table. Drop the
                        # grid and fall through to paragraph rendering.
                        _log.info(
                            f"  table block {tb.id} demoted to text "
                            f"(low-quality grid: {len(rows)} rows, "
                            f"{_table_fill_ratio(rows):.0%} fill)"
                        )
                        failed_tables.append(tb)
                    else:
                        # img2table failed completely on this block. Don't
                        # let the content silently disappear — re-OCR the
                        # whole table crop as a text block fallback so the
                        # text at least survives in flowing paragraph form.
                        failed_tables.append(tb)
                total_cells = sum(
                    sum(len(r.cells) for r in rows)
                    for rows in table_rows_map.values()
                )
                print(f"  [4/4] Tables: {len(tblocks)} tables, "
                      f"{sum(len(r) for r in table_rows_map.values())} rows, "
                      f"{total_cells} cells")
                if failed_tables:
                    print(
                        f"        ⚠ {len(failed_tables)} table(s) had no "
                        "structure detected — falling back to paragraph OCR"
                    )
            else:
                print("  [4/4] Tables: skipped")
            timings[f"p{pg+1}.tatr"] = time.perf_counter() - t0

            # Fallback for tables that img2table couldn't reconstruct.
            # Two-tier:
            #   1. If the OCR engine exposes ``recognize_with_boxes``,
            #      use ``image_table_reconstructor`` to build a real
            #      row/column grid from OCR line positions. This is the
            #      path that handles "full page is a scanned table"
            #      cases that come through as a Picture block.
            #   2. If grid reconstruction fails (too few rows/cols) or
            #      the engine has no boxes API, dump the OCR'd text as
            #      a flowing paragraph block — content is preserved,
            #      structure is lost. Same behaviour as before.
            for tb in failed_tables:
                if tb.crop is None:
                    continue

                recovered_rows: list[TableRow] = []
                # ── Tier 1: image-table reconstruction from OCR boxes ──
                if hasattr(self._ocr_engine, "recognize_with_boxes"):
                    try:
                        lines_with_boxes = self._ocr_engine.recognize_with_boxes(tb.crop)
                    except Exception as e:
                        _log.warning(
                            f"recognize_with_boxes failed on block {tb.id}: {e}"
                        )
                        lines_with_boxes = []
                    if lines_with_boxes:
                        try:
                            from image_table_reconstructor import (
                                reconstruct_table_from_lines,
                            )
                            recovered_rows = reconstruct_table_from_lines(
                                lines_with_boxes,
                                crop_size=tb.crop.size,
                            )
                        except Exception as e:
                            _log.warning(
                                f"image-table reconstruct failed on block "
                                f"{tb.id}: {e}"
                            )
                            recovered_rows = []
                if recovered_rows:
                    # OCR each cell in the recovered grid through the
                    # engine's standard cell path so we get the same
                    # gates / preprocessing as img2table's output.
                    table_rows_map[tb.id] = recovered_rows
                    _log.info(
                        f"  table reconstructed from OCR boxes: block {tb.id} "
                        f"→ {len(recovered_rows)} rows × "
                        f"{max(len(r.cells) for r in recovered_rows)} cols"
                    )
                    continue

                # ── Tier 2: paragraph dump (original behaviour) ────────
                try:
                    text, conf = self._ocr_engine.recognize_region(
                        tb.crop, line_separator="\n",
                    )
                except TypeError:
                    text, conf = self._ocr_engine.recognize_region(tb.crop)
                except Exception as e:
                    _log.warning(f"table-fallback OCR failed on block {tb.id}: {e}")
                    continue
                if not text or not text.strip():
                    continue
                tb.label = "text"  # rebrand so build_ir treats as text
                ocr_results.append(BlockOCRResult(
                    block=tb,
                    lines=[
                        OCRLine(text=ln.strip(), confidence=float(conf or 0.0))
                        for ln in text.split("\n") if ln.strip()
                    ],
                ))
                _log.info(
                    f"  table fallback (paragraph): block {tb.id} → "
                    f"{text.count(chr(10))+1} text line(s)"
                )

            page_ir = build_ir(img_path, blocks, ocr_results,
                               table_rows_map, page_index=pg,
                               figure_dir=self.output_dir / "figures")
            all_pages_ir.append(page_ir)

            # Debug artefacts for this page (if requested).
            if self._debug_dump_dir is not None:
                _debug_dump_page(
                    self._debug_dump_dir, pg, img_path,
                    blocks, ocr_results, table_rows_map, page_ir,
                )

            # ── Kick off translation for THIS page in the background ────────
            # Translation modifies page_ir in place (writes
            # ``translated_text`` on each block / cell). The worker thread
            # is single-threaded so we never have two pages mid-translate
            # simultaneously — that would race on shared LLM client state
            # and make Ollama serialise anyway.
            if translate_pool is not None:
                if translate_started_at is None:
                    translate_started_at = time.perf_counter()
                    # Print a TRANSLATION marker the UI's _TRANS_START_RE
                    # picks up so the translation progress bar can begin
                    # advancing immediately, in parallel with the OCR bar.
                    print(
                        f"\n  TRANSLATION ({self._direction}) — pipelined "
                        "in background, will run in parallel with remaining "
                        "page OCR"
                    )
                fut = translate_pool.submit(
                    self._translator.translate_page,
                    page_ir,
                    page_label = f"Page {pg + 1}/{num_pages}",
                )
                translate_futures.append((pg, fut))

        # ── Wait for all pipelined translations to finish ────────────────────
        sep = "─" * 56
        if translate_pool is not None:
            print(f"\n{sep}\n  TRANSLATION ({self._direction}) — waiting for "
                  f"pipeline to drain\n{sep}")
            n_done = 0
            for pg, fut in translate_futures:
                try:
                    fut.result(timeout=900)
                except Exception as e:
                    _log.warning(f"Translation page {pg+1} failed: {e}")
                n_done += 1
                if n_done % 2 == 0 or n_done == len(translate_futures):
                    print(f"  Page {pg+1}/{num_pages} translated "
                          f"({n_done}/{len(translate_futures)} done)")
            translate_pool.shutdown(wait=True)
            timings["translate"] = (
                time.perf_counter()
                - (translate_started_at if translate_started_at else t_start)
            )
        else:
            print(f"\n{sep}\n  TRANSLATION (skipped — --no-translate)\n{sep}")
            timings["translate"] = 0.0

        # Build output
        print(f"\n{sep}\n  BUILD OUTPUT\n{sep}")
        t0   = time.perf_counter()
        stem = source_path.stem

        # Translated markdown
        trans_md = [page_to_markdown(p, use_translated=True)
                    for p in all_pages_ir]
        out_md   = self.output_dir / f"{stem}_translated.md"
        out_md.write_text("\n\n---\n\n".join(trans_md), encoding="utf-8")
        print(f"  Markdown  → {out_md.name}  ({out_md.stat().st_size//1024} KB)")

        # Original OCR markdown (for debugging/quality check)
        orig_md     = [page_to_markdown(p, use_translated=False)
                       for p in all_pages_ir]
        out_md_orig = self.output_dir / f"{stem}_ocr.md"
        out_md_orig.write_text("\n\n---\n\n".join(orig_md), encoding="utf-8")
        print(f"  OCR debug → {out_md_orig.name}  ({out_md_orig.stat().st_size//1024} KB)")

        # DOCX — default path renders straight from IR with cell merges.
        # The docling fallback uses the markdown round-trip (lossy on spans).
        out_docx_name = (
            f"{stem}_translated.docx" if self._translate
            else f"{stem}_ocr.docx"
        )
        out_docx = self.output_dir / out_docx_name
        if self._table_engine == "docling":
            markdown_to_docx(
                trans_md if self._translate else orig_md, out_docx
            )
        else:
            from table_docx_writer import pages_to_docx
            pages_to_docx(
                all_pages_ir, out_docx, use_translated=self._translate
            )
        print(f"  DOCX      → {out_docx.name}  ({out_docx.stat().st_size//1024} KB)")

        # PDF
        out_pdf = docx_to_pdf(out_docx, self.output_dir)
        timings["build"] = time.perf_counter() - t0

        # Cleanup temp page PNGs
        for p in temp_imgs:
            if p.exists():
                p.unlink()

        total = time.perf_counter() - t_start
        _print_summary(timings, total)

        # Persist a machine-readable timing record alongside the page dumps so
        # speed regressions across engine/runtime changes are diff-able. Only
        # emitted when --debug-dump is active to avoid surprising file output
        # on production runs.
        if self._debug_dump_dir is not None:
            import json
            timings_path = self._debug_dump_dir / "timings.json"
            payload = {
                "source":        str(source_path),
                "num_pages":     num_pages,
                "dpi":           self._dpi,
                "ocr_engine":    self._ocr_engine_name,
                "table_engine":  self._table_engine,
                "yolo_model":    self._yolo_path,
                "translate":     self._translate,
                "total_seconds": round(total, 3),
                "stages":        {k: round(v, 3) for k, v in timings.items()},
            }
            timings_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            _log.info(f"Timings dumped → {timings_path}")

        return {"md": out_md, "docx": out_docx, "pdf": out_pdf}

    def _run_end2end(self, source_path: Path) -> dict[str, Path]:
        """
        Chế độ end2end: mỗi trang được rasterise rồi đưa THẲNG vào
        DeepSeek-OCR-2 (prompt <|grounding|>) — model tự làm layout + OCR +
        dựng bảng trong 1 lần gọi, KHÔNG qua YOLO/table_rec/IR như luồng
        "blocks".
        """
        import fitz
        from PIL import Image as _PIL
        import shutil as _shutil

        suffix    = source_path.suffix.lower()
        num_pages = len(fitz.open(str(source_path))) if suffix == ".pdf" else 1
        stem      = source_path.stem

        t_start = time.perf_counter()
        timings: dict[str, float] = {}

        print(f"\nSource: {source_path.name}  ({num_pages} pages)")
        print("Chế độ: end2end (DeepSeek-OCR-2 grounding, full-page) "
              "— bỏ qua YOLO/table-structure. "
              + ("Dịch thuật: BẬT." if self._translate else "Dịch thuật: TẮT."))

        pages_dir = self.output_dir / "pages"
        pages_dir.mkdir(parents=True, exist_ok=True)

        orig_page_mds:       list[str] = []
        orig_raw_groundings: list[str] = []
        trans_page_mds:      list[str] = []
        trans_raw_groundings:list[str] = []
        temp_imgs:           list[Path] = []

        # ── VÒNG LẶP 1: CHỈ CHẠY OCR ──
        for pg in range(num_pages):
            sep = "─" * 56
            print(f"\n{sep}\n  PAGE {pg+1}/{num_pages}\n{sep}")

            t0       = time.perf_counter()
            img_path = rasterise_page(source_path, pg, self._dpi, self.output_dir)
            if img_path.name.startswith("_page_"):
                temp_imgs.append(img_path)
            timings[f"p{pg+1}.raster"] = time.perf_counter() - t0
            print(f"  [1/2] Rasterised: {img_path.name}")

            persistent_page_path = pages_dir / f"{stem}_p{pg:04d}.png"
            try:
                _shutil.copyfile(str(img_path), str(persistent_page_path))
            except Exception as e:
                _log.warning(f"Không copy được ảnh trang {pg+1}: {e}")

            t0  = time.perf_counter()
            img = _PIL.open(str(img_path)).convert("RGB")
            
            raw_grounding, page_md = self._ocr_engine.ocr_full_page_markdown(img)
            timings[f"p{pg+1}.deepseek_e2e"] = time.perf_counter() - t0

            if not page_md.strip():
                _log.warning(f"Trang {pg+1}: DeepSeek-OCR-2 trả về rỗng.")
            print(f"  [2/2] DeepSeek-OCR-2: {len(page_md)} ký tự markdown "
                  f"trong {timings[f'p{pg+1}.deepseek_e2e']:.1f}s")

            orig_page_mds.append(f"## Page {pg + 1}\n\n{page_md}")
            orig_raw_groundings.append(f"## Page {pg + 1}\n\n{raw_grounding}")

        # ── VÒNG LẶP 2: CHỈ CHẠY DỊCH THUẬT ──
        if self._translate:
            sep = "─" * 56
            print(
                f"\n{sep}\n  TRANSLATION ({self._direction}) — dịch trực tiếp trên "
                f"markdown toàn bộ tài liệu (không qua IR)\n{sep}"
            )
            
            import re
            for pg in range(num_pages):
                t0 = time.perf_counter()
                
                # Dịch trên bản RAW (chứa thẻ toạ độ) để giữ nguyên cấu trúc Layout
                translated_raw = translate_ocr_markdown(
                    orig_raw_groundings[pg],
                    direction  = self._direction,
                    page_label = f"Page {pg + 1}/{num_pages}",
                )
                timings[f"p{pg+1}.translate"] = time.perf_counter() - t0
                
                # Dùng Regex xoá thẻ toạ độ để tạo bản sạch
                translated_md = re.sub(
                    r"^\s*<\|ref\|>.*?<\|/ref\|>\s*<\|det\|>.*?<\|/det\|>\s*\n?", 
                    "", 
                    translated_raw, 
                    flags=re.MULTILINE
                )
                
                trans_page_mds.append(translated_md.strip())
                trans_raw_groundings.append(translated_raw.strip())
        else:
            trans_page_mds = orig_page_mds.copy()
            trans_raw_groundings = orig_raw_groundings.copy()


        # ── Build output ─────────────────────────────────────────────────
        sep = "─" * 56
        print(f"\n{sep}\n  BUILD OUTPUT\n{sep}")
        t0 = time.perf_counter()

        out_md_orig = self.output_dir / f"{stem}_ocr.md"
        out_md_orig.write_text("\n\n---\n\n".join(orig_page_mds), encoding="utf-8")
        print(f"  OCR debug → {out_md_orig.name}  ({out_md_orig.stat().st_size//1024} KB)")

        out_md_raw = self.output_dir / f"{stem}_ocr_raw.md"
        out_md_raw.write_text("\n\n---\n\n".join(orig_raw_groundings), encoding="utf-8")
        print(f"  Raw MD    → {out_md_raw.name}  ({out_md_raw.stat().st_size//1024} KB)")

        out_docx_ocr = self.output_dir / f"{stem}_ocr.docx"
        markdown_to_docx(orig_page_mds, out_docx_ocr)
        print(f"  DOCX (OCR) → {out_docx_ocr.name}  ({out_docx_ocr.stat().st_size//1024} KB)")

        if self._translate:
            out_md_raw_trans = self.output_dir / f"{stem}_translated_raw.md"
            out_md_raw_trans.write_text("\n\n---\n\n".join(trans_raw_groundings), encoding="utf-8")
            print(f"  Raw Trans → {out_md_raw_trans.name}  ({out_md_raw_trans.stat().st_size//1024} KB)")
            
            out_md = self.output_dir / f"{stem}_translated.md"
            out_md.write_text("\n\n---\n\n".join(trans_page_mds), encoding="utf-8")
            print(f"  Markdown  → {out_md.name}  ({out_md.stat().st_size//1024} KB)")

            out_docx = self.output_dir / f"{stem}_translated.docx"
            markdown_to_docx(trans_page_mds, out_docx)
            print(f"  DOCX      → {out_docx.name}  ({out_docx.stat().st_size//1024} KB)")
        else:
            out_md   = out_md_orig
            out_docx = out_docx_ocr

        out_pdf = docx_to_pdf(out_docx, self.output_dir)
        timings["build"] = time.perf_counter() - t0

        for p in temp_imgs:
            if p.exists():
                p.unlink()

        total = time.perf_counter() - t_start
        _print_summary(timings, total)

        import json
        timing_file = self.output_dir / f"{stem}_timings.json"
        try:
            timing_file.write_text(
                json.dumps({"timings": timings, "total": total}, ensure_ascii=False, indent=2), 
                encoding="utf-8"
            )
        except Exception as e:
            _log.warning(f"Không thể lưu file timings: {e}")

        return {
            "md":        out_md,
            "md_raw":    out_md_raw,
            "docx":      out_docx,
            "pdf":       out_pdf,
            "pages_dir": pages_dir,
        }


# ══════════════════════════════════════════════════════════════════════════════
# § K  —  CLI / TIMING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _print_summary(timings: dict[str, float], total: float) -> None:
    W = 60
    print("\n" + "═" * W)
    print(f"{'PIPELINE TIMING SUMMARY':^{W}}")
    print("═" * W)
    for stage, elapsed in timings.items():
        bar = int((elapsed / total) * 20) if total > 0 else 0
        pct = elapsed / total * 100 if total > 0 else 0
        print(f"  {stage:<30} {elapsed:6.2f}s  {pct:4.0f}%  {'█'*bar}")
    print("─" * W)
    print(f"  {'TOTAL':<30} {total:6.2f}s  100%")
    print("═" * W)


class _TeeStream:
    """Tee a stream to a second sink (the log file).

    Used to mirror everything the pipeline prints with ``print(...)`` into the
    auto-generated run log alongside the structured Python logging records.
    Both ``write`` and ``flush`` proxy to both sinks; everything else falls
    through to the primary stream so attributes like ``isatty`` keep working.

    Encoding errors on the primary stream (most commonly when stdout is a
    pipe and Windows defaults to cp1252, which can't encode box-drawing
    glyphs like '─' or Vietnamese composed diacritics) are caught and the
    offending characters are replaced with '?' so the pipeline never
    crashes from a print statement.
    """

    def __init__(self, primary, mirror):
        self._primary = primary
        self._mirror  = mirror

    def write(self, data):
        # Mirror first — the log file is opened in utf-8 mode and is the
        # source of truth for the captured transcript.
        try:
            self._mirror.write(data)
            self._mirror.flush()
        except Exception:
            pass
        try:
            return self._primary.write(data)
        except UnicodeEncodeError:
            # Primary stream uses a narrower codec than the data.
            # Fall back to its encoding with replacement so we still print
            # *something* and never abort the run.
            enc = getattr(self._primary, "encoding", "ascii") or "ascii"
            safe = data.encode(enc, errors="replace").decode(enc, errors="replace")
            return self._primary.write(safe)

    def flush(self):
        try:
            self._mirror.flush()
        except Exception:
            pass
        try:
            self._primary.flush()
        except Exception:
            pass

    def __getattr__(self, name):
        return getattr(self._primary, name)


def _setup_run_logging(output_dir: Path, source_name: str) -> Path:
    """
    Configure the root logger + stdout tee so a complete transcript of the
    run is written to ``<output_dir>/logs/<source>_<timestamp>.log``.

    Returns the log file path so the caller can print it at the end.
    """
    from datetime import datetime

    # Windows pipes default to cp1252 which can't encode the box-drawing
    # characters and Vietnamese diacritics we print everywhere. Switch to
    # UTF-8 with replacement before anything else writes to stdout/stderr.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    logs_dir = Path(output_dir) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    stamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"{source_name}_{stamp}.log"

    # File handler for Python ``logging`` records.
    fh = logging.FileHandler(str(log_path), encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
    ))

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    # Replace any prior FileHandler so subsequent runs don't pile up.
    for h in list(root.handlers):
        if isinstance(h, logging.FileHandler):
            root.removeHandler(h)
    root.addHandler(fh)

    # Tee stdout / stderr into the same file so print()-based progress
    # output is captured too. The underlying file object is opened once and
    # shared so writes interleave correctly with the FileHandler.
    log_fp = open(log_path, "a", encoding="utf-8", buffering=1)
    sys.stdout = _TeeStream(sys.stdout, log_fp)
    sys.stderr = _TeeStream(sys.stderr, log_fp)

    logging.info(f"Run log: {log_path}")
    return log_path


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level  = logging.INFO,
        format = "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="OCR + translation pipeline (Snapdragon NPU)",
    )
    parser.add_argument("source",
                        help="Input PDF or image path")
    parser.add_argument("direction", nargs="?", default="vi->en",
                        choices=["vi->en", "en->vi"],
                        help="Translation direction (default: vi->en)")
    parser.add_argument("output_dir", nargs="?", default=None,
                        help="Output directory (default: <source>/output)")
    parser.add_argument(
        "--table-engine",
        choices=["img2table", "rules", "docling", "mineru"],
        default=os.environ.get("TABLE_ENGINE", "img2table"),
        help=(
            "Table structure recognizer. "
            "img2table (default) = xavctn/img2table CPU structure detector "
            "with whole-crop OCR via QAIRT EasyOCR (handles row/col spans); "
            "rules = OpenCV ruled-lines only (bordered tables); "
            "docling = Docling TableFormer (heavy optional dep, markdown "
            "round-trip)."
            "mineru = MinerU CLI table structure + built-in OCR text "
            "(chậm hơn — 1 subprocess riêng mỗi bảng)."
        ),
    )
    parser.add_argument("--dpi", type=int, default=150,
                        help="Rasterisation DPI (default: 150).")
    parser.add_argument(
        "--no-translate", action="store_true",
        help=(
            "Skip the LLM translation stage entirely. The DOCX renders raw OCR text "
            "(handy when the Nexa server is down or for stage-by-stage debugging)."
        ),
    )
    parser.add_argument(
        "--debug-dump", default=None,
        help=(
            "Directory to dump per-stage artefacts (page PNG, layout overlay PNG, "
            "layout.json, ocr.json, per-table crops + IR JSON, full ir.json). "
            "Use this to inspect each stage in isolation."
        ),
    )
    parser.add_argument(
        "--ocr-engine",
        choices=["easyocr", "easyocr-cpu", "vietocr", "tesseract", "deepseek"],
        default=os.environ.get("OCR_ENGINE", "easyocr-cpu"),
        help=(
            "OCR backend. "
            "'easyocr' = full Qualcomm EasyOCR on NPU (CRAFT detector + "
            "EasyOCR recognizer; fast but lossy on Vietnamese stacked tones). "
            "'easyocr-cpu' = standalone EasyOCR on CPU (no NPU, no QAIRT). "
            "'vietocr' (default) = hybrid EasyOCR-CRAFT detector on NPU + "
            "VietOCR recognizer on CPU (best Vietnamese accuracy). "
            "'tesseract' = Tesseract 5 on CPU (vie+eng); no NPU, useful as "
            "a fallback and comparison baseline."
        ),
    )
    parser.add_argument(
        "--vietocr-weights", default=os.environ.get("VIETOCR_WEIGHTS"),
        help=(
            "Path to local VietOCR .pth weights. If omitted, VietOCR will "
            "download to ~/.cache/vietocr/ on first run (~95 MB)."
        ),
    )
    parser.add_argument(
        "--vietocr-variant",
        choices=["vgg_transformer", "vgg_seq2seq"],
        default=os.environ.get("VIETOCR_VARIANT", "vgg_transformer"),
        help=(
            "VietOCR backbone. vgg_transformer (default) is most accurate; "
            "vgg_seq2seq is ~30%% faster on CPU with slightly lower accuracy."
        ),
    )
    parser.add_argument(
        "--vietocr-beam", action="store_true",
        help=(
            "Enable beam search in VietOCR (better accuracy on hard lines, "
            "~2-3× slower per line)."
        ),
    )
    parser.add_argument(
        "--yolo-model", default=None,
        help=(
            "Path to layout model weights. Default search order: "
            "doclayout_yolo_docstructbench_imgsz1024.pt (DocLayout-YOLO, "
            "recommended), yolo26m-doclaynet.pt, yolov8m-doclaynet.pt — "
            "first one that exists wins."
        ),
    )
    parser.add_argument(
        "--layout-backend",
        choices=["auto", "ultralytics", "doclayout"],
        default=os.environ.get("LAYOUT_BACKEND", "auto"),
        help=(
            "Layout model backend. 'auto' (default) picks doclayout-yolo "
            "if the weight filename contains 'doclayout', else ultralytics."
        ),
    )
    parser.add_argument(
        "--layout-conf", type=float, default=None,
        help=(
            "Confidence threshold for layout boxes. "
            "Default: 0.20 for DocLayout-YOLO, 0.25 for ultralytics."
        ),
    )
    parser.add_argument(
        "--layout-imgsz", type=int,
        default=int(os.environ.get("LAYOUT_IMGSZ", "1024")),
        help="Inference image size for the layout model (default 1024).",
    )
    parser.add_argument(
        "--engine-mode",
        choices=["blocks", "end2end"],
        default=os.environ.get("ENGINE_MODE", "blocks"),
        help=(
            "blocks (mặc định) = pipeline hiện có: YOLO layout → OCR per-block "
            "→ table structure → dịch. "
            "end2end = bỏ qua YOLO/table-structure/dịch, đưa thẳng ảnh cả trang "
            "vào DeepSeek-OCR-2 (prompt <|grounding|>), model tự làm layout+OCR+bảng "
            "trong 1 lần gọi. Chậm hơn, chưa hỗ trợ dịch, nhưng không phụ thuộc "
            "YOLO/TATR — dùng khi luồng 'blocks' cho kết quả kém trên trang khó."
        ),
    )
    parser.add_argument(
        "--ocr-figures", action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "OCR the contents of figure / Picture blocks too. ON by default — "
            "Vietnamese government letters often render the title text into "
            "the header banner image, which would otherwise be lost. "
            "Pass --no-ocr-figures to disable."
        ),
    )

    args = parser.parse_args()

    source_path = Path(args.source)
    direction   = args.direction
    output_dir  = (Path(args.output_dir) if args.output_dir
                   else source_path.parent / "output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Capture the entire run (logging + stdout/stderr) to a file in
    # output_dir/logs/. Path is shown again at the end of the run.
    _log_path = _setup_run_logging(output_dir, source_path.stem)

    if not source_path.exists():
        print(f"[ERROR] File not found: {source_path}")
        sys.exit(1)

    # ── Configuration ─────────────────────────────────────────────────────────
    # Resolve layout weights. Search order (first existing wins):
    #   1) Ultralytics DocLayNet 26m   (default — well-tested on Vietnamese docs)
    #   2) Ultralytics DocLayNet 8m    (fallback baseline)
    #   3) DocLayout-YOLO              (opt-in via --yolo-model)
    # An explicit --yolo-model always wins.
    def _resolve_yolo(cli_path: str | None) -> str:
        if cli_path:
            return cli_path
        for candidate in (
            "models/yolo/yolo26m-doclaynet.pt",
            "models/yolo/yolov8m-doclaynet.pt",
            "models/yolo/doclayout_yolo_docstructbench_imgsz1024.pt",
        ):
            if Path(candidate).exists():
                return candidate
        return "models/yolo/yolov8m-doclaynet.pt"  # let YOLO raise if missing

    YOLO_MODEL = _resolve_yolo(args.yolo_model)
    SDK        = r"C:\Qualcomm\AIStack\QAIRT\2.45.0.260326"
    MODEL_DIR  = (
        r"models\easyocr_qnn_vi"
        r"\easyocr-qnn_context_binary-float-qualcomm_snapdragon_x_elite"
    )
    print(f"YOLO model: {YOLO_MODEL}")

    pipeline = DocumentTranslationPipeline(
        yolo_model           = YOLO_MODEL,
        qairt_sdk            = SDK,
        model_dir            = MODEL_DIR,
        lang                 = "vi",
        direction            = direction,
        output_dir           = output_dir,
        dpi                  = args.dpi,
        table_engine         = args.table_engine,
        translate            = not args.no_translate,
        debug_dump           = args.debug_dump,
        ocr_engine           = args.ocr_engine,
        vietocr_weights      = args.vietocr_weights,
        vietocr_variant      = args.vietocr_variant,
        vietocr_beam         = args.vietocr_beam,
        layout_backend       = args.layout_backend,
        layout_conf          = args.layout_conf,
        layout_imgsz         = args.layout_imgsz,
        ocr_figures          = args.ocr_figures,
        engine_mode          = args.engine_mode,
    )

    outputs = pipeline.run(source_path)

    print("\nOutputs:")
    for key, path in outputs.items():
        size_kb = path.stat().st_size // 1024 if path.exists() else 0
        print(f"  {key.upper():<6}: {path}  ({size_kb} KB)")
    print(f"  LOG   : {_log_path}")