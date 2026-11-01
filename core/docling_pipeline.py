"""
core/docling_pipeline.py
========================

CPU-only document pipeline backed by **Docling** (layout + table
structure + text extraction) and **Tesseract** (OCR for scanned
regions). No NPU, no YOLO, no img2table, no CRAFT.

Architecture
------------
    PDF
     │
     ▼
   ┌─────────────────────────────────────────┐
   │  Split into ``batch_size``-page chunks  │   (pypdf)
   │  (memory bound — Docling holds whole    │
   │   document in RAM during conversion)    │
   └─────────────────────────────────────────┘
     │
     ▼
   ┌─────────────────────────────────────────┐
   │  Docling DocumentConverter              │
   │  ├─ layout detection                    │
   │  ├─ table-structure model (TableFormer) │
   │  ├─ reading-order resolver              │
   │  └─ OCR via TesseractCliOcrOptions      │
   │      lang=["vie","eng"]                 │
   └─────────────────────────────────────────┘
     │  (per-chunk export_to_dict)
     ▼
   ┌─────────────────────────────────────────┐
   │  Convert Docling JSON → IR              │
   │   * texts → IRBlock(type=<label>)       │
   │   * tables → IRBlock(type=table,        │
   │                       table_rows=…)     │
   │   * pictures → IRBlock(type=figure,     │
   │                         figure_path=…)  │
   └─────────────────────────────────────────┘
     │
     ▼
   list[DocumentPage]  ──► downstream translate + DOCX/MD/PDF export
                          (unchanged from the previous pipeline)

Why Docling
-----------
Docling's TableFormer + reading-order model are the only piece of the
open-source stack that consistently handles vertically-merged anchor
cells correctly — exactly the failure mode that took the YOLO/img2table
build a week to fight. Tesseract on CPU is the OCR engine used in the
Vietnamese-government-letter reference repo (``kotaemon``), so this
pipeline ports cleanly to other deployments.
"""

from __future__ import annotations

import io
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable

from ir_types import (
    DocumentPage,
    IRBlock,
    OCRLine,
    TableCell,
    TableRow,
)

# Defensive import: text_utils is a thin, dependency-free module, but we
# don't want a missing/renamed file to break Docling conversion — fall
# back to a no-op that returns the type unchanged.
try:
    from text_utils import reclassify_block_type
except ImportError:
    def reclassify_block_type(current_type: str, _text: str) -> str:
        return current_type

_log = logging.getLogger(__name__)


# ── Label normalisation ──────────────────────────────────────────────────────
#
# Docling's text labels (from docling.datamodel.base_models.DocItemLabel)
# don't quite match our IR's. Map them so the markdown / DOCX writers
# don't need to know about Docling.

_LABEL_MAP: dict[str, str] = {
    "title":              "title",
    "section_header":     "section_header",
    "page_header":        "page_header",
    "page_footer":        "page_footer",
    "caption":            "caption",
    "footnote":           "caption",
    "list_item":          "list_item",
    "formula":            "formula",
    "code":               "text",
    "checkbox_selected":  "text",
    "checkbox_unselected":"text",
    "text":               "text",
    "paragraph":          "text",
    # Tables & pictures get IRBlock.type set explicitly elsewhere — these
    # never reach _normalise_label.
}

_ABANDON_LABELS = {"page_footer"}   # kept for reference; no longer used to
                                     # skip blocks — see _item_to_block()


def _normalise_label(docling_label: str) -> str:
    return _LABEL_MAP.get(docling_label.lower(), "text")


# ── Main pipeline ────────────────────────────────────────────────────────────

class DoclingCPUPipeline:
    """
    End-to-end document conversion via Docling. Produces a
    ``list[DocumentPage]`` matching the IR the rest of the project
    consumes (translation, markdown/DOCX writers).

    Parameters
    ----------
    output_dir
        Directory where figure crops, debug dumps, and final outputs live.
    ocr_lang
        Tesseract language pack list. Default: ``("vie", "eng")``.
    batch_size
        Split the PDF into chunks of this many pages before handing each
        chunk to Docling. Keeps peak memory bounded on big docs and is
        recoverable per-chunk on failure. ``0`` disables batching.
    tesseract_cmd
        Optional override path to the Tesseract binary. When ``None``,
        Docling resolves via ``TESSERACT_CMD`` env var → ``$PATH``.
    do_cell_matching
        Forward to ``TableStructureOptions(do_cell_matching=...)``. ON
        gives Docling more accurate per-cell text attribution; OFF
        skips that step (faster, slightly worse on dense tables).
    """

    def __init__(
        self,
        output_dir:       str | Path = "output",
        ocr_lang:         Iterable[str] = ("vie", "eng"),
        batch_size:       int        = 5,
        tesseract_cmd:    str | None = None,
        do_cell_matching: bool       = True,
        do_ocr:           bool       = True,
        ocr_backend:      str        = "easyocr",
        figures_dir:      str | Path | None = None,
    ):
        self._output_dir       = Path(output_dir)
        self._ocr_lang         = tuple(ocr_lang)
        self._batch_size       = max(0, int(batch_size))
        self._tesseract_cmd    = tesseract_cmd or os.environ.get("TESSERACT_CMD")
        self._do_cell_matching = bool(do_cell_matching)
        self._do_ocr           = bool(do_ocr)
        # Which OCR backend Docling should drive. EasyOCR has materially
        # better Vietnamese diacritic accuracy than Tesseract; rapidocr
        # (ONNX port of PaddleOCR) is the strongest standalone option.
        # 'tesseract' is kept as a fallback when neither easyocr nor
        # rapidocr is installable.
        self._ocr_backend      = (ocr_backend or "easyocr").lower()
        self._figures_dir      = (Path(figures_dir)
                                  if figures_dir
                                  else self._output_dir / "figures")
        self._figures_dir.mkdir(parents=True, exist_ok=True)
        self._converter        = None     # lazily built

    # ────────────────────────────────────────────────────────────────────────
    # Lazy Docling setup
    # ────────────────────────────────────────────────────────────────────────

    def _ensure_converter(self) -> None:
        if self._converter is not None:
            return
        try:
            from docling.document_converter import (
                DocumentConverter,
                PdfFormatOption,
            )
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import (
                PdfPipelineOptions,
                TableStructureOptions,
            )
        except ImportError as e:
            raise RuntimeError(
                "Docling is not installed. Run: `pip install docling`."
            ) from e

        ocr_options = self._build_ocr_options()

        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr             = self._do_ocr
        pipeline_options.ocr_options        = ocr_options
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options = TableStructureOptions(
            do_cell_matching=self._do_cell_matching,
        )
        pipeline_options.images_scale         = 1.0
        pipeline_options.generate_page_images = True   # needed for figure crops

        self._converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
            }
        )
        _log.info(
            f"Docling converter ready "
            f"(ocr_backend={self._ocr_backend}, "
            f"ocr_lang={'+'.join(self._ocr_lang)}, "
            f"batch_size={self._batch_size}, "
            f"cell_matching={self._do_cell_matching})"
        )

    def _build_ocr_options(self):
        """
        Build the Docling OCR-options object matching ``self._ocr_backend``.

        Docling exposes several OCR engines through different option
        classes. We pick the right class and pass language tags in the
        engine-native format.
        """
        backend = self._ocr_backend
        if backend == "tesseract":
            from docling.datamodel.pipeline_options import TesseractCliOcrOptions
            # Tesseract config. Docling spawns ``tesseract`` CLI itself; we
            # set the env var explicitly so the subprocess inherits it.
            if self._tesseract_cmd:
                os.environ["TESSERACT_CMD"] = self._tesseract_cmd
                cmd_dir = str(Path(self._tesseract_cmd).parent)
                if cmd_dir not in os.environ.get("PATH", ""):
                    os.environ["PATH"] = cmd_dir + os.pathsep + os.environ.get("PATH", "")
            kwargs = {"lang": list(self._ocr_lang)}
            if self._tesseract_cmd:
                kwargs["tesseract_cmd"] = self._tesseract_cmd
            _log.info(f"OCR backend: Tesseract CLI (cmd={self._tesseract_cmd or 'PATH'})")
            return TesseractCliOcrOptions(**kwargs)

        if backend == "easyocr":
            try:
                from docling.datamodel.pipeline_options import EasyOcrOptions
            except ImportError as e:
                raise RuntimeError(
                    "EasyOCR backend requested but docling.EasyOcrOptions "
                    "isn't available. Install `easyocr` first: "
                    "`pip install easyocr`."
                ) from e
            # EasyOCR uses ISO-639-1-style codes ("vi", "en"), not the
            # Tesseract-style ("vie", "eng"). Translate on the fly.
            lang_map = {"vie": "vi", "eng": "en", "vi": "vi", "en": "en"}
            easy_langs = [lang_map.get(l.lower(), l) for l in self._ocr_lang]
            _log.info(f"OCR backend: EasyOCR (langs={easy_langs}, CPU)")
            try:
                return EasyOcrOptions(lang=easy_langs, use_gpu=False)
            except TypeError:
                # Older Docling versions: positional only
                return EasyOcrOptions(lang=easy_langs)

        if backend == "rapidocr":
            try:
                from docling.datamodel.pipeline_options import RapidOcrOptions
            except ImportError as e:
                raise RuntimeError(
                    "RapidOCR backend requested but docling.RapidOcrOptions "
                    "isn't available. Install `rapidocr-onnxruntime` first."
                ) from e
            _log.info(f"OCR backend: RapidOCR (ONNX)")
            return RapidOcrOptions()

        if backend == "auto":
            # Let Docling pick. As of docling v2 this defaults to easyocr.
            try:
                from docling.datamodel.pipeline_options import EasyOcrOptions
                _log.info("OCR backend: auto → EasyOCR")
                return EasyOcrOptions(lang=["vi", "en"])
            except Exception:
                from docling.datamodel.pipeline_options import TesseractCliOcrOptions
                _log.info("OCR backend: auto → Tesseract fallback")
                return TesseractCliOcrOptions(lang=list(self._ocr_lang))

        raise ValueError(
            f"Unknown ocr_backend={backend!r}. "
            "Expected one of: easyocr, rapidocr, tesseract, auto."
        )

    # ────────────────────────────────────────────────────────────────────────
    # Public API — single PDF → list[DocumentPage]
    # ────────────────────────────────────────────────────────────────────────

    def process(self, source_path: str | Path) -> list[DocumentPage]:
        """
        Convert ``source_path`` to a list of ``DocumentPage`` IR objects.
        Handles batching for large PDFs and gracefully continues past a
        failed chunk.
        """
        self._ensure_converter()
        source = Path(source_path)
        if not source.exists():
            raise FileNotFoundError(source)

        if source.suffix.lower() != ".pdf":
            # Single-file conversion (image input).
            return self._convert_single(source, page_offset=0)

        all_pages: list[DocumentPage] = []
        for chunk_path, page_offset in self._iter_pdf_batches(source):
            try:
                pages = self._convert_single(chunk_path, page_offset=page_offset)
                all_pages.extend(pages)
            except Exception as e:
                _log.error(
                    f"Docling chunk failed: {chunk_path} (offset={page_offset}): {e}",
                    exc_info=True,
                )
            finally:
                if chunk_path != source:
                    # Only unlink the temp chunk files, not the original.
                    try:
                        os.unlink(chunk_path)
                    except OSError:
                        pass

        all_pages.sort(key=lambda p: p.page_index)
        return all_pages

    # ────────────────────────────────────────────────────────────────────────
    # Internals
    # ────────────────────────────────────────────────────────────────────────

    def _iter_pdf_batches(self, source: Path) -> Iterable[tuple[Path, int]]:
        """
        Yield (chunk_path, page_offset) tuples. When batching is disabled
        (``batch_size == 0``) yields just (source, 0).
        """
        if self._batch_size <= 0:
            yield source, 0
            return
        try:
            from pypdf import PdfReader, PdfWriter
        except ImportError as e:
            raise RuntimeError(
                "pypdf required for batched PDF processing. "
                "Run: `pip install pypdf`"
            ) from e

        reader = PdfReader(str(source))
        total  = len(reader.pages)
        if total <= self._batch_size:
            yield source, 0
            return

        for start in range(0, total, self._batch_size):
            writer = PdfWriter()
            for j in range(start, min(start + self._batch_size, total)):
                writer.add_page(reader.pages[j])
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            tmp.close()
            with open(tmp.name, "wb") as fh:
                writer.write(fh)
            yield Path(tmp.name), start

    def _convert_single(
        self,
        chunk_path: Path,
        *,
        page_offset: int,
    ) -> list[DocumentPage]:
        """Run Docling on one PDF chunk and translate output to IR pages."""
        _log.info(f"Docling: converting {chunk_path.name} (offset={page_offset})")
        assert self._converter is not None
        result = self._converter.convert(str(chunk_path))
        result_dict = result.document.export_to_dict()
        _log.info(
            "Docling extracted — "
            f"texts: {len(result_dict.get('texts', []))}, "
            f"tables: {len(result_dict.get('tables', []))}, "
            f"pictures: {len(result_dict.get('pictures', []))}"
        )

        # Page metadata (size + raster). Docling indexes pages from 1
        # internally; we offset back to the source PDF's page numbering.
        page_meta: dict[int, dict] = {}
        for page_no_str, info in (result_dict.get("pages") or {}).items():
            try:
                pno = int(page_no_str) - 1 + page_offset   # 0-based
            except ValueError:
                continue
            page_meta[pno] = {
                "width_px":  float(info.get("size", {}).get("width",  0)),
                "height_px": float(info.get("size", {}).get("height", 0)),
            }

        # Group raw items by 0-based source page index.
        per_page_items: dict[int, list[tuple[str, dict]]] = {}

        def _add(pno: int, kind: str, obj: dict) -> None:
            per_page_items.setdefault(pno, []).append((kind, obj))

        for t in result_dict.get("texts", []):
            prov = (t.get("prov") or [{}])[0]
            pno  = (prov.get("page_no", 1) - 1) + page_offset
            _add(pno, "text", t)
        for tbl in result_dict.get("tables", []):
            prov = (tbl.get("prov") or [{}])[0]
            pno  = (prov.get("page_no", 1) - 1) + page_offset
            _add(pno, "table", tbl)
        for pic in result_dict.get("pictures", []):
            prov = (pic.get("prov") or [{}])[0]
            pno  = (prov.get("page_no", 1) - 1) + page_offset
            _add(pno, "picture", pic)

        ir_pages: list[DocumentPage] = []
        next_block_id = 0
        for pno in sorted(per_page_items):
            page_h = page_meta.get(pno, {}).get("height_px", 0.0)
            blocks: list[IRBlock] = []
            for kind, obj in per_page_items[pno]:
                blk = self._item_to_block(
                    kind, obj, page_no=pno, block_id=next_block_id,
                    chunk_path=chunk_path,
                    page_height=page_h,
                )
                if blk is not None:
                    blocks.append(blk)
                    next_block_id += 1

            blocks = _assign_reading_order(blocks)
            page = DocumentPage(
                page_index = pno,
                width_px   = page_meta.get(pno, {}).get("width_px",  0.0),
                height_px  = page_h,
                source_path= str(chunk_path),
                blocks     = blocks,
            )
            ir_pages.append(page)
            _log.info(
                f"IR page {pno}: {len(blocks)} blocks "
                f"({sum(1 for b in blocks if b.type=='table')} tables, "
                f"{sum(1 for b in blocks if b.type=='figure')} figures, "
                f"{sum(1 for b in blocks if b.skip)} skipped)"
            )
        return ir_pages

    def _item_to_block(
        self,
        kind:        str,
        obj:         dict,
        *,
        page_no:     int,
        block_id:    int,
        chunk_path:  Path,
        page_height: float = 0.0,
    ) -> IRBlock | None:
        """Dispatch on item kind and build an IRBlock.

        ``page_height`` is required to convert Docling's BOTTOMLEFT
        bboxes into TOPLEFT for downstream sorting / cropping. When
        omitted (0) we assume bboxes are already TOPLEFT.
        """
        bbox = _extract_bbox(obj, page_height=page_height)
        if kind == "text":
            text  = (obj.get("text") or "").strip()
            label = _normalise_label(obj.get("label") or "text")
            if not text:
                return None
            # Layout-model label is a guess — text that unambiguously
            # reads as "Chương .../Điều N." gets promoted to a real
            # heading even if Docling mislabelled it as plain "text".
            label = reclassify_block_type(label, text)
            # Previously: skip = label in _ABANDON_LABELS (page_footer),
            # which silently dropped running headers/footers (e.g.
            # "CÔNG BÁO/Số .../Ngày ..." + page number) from every
            # export. They're page furniture, not garbage — keep them
            # so the exported DOCX/Markdown preserves the source
            # document's own page-header/footer lines in place.
            return IRBlock(
                id            = block_id,
                type          = label,
                bbox          = bbox,
                ocr_lines     = [OCRLine(text=text, confidence=1.0)],
                original_text = text,
                skip          = False,
            )
        if kind == "table":
            rows = _docling_table_to_rows(obj)
            block_text = " | ".join(
                c.text for row in rows for c in row.cells
                if c.text and c.text.strip() and c.row_span and c.col_span
            )
            return IRBlock(
                id            = block_id,
                type          = "table",
                bbox          = bbox,
                table_rows    = rows,
                original_text = block_text,
            )
        if kind == "picture":
            crop_path = self._save_figure_crop(
                pdf_path = chunk_path,
                bbox     = bbox,
                obj      = obj,
                page_no  = page_no,
                block_id = block_id,
            )
            return IRBlock(
                id          = block_id,
                type        = "figure",
                bbox        = bbox,
                figure_path = str(crop_path) if crop_path else "",
            )
        return None

    def _save_figure_crop(
        self,
        *,
        pdf_path: Path,
        bbox:     tuple,
        obj:      dict,
        page_no:  int,
        block_id: int,
    ) -> Path | None:
        """
        Cut the figure region out of the source PDF as a PNG so the
        markdown / DOCX writers can embed it. Uses PyMuPDF (already a
        dep) because we don't want another image lib.
        """
        try:
            import fitz   # PyMuPDF
        except ImportError:
            _log.warning("PyMuPDF missing; figure crops disabled.")
            return None

        prov = (obj.get("prov") or [{}])[0]
        bb   = prov.get("bbox") or {}
        # Docling reports bbox as a dict with coord_origin "TOPLEFT" or
        # "BOTTOMLEFT". PyMuPDF Rect uses TOPLEFT (y=0 at top, y grows
        # downward). For PDF docs Docling defaults to BOTTOMLEFT.
        x0 = float(bb.get("l", 0)); t_raw = float(bb.get("t", 0))
        x1 = float(bb.get("r", 0)); b_raw = float(bb.get("b", 0))
        coord_origin = (bb.get("coord_origin") or "TOPLEFT").upper()

        # page index inside the chunk = docling 1-based page_no - 1
        page_in_chunk = (prov.get("page_no", 1) - 1)
        try:
            with fitz.open(str(pdf_path)) as doc:
                if page_in_chunk < 0 or page_in_chunk >= len(doc):
                    return None
                page = doc[page_in_chunk]
                if coord_origin == "BOTTOMLEFT":
                    # BOTTOMLEFT: t (PDF top) is the LARGER y value, b is
                    # smaller. Display top = h - t, display bottom = h - b.
                    h = page.rect.height
                    y0 = h - max(t_raw, b_raw)
                    y1 = h - min(t_raw, b_raw)
                else:
                    y0 = min(t_raw, b_raw)
                    y1 = max(t_raw, b_raw)
                rect = fitz.Rect(x0, y0, x1, y1)
                if rect.is_empty:
                    return None
                pix = page.get_pixmap(
                    clip = rect,
                    matrix = fitz.Matrix(2, 2),    # 2× scale for sharper crops
                )
                out = self._figures_dir / f"p{page_no:03d}_b{block_id:03d}.png"
                pix.save(str(out))
                return out
        except Exception as e:
            _log.warning(f"Figure crop failed (page={page_no}, block={block_id}): {e}")
            return None


# ── Helpers ──────────────────────────────────────────────────────────────────

def _extract_bbox(obj: dict, *, page_height: float = 0.0) -> tuple:
    """
    Pull (x0, y0, x1, y1) out of a Docling item, defaulting to ().

    Docling reports PDF bboxes in ``coord_origin == "BOTTOMLEFT"``
    (PDF convention: y=0 at the bottom of the page). Everything
    downstream — reading-order sorting, figure cropping in PyMuPDF —
    assumes TOPLEFT (y=0 at the top, y increases downward).

    When the bbox is BOTTOMLEFT and we know ``page_height``, we flip
    the y-axis. When it's already TOPLEFT, we just normalise so the
    returned ``(y0, y1)`` is ordered ``y0 <= y1`` regardless of which
    field Docling put which value in.

    Bug history: before this conversion existed, sorting by ``bbox[1]``
    ascending placed page-FOOTERS before page-HEADERS in reading order,
    because the footer's BOTTOMLEFT ``top`` field is numerically smaller
    than the header's. That's why the user saw "NGUYỄN VIẾT BẰNG /
    Lưu: VT, DVKT / Nơi nhận:" appear at the top of every page in the
    translated DOCX.
    """
    prov = (obj.get("prov") or [{}])[0]
    bb   = prov.get("bbox") or {}
    if not bb:
        return ()
    l = float(bb.get("l", 0))
    t = float(bb.get("t", 0))
    r = float(bb.get("r", 0))
    b = float(bb.get("b", 0))
    coord_origin = (bb.get("coord_origin") or "TOPLEFT").upper()
    if coord_origin == "BOTTOMLEFT" and page_height > 0:
        # Flip: PDF "top" (large y) becomes our small y near page-top,
        # PDF "bottom" (small y) becomes our large y near page-bottom.
        new_y0 = page_height - max(t, b)
        new_y1 = page_height - min(t, b)
        return (l, new_y0, r, new_y1)
    # Otherwise just normalise ordering so y0 <= y1.
    return (l, min(t, b), r, max(t, b))


def _docling_table_to_rows(table_obj: dict) -> list[TableRow]:
    """
    Convert Docling's ``table.data.grid`` (list of rows, each a list of
    cell dicts) into our IR ``list[TableRow]``. Docling already populates
    row_span / col_span for merged cells, so we don't have to infer them
    the way img2table forced us to.
    """
    grid = (table_obj.get("data") or {}).get("grid") or []
    rows_out: list[TableRow] = []
    for r_idx, row in enumerate(grid):
        cells_out: list[TableCell] = []
        for c_idx, cell in enumerate(row):
            text = (cell.get("text") or "").strip()
            rs = int(cell.get("row_span") or 1)
            cs = int(cell.get("col_span") or 1)
            bb = cell.get("bbox") or {}
            if isinstance(bb, dict):
                bbox = (
                    float(bb.get("l", 0)), float(bb.get("t", 0)),
                    float(bb.get("r", 0)), float(bb.get("b", 0)),
                )
            else:
                bbox = ()
            # Docling marks absorbed (merged-into) cells with the same
            # text as the anchor and matching bbox. We don't need to
            # detect that here — Docling already gives us span info.
            cells_out.append(TableCell(
                row      = r_idx,
                col      = c_idx,
                row_span = rs,
                col_span = cs,
                bbox     = bbox,
                text     = text,
            ))
        if cells_out:
            rows_out.append(TableRow(cells=cells_out))
    return rows_out


def _assign_reading_order(blocks: list[IRBlock]) -> list[IRBlock]:
    """
    Sort blocks top-to-bottom (and left-to-right as a secondary key).
    Docling's reading-order resolver runs inside the model, so we
    *could* trust the order it gave us — but we re-sort here to be
    robust against any reordering that happened during our per-page
    grouping above.
    """
    if not blocks:
        return blocks
    blocks_sorted = sorted(
        blocks,
        key=lambda b: (
            b.bbox[1] if len(b.bbox) >= 2 else 0.0,
            b.bbox[0] if len(b.bbox) >= 2 else 0.0,
        ),
    )
    for idx, b in enumerate(blocks_sorted):
        b.reading_order = idx
    return blocks_sorted


__all__ = ["DoclingCPUPipeline"]