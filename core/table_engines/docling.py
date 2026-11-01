"""
core/table_engines/docling.py
=============================

Optional Docling TableFormer reconstructor. Only loaded when the user
selects ``--table-engine docling``. Kept for A/B comparison; **note this
path goes through Docling's markdown export, so it cannot represent merged
cells** — present spans collapse to flat 1x1 cells. Useful only to verify
that the rest of the pipeline is wired correctly.

FIXED: docling >= 2.x moved pipeline configuration off the DocumentConverter
constructor. ``DocumentConverter(pipeline_options=...)`` no longer exists —
you must wrap options in ``PdfFormatOption`` and register them per
``InputFormat`` via ``format_options``. See:
https://docling-project.github.io/docling/reference/document_converter/
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

from .base import TableCell, TableRow, TableReconstructorBase

_log = logging.getLogger(__name__)


class DoclingReconstructor(TableReconstructorBase):
    name = "docling"

    def __init__(self, ocr_engine: Any, *, lang: str = "vi") -> None:
        super().__init__(ocr_engine)
        self.lang   = lang
        self._conv  = None

    def _load(self) -> None:
        if self._conv is not None:
            return
        try:
            from docling.document_converter import DocumentConverter, PdfFormatOption
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import (
                PdfPipelineOptions,
                TableFormerMode,
                EasyOcrOptions,
            )
            opts = PdfPipelineOptions()
            opts.do_ocr             = True
            opts.do_table_structure = True
            opts.table_structure_options.do_cell_matching = True
            opts.table_structure_options.mode             = TableFormerMode.ACCURATE
            opts.ocr_options = EasyOcrOptions(lang=[self.lang])

            # docling >= 2.x: pipeline_options không còn là kwarg trực tiếp của
            # DocumentConverter — phải bọc qua PdfFormatOption rồi gắn vào
            # format_options theo đúng InputFormat.
            self._conv = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=opts),
                }
            )
            _log.info("Docling TableFormer loaded")
        except Exception as e:
            _log.warning(f"Docling unavailable ({e})")

    def reconstruct(self, blk) -> list[TableRow]:
        self._load()
        if self._conv is None or getattr(blk, "crop", None) is None:
            return []

        with tempfile.TemporaryDirectory() as td:
            png = Path(td) / "table.png"
            blk.crop.save(str(png))
            try:
                result = self._conv.convert(str(png))
            except Exception as e:
                _log.warning(f"Docling convert failed: {e}")
                return []

        tables = list(result.document.tables) if hasattr(result.document, "tables") else []
        if not tables:
            return []
        md = tables[0].export_to_markdown()
        return _parse_pipe_markdown(md)


def _parse_pipe_markdown(md: str) -> list[TableRow]:
    """Lossy parse — flat grid only (Docling markdown cannot encode merges)."""
    lines = [l for l in md.splitlines() if l.strip().startswith("|")]
    data_lines = [
        l for l in lines
        if not all(c in "|-: \t" for c in l.strip())
    ]
    grid: list[TableRow] = []
    for r_idx, line in enumerate(data_lines):
        parts = [c.strip() for c in line.strip().strip("|").split("|")]
        tr = TableRow()
        for c_idx, txt in enumerate(parts):
            tr.cells.append(TableCell(
                row=r_idx, col=c_idx,
                bbox=(0.0, 0.0, 0.0, 0.0),
                text=txt,
            ))
        grid.append(tr)
    return grid


__all__ = ["DoclingReconstructor"]