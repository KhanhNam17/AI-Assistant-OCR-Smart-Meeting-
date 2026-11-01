"""
core/ir_types.py
================

Intermediate-representation dataclasses shared between the pipeline
(``docling_pipeline.py``), the translation engine
(``document_pipeline.TranslationEngine``), and the DOCX writer
(``table_docx_writer.py``).

Kept in a separate, dependency-free module so the writer doesn't have
to import the pipeline (and so renaming/replacing the pipeline doesn't
ripple into downstream importers).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ── OCR / per-line ───────────────────────────────────────────────────────────

@dataclass
class OCRLine:
    text:       str
    confidence: float
    polygon:    list[tuple[float, float]] = field(default_factory=list)


# ── Table cells / rows ───────────────────────────────────────────────────────

@dataclass
class TableCell:
    row:        int
    col:        int
    row_span:   int = 1
    col_span:   int = 1
    bbox:       tuple = ()
    text:       str  = ""
    translated: str  = ""


@dataclass
class TableRow:
    cells: list[TableCell] = field(default_factory=list)


# ── Block-level IR ───────────────────────────────────────────────────────────

@dataclass
class IRBlock:
    """One detected region on the page.

    Block ``type`` values:
        title / section_header / page_header / page_footer /
        text / list_item / caption / formula / figure / table
    """
    id:              int
    type:            str
    bbox:            tuple
    reading_order:   int             = 0
    ocr_lines:       list[OCRLine]   = field(default_factory=list)
    table_rows:      list[TableRow]  = field(default_factory=list)
    original_text:   str             = ""
    translated_text: str             = ""
    skip:            bool            = False
    figure_path:     str             = ""


@dataclass
class DocumentPage:
    page_index:  int
    width_px:    float
    height_px:   float
    source_path: str
    blocks:      list[IRBlock] = field(default_factory=list)


__all__ = [
    "OCRLine",
    "TableCell",
    "TableRow",
    "IRBlock",
    "DocumentPage",
]
