"""
core/table_engines — pluggable table reconstruction strategies.

Use :func:`build_reconstructor` to obtain a strategy by name. Each strategy
takes a layout block (with ``.crop`` and ``.bbox``) and returns the
canonical ``list[TableRow]`` from :mod:`core.table_engines.base`.

Strategies
----------
* ``img2table`` (default) — OpenCV structure detection via
  `xavctn/img2table`_, OCR on the whole crop via QAIRT EasyOCR.
  Handles bordered + borderless tables and reports row/col spans.
* ``rules`` — pure ruled-lines reconstruction. Fast but only works on
  bordered tables.
* ``docling`` — Docling TableFormer fallback (heavy, optional dep).
* ``mineru`` — MinerU CLI (layout + table structure), text taken from
  MinerU's own built-in OCR (no bbox per cell, so no re-OCR pass with
  our own recognizer). Slower — spawns a subprocess per table crop.
  Useful when img2table/rules mis-detect complex multi-level rowspan
  tables; see :mod:`core.table_engines.mineru_engine`.

The legacy TATR engines (``tatr``, ``legacy``, ``hybrid``) were removed
in favour of img2table — TATR's NPU port was brittle and the per-cell
crop OCR pattern produced garbled Vietnamese characters.

.. _xavctn/img2table: https://github.com/xavctn/img2table
"""

from __future__ import annotations

import logging
from typing import Any

from .base import (
    TableCell,
    TableRow,
    TableReconstructorBase,
    nms,
    ocr_cell_grid,
    tatr_iou,
    infer_rowspans_by_border,
)
from .ruled_lines import RuledLineReconstructor
# Note: docling, img2table, and mineru are intentionally NOT eagerly
# imported — they pull heavy deps (or spawn a subprocess). Loaded on
# demand inside build_reconstructor.

_log = logging.getLogger(__name__)


def build_reconstructor(
    name:       str,
    ocr_engine: Any,
    cfg:        dict | None = None,
) -> TableReconstructorBase:
    """
    Construct a table reconstruction strategy by name.

    Names: ``img2table`` (default), ``rules``, ``docling``, ``mineru``.
    """
    cfg  = cfg or {}
    name = (name or "img2table").lower()

    if name == "img2table":
        from .img2table_engine import Img2TableReconstructor
        return Img2TableReconstructor(ocr_engine)

    if name == "rules":
        return RuledLineReconstructor(ocr_engine)

    if name == "docling":
        from .docling import DoclingReconstructor
        return DoclingReconstructor(ocr_engine, lang=cfg.get("lang", "vi"))

    if name == "mineru":
        from .mineru_engine import MineruTableReconstructor
        return MineruTableReconstructor(
            ocr_engine,
            mineru_backend=cfg.get("mineru_backend", "pipeline"),
            keep_debug_files=cfg.get("mineru_keep_debug_files", False),
        )

    raise ValueError(
        f"Unknown table engine: {name!r} "
        "(expected img2table|rules|docling|mineru)"
    )


__all__ = [
    "TableCell",
    "TableRow",
    "TableReconstructorBase",
    "nms",
    "ocr_cell_grid",
    "tatr_iou",
    "infer_rowspans_by_border",
    "RuledLineReconstructor",
    "build_reconstructor",
]