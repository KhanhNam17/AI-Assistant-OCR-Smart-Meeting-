"""
core/table_engines/img2table_engine.py
======================================

Table reconstruction backed by `xavctn/img2table`_ for **structure
detection only**, with text filled in by our QAIRT EasyOCR engine
through the shared :func:`ocr_cell_grid` helper.

Why this split design
---------------------
We deliberately do **not** use img2table's built-in OCR adapter pattern
(``extract_tables(ocr=adapter)``). An earlier version did and produced
two regressions:

* **Speed.** Using an adapter means running our recognizer once per CRAFT
  word box on the whole crop — roughly 5–10× more recognizer calls than
  running it once per *cell*. A 10-page document went from ~3 min to
  ~20 min.
* **Mapping failures.** img2table's internal bbox coordinate space did
  not always line up with the pixel-space bboxes our recognizer returns,
  so words frequently failed to attach to cells. Markdown ended up empty.

Cleaner split:

    img2table       →  structure (cells, row/col spans, merged-cell info)
    recognize_region → text per cell crop (one call per cell, NPU-fast)

This is the same pattern :class:`RuledLineReconstructor` already uses.

.. _xavctn/img2table: https://github.com/xavctn/github.com/xavctn/img2table
"""

from __future__ import annotations

import io
import logging
from typing import Any

from .base import (
    TableCell,
    TableRow,
    TableReconstructorBase,
    ocr_cell_grid,
)

_log = logging.getLogger(__name__)


class Img2TableReconstructor(TableReconstructorBase):
    """
    Structure detection via img2table + per-cell EasyOCR.

    Parameters
    ----------
    ocr_engine
        Anything with ``recognize_region(PIL.Image) -> (str, float)``.
        Used by :func:`ocr_cell_grid` to fill cell text.
    borderless_tables
        ``True`` enables img2table's whitespace-based column detection —
        important for the kind of gridless layouts common in Vietnamese
        government letters.
    implicit_rows, implicit_columns
        Let img2table infer extra rows / columns from text positions when
        ruled lines are sparse. Default off; toggle on if a specific
        document loses rows.
    min_confidence
        img2table's *table* detection threshold (0–100). Lower = more
        permissive.
    """

    name = "img2table"

    def __init__(
        self,
        ocr_engine: Any,
        *,
        borderless_tables: bool = True,
        implicit_rows:     bool = False,
        implicit_columns:  bool = False,
        min_confidence:    int  = 50,
    ) -> None:
        super().__init__(ocr_engine)
        self._borderless    = borderless_tables
        self._implicit_rows = implicit_rows
        self._implicit_cols = implicit_columns
        self._min_conf      = min_confidence

        # Up-front dependency check so failures land at construction time.
        try:
            from img2table.document import Image as _Img  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "img2table is not installed. Run `pip install img2table` "
                "to enable the img2table table reconstructor."
            ) from e

    # ─────────────────────────────────────────────────────────────────────
    # Engine contract
    # ─────────────────────────────────────────────────────────────────────

    def reconstruct(self, blk) -> list[TableRow]:
        """
        Detect structure on ``blk.crop`` (PIL Image) with img2table, then
        OCR every cell with the shared engine via :func:`ocr_cell_grid`.

        Returns ``[]`` on any failure so the pipeline can still emit
        whatever IR it has — never crashes the per-page loop.
        """
        crop = blk.crop
        if crop is None:
            _log.debug("img2table: crop missing on block %s",
                       getattr(blk, "id", "?"))
            return []

        try:
            tables = self._detect_structure(crop)
        except Exception as e:
            _log.warning(f"img2table structure detection failed: {e}")
            tables = []

        # First-pass failure recovery. The page-1 table of vanban-1-10.pdf
        # only has horizontal rules, no vertical separators, and the
        # default borderless heuristic returns nothing. Retry once with
        # implicit_rows+implicit_columns ON and a lower min_confidence so
        # img2table infers structure from text-position whitespace.
        if not tables:
            _log.info(
                "img2table: first pass empty — retrying with implicit "
                "rows/columns + lower min_confidence"
            )
            try:
                tables = self._detect_structure(
                    crop,
                    implicit_rows     = True,
                    implicit_columns  = True,
                    borderless_tables = True,
                    min_confidence    = max(20, self._min_conf - 30),
                )
            except Exception as e:
                _log.warning(f"img2table aggressive retry failed: {e}")
                tables = []

        if not tables:
            _log.warning(
                "img2table: no structure detected even after aggressive "
                "retry. Table block will be emitted as a single OCR "
                "paragraph instead of a structured table."
            )
            return []

        # YOLO already cropped to a single table region. If img2table
        # finds more than one (rare), take the largest by area.
        chosen = max(tables, key=_table_area)

        grid = _img2table_to_grid(chosen)
        if not grid:
            _log.debug("img2table: chosen table has no rows")
            return []

        # Reuse the shared per-cell OCR helper for consistency with the
        # ruled-lines reconstructor.
        grid = ocr_cell_grid(grid, crop, self.ocr_engine, pad=3)

        n_cells = sum(len(r.cells) for r in grid)
        n_text  = sum(1 for r in grid for c in r.cells
                      if c.text and c.row_span and c.col_span)
        _log.info(
            "img2table: %d rows × %d cells (%d with text)",
            len(grid), n_cells, n_text,
        )
        return grid

    # ─────────────────────────────────────────────────────────────────────
    # Internals
    # ─────────────────────────────────────────────────────────────────────

    def _detect_structure(
        self, crop,
        *,
        implicit_rows:     bool | None = None,
        implicit_columns:  bool | None = None,
        borderless_tables: bool | None = None,
        min_confidence:    int  | None = None,
    ) -> list:
        """Render the PIL crop to PNG bytes and call img2table (no OCR).

        Kwargs override the instance defaults — used for the aggressive
        retry path when the default-settings pass returns nothing.
        """
        from img2table.document import Image as Img2TableImage

        buf = io.BytesIO()
        crop.save(buf, format="PNG")
        buf.seek(0)

        doc    = Img2TableImage(src=buf, detect_rotation=False)
        tables = doc.extract_tables(
            ocr               = None,                  # structure only
            implicit_rows     = (self._implicit_rows
                                 if implicit_rows is None else implicit_rows),
            implicit_columns  = (self._implicit_cols
                                 if implicit_columns is None else implicit_columns),
            borderless_tables = (self._borderless
                                 if borderless_tables is None else borderless_tables),
            min_confidence    = (self._min_conf
                                 if min_confidence is None else min_confidence),
        )
        return tables or []


# ─────────────────────────────────────────────────────────────────────────
# img2table.ExtractedTable → canonical IR
# ─────────────────────────────────────────────────────────────────────────

def _table_area(t) -> float:
    bb = t.bbox
    if hasattr(bb, "x1"):
        return float(max(0, bb.x2 - bb.x1) * max(0, bb.y2 - bb.y1))
    x1, y1, x2, y2 = bb
    return float(max(0, x2 - x1) * max(0, y2 - y1))


def _img2table_to_grid(table) -> list[TableRow]:
    """
    Convert ``ExtractedTable`` → ``list[TableRow]`` with row/col spans.

    img2table reports merged cells by repeating the same ``Cell`` object
    (or one with an identical bbox) across the positions it spans. We
    detect that and:
      * keep the *anchor* (top-left) instance with computed spans
      * mark absorbed copies as ``row_span = 0`` / ``col_span = 0`` so
        :func:`ocr_cell_grid` and the DOCX writer skip them — matches
        the convention used by every other reconstructor.

    Cells start with empty ``text``; :func:`ocr_cell_grid` fills it in
    using ``ocr_engine.recognize_region`` on each cell crop.
    """
    rows_out: list[TableRow] = []
    if not getattr(table, "content", None):
        return rows_out

    raw_grid: list[list[TableCell]] = []
    for r_idx, row in enumerate(table.content.values()):
        cells: list[TableCell] = []
        for c_idx, c in enumerate(row):
            bb = c.bbox
            if hasattr(bb, "x1"):
                x1, y1, x2, y2 = bb.x1, bb.y1, bb.x2, bb.y2
            else:
                x1, y1, x2, y2 = bb
            cells.append(TableCell(
                row      = r_idx,
                col      = c_idx,
                row_span = 1,
                col_span = 1,
                bbox     = (float(x1), float(y1), float(x2), float(y2)),
                # Leave text empty — ocr_cell_grid will fill it.
                text     = "",
            ))
        raw_grid.append(cells)

    if not raw_grid:
        return rows_out

    # Span detection by bbox dedup. Round to int pixels so float jitter
    # doesn't break the comparison.
    seen: dict[tuple, tuple[int, int]] = {}
    for r_idx, row in enumerate(raw_grid):
        for c_idx, cell in enumerate(row):
            key = (round(cell.bbox[0]), round(cell.bbox[1]),
                   round(cell.bbox[2]), round(cell.bbox[3]))
            if key in seen:
                anchor_r, anchor_c = seen[key]
                anchor = raw_grid[anchor_r][anchor_c]
                if r_idx > anchor_r:
                    anchor.row_span = max(anchor.row_span,
                                          r_idx - anchor_r + 1)
                    cell.row_span = 0
                if c_idx > anchor_c:
                    anchor.col_span = max(anchor.col_span,
                                          c_idx - anchor_c + 1)
                    cell.col_span = 0
            else:
                seen[key] = (r_idx, c_idx)

    for row in raw_grid:
        rows_out.append(TableRow(cells=row))
    return rows_out


__all__ = ["Img2TableReconstructor"]
