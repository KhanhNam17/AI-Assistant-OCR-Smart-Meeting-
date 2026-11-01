"""
core/table_engines/ruled_lines.py
=================================

Stage 1 of the hybrid recognizer: rule-based ruled-line detector.

Algorithm
---------
1. Grayscale → adaptive threshold (inverted; gridlines become white).
2. Morphological opening with thin horizontal / vertical kernels isolates
   only long ruling strokes.
3. Peak detection on row/column projections yields candidate line positions.
4. Cell bboxes = cartesian product of adjacent row × column bands.
5. Cell-merge detection: shared edge missing a line → the two cells are
   merged (row_span or col_span ≥ 2 on the anchor; absorbed cell gets 0/0).
6. Per-cell OCR via the shared ``ocr_cell_grid`` helper using the existing
   QAIRT-NPU engine — no new model.

Accept the result iff
    len(rows) >= 3  AND  len(cols) >= 2  AND  coverage >= 0.80
where ``coverage`` is the fraction of (row, col) intersections that actually
contain dark pixels in the line mask. If the bar is missed,
``reconstruct()`` returns ``[]`` so the caller can choose another engine
(typically ``Img2TableReconstructor``).
"""

from __future__ import annotations

import logging
from typing import Any

from .base import (
    TableCell,
    TableRow,
    TableReconstructorBase,
    infer_rowspans_by_border,
    ocr_cell_grid,
)

_log = logging.getLogger(__name__)


# ── Tunable thresholds ───────────────────────────────────────────────────────

_MIN_ROWS              = 2
_MIN_COLS              = 2
_COVERAGE_MIN          = 0.55
_PEAK_MIN_GAP_PX       = 14
_PEAK_REL_HEIGHT       = 0.55   # peak height >= 55% of max projection value
_PEAK_MIN_ABS_RATIO    = 0.40   # line must cover >=40% of opposite axis
_EDGE_LINE_COVERAGE    = 0.30   # < 30% → no line on that edge → merge
_BORDER_MARGIN_PX      = 3      # ignore N pixels at crop edge


def _peak_lines(
    profile,
    min_gap:        int,
    rel_height:     float,
    abs_min_value:  float = 0.0,
) -> list[int]:
    """
    Find peaks in a 1-D projection profile.

    A peak qualifies when the profile value is at least
    ``max(rel_height * profile.max(), abs_min_value)``. The absolute floor
    lets us reject low peaks that come from text-row pixels.
    """
    import numpy as np
    if profile.size == 0:
        return []
    p_max = float(profile.max())
    if p_max <= 0:
        return []
    thresh = max(p_max * rel_height, abs_min_value)
    above  = profile >= thresh
    peaks: list[int] = []
    i = 0
    n = profile.size
    while i < n:
        if not above[i]:
            i += 1
            continue
        j = i
        while j < n and above[j]:
            j += 1
        seg = profile[i:j]
        peaks.append(int(i + int(np.argmax(seg))))
        i = j
    if not peaks:
        return peaks
    merged = [peaks[0]]
    for p in peaks[1:]:
        if p - merged[-1] >= min_gap:
            merged.append(p)
        else:
            if profile[p] > profile[merged[-1]]:
                merged[-1] = p
    return merged


def _bracket(values: list[int], lo: int, hi: int) -> list[int]:
    """Ensure boundary lines exist at lo and hi (close to but not duplicating)."""
    out = sorted(values)
    if not out or out[0] - lo > _PEAK_MIN_GAP_PX:
        out.insert(0, lo)
    if not out or hi - out[-1] > _PEAK_MIN_GAP_PX:
        out.append(hi)
    return out


def _detect_grid(crop_gray):
    """Return (rows_y, cols_x, h_mask, v_mask, coverage)."""
    import cv2
    import numpy as np

    h, w = crop_gray.shape
    bw = cv2.adaptiveThreshold(
        crop_gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,
        25, -2,
    )

    h_kernel_len = max(10, w // 30)
    v_kernel_len = max(10, h // 30)
    horiz = cv2.morphologyEx(
        bw, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_len, 1)),
    )
    vert = cv2.morphologyEx(
        bw, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel_len)),
    )

    # A "real" horizontal rule should fill a large fraction of the crop width
    # (and likewise for vertical rules and crop height). The morph-opened
    # mask stores 255 per ruling pixel, so the absolute floor in projection
    # units is 255 * fraction_of_axis.
    abs_h = 255.0 * w * _PEAK_MIN_ABS_RATIO
    abs_v = 255.0 * h * _PEAK_MIN_ABS_RATIO

    rows_y = _peak_lines(
        horiz.sum(axis=1), _PEAK_MIN_GAP_PX, _PEAK_REL_HEIGHT, abs_h,
    )
    cols_x = _peak_lines(
        vert .sum(axis=0), _PEAK_MIN_GAP_PX, _PEAK_REL_HEIGHT, abs_v,
    )

    rows_y = _bracket(rows_y, _BORDER_MARGIN_PX, h - _BORDER_MARGIN_PX)
    cols_x = _bracket(cols_x, _BORDER_MARGIN_PX, w - _BORDER_MARGIN_PX)

    inter = cv2.bitwise_and(horiz, vert)
    expected = max(1, len(rows_y) * len(cols_x))
    found = 0
    for y in rows_y:
        y0 = max(0, y - 2); y1 = min(h, y + 3)
        for x in cols_x:
            x0 = max(0, x - 2); x1 = min(w, x + 3)
            if inter[y0:y1, x0:x1].any():
                found += 1
    coverage = found / expected
    return rows_y, cols_x, horiz, vert, coverage


def _edge_covered(mask, y0: int, y1: int, x0: int, x1: int, axis: str) -> float:
    """Coverage along a single edge segment in the line mask."""
    import numpy as np
    h, w = mask.shape
    y0 = max(0, min(h - 1, y0)); y1 = max(y0 + 1, min(h, y1))
    x0 = max(0, min(w - 1, x0)); x1 = max(x0 + 1, min(w, x1))
    if axis == "h":  # horizontal segment between (x0..x1) at row y0..y1
        seg = mask[y0:y1, x0:x1]
        # collapse rows; count cols with any line pixel
        col_hit = (seg > 0).any(axis=0)
        if col_hit.size == 0:
            return 0.0
        return float(col_hit.mean())
    else:  # vertical segment
        seg = mask[y0:y1, x0:x1]
        row_hit = (seg > 0).any(axis=1)
        if row_hit.size == 0:
            return 0.0
        return float(row_hit.mean())


class RuledLineReconstructor(TableReconstructorBase):
    """Stage 1 — OpenCV ruled-line table reconstructor."""

    name = "rules"

    def __init__(
        self,
        ocr_engine: Any,
        *,
        min_rows:      int   = _MIN_ROWS,
        min_cols:      int   = _MIN_COLS,
        coverage_min:  float = _COVERAGE_MIN,
    ) -> None:
        super().__init__(ocr_engine)
        self.min_rows     = min_rows
        self.min_cols     = min_cols
        self.coverage_min = coverage_min

    # Diagnostic confidence from the last reconstruct() call.
    last_coverage: float = 0.0
    last_rows:     int   = 0
    last_cols:     int   = 0

    def reconstruct(self, blk) -> list[TableRow]:
        import numpy as np

        crop = getattr(blk, "crop", None)
        if crop is None:
            return []
        crop = crop.convert("RGB")
        gray = np.array(crop.convert("L"), dtype=np.uint8)

        rows_y, cols_x, h_mask, v_mask, coverage = _detect_grid(gray)
        self.last_coverage = coverage
        self.last_rows     = max(0, len(rows_y) - 1)
        self.last_cols     = max(0, len(cols_x) - 1)

        n_rows = self.last_rows
        n_cols = self.last_cols

        if (n_rows < self.min_rows
                or n_cols < self.min_cols
                or coverage < self.coverage_min):
            _log.info(
                "  ruled-lines: declined "
                f"(rows={n_rows}, cols={n_cols}, coverage={coverage:.2f})"
            )
            return []

        _log.info(
            f"  ruled-lines: {n_rows} rows x {n_cols} cols "
            f"(coverage={coverage:.2f})"
        )

        # Build raw cell grid.
        grid: list[TableRow] = []
        for r_idx in range(n_rows):
            y0, y1 = rows_y[r_idx], rows_y[r_idx + 1]
            tr = TableRow()
            for c_idx in range(n_cols):
                x0, x1 = cols_x[c_idx], cols_x[c_idx + 1]
                tr.cells.append(TableCell(
                    row=r_idx, col=c_idx,
                    bbox=(float(x0), float(y0), float(x1), float(y1)),
                ))
            grid.append(tr)

        # Horizontal merges: cell looks right; if right edge has no v_mask line,
        # absorb the right neighbour into this anchor (col_span += 1).
        for r_idx in range(n_rows):
            row = grid[r_idx].cells
            c_idx = 0
            while c_idx < len(row):
                anchor = row[c_idx]
                if anchor.col_span == 0:
                    c_idx += 1
                    continue
                nxt = c_idx + anchor.col_span
                while nxt < len(row):
                    edge_x = cols_x[nxt]   # boundary between anchor and nxt
                    ey0    = rows_y[r_idx] + 2
                    ey1    = rows_y[r_idx + 1] - 2
                    cov    = _edge_covered(
                        v_mask, ey0, ey1, edge_x - 2, edge_x + 3, "v")
                    if cov >= _EDGE_LINE_COVERAGE:
                        break
                    # Merge nxt into anchor
                    absorbed = row[nxt]
                    anchor.col_span += 1
                    anchor.bbox = (
                        anchor.bbox[0], anchor.bbox[1],
                        absorbed.bbox[2], anchor.bbox[3],
                    )
                    absorbed.row_span = 0
                    absorbed.col_span = 0
                    absorbed.text     = ""
                    nxt += 1
                c_idx = nxt

        # Vertical merges: defer to the shared border-probe helper so the
        # convention matches the legacy path exactly (anchor keeps row_span>=1,
        # absorbed cells set to 0/0). Operate on horizontal mask too as a
        # fast sanity pass before the helper.
        grid = infer_rowspans_by_border(grid, crop, n_cols)

        # OCR every anchor cell.
        grid = ocr_cell_grid(grid, crop, self.ocr_engine, pad=3)
        return grid


__all__ = ["RuledLineReconstructor"]
