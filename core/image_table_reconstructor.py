"""
core/image_table_reconstructor.py
=================================

Reconstruct a table from OCR line positions when img2table's border-
detection fails. Used as a fallback for borderless tables and for
full-page scanned tables that come through as a single ``Picture``
layout block.

Algorithm
---------
1. Caller hands us a flat list of ``OCRLine`` (each with a polygon).
2. Group lines vertically into rows by y-overlap clustering.
3. Estimate column count from the median lines-per-row.
4. Build column boundaries by 1-D clustering line-left x positions
   across all lines.
5. For each row × column slot, pick the line whose left-x falls into
   that column's bucket. Multiple lines collapse into the same cell
   with ``" "`` separation (preserves multi-word content).
6. Emit a ``list[TableRow]`` matching the IR contract that
   ``table_docx_writer`` expects.

This deliberately doesn't try to detect row spans or column spans —
img2table is better at that when it does work. The output here is
always a flat grid, which is the right behaviour for the
"full-page image is actually a table" case the user described:
those tables have very regular grid structure precisely because
they have no spans.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from table_engines.base import TableCell, TableRow

logger = logging.getLogger(__name__)


# ── Tunables ─────────────────────────────────────────────────────────────────

# Minimum dimensions to call the result a "real" table. Anything smaller
# is rejected so callers can keep their paragraph fallback for short
# regions.
_MIN_ROWS = 3
_MIN_COLS = 2

# y-overlap fraction for a line to join an existing row.
_ROW_OVERLAP_THRESH = 0.45

# When clustering x-left positions for column boundaries, two x's within
# this fraction of the crop width fall into the same column.
_COL_X_GAP_FRAC = 0.04


# ── Public API ───────────────────────────────────────────────────────────────

def reconstruct_table_from_lines(
    lines:     list,
    crop_size: tuple[int, int],
    *,
    min_rows:  int = _MIN_ROWS,
    min_cols:  int = _MIN_COLS,
) -> list[TableRow]:
    """
    Build a ``list[TableRow]`` grid from a flat list of OCR lines.

    Returns an empty list if the reconstruction doesn't meet the
    ``min_rows`` × ``min_cols`` quality bar, so the caller can fall
    back to its original paragraph dump.

    Parameters
    ----------
    lines
        ``list[OCRLine]`` with ``.polygon`` populated. The polygon must
        be a sequence of ``(x, y)`` tuples (4 corners typical).
    crop_size
        ``(width, height)`` of the source crop, used to normalise the
        column-clustering distance threshold.
    """
    if not lines:
        return []
    w, _h = crop_size
    if w <= 0:
        return []

    # 1) Cluster into rows.
    rows = _group_lines_by_row(lines, overlap_thresh=_ROW_OVERLAP_THRESH)
    if len(rows) < min_rows:
        logger.debug(
            f"reconstruct_table_from_lines: only {len(rows)} rows clustered, "
            f"need ≥{min_rows} — declining."
        )
        return []

    # 2) Estimate column count. Look at the distribution of line counts
    #    per row — pages that are really tables have a strong mode at
    #    the true column count.
    counts   = sorted(len(r) for r in rows)
    median_n = counts[len(counts) // 2]
    n_cols   = max(min_cols, median_n)

    # 3) Column boundaries from clustered line-left positions.
    boundaries = _column_boundaries_from_lines(
        lines, n_cols=n_cols, crop_width=float(w),
    )
    if len(boundaries) <= 1:
        logger.debug(
            "reconstruct_table_from_lines: no usable column boundaries"
        )
        return []
    n_cols_final = len(boundaries) - 1
    if n_cols_final < min_cols:
        return []

    # 4) Assign each line to its (row, col) slot.
    grid: list[TableRow] = []
    for r_idx, row_lines in enumerate(rows):
        cells: list[TableCell] = [
            TableCell(row=r_idx, col=c, row_span=1, col_span=1, bbox=(), text="")
            for c in range(n_cols_final)
        ]
        for ln in row_lines:
            x_left = _line_x_left(ln)
            col_idx = _bucket_for_x(x_left, boundaries)
            if col_idx is None:
                continue
            existing = cells[col_idx].text
            cells[col_idx].text = (
                f"{existing} {ln.text}".strip() if existing else ln.text
            )
            # Track the union bbox of all lines that landed in this cell.
            cells[col_idx].bbox = _union_bbox(cells[col_idx].bbox, _line_bbox(ln))
        grid.append(TableRow(cells=cells))

    n_with_text = sum(1 for r in grid for c in r.cells if c.text.strip())
    logger.info(
        f"image-table reconstruct: {len(grid)} rows × {n_cols_final} cells "
        f"({n_with_text} with text)"
    )
    return grid


# ── Row clustering ───────────────────────────────────────────────────────────

def _group_lines_by_row(
    lines: list,
    *,
    overlap_thresh: float = _ROW_OVERLAP_THRESH,
) -> list[list]:
    """
    Group lines into rows by y-overlap. A line joins an existing row
    when their y-extents overlap by ≥ ``overlap_thresh`` of the smaller
    height. Returned rows are sorted top-to-bottom, each row sorted
    left-to-right.
    """
    sorted_lines = sorted(lines, key=_line_y_mid)
    rows:        list[list] = []
    row_ranges:  list[tuple[float, float]] = []
    for ln in sorted_lines:
        y0, y1 = _line_y_extent(ln)
        h = max(1e-3, y1 - y0)
        placed = False
        for i in range(len(rows) - 1, -1, -1):
            ry0, ry1 = row_ranges[i]
            overlap  = max(0.0, min(y1, ry1) - max(y0, ry0))
            if overlap / min(h, ry1 - ry0) >= overlap_thresh:
                rows[i].append(ln)
                row_ranges[i] = (min(ry0, y0), max(ry1, y1))
                placed = True
                break
        if not placed:
            rows.append([ln])
            row_ranges.append((y0, y1))
    for row in rows:
        row.sort(key=_line_x_left)
    rows.sort(key=lambda r: float(np.mean([_line_y_mid(l) for l in r])))
    return rows


# ── Column-boundary clustering ──────────────────────────────────────────────

def _column_boundaries_from_lines(
    lines:      list,
    *,
    n_cols:     int,
    crop_width: float,
) -> list[float]:
    """
    Pick ``n_cols + 1`` x-boundaries by 1-D clustering line-left
    positions, then choose split points between adjacent clusters.

    Boundaries start at x=0 and end at x=crop_width so every line's
    x-left falls into exactly one bucket.
    """
    xs = sorted(_line_x_left(ln) for ln in lines)
    if not xs:
        return []

    # 1-D agglomerative-ish clustering: walk sorted xs, merge into the
    # current cluster when the gap to the previous x is small. The
    # threshold is a fraction of the crop width.
    gap_thresh = max(2.0, crop_width * _COL_X_GAP_FRAC)
    clusters: list[list[float]] = [[xs[0]]]
    for x in xs[1:]:
        if x - clusters[-1][-1] <= gap_thresh:
            clusters[-1].append(x)
        else:
            clusters.append([x])
    # Use cluster centroids as column "anchor" positions, then derive
    # boundaries as midpoints between adjacent anchors.
    anchors = [float(np.mean(c)) for c in clusters]
    if len(anchors) < 2:
        # Only one cluster — fall back to even-width columns.
        anchors = [
            crop_width * (i + 0.5) / n_cols
            for i in range(max(1, n_cols))
        ]
    boundaries: list[float] = [0.0]
    for i in range(len(anchors) - 1):
        boundaries.append((anchors[i] + anchors[i + 1]) / 2.0)
    boundaries.append(crop_width)
    return boundaries


def _bucket_for_x(x: float, boundaries: list[float]) -> int | None:
    """Find which bucket ``x`` falls into. Boundaries are sorted, with
    ``len(boundaries) - 1`` buckets total."""
    if not boundaries:
        return None
    for i in range(len(boundaries) - 1):
        lo, hi = boundaries[i], boundaries[i + 1]
        if lo <= x < hi:
            return i
    return len(boundaries) - 2


# ── Geometry helpers ────────────────────────────────────────────────────────

def _line_y_mid(line) -> float:
    if not line.polygon:
        return 0.0
    ys = [p[1] for p in line.polygon]
    return (min(ys) + max(ys)) / 2.0


def _line_y_extent(line) -> tuple[float, float]:
    if not line.polygon:
        return (0.0, 0.0)
    ys = [p[1] for p in line.polygon]
    return (min(ys), max(ys))


def _line_x_left(line) -> float:
    if not line.polygon:
        return 0.0
    return min(p[0] for p in line.polygon)


def _line_bbox(line) -> tuple[float, float, float, float]:
    if not line.polygon:
        return (0.0, 0.0, 0.0, 0.0)
    xs = [p[0] for p in line.polygon]
    ys = [p[1] for p in line.polygon]
    return (min(xs), min(ys), max(xs), max(ys))


def _union_bbox(a, b) -> tuple[float, float, float, float]:
    if not a:
        return b
    if not b:
        return a
    return (min(a[0], b[0]), min(a[1], b[1]),
            max(a[2], b[2]), max(a[3], b[3]))


__all__ = ["reconstruct_table_from_lines"]
