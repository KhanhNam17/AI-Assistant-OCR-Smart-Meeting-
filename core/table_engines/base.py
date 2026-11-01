"""
core/table_engines/base.py
==========================

Canonical IR types and shared helpers for every table reconstruction engine.

Engines (ruled-lines, TATR-NPU, TATR-CPU legacy, Docling) all return
``list[TableRow]`` so the rest of the pipeline does not care which one ran.

Re-exported by ``core.document_pipeline`` for backwards compatibility.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Protocol

_log = logging.getLogger(__name__)


# ── IR ────────────────────────────────────────────────────────────────────────

@dataclass
class TableCell:
    row:        int
    col:        int
    row_span:   int   = 1
    col_span:   int   = 1
    bbox:       tuple = (0.0, 0.0, 0.0, 0.0)
    text:       str   = ""
    translated: str   = ""


@dataclass
class TableRow:
    cells: list[TableCell] = field(default_factory=list)


# ── Engine interface ─────────────────────────────────────────────────────────

class TableBlock(Protocol):
    """Minimum surface needed from a LayoutBlock (avoids circular import)."""
    crop: Any
    bbox: tuple


class TableReconstructorBase(ABC):
    """Engine contract: take a layout block, return a row/cell grid."""

    name: str = "base"

    def __init__(self, ocr_engine: Any) -> None:
        self.ocr_engine = ocr_engine

    @abstractmethod
    def reconstruct(self, blk: TableBlock) -> list[TableRow]:
        """Return rows; empty list signals "engine declined / fall through"."""
        ...


# ── Geometry helpers (shared between table engines) ──────────────────────────

def tatr_iou(a: tuple, b: tuple) -> float:
    ix0 = max(a[0], b[0]); iy0 = max(a[1], b[1])
    ix1 = min(a[2], b[2]); iy1 = min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter  = (ix1 - ix0) * (iy1 - iy0)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union  = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def nms(objs: list[dict], iou: float = 0.5) -> list[dict]:
    """Greedy NMS for row/column candidate boxes (dicts with 'bbox','score')."""
    kept: list[dict] = []
    for o in sorted(objs, key=lambda x: x["score"], reverse=True):
        if not any(tatr_iou(o["bbox"], k["bbox"]) > iou for k in kept):
            kept.append(o)
    return kept


# ── Span recovery via border-line probe ──────────────────────────────────────

def infer_rowspans_by_border(
    grid:     list[TableRow],
    crop,
    num_cols: int,
) -> list[TableRow]:
    """
    For each column independently, walk row boundaries; if no dark horizontal
    line exists in that column at the boundary y, the rows share a merged
    cell — absorb into the anchor row.

    DARK=60 / FRAC=0.25 matches the heuristic that's been working on the
    Vietnamese legal corpus today (see document_pipeline.py original site).
    Anchor convention: anchor cell keeps row_span >= 1; absorbed cells get
    row_span=0, col_span=0, text=''.
    """
    import numpy as np

    if not grid or crop is None or len(grid) < 2:
        return grid

    DARK = 60
    FRAC = 0.25

    cw, ch = crop.size
    gray   = np.array(crop.convert("L"), dtype=np.uint8)

    col_x: list[tuple[int, int]] = []
    for c in range(num_cols):
        if c < len(grid[0].cells):
            cell = grid[0].cells[c]
            col_x.append((max(0, int(cell.bbox[0]) + 2),
                          min(cw, int(cell.bbox[2]) - 2)))
        else:
            col_x.append((0, cw))

    def has_border(y: int, x0: int, x1: int) -> bool:
        y = max(0, min(ch - 1, y))
        if x1 <= x0 + 4:
            return False
        return float((gray[y, x0:x1] < DARK).mean()) >= FRAC

    for c_idx in range(num_cols):
        x0, x1 = col_x[c_idx]
        anchor = 0
        for r_idx in range(1, len(grid)):
            if c_idx >= len(grid[r_idx].cells):
                continue
            cell = grid[r_idx].cells[c_idx]
            if cell.row_span == 0:
                continue
            if has_border(int(cell.bbox[1]), x0, x1):
                anchor = r_idx
            else:
                anc = grid[anchor].cells[c_idx]
                if anc.row_span != 0:
                    anc.row_span += 1
                    cell.row_span = 0
                    cell.col_span = 0
                    cell.text     = ""
    return grid


# ── Per-cell OCR helper (shared by ruled-lines + TATR engines) ───────────────

def ocr_cell_grid(
    grid:       list[TableRow],
    crop,
    ocr_engine,
    pad:        int = 3,
) -> list[TableRow]:
    """OCR every anchor cell in-place. Cells with row_span==0 or col_span==0
    are absorbed mergees and are skipped. Bbox is interpreted in crop pixels.

    Each cell crop is passed through
    :func:`core.cell_preprocess.prepare_cell_for_ocr` first — that strips
    the ruled-line border, upscales tiny cells with INTER_CUBIC, and pads
    a clean white margin. Without this step the recognizer hallucinates
    long trailing garbage like ``cYellowbETY`` because its internal
    background-fill computation gets contaminated by border pixels.
    """
    if crop is None:
        return grid

    # Local import keeps core.table_engines.base free of an OpenCV
    # requirement at import time — some entrypoints construct an engine
    # without ever calling ocr_cell_grid.
    try:
        from cell_preprocess import prepare_cell_for_ocr
    except ImportError:
        # Tolerant fallback so older deployments keep working; we just
        # lose the quality boost.
        def prepare_cell_for_ocr(c):
            return c

    # Confidence threshold for treating short OCR output as noise.
    # Concrete failure mode we're protecting against: tiny near-empty
    # cells (one checkmark, one dot) where the recognizer hallucinates a
    # 1-3 char Latin string like "L", "=", "3S" with very low confidence.
    # A real ASCII character on a clean crop runs ~0.7+ confidence.
    _SHORT_LEN_GATE  = 3
    _LOW_CONF_GATE   = 0.50

    # Width-aware gate for narrow cells (typically column-zero
    # numbering: "1", "1.1", "2", ...). Threshold was 0.70 — calibrated
    # for the old EasyOCR recognizer which gave high-confidence
    # hallucinations. VietOCR is more honest on tiny crops: a legitimate
    # single-digit "1"/"2"/"3" routinely scores 0.50-0.65 because the
    # model was trained on word-level lines, not single chars. Dropping
    # at 0.70 ate every STT (numbering) column. The _SHORT_LEN_GATE
    # below at conf<0.50 still catches the actual hallucinations.
    _NARROW_CELL_PX        = 80
    _NARROW_CELL_CONF_GATE = 0.45

    cw, ch = crop.size
    for row in grid:
        for cell in row.cells:
            if cell.row_span == 0 or cell.col_span == 0:
                continue
            x0, y0, x1, y1 = cell.bbox
            cx0 = max(0, int(x0) - pad)
            cy0 = max(0, int(y0) - pad)
            cx1 = min(cw, int(x1) + pad)
            cy1 = min(ch, int(y1) + pad)
            if cx1 - cx0 < 4 or cy1 - cy0 < 4:
                continue
            ccrop = crop.crop((cx0, cy0, cx1, cy1))
            try:
                cleaned = prepare_cell_for_ocr(ccrop)
            except Exception as e:
                _log.debug(f"cell preprocess failed (passing raw crop): {e}")
                cleaned = ccrop

            # Empty-cell short-circuit. ``prepare_cell_for_ocr`` returns
            # ``None`` when ink ratio is below ~2 % — feeding such a
            # crop to the recognizer produces hallucinated Latin like
            # "3377" / "L" / "=".
            if cleaned is None:
                cell.text = ""
                continue

            try:
                # Multi-line cells (e.g. "Tên thiết bị\nMã hiệu\nSố
                # lượng") are common in Vietnamese government tables.
                # Passing line_separator="\n" preserves the visual
                # breaks; the DOCX and Markdown writers turn those into
                # real in-cell line breaks downstream.
                #
                # Older engines without the kwarg still work — we fall
                # back to a plain call.
                try:
                    text, conf = ocr_engine.recognize_region(
                        cleaned, line_separator="\n"
                    )
                except TypeError:
                    text, conf = ocr_engine.recognize_region(cleaned)
            except Exception as e:
                _log.debug(f"cell OCR failed: {e}")
                text, conf = "", 0.0

            text = text.strip()
            conf_f = float(conf or 0.0)
            cell_w = cx1 - cx0

            # Confidence-gated suppression for residual hallucinations on
            # cells that *passed* the ink-ratio test but still look like
            # noise (e.g. a faint checkmark stroke). Short output + low
            # confidence = drop.
            if text and len(text) <= _SHORT_LEN_GATE and conf_f < _LOW_CONF_GATE:
                _log.debug(
                    "cell OCR suppressed: %r conf=%.3f (short+lowconf)",
                    text, conf_f,
                )
                text = ""

            # Width-aware gate. Narrow cells (typically column-zero
            # numbering) hallucinate at higher confidence than the short
            # gate catches, but the threshold for "trustworthy" output
            # on those crops is higher too. Drop anything below 0.70.
            elif text and cell_w < _NARROW_CELL_PX and conf_f < _NARROW_CELL_CONF_GATE:
                _log.debug(
                    "narrow-cell OCR suppressed: %r conf=%.3f (w=%d < %d)",
                    text, conf_f, cell_w, _NARROW_CELL_PX,
                )
                text = ""

            cell.text = text

    # ── Span-anchor retry pass ──────────────────────────────────────────
    # When img2table DID detect a vertical merge, the anchor cell holds
    # row_span ≥ 2 and the absorbed cells have row_span == 0. The anchor's
    # bbox covers the full merged region. Problem: that region is mostly
    # whitespace with one centred line of text — ink ratio sits below
    # the 2 % empty-cell threshold, so prepare_cell_for_ocr returned None
    # in the main loop and we never even ran the recognizer.
    #
    # Concrete cases on vanban-pages-1.pdf page 3:
    #   * "WAN" (3-row span)
    #   * "Giao thức mạng và tính năng nâng cao" (11-row span)
    #   * "Tính năng Wireless" (9-row span)
    #
    # Retry those with skip_empty_check=True so we go straight to OCR.
    # CRAFT will localise the line, VietOCR will read it. The cost is
    # one extra recognizer call per missed anchor — handful per page.
    _retry_span_anchors(
        grid, crop, ocr_engine, prepare_cell_for_ocr,
        short_len_gate=_SHORT_LEN_GATE,
        low_conf_gate=_LOW_CONF_GATE,
        narrow_cell_px=_NARROW_CELL_PX,
        narrow_cell_conf_gate=_NARROW_CELL_CONF_GATE,
        pad=pad,
    )

    # ── Run-based recovery (kept as defence-in-depth) ───────────────────
    # Handles the case where img2table did NOT detect the merge and
    # produced N adjacent empty single-row cells. Walks each column for
    # runs of ≥2 consecutive empty cells with row_span > 0 and OCRs their
    # union. Independent from the anchor-retry above.
    _recover_column_anchors(
        grid, crop, ocr_engine, prepare_cell_for_ocr,
        short_len_gate=_SHORT_LEN_GATE,
        low_conf_gate=_LOW_CONF_GATE,
        narrow_cell_px=_NARROW_CELL_PX,
        narrow_cell_conf_gate=_NARROW_CELL_CONF_GATE,
        pad=pad,
    )
    return grid


def _retry_span_anchors(
    grid:                  list[TableRow],
    crop,
    ocr_engine,
    prepare_cell_for_ocr,
    *,
    short_len_gate:        int,
    low_conf_gate:         float,
    narrow_cell_px:        int,
    narrow_cell_conf_gate: float,
    pad:                   int = 3,
) -> None:
    """
    Re-OCR every anchor cell with ``row_span > 1`` or ``col_span > 1``
    whose text came back empty in the main pass. Bypasses the ink-ratio
    short-circuit because vertically-merged anchors have low ink ratios
    by design (one line of text, many rows of whitespace).
    """
    if crop is None or not grid:
        return
    cw, ch = crop.size

    for r_idx, row in enumerate(grid):
        for cell in row.cells:
            if cell.row_span <= 1 and cell.col_span <= 1:
                continue
            if (cell.text or "").strip():
                continue
            if cell.bbox is None:
                continue
            x0, y0, x1, y1 = cell.bbox
            cx0 = max(0, int(x0) - pad)
            cy0 = max(0, int(y0) - pad)
            cx1 = min(cw, int(x1) + pad)
            cy1 = min(ch, int(y1) + pad)
            if cx1 - cx0 < 8 or cy1 - cy0 < 16:
                continue

            ccrop = crop.crop((cx0, cy0, cx1, cy1))
            try:
                cleaned = prepare_cell_for_ocr(ccrop, skip_empty_check=True)
            except TypeError:
                # Older signature without skip_empty_check kwarg.
                cleaned = prepare_cell_for_ocr(ccrop)
            except Exception as e:
                _log.debug(f"span-anchor preprocess failed: {e}")
                cleaned = ccrop
            if cleaned is None:
                cleaned = ccrop

            try:
                try:
                    text, conf = ocr_engine.recognize_region(
                        cleaned, line_separator="\n",
                    )
                except TypeError:
                    text, conf = ocr_engine.recognize_region(cleaned)
            except Exception as e:
                _log.debug(f"span-anchor OCR failed: {e}")
                continue

            text   = (text or "").strip()
            conf_f = float(conf or 0.0)
            cell_w = cx1 - cx0
            if not text or len(text) < 2:
                continue
            # Same gates as the per-cell path — symmetric quality bar.
            if len(text) <= short_len_gate and conf_f < low_conf_gate:
                continue
            if cell_w < narrow_cell_px and conf_f < narrow_cell_conf_gate:
                continue

            cell.text = text
            _log.info(
                "span-anchor retry: row=%d span=%dx%d → %r (conf=%.2f)",
                r_idx, cell.row_span, cell.col_span,
                text[:60], conf_f,
            )


def _recover_column_anchors(
    grid:                  list[TableRow],
    crop,
    ocr_engine,
    prepare_cell_for_ocr,
    *,
    short_len_gate:        int,
    low_conf_gate:         float,
    narrow_cell_px:        int,
    narrow_cell_conf_gate: float,
    pad:                   int = 3,
) -> None:
    """
    Find runs of consecutive empty cells in each column and try a single
    union-bbox OCR pass to recover the merged-anchor text. Mutates the
    grid in place. Conservative — only writes a recovered anchor if the
    union OCR returns ≥2 chars at decent confidence.
    """
    if crop is None or not grid:
        return

    cw, ch = crop.size
    ncols  = max((len(r.cells) for r in grid), default=0)

    for c_idx in range(ncols):
        # Collect (row_idx, cell) pairs for this column, with the cell's
        # current state. Cells beyond a row's length are skipped.
        col: list[tuple[int, "TableCell"]] = []
        for r_idx, row in enumerate(grid):
            if c_idx < len(row.cells):
                col.append((r_idx, row.cells[c_idx]))

        # Walk and find runs of ≥2 consecutive empty cells. A cell counts
        # as "empty" if its text is blank AND its bbox has non-trivial
        # height (skip absorbed cells that were marked row_span=0 by the
        # original img2table detection — they already point at a real
        # anchor elsewhere).
        i = 0
        while i < len(col):
            r_i, cell_i = col[i]
            # Note: row_span/col_span can be 0 (absorbed by an anchor) or
            # >=1 (anchor). For the run-based recovery we accept both —
            # the anchor-retry pass above already handled the multi-span
            # anchor case, so here we want to catch the *other* failure
            # mode where img2table left adjacent empty cells without
            # detecting the merge at all.
            is_empty = (
                not (cell_i.text or "").strip()
                and cell_i.bbox is not None
                and (cell_i.bbox[3] - cell_i.bbox[1]) >= 8
            )
            if not is_empty:
                i += 1
                continue
            # Extend the run while subsequent cells are also empty AND
            # vertically adjacent (bottom of cell_k ≈ top of cell_k+1).
            run = [(r_i, cell_i)]
            j = i + 1
            while j < len(col):
                r_j, cell_j = col[j]
                if r_j != run[-1][0] + 1:
                    break
                same_state = (
                    not (cell_j.text or "").strip()
                    and cell_j.bbox is not None
                )
                if not same_state:
                    break
                # Reject big vertical gaps — those are real layout breaks,
                # not a hidden merge. Accept up to 6 px of gap (border
                # pixels) between consecutive sub-cells.
                prev_y1 = run[-1][1].bbox[3]
                this_y0 = cell_j.bbox[1]
                if this_y0 - prev_y1 > 6:
                    break
                run.append((r_j, cell_j))
                j += 1

            if len(run) >= 2:
                # Compute union bbox and OCR it.
                xs0 = min(c.bbox[0] for _, c in run)
                ys0 = min(c.bbox[1] for _, c in run)
                xs1 = max(c.bbox[2] for _, c in run)
                ys1 = max(c.bbox[3] for _, c in run)
                cx0 = max(0, int(xs0) - pad)
                cy0 = max(0, int(ys0) - pad)
                cx1 = min(cw, int(xs1) + pad)
                cy1 = min(ch, int(ys1) + pad)
                if cx1 - cx0 >= 8 and cy1 - cy0 >= 16:
                    text, conf = _ocr_union(
                        crop, (cx0, cy0, cx1, cy1),
                        ocr_engine, prepare_cell_for_ocr,
                    )
                    text = text.strip()
                    conf_f = float(conf or 0.0)
                    cell_w = cx1 - cx0
                    # Same gates as the per-cell path — symmetric quality
                    # bar so we don't introduce hallucinations.
                    if text and len(text) <= short_len_gate \
                            and conf_f < low_conf_gate:
                        text = ""
                    elif text and cell_w < narrow_cell_px \
                            and conf_f < narrow_cell_conf_gate:
                        text = ""
                    if text and len(text) >= 2:
                        anchor_r, anchor_c = run[0]
                        run[0][1].text     = text
                        run[0][1].row_span = len(run)
                        for k in range(1, len(run)):
                            run[k][1].text     = ""
                            run[k][1].row_span = 0
                            run[k][1].col_span = 0
                        _log.info(
                            "span-anchor recovered: col=%d rows=%d..%d "
                            "→ %r (conf=%.2f)",
                            c_idx, anchor_r, anchor_r + len(run) - 1,
                            text[:60], conf_f,
                        )
            i = j


def _ocr_union(
    crop,
    box: tuple[int, int, int, int],
    ocr_engine,
    prepare_cell_for_ocr,
) -> tuple[str, float]:
    """OCR a crop region; same code path as the per-cell helper.
    Kept separate so it stays trivially mockable in tests."""
    cx0, cy0, cx1, cy1 = box
    sub = crop.crop((cx0, cy0, cx1, cy1))
    try:
        cleaned = prepare_cell_for_ocr(sub)
    except Exception:
        cleaned = sub
    if cleaned is None:
        return "", 0.0
    try:
        try:
            return ocr_engine.recognize_region(cleaned, line_separator="\n")
        except TypeError:
            return ocr_engine.recognize_region(cleaned)
    except Exception as e:
        _log.debug(f"union OCR failed: {e}")
        return "", 0.0


__all__ = [
    "TableCell",
    "TableRow",
    "TableBlock",
    "TableReconstructorBase",
    "tatr_iou",
    "nms",
    "infer_rowspans_by_border",
    "ocr_cell_grid",
]
