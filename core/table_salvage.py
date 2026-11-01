"""
core/table_salvage.py — reclassify YOLO-mistagged tables.

YOLO DocLayNet (both 8m and 26m variants) sometimes labels tables as
``Figure`` / ``Picture`` / ``Text``. When that happens the OCR runner
drops them on the floor (``_NON_TEXT_LABELS`` skip), and the user sees a
table-shaped hole in the output.

This module runs ``img2table`` on the **full page image** once per page,
then for each detected table region:

* If a YOLO block overlaps the region (IoU ≥ threshold) and isn't already
  labeled ``Table``, **relabel it ``Table``**.
* If no YOLO block overlaps, **synthesize a new ``Table`` block** at the
  detected coordinates.

Two safety rails:

1. We only act on img2table regions that look genuinely tabular — at
   least 3 rows AND 2 columns. A 2-cell grid won't trigger anything,
   so small images with caption strips stay as figures.
2. We never *remove* blocks. We only relabel or add. So if anything
   misfires the worst case is "a figure got an extra failed OCR run".
"""

from __future__ import annotations

import io
import logging
from typing import Any

_log = logging.getLogger(__name__)

# Minimum structural complexity for an img2table region to count as
# "definitely a table".
_MIN_ROWS = 3
_MIN_COLS = 2

# IoU threshold between an img2table region and a YOLO block. 0.30 is
# loose enough to catch a YOLO bbox that's a bit wider than the table
# (header drift) but tight enough to avoid grabbing an adjacent paragraph.
_IOU_RELABEL = 0.30


def salvage_tables(
    page_pil,
    blocks: list,
    *,
    block_cls,
    min_confidence: int = 50,
    borderless: bool = True,
) -> list:
    """
    Run img2table on the full page image and reconcile against ``blocks``.

    Parameters
    ----------
    page_pil
        Full-page PIL image (RGB).
    blocks
        List of ``LayoutBlock`` from YOLO. Mutated in place — labels can
        change, and new blocks may be appended.
    block_cls
        The ``LayoutBlock`` dataclass (passed in to keep this module free
        of a back-import from ``document_pipeline``).

    Returns
    -------
    The (possibly extended) blocks list, ready for OCR / table
    reconstruction. IDs are renumbered so downstream code sees a clean
    sequence.
    """
    try:
        from img2table.document import Image as Img2TableImage
    except ImportError:
        _log.warning("table_salvage: img2table not installed; skipping")
        return blocks

    if page_pil is None or not blocks:
        return blocks

    # ── Detect tables on the full page ──────────────────────────────────────
    try:
        buf = io.BytesIO()
        page_pil.save(buf, format="PNG")
        buf.seek(0)
        doc = Img2TableImage(src=buf, detect_rotation=False)
        tables = doc.extract_tables(
            ocr               = None,
            implicit_rows     = False,
            implicit_columns  = False,
            borderless_tables = borderless,
            min_confidence    = min_confidence,
        )
    except Exception as e:
        _log.warning(f"table_salvage: img2table failed on page ({e}); skipping")
        return blocks

    if not tables:
        _log.debug("table_salvage: no tables found by img2table on this page")
        return blocks

    # ── Filter to "definitely tabular" structures ───────────────────────────
    real_tables = []
    for t in tables:
        n_rows = _count_rows(t)
        n_cols = _count_cols(t)
        if n_rows >= _MIN_ROWS and n_cols >= _MIN_COLS:
            real_tables.append((t, n_rows, n_cols))

    if not real_tables:
        _log.debug(
            "table_salvage: img2table found %d candidate(s) "
            "but none met >= %dr×%dc threshold",
            len(tables), _MIN_ROWS, _MIN_COLS,
        )
        return blocks

    _log.info(
        "table_salvage: %d definite table(s) on full page (min %dr×%dc)",
        len(real_tables), _MIN_ROWS, _MIN_COLS,
    )

    # ── Reconcile against YOLO blocks ───────────────────────────────────────
    salvage_count   = 0  # relabeled existing blocks
    synthesise_count = 0  # added brand-new Table blocks

    for tbl, n_rows, n_cols in real_tables:
        bb = tbl.bbox
        if hasattr(bb, "x1"):
            tx0, ty0, tx1, ty1 = bb.x1, bb.y1, bb.x2, bb.y2
        else:
            tx0, ty0, tx1, ty1 = bb

        # Find the YOLO block that overlaps the most.
        best_idx = -1
        best_iou = 0.0
        for i, blk in enumerate(blocks):
            iou = _iou((tx0, ty0, tx1, ty1), blk.bbox)
            if iou > best_iou:
                best_iou = iou
                best_idx = i

        if best_idx >= 0 and best_iou >= _IOU_RELABEL:
            target = blocks[best_idx]
            if target.label.lower() != "table":
                _log.info(
                    "  table_salvage: block #%d %r → 'Table' (iou=%.2f, %dr×%dc)",
                    target.id, target.label, best_iou, n_rows, n_cols,
                )
                target.label = "Table"
                salvage_count += 1
        else:
            # No YOLO block matched — synthesise one.
            new_id = max((b.id for b in blocks), default=-1) + 1
            crop_box = (max(0, int(tx0)), max(0, int(ty0)),
                        min(page_pil.width,  int(tx1)),
                        min(page_pil.height, int(ty1)))
            try:
                crop = page_pil.crop(crop_box)
            except Exception:
                crop = None
            new_blk = block_cls(
                id    = new_id,
                label = "Table",
                bbox  = (float(tx0), float(ty0), float(tx1), float(ty1)),
                crop  = crop,
            )
            blocks.append(new_blk)
            synthesise_count += 1
            _log.info(
                "  table_salvage: synthesised Table block at "
                "(%d,%d)-(%d,%d) %dr×%dc",
                int(tx0), int(ty0), int(tx1), int(ty1), n_rows, n_cols,
            )

    # Renumber so block IDs stay dense and ascending.
    if salvage_count or synthesise_count:
        blocks.sort(key=lambda b: (round(b.bbox[1] / 20) * 20, b.bbox[0]))
        for idx, b in enumerate(blocks):
            b.id = idx
        _log.info(
            "table_salvage: %d relabeled, %d synthesised "
            "(total blocks now %d)",
            salvage_count, synthesise_count, len(blocks),
        )

    return blocks


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _count_rows(table) -> int:
    return len(getattr(table, "content", {}) or {})


def _count_cols(table) -> int:
    content = getattr(table, "content", None) or {}
    if not content:
        return 0
    return max((len(row) for row in content.values()), default=0)


def _iou(a: tuple, b: tuple) -> float:
    """IoU on (x0, y0, x1, y1) tuples."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0); iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1); iy1 = min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter  = (ix1 - ix0) * (iy1 - iy0)
    area_a = max(0.0, (ax1 - ax0)) * max(0.0, (ay1 - ay0))
    area_b = max(0.0, (bx1 - bx0)) * max(0.0, (by1 - by0))
    union  = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


__all__ = ["salvage_tables"]
