"""
core/cell_preprocess.py — defensive preprocessing for table-cell OCR crops.

Why this exists
---------------
Body-text OCR works fine: the layout detector hands us a cell-free crop of
a paragraph and the recognizer's preprocessor (`split_and_preprocess_crop`)
does the right thing. Table cells are a different beast:

1. The cell crop includes the surrounding ruled lines as a frame of dark
   pixels.
2. ``split_and_preprocess_crop`` computes the canvas's background fill
   colour from the *top and bottom 1–3 rows* of the crop. When those rows
   ARE the cell's black border, the fill goes dark — and the recognizer
   sees a near-black canvas with a small island of text. That's well
   outside its training distribution, and you get long trailing junk
   tokens like ``cYellowbETY``.
3. Many table cells are 18–22 px tall at 150 DPI. The recognizer needs
   height 64; the resize that gets us there uses INTER_LINEAR which
   blurs the already-tiny Vietnamese diacritics into illegible blobs.

This module patches both, *only* on the table-cell code path. Body text
OCR is untouched.

Public API
----------
:func:`prepare_cell_for_ocr` — single entry point used by
:func:`core.table_engines.base.ocr_cell_grid`.
"""

from __future__ import annotations

import logging
from typing import Final

import cv2
import numpy as np
from PIL import Image

_log = logging.getLogger(__name__)

# Tuning constants. Calibrated against scanned Vietnamese government
# tables (150 DPI), where ruled-line borders are typically 1-2 px thick
# and anti-aliased to gray 120-150 (not pure black).
#
# History: we started with _DARK_PIXEL_THRESH=100 and missed nearly every
# anti-aliased border, which caused the recognizer to read those border
# strokes as Latin characters ("]", "L", "Z", "1", "3L", "Eo", ...) at
# the leading/trailing edges of each cell's transcription.

#: Pixel below this 0–255 value counts as "dark". 200 catches even
#: heavily anti-aliased gray strokes (real-world borders sit at ~120-185
#: depending on scanner contrast). Text rows are still safe because
#: text strokes occupy only 15-30 % of a row, far below the ratio gate.
_DARK_PIXEL_THRESH: Final[int] = 200

#: Fraction of a row/column that must be dark before we call it a
#: border line. 0.65 with threshold 200 separates ruled lines
#: (uniformly dark, ratio 0.85+) from text rows (sparse strokes,
#: typically 0.20-0.40).
_BORDER_DARK_RATIO: Final[float] = 0.65

#: We consider stripping any dark line found in the outer 25 % of each
#: edge — not just consecutive lines from the very edge. Many scanned
#: documents have 2-3 px of whitespace margin between the cell bbox
#: edge and the ruled line; the old "consecutive-from-edge" algorithm
#: missed those entirely.
_BORDER_SCAN_FRACTION: Final[float] = 0.25
_BORDER_SCAN_MAX_PX:   Final[int]   = 10

#: After detecting and stripping borders, crop 4 more pixels off each
#: edge as defense-in-depth. Vietnamese government tables consistently
#: have ≥5 px gap between text and ruled lines, so this never eats
#: content; it does mop up borders the threshold-based scan missed
#: (e.g. col-64 anti-aliased strokes with mean 181 that sit just above
#: the threshold).
_POST_STRIP_INSET_PX: Final[int] = 4

#: Cells shorter than this get upscaled with INTER_CUBIC before OCR.
_MIN_OCR_HEIGHT: Final[int] = 32

#: Pixels below this 0–255 value count as "ink" when deciding whether a
#: cell is empty. A printed character has 20-50% ink ratio at this
#: threshold; an empty cell with a faint scan artifact stays well below
#: 2%.
_INK_PIXEL_THRESH: Final[int] = 150

#: Cells with less ink than this fraction are considered empty — the
#: recognizer cannot output "nothing", so it hallucinates Latin glyphs
#: when given a blank canvas. Skip them.
_EMPTY_CELL_INK_RATIO: Final[float] = 0.02


def prepare_cell_for_ocr(
    crop: Image.Image,
    *,
    skip_empty_check: bool = False,
) -> Image.Image | None:
    """
    Return a cleaned PIL image suitable for ``QAIRTEasyOCREngine.recognize_region``,
    or ``None`` if the cell is judged empty (skip OCR entirely).

    Steps:

    1. Convert to grayscale (numpy uint8).
    2. Strip outer border lines via :func:`_strip_outer_borders`.
    3. Compute ink ratio on the inner content. If it's below
       :data:`_EMPTY_CELL_INK_RATIO`, return ``None`` — the recognizer
       cannot output "nothing" and will hallucinate Latin glyphs on a
       blank canvas.
    4. If the result is shorter than ``_MIN_OCR_HEIGHT``, upscale with
       INTER_CUBIC keeping aspect ratio.
    5. Add a small white margin so the recognizer's CRAFT-style padding
       has a clean background to read.

    ``skip_empty_check``: bypass step 3. Used by the span-anchor retry
    pass in ``ocr_cell_grid`` because tall vertically-merged anchor
    cells contain one centred line of text drowned in many rows of
    whitespace — ink ratio sits below the 2 % threshold and the cell
    gets falsely rejected. CRAFT detection in the recognizer still
    handles the "find the line" problem correctly when we let it.

    Failure-safe: any exception falls back to the original crop so the
    caller can't be broken by this module.
    """
    try:
        arr = np.array(crop.convert("L"))
    except Exception as e:
        _log.debug("prepare_cell_for_ocr: convert failed: %s", e)
        return crop

    if arr.size == 0 or arr.ndim != 2:
        return crop
    if arr.shape[0] < 4 or arr.shape[1] < 4:
        return crop

    stripped = _strip_outer_borders(arr)
    if stripped is None or stripped.shape[0] < 4 or stripped.shape[1] < 4:
        # Stripping ate everything — fall back to the raw crop.
        stripped = arr

    # Empty-cell short-circuit. Done AFTER border strip so the borders
    # themselves don't count as ink. Bypassed for multi-row/col anchors
    # where one line of text drowns in many rows of whitespace and the
    # ink ratio would falsely flag the cell as empty.
    if not skip_empty_check:
        ink_ratio = float((stripped < _INK_PIXEL_THRESH).mean())
        if ink_ratio < _EMPTY_CELL_INK_RATIO:
            _log.debug(
                "prepare_cell_for_ocr: empty cell skipped (ink=%.3f < %.3f)",
                ink_ratio, _EMPTY_CELL_INK_RATIO,
            )
            return None

    upscaled = _upscale_if_small(stripped)

    # Add 4 px white margin so the recognizer never has text touching the
    # canvas edge after its own internal padding.
    h, w = upscaled.shape
    bordered = np.full((h + 8, w + 8), 255, dtype=np.uint8)
    bordered[4:4 + h, 4:4 + w] = upscaled

    return Image.fromarray(bordered, mode="L")


# ── Internals ────────────────────────────────────────────────────────────────

def _strip_outer_borders(gray: np.ndarray) -> np.ndarray:
    """
    Trim dark border rows/columns that live in the outer 25 % of each edge.

    Algorithm
    ---------
    1. Compute, for every row/col, whether it qualifies as a "border
       line" — meaning ``fraction(pixel < _DARK_PIXEL_THRESH) >=
       _BORDER_DARK_RATIO``.
    2. In the outer ``_BORDER_SCAN_MAX_PX`` rows/cols of each edge,
       find the **deepest** border line (not the first). Strip
       everything from the edge up to and including that line. This is
       important because scanned cells often have 2-3 px of whitespace
       margin between the bbox edge and the ruled line, so the old
       "consecutive-from-edge" approach (``_count_leading_true``)
       missed the border entirely.
    3. Apply an unconditional ``_POST_STRIP_INSET_PX`` safety inset on
       every side, as long as ≥8 px remain on each axis. This mops up
       borders that slipped past the threshold scan — important for
       1-px-wide anti-aliased strokes that mean ~181 (just above the
       200 threshold).
    """
    h, w = gray.shape
    if h < 8 or w < 8:
        return gray

    is_dark_row = (gray < _DARK_PIXEL_THRESH).mean(axis=1) >= _BORDER_DARK_RATIO
    is_dark_col = (gray < _DARK_PIXEL_THRESH).mean(axis=0) >= _BORDER_DARK_RATIO

    max_scan_y = min(_BORDER_SCAN_MAX_PX, int(h * _BORDER_SCAN_FRACTION))
    max_scan_x = min(_BORDER_SCAN_MAX_PX, int(w * _BORDER_SCAN_FRACTION))

    top    = _find_deepest_dark_from_edge(is_dark_row, max_scan_y, from_start=True)
    bottom = _find_deepest_dark_from_edge(is_dark_row, max_scan_y, from_start=False)
    left   = _find_deepest_dark_from_edge(is_dark_col, max_scan_x, from_start=True)
    right  = _find_deepest_dark_from_edge(is_dark_col, max_scan_x, from_start=False)

    # Unconditional safety inset on each side, applied only if the
    # remaining region after inset is still meaningful (≥ 8 px).
    inset = _POST_STRIP_INSET_PX
    if (h - top - bottom - 2 * inset) >= 8 and (w - left - right - 2 * inset) >= 8:
        top    += inset
        bottom += inset
        left   += inset
        right  += inset

    if top + bottom + left + right == 0:
        return gray  # nothing to do

    h_end = h - bottom if bottom else h
    w_end = w - right  if right  else w
    return gray[top:h_end, left:w_end]


def _find_deepest_dark_from_edge(
    is_dark_arr,
    max_scan: int,
    *,
    from_start: bool,
) -> int:
    """
    Look in the outer ``max_scan`` items of ``is_dark_arr`` for the
    deepest position flagged as "dark line". Return the number of
    items to strip from that edge.

    ``from_start=True``  scans ``is_dark_arr[:max_scan]`` from index 0
    outward; the returned count strips ``is_dark_arr[0:count]``.

    ``from_start=False`` scans ``is_dark_arr[-max_scan:]`` from the
    end inward; the returned count strips the last ``count`` items.

    If no dark line is found in the scan range, returns 0.
    """
    if max_scan <= 0:
        return 0
    last_dark_offset = -1
    if from_start:
        window = is_dark_arr[:max_scan]
        for i, v in enumerate(window):
            if v:
                last_dark_offset = i
        return last_dark_offset + 1 if last_dark_offset >= 0 else 0
    else:
        # Reverse so [0] is at the edge.
        window = is_dark_arr[-max_scan:][::-1]
        for i, v in enumerate(window):
            if v:
                last_dark_offset = i
        return last_dark_offset + 1 if last_dark_offset >= 0 else 0


def _upscale_if_small(gray: np.ndarray) -> np.ndarray:
    """
    Bring crops up to ``_MIN_OCR_HEIGHT`` with INTER_CUBIC.

    INTER_CUBIC preserves stroke sharpness much better than the default
    INTER_LINEAR used by the recognizer's own resize. Critical for the
    tonal diacritics on Vietnamese vowels.
    """
    h, w = gray.shape
    if h >= _MIN_OCR_HEIGHT:
        return gray
    scale = _MIN_OCR_HEIGHT / float(h)
    new_w = max(1, int(round(w * scale)))
    return cv2.resize(gray, (new_w, _MIN_OCR_HEIGHT),
                      interpolation=cv2.INTER_CUBIC)


__all__ = ["prepare_cell_for_ocr"]
