"""
core/easyocr_cpu_engine.py
==========================

Pure CPU EasyOCR backend.

Distinct from ``easyocr_qairt_engine`` which calls Qualcomm's NPU-
compiled EasyOCR build via ``qnn-net-run.exe``. This module uses the
upstream ``easyocr`` Python package — works on any machine, no QAIRT
SDK required.

When to pick this engine
------------------------
* You're on a machine without the Snapdragon NPU (laptop, CI, server).
* You want a single dependency that handles both detection *and*
  recognition without the hybrid-CRAFT-then-VietOCR plumbing.
* You need ``recognize_with_boxes`` — per-line bboxes — because the
  pipeline falls back to it when img2table can't reconstruct a table.

When NOT to pick it
-------------------
* You have working QAIRT: ``easyocr_qairt_engine`` is ~3× faster.
* You need top Vietnamese diacritic accuracy: ``vietocr`` hybrid wins.

Interface
---------
Matches the engine contract used by ``BlockOCRRunner`` and
``ocr_cell_grid``:

    recognize_region(crop, *, line_separator=" ") -> (text, conf)
    recognize_with_boxes(crop)                    -> list[OCRLine]

``OCRLine.polygon`` is populated with the 4-corner CRAFT box so the
image-table reconstructor can cluster lines into rows / columns.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import Any, Iterable

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# ── Tunables ─────────────────────────────────────────────────────────────────

_MIN_LINE_CONFIDENCE = 0.25   # EasyOCR returns 0..1
_MIN_PRED_CHARS      = 1


# ── Lightweight text cleanup (mirrors the other engines) ────────────────────

_CONFUSABLE = {
    "“": '"', "”": '"',
    "‘": "'", "’": "'",
}


def _restore_special_chars(text: str) -> str:
    text = re.sub(r"(\d)\s*\?\s*C\s*\?\s*(\d)", r"\1°C - \2", text)
    text = re.sub(r"(\d)\s*\?\s*C", r"\1°C", text)
    text = re.sub(r"(%)\s*\?\s*(\d)", r"\1 - \2", text)
    text = re.sub(r"(\d)\s+\?\s+(\d)", r"\1 - \2", text)
    return text


def _clean_text(text: str) -> str:
    if not text:
        return text
    text = unicodedata.normalize("NFC", text)
    text = "".join(_CONFUSABLE.get(ch, ch) for ch in text)
    text = _restore_special_chars(text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


# ── Public engine ────────────────────────────────────────────────────────────

class EasyOCRCPUEngine:
    """
    EasyOCR running on CPU. Lazily instantiates the reader on first
    use because the model load is expensive (~1.5 s + 70 MB to download
    on first ever run).
    """

    def __init__(
        self,
        workspace_dir:  str | None       = None,   # ignored (compat)
        qairt_sdk_path: str | None       = None,   # ignored (compat)
        model_dir:      str | None       = None,   # ignored (compat)
        lang:           str              = "vi",
        easyocr_langs:  Iterable[str] | None = None,
        download_enabled: bool           = True,
        gpu:            bool             = False,
    ):
        # Normalise language list. The pipeline calls us with lang="vi"
        # or "en". EasyOCR uses ISO-639-1 codes too; "vi", "en" both work.
        # We always include English so embedded Latin (model numbers,
        # IEEE specs) gets recognised reliably.
        if easyocr_langs is not None:
            langs = list(easyocr_langs)
        else:
            primary = (lang or "vi").lower()
            primary = {"vie": "vi", "eng": "en"}.get(primary, primary)
            langs = [primary] if primary == "en" else [primary, "en"]
        self._langs            = langs
        self._download_enabled = bool(download_enabled)
        self._gpu              = bool(gpu)
        self._reader           = None     # lazy

        logger.info(
            f"EasyOCRCPUEngine | langs={self._langs} "
            f"gpu={self._gpu} download={self._download_enabled}"
        )

    # ────────────────────────────────────────────────────────────────────────
    # Lazy reader build
    # ────────────────────────────────────────────────────────────────────────

    def _get_reader(self):
        if self._reader is not None:
            return self._reader
        try:
            import easyocr
        except ImportError as e:
            raise RuntimeError(
                "easyocr is not installed. Run: `pip install easyocr`. "
                "First call downloads the CRAFT detector + recognizer "
                "models (~70 MB) to ~/.EasyOCR/ ."
            ) from e
        logger.info(
            "EasyOCRCPUEngine: loading Reader "
            f"(langs={self._langs}, gpu={self._gpu})"
        )
        self._reader = easyocr.Reader(
            self._langs,
            gpu                = self._gpu,
            download_enabled   = self._download_enabled,
            verbose            = False,
        )
        logger.info("EasyOCRCPUEngine: ready")
        return self._reader

    # ────────────────────────────────────────────────────────────────────────
    # Public API — recognize_region (engine contract)
    # ────────────────────────────────────────────────────────────────────────

    def recognize_region(
        self,
        crop_image,
        *,
        line_separator: str = " ",
    ) -> tuple[str, float]:
        """
        OCR a pre-cropped region and return ``(joined_text, mean_conf)``.

        EasyOCR's ``readtext`` already produces per-line text with
        bboxes — we group them top-to-bottom (then left-to-right within
        a row) and join with the requested separator.
        """
        lines = self.recognize_with_boxes(crop_image)
        if not lines:
            return "", 0.0
        # Group lines into rows by Y-midpoint clusters, then within
        # a row sort left-to-right.
        rows  = _group_lines_by_row(lines)
        texts = [
            " ".join(ln.text for ln in row) for row in rows
        ]
        confs = [ln.confidence for ln in lines if ln.confidence > 0]
        full  = line_separator.join(t for t in texts if t)
        mean  = float(np.mean(confs)) if confs else 0.0
        logger.debug(
            f"  recognize_region: '{full[:80]}' "
            f"(lines={len(lines)}, rows={len(rows)}, conf={mean:.3f})"
        )
        return full, mean

    # ────────────────────────────────────────────────────────────────────────
    # Public API — recognize_with_boxes (for image-table reconstruction)
    # ────────────────────────────────────────────────────────────────────────

    def recognize_with_boxes(self, crop_image):
        """
        Run EasyOCR detection + recognition on ``crop_image`` and return
        a list of :class:`OCRLine` with the per-line polygon attached.

        Used by ``image_table_reconstructor`` to build a row/column
        grid from raw OCR line positions when img2table can't infer
        table structure from borders.
        """
        # Lazy import so the IR types are only required when this code
        # path actually runs.
        from document_pipeline import OCRLine

        pil = _to_rgb_pil(crop_image)
        if pil is None:
            return []
        w, h = pil.size
        if h < 4 or w < 4:
            return []

        reader = self._get_reader()
        try:
            results = reader.readtext(
                np.array(pil),
                detail        = 1,
                paragraph     = False,   # we want per-line, not merged
            )
        except Exception as e:
            logger.warning(f"EasyOCR readtext failed: {e}")
            return []

        out: list = []
        for box, text, conf in results:
            cleaned = _clean_text(text)
            if not cleaned or len(cleaned) < _MIN_PRED_CHARS:
                continue
            if conf is not None and float(conf) < _MIN_LINE_CONFIDENCE:
                continue
            # EasyOCR returns a 4-point polygon (top-left, top-right,
            # bottom-right, bottom-left).
            poly = [(float(p[0]), float(p[1])) for p in box]
            out.append(OCRLine(
                text       = cleaned,
                confidence = float(conf or 0.0),
                polygon    = poly,
            ))
        return out


# ── Helpers ──────────────────────────────────────────────────────────────────

def _to_rgb_pil(crop) -> Image.Image | None:
    if crop is None:
        return None
    if isinstance(crop, Image.Image):
        return crop.convert("RGB") if crop.mode != "RGB" else crop
    if isinstance(crop, np.ndarray):
        arr = crop
        if arr.size == 0:
            return None
        if arr.ndim == 2:
            import cv2
            arr = cv2.cvtColor(arr.astype(np.uint8, copy=False),
                               cv2.COLOR_GRAY2RGB)
        elif arr.shape[2] == 4:
            import cv2
            arr = cv2.cvtColor(arr.astype(np.uint8, copy=False),
                               cv2.COLOR_RGBA2RGB)
        return Image.fromarray(arr.astype(np.uint8, copy=False), mode="RGB")
    raise TypeError(f"Unsupported crop type: {type(crop)!r}")


def _line_y_mid(line) -> float:
    """Y-midpoint of a line's polygon. ``line.polygon`` is a list of
    ``(x, y)`` tuples (typically 4 corners)."""
    if not line.polygon:
        return 0.0
    ys = [p[1] for p in line.polygon]
    return (min(ys) + max(ys)) / 2.0


def _line_x_left(line) -> float:
    if not line.polygon:
        return 0.0
    return min(p[0] for p in line.polygon)


def _group_lines_by_row(lines, *, row_overlap_thresh: float = 0.5):
    """
    Cluster a flat list of OCRLines into rows by vertical overlap. Each
    returned row is itself sorted left-to-right.

    A line joins an existing row when its y-extent overlaps the row's
    y-extent by ≥ ``row_overlap_thresh`` of the smaller height.
    """
    if not lines:
        return []
    # Sort top-to-bottom first so we can incrementally build rows.
    sorted_lines = sorted(lines, key=_line_y_mid)

    rows: list[list] = []
    row_ranges: list[tuple[float, float]] = []
    for ln in sorted_lines:
        ys = [p[1] for p in ln.polygon] if ln.polygon else [0, 0]
        y0, y1 = min(ys), max(ys)
        h = max(1e-3, y1 - y0)
        placed = False
        for i in range(len(rows) - 1, -1, -1):
            ry0, ry1 = row_ranges[i]
            ov_lo, ov_hi = max(y0, ry0), min(y1, ry1)
            overlap = max(0.0, ov_hi - ov_lo)
            if overlap / min(h, ry1 - ry0) >= row_overlap_thresh:
                rows[i].append(ln)
                row_ranges[i] = (min(ry0, y0), max(ry1, y1))
                placed = True
                break
        if not placed:
            rows.append([ln])
            row_ranges.append((y0, y1))

    # Sort each row left-to-right.
    for row in rows:
        row.sort(key=_line_x_left)
    # Sort rows top-to-bottom by the average y of the row.
    rows.sort(key=lambda r: np.mean([_line_y_mid(l) for l in r]))
    return rows


__all__ = ["EasyOCRCPUEngine"]
