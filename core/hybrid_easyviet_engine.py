"""
core/hybrid_easyviet_engine.py
==============================

Hybrid OCR engine: **EasyOCR CRAFT detector on NPU**  +
**VietOCR recognizer on CPU**.

Why this composition
--------------------
Detector and recognizer have very different hardware affinities.

  • CRAFT detector
        - Fixed input shape (608×800), pure CNN, dense heatmap output.
        - Quantises cleanly to INT8 → big speedup on the Snapdragon NPU
          with no accuracy loss.

  • VietOCR recognizer
        - Dynamic input width (one tensor per line, arbitrary length).
        - Sequence model (Transformer / Seq2Seq) trained on ~10M
          Vietnamese lines — handles stacked tones (ế, ặ, ợ, ữ, …)
          correctly out of the box.
        - INT8 quantisation mangles the decoder, so it stays on CPU
          FP32 where the accuracy is preserved.

Flow per layout region (called by ``BlockOCRRunner`` in the pipeline):

    YOLO region crop  ──► [EasyOCR CRAFT on NPU]  ──► line bboxes
                                                       │
                                                       ▼
                          ┌───────────── for each line ────────────┐
                          │ warp → line image → VietOCR (CPU FP32) │
                          └────────────────────────────────────────┘
                                                       │
                                                       ▼
                                          ``(joined_text, mean_conf)``

This matches the ``recognize_region(crop) -> (text, conf)`` contract that
``document_pipeline.py``, ``block_ocr.py``, and the img2table table
reconstructor already consume — no caller changes needed.

Fallback
--------
If CRAFT finds zero text boxes in a crop (very small captions, noisy
backgrounds), we fall back to VietOCR's internal horizontal-projection
line splitter so the region still gets read instead of returning empty.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from PIL import Image

from easyocr_qairt_engine import (
    QAIRTEasyOCREngine,
    four_point_transform,
)
from vietocr_engine import VietOCREngine

logger = logging.getLogger(__name__)


# Horizontal padding (in pixels) added around each CRAFT line crop before
# handing it to VietOCR. CRAFT boxes are sometimes tight on ascenders /
# diacritics; a small pad gives the recognizer breathing room.
_LINE_PAD_H_PX = 4
_LINE_PAD_V_PX = 6


class HybridEasyVietEngine:
    """
    Drop-in replacement for ``QAIRTEasyOCREngine`` / ``VietOCREngine``.

    Constructor signature matches the other engines so the factory in
    ``document_pipeline.py`` can swap by name without re-plumbing.
    Unused kwargs (kept for compat) are accepted and ignored.
    """

    def __init__(
        self,
        workspace_dir:   str,
        qairt_sdk_path:  str,
        model_dir:       str,
        lang:            str               = "vi",
        vietocr_weights: str | None        = None,
        vietocr_variant: str               = "vgg_transformer",
        vietocr_beam:    bool              = False,
    ):
        self.lang     = lang
        self._easy    = QAIRTEasyOCREngine(
            workspace_dir  = workspace_dir,
            qairt_sdk_path = qairt_sdk_path,
            model_dir      = model_dir,
            lang           = lang,
        )
        self._viet    = VietOCREngine(
            lang          = lang,
            weights_path  = vietocr_weights,
            model_variant = vietocr_variant,
            device        = "cpu",
            beam_search   = vietocr_beam,
        )
        logger.info(
            "HybridEasyVietEngine | CRAFT detector on NPU, "
            f"VietOCR recognizer on CPU ({vietocr_variant})"
        )

    # ────────────────────────────────────────────────────────────────────────
    # Public API — matches the other engines' contract
    # ────────────────────────────────────────────────────────────────────────

    def recognize_region(
        self,
        crop_image,
        *,
        line_separator: str = " ",
    ) -> tuple[str, float]:
        """
        Detect text lines on NPU, recognise each on CPU, return joined text.

        Args:
            crop_image: PIL Image of the pre-cropped YOLO region.
            line_separator: ``" "`` for body text, ``"\\n"`` for table cells
                so the DOCX / Markdown writers preserve cell line breaks.

        Returns:
            (text, mean_confidence)
        """
        pil = _to_rgb_pil(crop_image)
        if pil is None:
            return "", 0.0

        w, h = pil.size
        if h < 4 or w < 4:
            return "", 0.0

        try:
            boxes, _ = self._easy.detect_lines(pil, merge_into_lines=True)
        except Exception as e:
            logger.warning(f"CRAFT detect failed, falling back to VietOCR only: {e}")
            return self._viet.recognize_region(
                pil, line_separator=line_separator
            )

        # CRAFT found nothing — short caption, blank crop, noisy bg, etc.
        # Fall back to VietOCR's internal projection splitter so we still
        # try to read the region instead of dropping it silently.
        if not boxes:
            logger.debug("  CRAFT: 0 lines → VietOCR projection fallback")
            return self._viet.recognize_region(
                pil, line_separator=line_separator
            )

        # Sort boxes top-to-bottom (CRAFT returns them in heatmap-scan order,
        # which is *usually* but not always reading order on tight crops).
        boxes_sorted = sorted(boxes, key=lambda b: float(b[:, 1].mean()))

        rgb = np.array(pil, dtype=np.uint8)
        oh, ow = rgb.shape[:2]

        texts: list[str] = []
        confs: list[float] = []
        for box in boxes_sorted:
            # Expand the box slightly so VietOCR sees full ascenders/diacritics.
            box_c = box.copy()
            box_c[0] += [-_LINE_PAD_H_PX, -_LINE_PAD_V_PX]
            box_c[1] += [ _LINE_PAD_H_PX, -_LINE_PAD_V_PX]
            box_c[2] += [ _LINE_PAD_H_PX,  _LINE_PAD_V_PX]
            box_c[3] += [-_LINE_PAD_H_PX,  _LINE_PAD_V_PX]
            box_c[:, 0] = np.clip(box_c[:, 0], 0, ow - 1)
            box_c[:, 1] = np.clip(box_c[:, 1], 0, oh - 1)

            line_arr = four_point_transform(rgb, box_c)
            if line_arr.size == 0 or line_arr.shape[0] < 4 or line_arr.shape[1] < 4:
                continue

            line_pil = Image.fromarray(line_arr.astype(np.uint8, copy=False),
                                       mode="RGB")
            # Each line crop is already single-line, so VietOCR's internal
            # splitter passes it through. The ink-ratio and confidence
            # guards inside VietOCREngine still apply.
            text, conf = self._viet.recognize_region(
                line_pil, line_separator=line_separator,
            )
            if text:
                texts.append(text)
                confs.append(conf)

        full_text = line_separator.join(texts)
        mean_conf = float(np.mean(confs)) if confs else 0.0
        logger.debug(
            f"  recognize_region: '{full_text[:80]}' "
            f"(lines={len(texts)}, conf={mean_conf:.3f})"
        )
        return full_text, mean_conf

    # ────────────────────────────────────────────────────────────────────────
    # recognize_with_boxes — for image_table_reconstructor
    # ────────────────────────────────────────────────────────────────────────

    def recognize_with_boxes(self, crop_image):
        """
        Run CRAFT detection + VietOCR per-line recognition and return a
        flat list of ``OCRLine`` with each line's 4-point polygon
        populated. Used by the image-table fallback to build a grid
        from raw line positions when img2table fails.
        """
        from document_pipeline import OCRLine

        pil = _to_rgb_pil(crop_image)
        if pil is None:
            return []
        try:
            boxes, _ = self._easy.detect_lines(pil, merge_into_lines=True)
        except Exception as e:
            logger.warning(f"recognize_with_boxes: CRAFT detect failed: {e}")
            return []
        if not boxes:
            return []

        rgb = np.array(pil, dtype=np.uint8)
        oh, ow = rgb.shape[:2]

        out: list = []
        for box in boxes:
            box_c = box.copy()
            box_c[0] += [-_LINE_PAD_H_PX, -_LINE_PAD_V_PX]
            box_c[1] += [ _LINE_PAD_H_PX, -_LINE_PAD_V_PX]
            box_c[2] += [ _LINE_PAD_H_PX,  _LINE_PAD_V_PX]
            box_c[3] += [-_LINE_PAD_H_PX,  _LINE_PAD_V_PX]
            box_c[:, 0] = np.clip(box_c[:, 0], 0, ow - 1)
            box_c[:, 1] = np.clip(box_c[:, 1], 0, oh - 1)

            line_arr = four_point_transform(rgb, box_c)
            if line_arr.size == 0 or line_arr.shape[0] < 4 or line_arr.shape[1] < 4:
                continue
            line_pil = Image.fromarray(
                line_arr.astype(np.uint8, copy=False), mode="RGB"
            )
            text, conf = self._viet.recognize_region(line_pil)
            text = (text or "").strip()
            if not text:
                continue
            poly = [(float(p[0]), float(p[1])) for p in box]
            out.append(OCRLine(
                text       = text,
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


__all__ = ["HybridEasyVietEngine"]
