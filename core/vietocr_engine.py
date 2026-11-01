"""
core/vietocr_engine.py
======================

VietOCR (pbcquoc/vietocr) recognizer — Vietnamese-tuned transformer / seq2seq
OCR that handles stacked diacritics (ế, ặ, ợ, ữ, ...) correctly.

Why this engine
---------------
The Qualcomm-compiled EasyOCR recognizer is trained on a generic Latin
corpus and loses accuracy on Vietnamese stacked tones, especially after
INT8 quantisation for the NPU. VietOCR is trained on ~10M Vietnamese text
lines (printed + scene + scan) and reaches >95% line accuracy on
Vietnamese documents. It runs on CPU FP32 so quantisation never enters
the picture.

Architecture in this project
----------------------------
* Detector → unchanged (YOLO layout + your existing CRAFT/PP-OCRv5 detector
  on NPU). VietOCR only does *recognition*.
* Recognizer → VietOCR Transformer / Seq2Seq on **CPU FP32**. The model
  is dynamic-width, so long lines are not squashed to 320 px.

Interface contract
------------------
Matches ``QAIRTEasyOCREngine.recognize_region`` exactly:

    recognize_region(crop_image: PIL.Image | np.ndarray) -> tuple[str, float]

so it is a drop-in replacement. In the production pipeline this engine is
wrapped by ``HybridEasyVietEngine`` which runs the EasyOCR CRAFT detector
on NPU and delegates per-line recognition here.

Install
-------
    pip install vietocr

Weights (~95 MB) auto-download to ``~/.cache/vietocr/`` on first call. To
pin a local copy, pass ``weights_path=...`` to the constructor.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# ── Tunables (tightened to suppress phantom predictions on sliver crops) ────

_LINE_MIN_HEIGHT_PX     = 14      # at 150 DPI, real lines are ≥18 px; <14 = noise
_LINE_PROJECTION_THRESH = 0.18    # row-sum ratio that counts as "text" (was 0.08
                                  # — caused faint shadow rows to register as
                                  # standalone lines, VietOCR then hallucinates)
_INK_PIXEL_THRESH       = 180     # < this gray value = ink
_MAX_SINGLE_LINE_HEIGHT = 80      # taller crops get line-split

# Quality filters applied AFTER VietOCR predicts on each line strip
_MIN_LINE_CONFIDENCE    = 0.40    # drop "I'm guessing" predictions (VietOCR
                                  # returns Bayesian probs; <0.4 is junk)
_MIN_LINE_INK_RATIO     = 0.015   # require ≥1.5% ink pixels in the strip
                                  # before even calling VietOCR
_MIN_PRED_CHARS         = 1       # was 2; raised back to 1 because legitimate
                                  # table cells contain single digits ("1",
                                  # "2", ..., "9") and we were dropping them.
                                  # The confidence + ink-ratio guards above
                                  # are sufficient to suppress real hallucinations.


# ── Public engine ────────────────────────────────────────────────────────────

class VietOCREngine:
    """
    Vietnamese-tuned recognizer, CPU FP32, dynamic input width.

    Constructor signature accepts the same kwargs as the other engines so the
    factory in ``document_pipeline.py`` can swap by name without re-plumbing.
    Unused kwargs are accepted and ignored (workspace_dir, qairt_sdk_path).
    """

    def __init__(
        self,
        workspace_dir:  str | None = None,   # ignored (compat)
        qairt_sdk_path: str | None = None,   # ignored (compat)
        model_dir:      str | None = None,   # ignored (compat)
        lang:           str        = "vi",
        weights_path:   str | None = None,
        model_variant:  str        = "vgg_transformer",
        device:         str        = "cpu",
        beam_search:    bool       = False,
    ):
        # ``lang`` is informational — VietOCR is Vietnamese-only by design.
        # We accept ``en`` so a mixed corpus pipeline can opt-in without
        # raising, but quality on pure English will be lower than EasyOCR.
        if lang not in ("vi", "en"):
            logger.warning(
                f"VietOCREngine: lang={lang!r} not validated — "
                "model is trained on Vietnamese."
            )

        if model_variant not in ("vgg_transformer", "vgg_seq2seq"):
            raise ValueError(
                f"model_variant must be 'vgg_transformer' or 'vgg_seq2seq', "
                f"got {model_variant!r}"
            )

        self._model_variant = model_variant
        self._device        = device
        self._beam_search   = beam_search
        self._weights_path  = Path(weights_path) if weights_path else None

        # Lazy: don't import vietocr / torch until first use. Keeps startup
        # fast when the user runs --ocr-engine easyocr instead.
        self._predictor: Any = None

        logger.info(
            f"VietOCREngine | variant={model_variant} device={device} "
            f"beam_search={beam_search} weights={self._weights_path}"
        )

    # ────────────────────────────────────────────────────────────────────────
    # Public API
    # ────────────────────────────────────────────────────────────────────────

    def recognize_region(
        self,
        crop_image,
        *,
        line_separator: str = " ",
    ) -> tuple[str, float]:
        """
        Recognise text from a pre-cropped layout region (single or multi-line).

        Steps:
          1. Coerce input to RGB PIL Image.
          2. If the crop is taller than one line, split into line-strips via
             horizontal-projection profile.
          3. Run VietOCR ``predict(..., return_prob=True)`` on each line.
          4. Concatenate with ``line_separator``; mean the per-line probs.

        Args:
            crop_image: PIL.Image or ndarray (HWC, RGB by project convention).
            line_separator: ``" "`` for body text, ``"\\n"`` for table cells
                so the downstream DOCX / Markdown writers preserve cell
                line breaks.

        Returns:
            (text, confidence) — confidence is the mean per-line probability
            returned by VietOCR.
        """
        pil = _to_rgb_pil(crop_image)
        if pil is None:
            return "", 0.0

        w, h = pil.size
        if h < 4 or w < 4:
            return "", 0.0

        line_crops = _split_into_line_crops(pil)
        if not line_crops:
            line_crops = [pil]

        predictor = self._get_predictor()

        texts: list[str] = []
        confs: list[float] = []
        for line in line_crops:
            # Ink-ratio guard: VietOCR is a generative model and will return
            # plausible-looking Vietnamese for a near-blank strip, which is
            # the root cause of "phantom" tokens like "Thuận"/"Thuật" appearing
            # between real lines. Skip strips that are mostly background.
            if not _has_enough_ink(line):
                continue
            try:
                result = predictor.predict(line, return_prob=True)
            except Exception as e:
                logger.warning(f"VietOCR predict failed on a line: {e}")
                continue

            # ── Defensive return-shape normalisation ───────────────────────
            # VietOCR's contract is "(text, prob) when return_prob=True", but
            # in practice beam-search mode (beamsearch=True in the config)
            # returns ``prob=None`` because the beam path skips the per-token
            # softmax accumulation. A few older versions return the plain
            # string. Coerce both to a uniform (text, prob_or_None).
            if isinstance(result, tuple):
                text = result[0] if len(result) > 0 else ""
                prob = result[1] if len(result) > 1 else None
            else:
                text, prob = result, None
            prob = float(prob) if prob is not None else None

            text = _clean_text(text)
            if not text or len(text) < _MIN_PRED_CHARS:
                continue

            # Confidence gate only applies when we actually got a probability.
            # With beam search prob is None and we trust the recognizer
            # output as-is (the ink-ratio guard above already filters the
            # near-blank inputs that would tempt hallucination).
            if prob is not None and prob < _MIN_LINE_CONFIDENCE:
                logger.debug(
                    f"  drop low-conf line: '{text[:40]}' (p={prob:.2f})"
                )
                continue
            texts.append(text)
            confs.append(prob if prob is not None else 1.0)

        full_text = line_separator.join(texts)
        mean_conf = float(np.mean(confs)) if confs else 0.0
        logger.debug(
            f"  recognize_region: '{full_text[:80]}' (conf={mean_conf:.3f})"
        )
        return full_text, mean_conf

    # ────────────────────────────────────────────────────────────────────────
    # Internal — lazy model load
    # ────────────────────────────────────────────────────────────────────────

    def _get_predictor(self):
        if self._predictor is not None:
            return self._predictor

        try:
            from vietocr.tool.predictor import Predictor
            from vietocr.tool.config import Cfg
        except ImportError as e:
            raise RuntimeError(
                "VietOCREngine requires the `vietocr` package. Install with:\n"
                "    pip install vietocr\n"
                "On ARM64 Windows you may also need a CPU PyTorch wheel; see "
                "https://pytorch.org/get-started/locally/ ."
            ) from e

        cfg = Cfg.load_config_from_name(self._model_variant)
        if self._weights_path is not None:
            if not self._weights_path.exists():
                raise FileNotFoundError(
                    f"VietOCR weights not found: {self._weights_path}"
                )
            cfg["weights"] = str(self._weights_path)
            cfg["pretrain"] = None  # disable HTTP fetch when weights are local
        cfg["device"] = self._device
        cfg["predictor"]["beamsearch"] = self._beam_search

        logger.info(
            f"VietOCR: loading predictor (this downloads "
            f"~95 MB on first run if no local weights)"
        )
        self._predictor = Predictor(cfg)
        logger.info("VietOCR: predictor ready")
        return self._predictor


# ── Helpers ──────────────────────────────────────────────────────────────────

def _to_rgb_pil(crop) -> Image.Image | None:
    """Accept PIL.Image or ndarray (HWC, RGB by convention); return RGB PIL."""
    if crop is None:
        return None
    if isinstance(crop, Image.Image):
        return crop.convert("RGB") if crop.mode != "RGB" else crop
    if isinstance(crop, np.ndarray):
        arr = crop
        if arr.size == 0:
            return None
        if arr.ndim == 2:
            arr = cv2.cvtColor(arr.astype(np.uint8, copy=False),
                               cv2.COLOR_GRAY2RGB)
        elif arr.shape[2] == 4:
            arr = cv2.cvtColor(arr.astype(np.uint8, copy=False),
                               cv2.COLOR_RGBA2RGB)
        return Image.fromarray(arr.astype(np.uint8, copy=False), mode="RGB")
    raise TypeError(f"Unsupported crop type: {type(crop)!r}")


def _has_enough_ink(pil: Image.Image) -> bool:
    """
    Return False if the strip is mostly background — VietOCR will happily
    hallucinate text on near-blank crops, so we guard the call site.
    """
    gray = np.array(pil.convert("L"), dtype=np.uint8)
    if gray.size == 0:
        return False
    ratio = float((gray < _INK_PIXEL_THRESH).sum()) / float(gray.size)
    return ratio >= _MIN_LINE_INK_RATIO


def _split_into_line_crops(pil: Image.Image) -> list[Image.Image]:
    """
    Split a multi-line region into single-line PIL crops via horizontal-
    projection profile. Single-line crops (height <= _MAX_SINGLE_LINE_HEIGHT)
    pass through unchanged.

    Used as a fallback when the hybrid engine's CRAFT detector finds no
    text boxes in a region (very small caption crops, noisy backgrounds).
    """
    w, h = pil.size
    if h <= _MAX_SINGLE_LINE_HEIGHT:
        return [pil]

    gray = np.array(pil.convert("L"), dtype=np.uint8)
    ink  = (gray < _INK_PIXEL_THRESH).astype(np.uint8)
    row_sum = ink.sum(axis=1).astype(np.float32)
    if row_sum.max() <= 0:
        return [pil]

    thresh = row_sum.max() * _LINE_PROJECTION_THRESH
    is_text = row_sum > thresh

    # Build non-overlapping bands. Previous version padded each band by ±2 px
    # which caused descenders to appear in both line N and line N+1's strip
    # — VietOCR then read the overlapping content twice ("ngày ngày" etc.)
    # We now emit strictly contiguous bands with no padding.
    bands: list[tuple[int, int]] = []
    in_band = False
    start = 0
    for y in range(h):
        if is_text[y] and not in_band:
            in_band = True
            start = y
        elif (not is_text[y]) and in_band:
            in_band = False
            if y - start >= _LINE_MIN_HEIGHT_PX:
                bands.append((start, y))
    if in_band and h - start >= _LINE_MIN_HEIGHT_PX:
        bands.append((start, h))

    if not bands:
        return [pil]

    return [pil.crop((0, y0, w, y1)) for (y0, y1) in bands]


_CONFUSABLE = {
    "“": '"', "”": '"',
    "‘": "'", "’": "'",
}

# VietOCR's vocab is fixed and does NOT include some characters that
# appear routinely in Vietnamese technical documents:
#   °  (U+00B0, degree sign)            — temperatures
#   –  (U+2013, en dash) / — (em dash)  — ranges, lists
#   ×  (U+00D7, multiplication sign)    — antenna specs like "2×2"
#
# The recognizer emits "?" as a fallback when it sees an unmappable glyph
# (e.g. "0°C-40°C" → "0?C-40?C"). We recover the most common patterns
# below with conservative regexes so we only rewrite when the context
# unambiguously identifies the missing glyph. Ambiguous cases (a lone "?"
# next to letters) are left alone so we don't corrupt legitimate question
# marks.
def _restore_special_chars(text: str) -> str:
    # Temperatures.  "0?C?40?C"  →  "0°C - 40°C"
    text = re.sub(r"(\d)\s*\?\s*C\s*\?\s*(\d)", r"\1°C - \2", text)
    text = re.sub(r"(\d)\s*\?\s*C", r"\1°C", text)
    # Humidity / percentage ranges.  "5%?90%"  →  "5% - 90%"
    text = re.sub(r"(%)\s*\?\s*(\d)", r"\1 - \2", text)
    # Generic numeric ranges with surrounding spaces.  "100 ? 200" → "100 - 200"
    text = re.sub(r"(\d)\s+\?\s+(\d)", r"\1 - \2", text)
    return text


def _clean_text(text: str) -> str:
    if not text:
        return text
    text = unicodedata.normalize("NFC", text)
    text = "".join(_CONFUSABLE.get(ch, ch) for ch in text)
    text = _restore_special_chars(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


__all__ = ["VietOCREngine"]


# ── Standalone smoke test ────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import time

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    img_path = sys.argv[1] if len(sys.argv) > 1 else "test_line_3.png"
    if not Path(img_path).exists():
        print(f"[ERROR] Image not found: {img_path}")
        sys.exit(1)

    engine = VietOCREngine(lang="vi", device="cpu")

    img = Image.open(img_path).convert("RGB")
    print(f"Input: {img.size[0]}×{img.size[1]} px")

    t0 = time.time()
    text, conf = engine.recognize_region(img)
    elapsed = time.time() - t0

    print(f"\n✅ recognize_region in {elapsed:.2f}s:")
    print(f"  '{text}'  (conf={conf:.3f})")
