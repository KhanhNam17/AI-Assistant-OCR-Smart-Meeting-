"""
core/tesseract_engine.py
========================

Tesseract OCR backend — a no-NPU CPU baseline that works on any machine
with the Tesseract binary + ``vie.traineddata`` installed.

Why this exists
---------------
* **Fallback** when the QAIRT SDK / NPU path isn't available.
* **Comparison baseline** against the hybrid (CRAFT-NPU + VietOCR-CPU)
  engine. When a cell or block comes out empty under one engine, flip
  ``--ocr-engine tesseract`` and re-run to localise the failure:
  - both empty  → bbox / preprocessing problem (not the recognizer)
  - only one empty → recognizer-specific failure mode

Architecture
------------
Tesseract is a **detector + recognizer in one binary**. There's no
CRAFT pass here — the whole pre-cropped layout region is handed to
Tesseract which does its own internal line detection.

Interface contract
------------------
Matches ``QAIRTEasyOCREngine.recognize_region`` and
``HybridEasyVietEngine.recognize_region`` exactly:

    recognize_region(crop: PIL.Image | np.ndarray, *,
                     line_separator: str = " ") -> tuple[str, float]

so it drops into the existing factory in ``document_pipeline.py``
behind ``--ocr-engine tesseract``.

System requirements
-------------------
1. Tesseract 5.x binary on PATH. Windows install:

       winget install UB-Mannheim.TesseractOCR

   …or the installer from
   https://github.com/UB-Mannheim/tesseract/wiki — make sure the
   "Vietnamese" language pack is checked during install.

2. Vietnamese traineddata. If the installer didn't drop it,
   manually place ``vie.traineddata`` into ``<tesseract>/tessdata/``.
   File is at
   https://github.com/tesseract-ocr/tessdata/raw/main/vie.traineddata
   (~1.2 MB).

3. Python: ``pip install pytesseract``.

Override the Tesseract executable path with the ``TESSERACT_CMD``
environment variable if the binary isn't on PATH.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import statistics
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# Standard install locations probed when ``tesseract`` isn't on PATH and
# ``TESSERACT_CMD`` isn't set. The installer from UB-Mannheim drops the
# binary in one of the first two; chocolatey / winget use either. Local-
# install variants (per-user) tend to land under %LOCALAPPDATA%.
_WINDOWS_DEFAULT_PATHS: tuple[str, ...] = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
    os.path.expandvars(r"%USERPROFILE%\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"),
    os.path.expandvars(r"%PROGRAMFILES%\Tesseract-OCR\tesseract.exe"),
)


def _discover_tesseract_binary() -> str | None:
    """Return the path to a usable tesseract.exe or None.

    Resolution order:
      1. ``TESSERACT_CMD`` env var (explicit override)
      2. ``shutil.which("tesseract")`` — anywhere on PATH
      3. Standard Windows install locations
    """
    env_override = os.environ.get("TESSERACT_CMD")
    if env_override and Path(env_override).exists():
        return env_override
    found = shutil.which("tesseract")
    if found:
        return found
    if os.name == "nt":
        for candidate in _WINDOWS_DEFAULT_PATHS:
            if candidate and Path(candidate).exists():
                logger.info(
                    f"TesseractEngine: auto-discovered binary at {candidate} "
                    "(not on PATH). Tip: add its folder to PATH or set "
                    "TESSERACT_CMD to silence this lookup."
                )
                return candidate
    return None


# ── Tunables ─────────────────────────────────────────────────────────────────

# Page Segmentation Mode passed to Tesseract:
#   3 = fully automatic page segmentation (default)
#   6 = assume a single uniform block of text — what we want for YOLO
#       region crops, which are paragraph / cell sized.
#   7 = single text line (used for very narrow table cells)
_PSM_BLOCK     = 6
_PSM_LINE      = 7
_NARROW_LINE_H = 36   # crops <= this px tall get psm 7

# Confidence-related: Tesseract returns per-token confidences in image_to_data.
# We average them per call and return the mean as the engine's overall conf.
_MIN_PRED_CHARS = 1   # match the other engines


# ── Confusable / glyph cleanup (mirrors vietocr_engine._restore_special_chars)
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

class TesseractEngine:
    """
    Tesseract-based recognizer. Drop-in replacement for the other engines.

    Constructor signature is intentionally permissive so the factory in
    ``document_pipeline.py`` can swap by name without re-plumbing. Unused
    kwargs (kept for compat with the other engines' signatures) are
    silently ignored.
    """

    def __init__(
        self,
        workspace_dir:  str | None = None,   # ignored (compat)
        qairt_sdk_path: str | None = None,   # ignored (compat)
        model_dir:      str | None = None,   # ignored (compat)
        lang:           str        = "vi",
        tesseract_lang: str | None = None,
        tesseract_cmd:  str | None = None,
    ):
        # Pin the binary path. Priority:
        #   1. tesseract_cmd kwarg (caller override)
        #   2. TESSERACT_CMD env var
        #   3. PATH lookup
        #   4. Standard Windows install locations
        cmd = tesseract_cmd or _discover_tesseract_binary()
        if cmd is None:
            raise RuntimeError(
                "Tesseract binary not found. Checked PATH and standard "
                "Windows install locations:\n"
                + "\n".join(f"  - {p}" for p in _WINDOWS_DEFAULT_PATHS)
                + "\n\nFix one of:\n"
                "  • Add the Tesseract install folder to PATH, OR\n"
                "  • Set TESSERACT_CMD in .env to the full path of "
                "tesseract.exe\n"
                "    e.g. TESSERACT_CMD=C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
            )

        try:
            import pytesseract  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "pytesseract is not installed. Run: `pip install pytesseract`."
            ) from e

        import pytesseract as _pt
        _pt.pytesseract.tesseract_cmd = cmd
        self._pt = _pt

        # Default to "vie+eng" so embedded English in Vietnamese documents
        # (model numbers, IEEE specs, etc.) doesn't trip the Vietnamese
        # language model.
        if tesseract_lang:
            self._tess_lang = tesseract_lang
        elif lang == "en":
            self._tess_lang = "eng"
        else:
            self._tess_lang = "vie+eng"

        # Verify Vietnamese traineddata exists. Fail loud at construction
        # so the user fixes it once, not per-cell.
        self._verify_traineddata()

        logger.info(
            f"TesseractEngine | lang={self._tess_lang} | "
            f"binary={cmd}"
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
        Run Tesseract on a pre-cropped region and return ``(text, conf)``.

        Multi-line regions (banners, paragraphs, table cells with stacked
        content) are handled by Tesseract's own internal line detection —
        we pass them through with ``--psm 6`` so the output preserves
        line boundaries. The caller's ``line_separator`` then joins those
        lines: ``" "`` for inline-flow paragraphs, ``"\\n"`` for cells
        where the DOCX writer should keep the breaks.
        """
        pil = _to_rgb_pil(crop_image)
        if pil is None:
            return "", 0.0
        w, h = pil.size
        if h < 4 or w < 4:
            return "", 0.0

        psm = _PSM_LINE if h <= _NARROW_LINE_H else _PSM_BLOCK
        config = f"--oem 3 --psm {psm}"

        # image_to_data gives us per-token text + confidence so we can
        # filter low-conf garbage and compute a real mean confidence.
        try:
            data = self._pt.image_to_data(
                pil,
                lang   = self._tess_lang,
                config = config,
                output_type = self._pt.Output.DICT,
            )
        except Exception as e:
            logger.warning(f"Tesseract failed on a region: {e}")
            return "", 0.0

        # Group tokens by line so we preserve Tesseract's own line-break
        # decisions; later we'll join with line_separator.
        lines_by_key: dict[tuple, list[tuple[str, float]]] = {}
        n_tokens = len(data.get("text", []))
        for i in range(n_tokens):
            txt = (data["text"][i] or "").strip()
            if not txt:
                continue
            try:
                conf = float(data["conf"][i])
            except (TypeError, ValueError):
                conf = -1.0
            if conf < 0:                   # Tesseract uses -1 for "no info"
                continue
            key = (
                data.get("block_num",  [0])[i],
                data.get("par_num",    [0])[i],
                data.get("line_num",   [0])[i],
            )
            lines_by_key.setdefault(key, []).append((txt, conf))

        if not lines_by_key:
            return "", 0.0

        # Preserve the natural reading order Tesseract emitted them in.
        ordered_keys = sorted(lines_by_key.keys())
        line_strings: list[str] = []
        all_confs:    list[float] = []
        for key in ordered_keys:
            toks = lines_by_key[key]
            line_text = " ".join(t for t, _ in toks).strip()
            line_text = _clean_text(line_text)
            if not line_text or len(line_text) < _MIN_PRED_CHARS:
                continue
            line_strings.append(line_text)
            all_confs.extend(c for _, c in toks)

        if not line_strings:
            return "", 0.0

        full_text = line_separator.join(line_strings)
        # Tesseract conf is 0-100. Normalise to 0-1 to match the other
        # engines (VietOCR/EasyOCR return Bayesian probabilities).
        mean_conf = float(statistics.mean(all_confs)) / 100.0 if all_confs else 0.0
        logger.debug(
            f"  recognize_region: '{full_text[:80]}' "
            f"(lines={len(line_strings)}, conf={mean_conf:.3f})"
        )
        return full_text, mean_conf

    # ────────────────────────────────────────────────────────────────────────
    # recognize_with_boxes — for image_table_reconstructor
    # ────────────────────────────────────────────────────────────────────────

    def recognize_with_boxes(self, crop_image):
        """
        Run Tesseract and return one ``OCRLine`` per visible text line
        with its bounding rectangle attached. Used by the image-table
        fallback when img2table can't infer table structure.
        """
        from document_pipeline import OCRLine

        pil = _to_rgb_pil(crop_image)
        if pil is None:
            return []
        w, h = pil.size
        if h < 4 or w < 4:
            return []

        psm = _PSM_LINE if h <= _NARROW_LINE_H else _PSM_BLOCK
        config = f"--oem 3 --psm {psm}"
        try:
            data = self._pt.image_to_data(
                pil, lang=self._tess_lang, config=config,
                output_type=self._pt.Output.DICT,
            )
        except Exception as e:
            logger.warning(f"Tesseract recognize_with_boxes failed: {e}")
            return []

        # Group tokens into lines; each line gets a bounding rectangle
        # computed as the union of its tokens' bboxes.
        lines_by_key: dict[tuple, dict] = {}
        n_tokens = len(data.get("text", []))
        for i in range(n_tokens):
            txt = (data["text"][i] or "").strip()
            if not txt:
                continue
            try:
                conf = float(data["conf"][i])
            except (TypeError, ValueError):
                conf = -1.0
            if conf < 0:
                continue
            key = (
                data.get("block_num", [0])[i],
                data.get("par_num",   [0])[i],
                data.get("line_num",  [0])[i],
            )
            x = int(data["left"][i]);    y = int(data["top"][i])
            ww = int(data["width"][i]);  hh = int(data["height"][i])
            entry = lines_by_key.setdefault(key, {
                "tokens": [], "confs": [], "x0": x, "y0": y,
                "x1": x + ww, "y1": y + hh,
            })
            entry["tokens"].append(txt)
            entry["confs"].append(conf)
            entry["x0"] = min(entry["x0"], x)
            entry["y0"] = min(entry["y0"], y)
            entry["x1"] = max(entry["x1"], x + ww)
            entry["y1"] = max(entry["y1"], y + hh)

        out: list = []
        for entry in lines_by_key.values():
            line_text = _clean_text(" ".join(entry["tokens"]))
            if not line_text:
                continue
            poly = [
                (float(entry["x0"]), float(entry["y0"])),
                (float(entry["x1"]), float(entry["y0"])),
                (float(entry["x1"]), float(entry["y1"])),
                (float(entry["x0"]), float(entry["y1"])),
            ]
            conf = (statistics.mean(entry["confs"]) / 100.0
                    if entry["confs"] else 0.0)
            out.append(OCRLine(
                text=line_text, confidence=float(conf), polygon=poly,
            ))
        return out

    # ────────────────────────────────────────────────────────────────────────
    # Internal
    # ────────────────────────────────────────────────────────────────────────

    def _verify_traineddata(self) -> None:
        """Fail fast at startup if vie.traineddata is missing."""
        try:
            available = set(self._pt.get_languages(config=""))
        except Exception as e:
            logger.warning(
                f"TesseractEngine: could not list languages ({e}); "
                "proceeding without verification."
            )
            return
        for needed in self._tess_lang.split("+"):
            if needed and needed not in available:
                raise RuntimeError(
                    f"Tesseract language pack '{needed}' is not installed. "
                    f"Available: {sorted(available)}. Drop "
                    f"{needed}.traineddata into your Tesseract tessdata/ "
                    "folder (download from "
                    "https://github.com/tesseract-ocr/tessdata/)."
                )


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


__all__ = ["TesseractEngine"]


# ── Standalone smoke test ────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import time

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    img_path = sys.argv[1] if len(sys.argv) > 1 else "test_line_3.png"
    from pathlib import Path
    if not Path(img_path).exists():
        print(f"[ERROR] Image not found: {img_path}")
        sys.exit(1)

    engine = TesseractEngine(lang="vi")
    img    = Image.open(img_path).convert("RGB")
    print(f"Input: {img.size[0]}×{img.size[1]} px")

    t0 = time.time()
    text, conf = engine.recognize_region(img)
    print(f"\n✅ recognize_region in {time.time() - t0:.2f}s:")
    print(f"  '{text}'  (conf={conf:.3f})")
