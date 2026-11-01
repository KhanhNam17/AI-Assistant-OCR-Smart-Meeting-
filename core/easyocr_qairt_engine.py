"""
core/easyocr_engine_qairt.py  —  v4  (Layout-centric refactor)

KEY CHANGE from v3
──────────────────
Added `recognize_region(crop_image)` — a recognition-only path that skips
CRAFT detection entirely. This is used by region_ocr.py when Docling has
already provided the bounding boxes (layout-centric pipeline).

Old flow (OCR-centric, WRONG when called from region_ocr):
  region_ocr crops a Docling block → engine.recognize(crop)
    → CRAFT runs AGAIN on the crop → sub-boxes in crop-local coords
    → combine_detections concatenates text (bbox is wasted)
    → double-stage detection with no coord benefit

New flow (layout-centric, CORRECT):
  Docling → layout blocks (semantic bboxes)
    → region_ocr crops each block
       → engine.recognize_region(crop)   ← NEW: recognition only
          → split into chunks (for long lines)
             → CTC decode
                → clean text

The full `recognize(full_image)` method is UNCHANGED and still works for
standalone use (e.g. testing a single image without Docling).

Model: qualcomm/EasyOCR (NPU-optimized, Snapdragon X Elite)

  Detector  (detector.bin):
    Input : image         (1, 608, 800, 3) NHWC float32
    Output: results       (1, 304, 400, 2) float32  ← heatmap (text + link)

  Recognizer (recognizer.bin):
    ┌─ English model (easyocr_qnn / english_g2):
    │   Output: output_preds  (1, 199, 97)     float32   vocab = 97
    └─ Vietnamese model (easyocr_qnn_vi / latin_g2):
        Output: output_preds  (1, 199, 352)    float32   vocab = 352
"""

from __future__ import annotations

import os
import math
import shutil
import subprocess
import logging
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# Model I/O constants
# ══════════════════════════════════════════════════════════════════════════════

DET_H, DET_W = 608, 800
MAP_H, MAP_W = 304, 400
REC_H        = 64
REC_W        = 800
REC_TIME     = 199

TEXT_THRESH  = 0.7
LINK_THRESH  = 0.4
LOW_TEXT     = 0.4
CANVAS_SIZE  = 2560
MAG_RATIO    = 1.0
REC_MEAN     = 0.5
REC_STD      = 0.5

# ══════════════════════════════════════════════════════════════════════════════
# Character sets
# ══════════════════════════════════════════════════════════════════════════════

_ENGLISH_G2 = (
    " !\"#$%&'()*+,-./0123456789:;<=>?@"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "[\\]^_`"
    "abcdefghijklmnopqrstuvwxyz"
    "{|}~\u20ac"
)
CHARSET_EN: list[str] = ["[blank]"] + list(_ENGLISH_G2)
assert len(CHARSET_EN) == 97, f"EN charset must be 97, got {len(CHARSET_EN)}"

_LATIN_G2 = (
    " !\"#$%&'()*+,-./0123456789:;<=>?@"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "[\\]^_`"
    "abcdefghijklmnopqrstuvwxyz"
    "{|}~"
    "\xaaÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝÞß"
    "àáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿ"
    "\u0100\u0101\u0102\u0103\u0104\u0105\u0106\u0107\u010c\u010d"
    "\u010e\u010f\u0110\u0111\u0112\u0113\u0116\u0117\u0118\u0119"
    "\u011a\u011b\u011e\u011f\u0128\u0129\u012a\u012b\u012e\u012f"
    "\u0130\u0131\u0136\u0137\u0139\u013a\u013b\u013c\u013d\u013e"
    "\u0141\u0142\u0143\u0144\u0145\u0146\u0147\u0148\u0152\u0153"
    "\u0154\u0155\u0158\u0159\u015a\u015b\u015e\u015f\u0160\u0161"
    "\u0164\u0165\u0168\u0169\u016a\u016b\u016e\u016f\u0172\u0173"
    "\u0178\u0179\u017a\u017b\u017c\u017d\u017e"
    "\u018f\u01a0\u01a1\u01af\u01b0\u0218\u0219\u021a\u021b"
    "\u0259\u0307"
    "\u1e0c\u1e0d\u1e36\u1e37\u1e40\u1e41\u1e42\u1e43\u1e44\u1e45"
    "\u1e46\u1e47\u1e6c\u1e6d"
    "\u1ea0\u1ea1\u1ea2\u1ea3\u1ea4\u1ea5\u1ea6\u1ea7\u1ea8\u1ea9"
    "\u1eaa\u1eab\u1eac\u1ead\u1eae\u1eaf\u1eb0\u1eb1\u1eb2\u1eb3"
    "\u1eb4\u1eb5\u1eb6\u1eb7\u1eb8\u1eb9\u1eba\u1ebb\u1ebc\u1ebd"
    "\u1ebe\u1ebf\u1ec0\u1ec1\u1ec2\u1ec3\u1ec4\u1ec5\u1ec6\u1ec7"
    "\u1ec8\u1ec9\u1eca\u1ecb\u1ecc\u1ecd\u1ece\u1ecf\u1ed0\u1ed1"
    "\u1ed2\u1ed3\u1ed4\u1ed5\u1ed6\u1ed7\u1ed8\u1ed9\u1eda\u1edb"
    "\u1edc\u1edd\u1ede\u1edf\u1ee0\u1ee1\u1ee2\u1ee3\u1ee4\u1ee5"
    "\u1ee6\u1ee7\u1ee8\u1ee9\u1eea\u1eeb\u1eec\u1eed\u1eee\u1eef"
    "\u1ef0\u1ef1\u1ef2\u1ef3\u1ef4\u1ef5\u1ef6\u1ef7\u1ef8\u1ef9"
    "\u20ac"
)
CHARSET_VI: list[str] = ["[blank]"] + list(_LATIN_G2)
assert len(CHARSET_VI) == 352, f"VI charset must be 352, got {len(CHARSET_VI)}"


def build_charset(lang: str) -> list[str]:
    if lang in ("en", "english"):
        return CHARSET_EN
    elif lang in ("vi", "vietnamese", "latin"):
        return CHARSET_VI
    else:
        raise ValueError(f"Unknown lang '{lang}'. Use 'en' (vocab=97) or 'vi' (vocab=352).")


# ══════════════════════════════════════════════════════════════════════════════
# EasyOCR preprocessing helpers
# ══════════════════════════════════════════════════════════════════════════════

def normalize_mean_variance(
    in_img: np.ndarray,
    mean: tuple = (0.485, 0.456, 0.406),
    variance: tuple = (0.229, 0.224, 0.225),
) -> np.ndarray:
    img = in_img.copy().astype(np.float32)
    img -= np.array([mean[0] * 255.0, mean[1] * 255.0, mean[2] * 255.0], dtype=np.float32)
    img /= np.array([variance[0] * 255.0, variance[1] * 255.0, variance[2] * 255.0], dtype=np.float32)
    return img


def resize_aspect_ratio(
    img: np.ndarray,
    square_size: int,
    interpolation: int = cv2.INTER_LINEAR,
    mag_ratio: float = 1.0,
) -> tuple[np.ndarray, float, tuple[int, int]]:
    height, width, _ = img.shape
    target_size = mag_ratio * square_size
    ratio = target_size / max(height, width)
    if ratio > 1.0:
        ratio = 1.0
    target_h = int(height * ratio)
    target_w = int(width  * ratio)
    proc = cv2.resize(img, (target_w, target_h), interpolation=interpolation)
    target_h32 = math.ceil(target_h / 32) * 32
    target_w32 = math.ceil(target_w / 32) * 32
    padded = np.zeros((target_h32, target_w32, 3), dtype=proc.dtype)
    padded[:target_h, :target_w] = proc
    return padded, ratio, (target_h32 // 2, target_w32 // 2)


# ══════════════════════════════════════════════════════════════════════════════
# CRAFT post-processing
# ══════════════════════════════════════════════════════════════════════════════

def get_det_boxes_core(
    textmap: np.ndarray,
    linkmap: np.ndarray,
    text_threshold: float,
    link_threshold: float,
    low_text: float,
) -> list[np.ndarray]:
    text_score_comb = np.clip(textmap + linkmap, 0.0, 1.0)
    _, text_score_comb_bin = cv2.threshold(
        (text_score_comb * 255).astype(np.uint8),
        int(low_text * 255), 255, cv2.THRESH_BINARY,
    )
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        text_score_comb_bin, connectivity=4
    )
    det_boxes: list[np.ndarray] = []
    for k in range(1, num_labels):
        size = stats[k, cv2.CC_STAT_AREA]
        if size < 10:
            continue
        comp_mask = (labels == k).astype(bool)
        if textmap[comp_mask].max() < text_threshold:
            continue
        segmap = np.zeros_like(textmap, dtype=np.uint8)
        segmap[comp_mask] = 255
        niter  = max(1, int(math.sqrt(size * min(
            stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT])) * 0.03))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1 + niter, 1 + niter))
        segmap = cv2.dilate(segmap, kernel)
        contours, _ = cv2.findContours(segmap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        cnt  = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(cnt)
        box  = cv2.boxPoints(rect)
        box  = np.intp(np.round(box))
        w = np.linalg.norm(box[0] - box[1])
        h = np.linalg.norm(box[1] - box[2])
        if min(w, h) < 1.0:
            continue
        det_boxes.append(box.astype(np.float32))
    return det_boxes


def adjust_result_coordinates(
    boxes_heatmap: list[np.ndarray],
    ratio_w: float,
    ratio_h: float,
) -> list[np.ndarray]:
    if not boxes_heatmap:
        return []
    result = []
    for box in boxes_heatmap:
        box = box.copy()
        box[:, 0] *= (1.0 / ratio_w) * 2.0
        box[:, 1] *= (1.0 / ratio_h) * 2.0
        result.append(box)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Recognizer preprocessing helpers
# ══════════════════════════════════════════════════════════════════════════════

def expand_bbox(
    box: np.ndarray,
    orig_h: int,
    orig_w: int,
    pad_v_ratio: float = 0.20,
    pad_h_ratio: float = 0.05,
) -> np.ndarray:
    box_h = float(box[:, 1].max() - box[:, 1].min())
    pad_v = max(2, int(box_h * pad_v_ratio))
    pad_h = max(1, int(box_h * pad_h_ratio))
    expanded = box.copy()
    expanded[0] = [np.clip(box[0, 0] - pad_h, 0, orig_w - 1), np.clip(box[0, 1] - pad_v, 0, orig_h - 1)]
    expanded[1] = [np.clip(box[1, 0] + pad_h, 0, orig_w - 1), np.clip(box[1, 1] - pad_v, 0, orig_h - 1)]
    expanded[2] = [np.clip(box[2, 0] + pad_h, 0, orig_w - 1), np.clip(box[2, 1] + pad_v, 0, orig_h - 1)]
    expanded[3] = [np.clip(box[3, 0] - pad_h, 0, orig_w - 1), np.clip(box[3, 1] + pad_v, 0, orig_h - 1)]
    return expanded


def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    rect = _order_points(pts)
    tl, tr, br, bl = rect
    max_w = max(int(np.linalg.norm(br - bl)), int(np.linalg.norm(tr - tl)))
    max_h = max(int(np.linalg.norm(tr - br)), int(np.linalg.norm(tl - bl)))
    if max_w < 1 or max_h < 1:
        return image
    dst = np.array([[0, 0], [max_w - 1, 0], [max_w - 1, max_h - 1], [0, max_h - 1]], dtype=np.float32)
    M      = cv2.getPerspectiveTransform(rect.astype(np.float32), dst)
    warped = cv2.warpPerspective(image, M, (max_w, max_h))
    return warped


def _order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)
    s    = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


_CONFUSABLE: dict[str, str] = {
    'ş': 'g',   'ș': 's',   'ț': 't',   'ğ': 'g',
    'ħ': 'h',   'ł': 'l',   'ø': 'o',   'ð': 'd',
    'ē': 'a',   'ı': '',    'ŕ': '',    'ƀ': '',
    '\u2019': '', '\u2018': '',
    '\u0307': '', '\u0308': '',
    '\u030a': '', '\u0323': '',
}


def clean_ocr_text(text: str) -> str:
    import unicodedata, re
    if not text:
        return text
    text = text.replace("~", "").replace("_", " ").replace("|", "").replace("&#124;", "")
    text = unicodedata.normalize('NFC', text)
    buf: list[str] = []
    for ch in text:
        buf.append(_CONFUSABLE.get(ch, ch))
    text = ''.join(buf)
    text = re.sub(r'([bcdghklmnpqrstvwxy])\1+', r'\1', text, flags=re.IGNORECASE)
    text = re.sub(r' +', ' ', text).strip()
    return text


def merge_boxes_into_lines(boxes: list[np.ndarray], y_overlap_thresh: float = 0.4) -> list[np.ndarray]:
    if not boxes:
        return []

    def box_yrange(b):
        ys = b[:, 1]
        return float(ys.min()), float(ys.max())

    def y_overlap_ratio(r1, r2):
        lo = max(r1[0], r2[0])
        hi = min(r1[1], r2[1])
        overlap = max(0.0, hi - lo)
        h1 = max(1e-6, r1[1] - r1[0])
        h2 = max(1e-6, r2[1] - r2[0])
        return overlap / min(h1, h2)

    boxes_sorted = sorted(boxes, key=lambda b: (b[:, 1].mean(), b[:, 0].mean()))
    lines = []
    line_yranges = []

    for box in boxes_sorted:
        by0, by1 = box_yrange(box)
        placed = False
        for g_idx in range(len(lines) - 1, -1, -1):
            gy0, gy1 = line_yranges[g_idx]
            if y_overlap_ratio((by0, by1), (gy0, gy1)) >= y_overlap_thresh:
                lines[g_idx].append(box)
                line_yranges[g_idx] = (min(gy0, by0), max(gy1, by1))
                placed = True
                break
        if not placed:
            lines.append([box])
            line_yranges.append((by0, by1))

    result = []
    for group in lines:
        if not group:
            continue
        all_pts = np.concatenate(group, axis=0)
        x0 = float(all_pts[:, 0].min())
        y0 = float(all_pts[:, 1].min())
        x1 = float(all_pts[:, 0].max())
        y1 = float(all_pts[:, 1].max())
        hull = np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]], dtype=np.float32)
        result.append(hull)

    result.sort(key=lambda b: b[:, 1].min())
    return result


def split_and_preprocess_crop(
    gray_crop: np.ndarray,
    target_h: int = REC_H,
    target_w: int = REC_W,
    pad_lr_px: int = 40,
) -> list[np.ndarray]:
    """
    Scale crop to target_h (aspect-ratio preserved), split into non-overlapping
    chunks, return list of (1, H, W, 1) float32 tensors.

    No overlap is used — each pixel belongs to exactly one chunk — to avoid
    the CTC duplicate-character bug at chunk boundaries.
    """
    if gray_crop.size == 0 or gray_crop.shape[0] == 0 or gray_crop.shape[1] == 0:
        return [np.zeros((1, target_h, target_w, 1), dtype=np.float32)]

    h, w     = gray_crop.shape[:2]
    ratio    = target_h / float(h)
    scaled_w = int(w * ratio)
    if scaled_w <= 0:
        return [np.zeros((1, target_h, target_w, 1), dtype=np.float32)]

    resized = cv2.resize(gray_crop, (scaled_w, target_h), interpolation=cv2.INTER_LINEAR)
    norm    = (resized.astype(np.float32) / 255.0 - REC_MEAN) / REC_STD

    border_px = max(1, min(3, h // 4))
    bg_u8     = float(np.median(np.concatenate([
        gray_crop[:border_px, :].ravel(),
        gray_crop[-border_px:, :].ravel(),
    ])))
    bg_norm   = (bg_u8 / 255.0 - REC_MEAN) / REC_STD

    content_w = target_w - 2 * pad_lr_px
    chunks    = []
    x         = 0

    while True:
        x_end    = min(x + content_w, scaled_w)
        actual_w = x_end - x
        canvas = np.full((target_h, target_w), bg_norm, dtype=np.float32)
        canvas[:, pad_lr_px : pad_lr_px + actual_w] = norm[:, x:x_end]
        chunks.append(canvas[np.newaxis, :, :, np.newaxis])
        if x_end >= scaled_w:
            break
        x += content_w

    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# Data types
# ══════════════════════════════════════════════════════════════════════════════

class DetectionResult(NamedTuple):
    bbox:       np.ndarray   # (4, 2) corners in original image coords
    text:       str
    confidence: float


# ══════════════════════════════════════════════════════════════════════════════
# Main engine
# ══════════════════════════════════════════════════════════════════════════════

class QAIRTEasyOCREngine:
    """
    EasyOCR inference on Qualcomm NPU via qnn-net-run.exe.

    Two public recognition methods:

    1. recognize(pil_image) → list[DetectionResult]
       Full pipeline: CRAFT tiled detection + chunked CTC recognition.
       Use for standalone images where you have no layout info.

    2. recognize_region(pil_crop) → (text, confidence)     ← NEW
       Recognition only — NO detection, NO CRAFT.
       Use when the caller (e.g. region_ocr.py) already has the bbox
       from Docling layout analysis. The crop IS the region.
       This is the correct call in the layout-centric pipeline.
    """

    def __init__(
        self,
        workspace_dir:  str,
        qairt_sdk_path: str,
        model_dir:      str,
        lang:           str = "vi",
    ):
        self.workspace  = Path(workspace_dir)
        self.model_dir  = Path(model_dir)
        self.charset    = build_charset(lang)
        self.rec_vocab  = len(self.charset)

        sdk = Path(qairt_sdk_path)
        self.qairt_bin = sdk / "bin" / "x86_64-windows-msvc" / "qnn-net-run.exe"
        self.backend   = sdk / "lib" / "arm64x-windows-msvc"  / "QnnHtp.dll"
        self.skel_dir  = sdk / "lib" / "hexagon-v73"          / "unsigned"

        self.in_dir  = self.workspace / "easy_input"
        self.out_dir = self.workspace / "easy_output"

        self._ratio_w: float = 1.0
        self._ratio_h: float = 1.0

        logger.info(
            f"QAIRTEasyOCREngine | lang={lang} vocab={self.rec_vocab} "
            f"| model_dir={model_dir}"
        )

    # ══════════════════════════════════════════════════════════════════════════
    # PUBLIC API — Layout-centric path (NEW)
    # ══════════════════════════════════════════════════════════════════════════

    def recognize_region(
        self,
        crop_image: Image.Image,
        *,
        line_separator: str = " ",
    ) -> tuple[str, float]:
        """
        Recognition-only path for use with Docling layout analysis.

        The crop is a pre-extracted region from region_ocr.py — Docling has
        already determined WHERE the text is; we only need to read WHAT it says.

        Steps:
          1. Convert PIL crop to grayscale numpy array.
          2. If the crop has multiple lines (tall height), split it into
             horizontal slices so each slice is a single text line.
          3. For each line slice, call split_and_preprocess_crop() to handle
             long lines (chunking without overlap).
          4. Run the recognizer on each chunk via NPU.
          5. Concatenate and clean the results.

        Args:
            crop_image: PIL Image of the pre-cropped layout region.
            line_separator: String used to join consecutive OCR lines.
                Default ``" "`` matches the historical behaviour (body
                text). Pass ``"\\n"`` from table-cell OCR so multi-line
                cells preserve their vertical structure — the DOCX and
                Markdown writers translate ``\\n`` into proper cell-line
                breaks downstream.

        Returns:
            (text, confidence) tuple.
        """
        self._clear_workspace()

        img_rgb  = np.array(crop_image.convert("RGB"), dtype=np.uint8)
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        crop_h, crop_w = img_gray.shape[:2]

        if crop_h < 4 or crop_w < 4:
            return "", 0.0

        # ── Split multi-line region into individual line crops ─────────────────
        # A Docling "text" block can span multiple lines. We estimate line height
        # from the crop height and split into rows. Single-line blocks go through
        # unchanged.
        line_crops = self._split_into_line_crops(img_gray)

        full_text  = ""
        all_confs: list[float] = []

        for line_idx, line_gray in enumerate(line_crops):
            if line_gray.shape[0] < 4 or line_gray.shape[1] < 4:
                continue

            # Handle long lines by chunking (no overlap to avoid CTC duplicates)
            chunks = split_and_preprocess_crop(line_gray)

            line_text = ""
            line_confs: list[float] = []

            for chunk_idx, chunk_tensor in enumerate(chunks):
                rec_out = self._run_qnn(
                    "recognizer",
                    {"image": chunk_tensor},
                    f"region_{line_idx}_{chunk_idx}",
                )
                chunk_text, conf = self._decode_recognizer(rec_out)
                line_text += chunk_text
                line_confs.append(conf)

            cleaned = clean_ocr_text(line_text)
            if cleaned:
                full_text  += (line_separator if full_text else "") + cleaned
                all_confs.extend(line_confs)

        avg_conf = float(np.mean(all_confs)) if all_confs else 0.0
        logger.debug(f"  recognize_region: '{full_text[:80]}' (conf={avg_conf:.3f})")
        return full_text, avg_conf

    # ══════════════════════════════════════════════════════════════════════════
    # PUBLIC API — Standalone full-image path (UNCHANGED from v3)
    # ══════════════════════════════════════════════════════════════════════════

    def recognize(self, image: Image.Image) -> list[DetectionResult]:
        """
        Full CRAFT tiled detection + CTC recognition pipeline.

        Use this when you have a raw image and no layout information.
        When called from region_ocr.py (Docling pipeline), use
        recognize_region() instead — it skips the unnecessary detection pass.
        """
        self._clear_workspace()

        img_rgb  = np.array(image.convert("RGB"), dtype=np.uint8)
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        orig_h, orig_w = img_rgb.shape[:2]

        # ── Tiled detection ────────────────────────────────────────────────────
        img_norm = img_rgb.astype(np.float32) / 255.0
        img_norm = (img_norm - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])

        pad_h = (DET_H - orig_h % DET_H) % DET_H
        pad_w = (DET_W - orig_w % DET_W) % DET_W
        img_padded = np.pad(img_norm, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=0)
        pad_h_len, pad_w_len = img_padded.shape[:2]

        heatmap_full = np.zeros((pad_h_len // 2, pad_w_len // 2, 2), dtype=np.float32)

        for y in range(0, pad_h_len, DET_H):
            for x in range(0, pad_w_len, DET_W):
                tile = img_padded[y:y+DET_H, x:x+DET_W]
                tile_input = np.expand_dims(tile, axis=0).astype(np.float32)
                det_out_path = self._run_qnn("detector", {"image": tile_input}, f"det_{y}_{x}")
                score_text, score_link = self._read_detector_output(det_out_path)
                hy, hx = y // 2, x // 2
                heatmap_full[hy:hy+MAP_H, hx:hx+MAP_W, 0] = score_text
                heatmap_full[hy:hy+MAP_H, hx:hx+MAP_W, 1] = score_link

        heatmap_full = heatmap_full[:orig_h // 2, :orig_w // 2]
        score_text = heatmap_full[:, :, 0]
        score_link = heatmap_full[:, :, 1]

        boxes_hm = get_det_boxes_core(score_text, score_link, TEXT_THRESH, LINK_THRESH, LOW_TEXT)
        if not boxes_hm:
            return []

        boxes_orig  = adjust_result_coordinates(boxes_hm, 1.0, 1.0)
        boxes_lines = merge_boxes_into_lines(boxes_orig, y_overlap_thresh=0.4)
        logger.info(f"Tiled Detection: {len(boxes_orig)} boxes -> Merged into {len(boxes_lines)} lines")

        # ── Chunked recognition ────────────────────────────────────────────────
        results: list[DetectionResult] = []
        for i, box in enumerate(boxes_lines):
            pad_h_px, pad_v_px = 5, 8
            box_c = box.copy()
            box_c[0] += [-pad_h_px, -pad_v_px]
            box_c[1] += [ pad_h_px, -pad_v_px]
            box_c[2] += [ pad_h_px,  pad_v_px]
            box_c[3] += [-pad_h_px,  pad_v_px]
            box_c[:, 0] = np.clip(box_c[:, 0], 0, orig_w - 1)
            box_c[:, 1] = np.clip(box_c[:, 1], 0, orig_h - 1)

            crop = four_point_transform(img_gray, box_c)
            if crop.size == 0 or crop.shape[0] < 2 or crop.shape[1] < 2:
                continue

            rec_inputs = split_and_preprocess_crop(crop)
            full_text  = ""
            confs      = []

            for j, rec_input in enumerate(rec_inputs):
                rec_out = self._run_qnn("recognizer", {"image": rec_input}, f"rec_{i}_{j}")
                text_chunk, conf = self._decode_recognizer(rec_out)
                full_text += text_chunk
                confs.append(conf)

            text = clean_ocr_text(full_text)
            conf = sum(confs) / len(confs) if confs else 0.0
            logger.debug(f"  line {i}: '{text}' (conf={conf:.3f})")

            if text:
                results.append(DetectionResult(bbox=box_c, text=text, confidence=conf))

        return results

    def recognize_text_only(self, image: Image.Image) -> list[str]:
        return [r.text for r in self.recognize(image)]

    # ══════════════════════════════════════════════════════════════════════════
    # PUBLIC API — Detection-only (CRAFT on NPU, no recognizer)
    # ══════════════════════════════════════════════════════════════════════════

    def detect_lines(
        self,
        image: Image.Image,
        *,
        merge_into_lines: bool = True,
    ) -> tuple[list[np.ndarray], np.ndarray]:
        """
        Run CRAFT tiled detection on the NPU and return text-line boxes — no
        recognizer pass. Used by the EasyOCR-detect + VietOCR-recognize hybrid
        so the NPU does the heavy fixed-shape stage and the CPU does the
        dynamic-width recognition where Vietnamese accuracy matters.

        Args:
            image: source RGB PIL image (typically a YOLO region crop).
            merge_into_lines: if True, CRAFT word/char boxes are merged into
                horizontal text lines (default — what the recognizer wants).
                Pass False to get the raw CRAFT boxes (one per word/cluster).

        Returns:
            (boxes, img_gray)
              boxes    — list of (4, 2) float32 polygons in image coords.
              img_gray — the grayscale source image, for the caller to crop
                         via four_point_transform without re-converting.
        """
        self._clear_workspace()

        img_rgb  = np.array(image.convert("RGB"), dtype=np.uint8)
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        orig_h, orig_w = img_rgb.shape[:2]

        # Normalised, padded tile grid — identical preprocessing to recognize()
        img_norm = img_rgb.astype(np.float32) / 255.0
        img_norm = (img_norm - np.array([0.485, 0.456, 0.406])) \
                   / np.array([0.229, 0.224, 0.225])

        pad_h = (DET_H - orig_h % DET_H) % DET_H
        pad_w = (DET_W - orig_w % DET_W) % DET_W
        img_padded = np.pad(
            img_norm,
            ((0, pad_h), (0, pad_w), (0, 0)),
            mode="constant", constant_values=0,
        )
        pad_h_len, pad_w_len = img_padded.shape[:2]

        heatmap_full = np.zeros(
            (pad_h_len // 2, pad_w_len // 2, 2), dtype=np.float32
        )
        for y in range(0, pad_h_len, DET_H):
            for x in range(0, pad_w_len, DET_W):
                tile        = img_padded[y:y + DET_H, x:x + DET_W]
                tile_input  = np.expand_dims(tile, axis=0).astype(np.float32)
                det_out     = self._run_qnn(
                    "detector", {"image": tile_input}, f"det_{y}_{x}",
                )
                score_text, score_link = self._read_detector_output(det_out)
                hy, hx = y // 2, x // 2
                heatmap_full[hy:hy + MAP_H, hx:hx + MAP_W, 0] = score_text
                heatmap_full[hy:hy + MAP_H, hx:hx + MAP_W, 1] = score_link

        heatmap_full = heatmap_full[:orig_h // 2, :orig_w // 2]
        score_text   = heatmap_full[:, :, 0]
        score_link   = heatmap_full[:, :, 1]

        boxes_hm = get_det_boxes_core(
            score_text, score_link, TEXT_THRESH, LINK_THRESH, LOW_TEXT,
        )
        if not boxes_hm:
            return [], img_gray

        boxes_orig = adjust_result_coordinates(boxes_hm, 1.0, 1.0)
        if merge_into_lines:
            boxes_out = merge_boxes_into_lines(boxes_orig, y_overlap_thresh=0.4)
        else:
            boxes_out = boxes_orig

        logger.info(
            f"detect_lines: {len(boxes_orig)} raw CRAFT boxes -> "
            f"{len(boxes_out)} {'lines' if merge_into_lines else 'words'}"
        )
        return boxes_out, img_gray

    # ══════════════════════════════════════════════════════════════════════════
    # PUBLIC API — Word-granular detection (no line merging)
    # ══════════════════════════════════════════════════════════════════════════

    def recognize_words(self, image: Image.Image) -> list[DetectionResult]:
        """
        Like :meth:`recognize` but **does not** merge CRAFT boxes into lines.

        ``recognize()`` is built for free-form documents where merging
        adjacent boxes into single text lines is desirable. For tables
        that's catastrophic — the cells in one row sit on the same Y level
        and would all be merged into one giant box spanning the full width,
        making per-cell text assignment impossible.

        This method preserves the raw CRAFT box per text region (typically
        word- or short-phrase-level) and runs the recognizer on each one
        independently. Output is a list of :class:`DetectionResult` — same
        shape as :meth:`recognize`, just with finer granularity. Used by
        the img2table table reconstructor to map words to cells.
        """
        self._clear_workspace()

        img_rgb  = np.array(image.convert("RGB"), dtype=np.uint8)
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        orig_h, orig_w = img_rgb.shape[:2]

        # ── Tiled CRAFT detection (identical to recognize) ──────────────────
        img_norm = img_rgb.astype(np.float32) / 255.0
        img_norm = (img_norm - np.array([0.485, 0.456, 0.406])) \
                   / np.array([0.229, 0.224, 0.225])

        pad_h = (DET_H - orig_h % DET_H) % DET_H
        pad_w = (DET_W - orig_w % DET_W) % DET_W
        img_padded = np.pad(
            img_norm,
            ((0, pad_h), (0, pad_w), (0, 0)),
            mode="constant", constant_values=0,
        )
        pad_h_len, pad_w_len = img_padded.shape[:2]

        heatmap_full = np.zeros(
            (pad_h_len // 2, pad_w_len // 2, 2), dtype=np.float32
        )
        for y in range(0, pad_h_len, DET_H):
            for x in range(0, pad_w_len, DET_W):
                tile        = img_padded[y:y + DET_H, x:x + DET_W]
                tile_input  = np.expand_dims(tile, axis=0).astype(np.float32)
                det_out     = self._run_qnn(
                    "detector", {"image": tile_input}, f"det_{y}_{x}",
                )
                score_text, score_link = self._read_detector_output(det_out)
                hy, hx = y // 2, x // 2
                heatmap_full[hy:hy + MAP_H, hx:hx + MAP_W, 0] = score_text
                heatmap_full[hy:hy + MAP_H, hx:hx + MAP_W, 1] = score_link

        heatmap_full = heatmap_full[:orig_h // 2, :orig_w // 2]
        score_text   = heatmap_full[:, :, 0]
        score_link   = heatmap_full[:, :, 1]

        boxes_hm = get_det_boxes_core(
            score_text, score_link, TEXT_THRESH, LINK_THRESH, LOW_TEXT,
        )
        if not boxes_hm:
            return []

        boxes_orig = adjust_result_coordinates(boxes_hm, 1.0, 1.0)
        # NOTE: deliberately *no* merge_boxes_into_lines() call here.
        logger.info(f"recognize_words: {len(boxes_orig)} raw CRAFT boxes")

        # ── Recognize each box individually ─────────────────────────────────
        results: list[DetectionResult] = []
        for i, box in enumerate(boxes_orig):
            pad_h_px, pad_v_px = 4, 6
            box_c = box.copy()
            box_c[0] += [-pad_h_px, -pad_v_px]
            box_c[1] += [ pad_h_px, -pad_v_px]
            box_c[2] += [ pad_h_px,  pad_v_px]
            box_c[3] += [-pad_h_px,  pad_v_px]
            box_c[:, 0] = np.clip(box_c[:, 0], 0, orig_w - 1)
            box_c[:, 1] = np.clip(box_c[:, 1], 0, orig_h - 1)

            crop = four_point_transform(img_gray, box_c)
            if crop.size == 0 or crop.shape[0] < 2 or crop.shape[1] < 2:
                continue

            rec_inputs = split_and_preprocess_crop(crop)
            full_text  = ""
            confs:     list[float] = []
            for j, rec_input in enumerate(rec_inputs):
                rec_out          = self._run_qnn(
                    "recognizer", {"image": rec_input}, f"rec_{i}_{j}",
                )
                text_chunk, conf = self._decode_recognizer(rec_out)
                full_text       += text_chunk
                confs.append(conf)

            text = clean_ocr_text(full_text)
            conf = sum(confs) / len(confs) if confs else 0.0
            if text:
                results.append(
                    DetectionResult(bbox=box_c, text=text, confidence=conf)
                )

        return results

    # ══════════════════════════════════════════════════════════════════════════
    # PUBLIC API — recognize_with_boxes  (for image_table_reconstructor)
    # ══════════════════════════════════════════════════════════════════════════

    def recognize_with_boxes(self, crop_image):
        """
        Run CRAFT detection + per-line recognition and return a flat
        list of ``OCRLine`` with each line's 4-point polygon populated.
        Thin adapter on top of :meth:`recognize_words` — converts each
        ``DetectionResult`` into the engine-agnostic ``OCRLine`` shape
        the image-table fallback expects.
        """
        from document_pipeline import OCRLine

        try:
            results = self.recognize_words(crop_image)
        except Exception as e:
            logger.warning(f"recognize_with_boxes failed: {e}")
            return []

        out: list = []
        for r in results:
            if not (r.text or "").strip():
                continue
            poly = [(float(p[0]), float(p[1])) for p in r.bbox]
            out.append(OCRLine(
                text       = r.text,
                confidence = float(r.confidence or 0.0),
                polygon    = poly,
            ))
        return out

    # ══════════════════════════════════════════════════════════════════════════
    # PRIVATE: Line splitting for multi-line region crops
    # ══════════════════════════════════════════════════════════════════════════

    def _split_into_line_crops(
        self,
        gray: np.ndarray,
        min_line_h_px: int = 12,
        max_line_h_px: int = 80,
    ) -> list[np.ndarray]:
        """
        Split a multi-line grayscale crop into individual line images.

        Strategy: if the crop height is <= max_line_h_px (e.g. 80px at 150dpi
        ≈ a single text line in a normal document), return it as-is.

        For taller crops (paragraphs), use horizontal projection profile:
          - Binarize with Otsu
          - Compute row-wise sum of foreground pixels
          - Find valleys (gaps between text lines) and split there

        Falls back to equal-height slicing if projection analysis fails.

        Note: Docling's region bbox usually covers exactly one paragraph /
        heading, so multi-line splits are rare but do occur for dense paragraphs.
        """
        h, w = gray.shape[:2]

        # Single-line region — return as-is (most common case from Docling)
        if h <= max_line_h_px:
            return [gray]

        # ── Horizontal projection profile ──────────────────────────────────────
        _, binarized = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        row_sums = binarized.sum(axis=1).astype(np.float32)

        # Smooth to reduce noise
        kernel_size = max(3, min(11, h // 10) | 1)  # must be odd
        smoothed = cv2.GaussianBlur(row_sums.reshape(-1, 1), (1, kernel_size), 0).ravel()

        # Find valley (gap) rows: rows where smoothed sum < threshold
        # Threshold: 5% of maximum row activity
        gap_thresh = smoothed.max() * 0.05
        is_gap = smoothed < gap_thresh

        # Find contiguous text regions (non-gap rows)
        lines: list[np.ndarray] = []
        in_line = False
        line_start = 0

        for row_idx in range(h):
            if not is_gap[row_idx] and not in_line:
                in_line    = True
                line_start = row_idx
            elif is_gap[row_idx] and in_line:
                in_line = False
                line_h  = row_idx - line_start
                if line_h >= min_line_h_px:
                    # Add small vertical padding to avoid clipping ascenders/descenders
                    pad = max(2, int(line_h * 0.15))
                    y0 = max(0, line_start - pad)
                    y1 = min(h, row_idx + pad)
                    lines.append(gray[y0:y1, :])

        # Flush last line
        if in_line:
            line_h = h - line_start
            if line_h >= min_line_h_px:
                pad = max(2, int(line_h * 0.15))
                y0  = max(0, line_start - pad)
                lines.append(gray[y0:h, :])

        if not lines:
            # Projection failed — return the whole crop unchanged
            logger.debug("  _split_into_line_crops: projection failed, using full crop")
            return [gray]

        logger.debug(f"  _split_into_line_crops: {h}px crop → {len(lines)} line(s)")
        return lines

    # ══════════════════════════════════════════════════════════════════════════
    # Detection preprocessing (used only by recognize())
    # ══════════════════════════════════════════════════════════════════════════

    def _preprocess_detector(
        self, img_rgb: np.ndarray,
    ) -> tuple[np.ndarray, float, tuple[int, int]]:
        normalized = normalize_mean_variance(img_rgb)
        resized, ratio, size_heatmap = resize_aspect_ratio(
            normalized, CANVAS_SIZE, cv2.INTER_LINEAR, MAG_RATIO,
        )
        det_h, det_w = resized.shape[:2]
        if det_h != DET_H or det_w != DET_W:
            resized       = cv2.resize(resized, (DET_W, DET_H), interpolation=cv2.INTER_LINEAR)
            self._ratio_w = ratio / (det_w / DET_W)
            self._ratio_h = ratio / (det_h / DET_H)
        else:
            self._ratio_w = ratio
            self._ratio_h = ratio
        return resized[np.newaxis, ...].astype(np.float32), ratio, size_heatmap

    def _read_detector_output(self, out_dir: Path) -> tuple[np.ndarray, np.ndarray]:
        f    = self._find_file(out_dir / "Result_0", "results.raw")
        data = np.fromfile(f, dtype=np.float32)
        expected = MAP_H * MAP_W * 2
        if data.size != expected:
            raise ValueError(f"Detector: expected {expected} floats, got {data.size}")
        hm = data.reshape(MAP_H, MAP_W, 2)
        return hm[:, :, 0], hm[:, :, 1]

    # ══════════════════════════════════════════════════════════════════════════
    # CTC decoder
    # ══════════════════════════════════════════════════════════════════════════

    def _decode_recognizer(
        self, out_dir: Path, debug: bool = False,
    ) -> tuple[str, float]:
        f    = self._find_file(out_dir / "Result_0", "output_preds.raw")
        data = np.fromfile(f, dtype=np.float32)

        expected = REC_TIME * self.rec_vocab
        if data.size != expected:
            raise ValueError(
                f"Recognizer: expected {expected} floats, got {data.size}. "
                f"Ensure lang= matches the exported model (vocab={self.rec_vocab})."
            )

        logits    = data.reshape(REC_TIME, self.rec_vocab)
        probs     = _softmax(logits)
        token_ids = np.argmax(probs, axis=1)
        confs     = probs[np.arange(REC_TIME), token_ids]

        chars:  list[str]   = []
        scores: list[float] = []
        prev = -1
        for t, tid in enumerate(token_ids.tolist()):
            if tid == 0 or tid == prev:
                prev = tid
                continue
            if 0 < tid < len(self.charset):
                c = self.charset[tid]
                if c and c not in ("[blank]", "[pad]"):
                    chars.append(c)
                    scores.append(float(confs[t]))
            prev = tid

        return "".join(chars), float(np.mean(scores)) if scores else 0.0

    # ══════════════════════════════════════════════════════════════════════════
    # QNN subprocess runner
    # ══════════════════════════════════════════════════════════════════════════

    # Class-level counter so log lines can show "NPU call #N" — useful
    # when the user wants to verify the NPU is actually being hit.
    _npu_call_counter: int = 0

    def _run_qnn(
        self,
        model_name:    str,
        input_tensors: dict[str, np.ndarray],
        out_subdir:    str = "",
    ) -> Path:
        # ── Pre-flight sanity: SDK binary + model context must exist ──
        # Catches the "silent CPU fallback" failure mode where qnn-net-run
        # is missing or the .bin model file got moved. Without this, the
        # only signal was a generic subprocess error in the log.
        if not self.qairt_bin.exists():
            raise RuntimeError(
                f"NPU subprocess missing: {self.qairt_bin}. "
                "Check QAIRT SDK installation."
            )
        ctx_bin = self.model_dir / f"{model_name}.bin"
        if not ctx_bin.exists():
            raise RuntimeError(
                f"NPU context binary missing: {ctx_bin}. "
                "Re-download the EasyOCR Snapdragon model from "
                "models/easyocr_qnn_vi/."
            )

        raw_paths: dict[str, Path] = {}
        for name, arr in input_tensors.items():
            p = self.in_dir / f"{model_name}_{name}.raw"
            np.ascontiguousarray(arr, dtype=np.float32).tofile(p)
            raw_paths[name] = p

        list_content = " ".join(
            f"{name}:={path.resolve()}" for name, path in raw_paths.items()
        )
        list_file = self.workspace / f"{model_name}_input_list.txt"
        list_file.write_text(list_content, encoding="ascii")

        output_dir = self.out_dir / (out_subdir or model_name)

        env = os.environ.copy()
        env["ADSP_LIBRARY_PATH"] = f"{self.skel_dir};;{self.backend.parent};;"
        env["PATH"]              = f"{self.backend.parent};" + env.get("PATH", "")

        cmd = [
            str(self.qairt_bin),
            "--backend",          str(self.backend),
            "--retrieve_context", str(ctx_bin),
            "--input_list",       str(list_file),
            "--output_dir",       str(output_dir),
            "--use_native_input_files",
            "--use_native_output_files",
        ]
        logger.debug("CMD: " + " ".join(cmd))

        # ── Run on NPU and log an unambiguous proof-of-execution line ──
        import time as _time
        QAIRTEasyOCREngine._npu_call_counter += 1
        n  = QAIRTEasyOCREngine._npu_call_counter
        t0 = _time.perf_counter()
        r  = subprocess.run(cmd, env=env, capture_output=True, text=True)
        dt = _time.perf_counter() - t0

        if r.returncode != 0:
            logger.error(
                f"❌ NPU call #{n} [{model_name}] FAILED (rc={r.returncode}, "
                f"{dt*1000:.0f} ms)"
            )
            logger.error(f"STDERR:\n{r.stderr}")
            r.check_returncode()
        else:
            logger.info(
                f"✓ NPU call #{n} [{model_name}] ok ({dt*1000:.0f} ms) "
                f"via {self.qairt_bin.name}"
            )

        return output_dir

    # ══════════════════════════════════════════════════════════════════════════
    # Utilities
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _find_file(directory: Path, base_name: str) -> Path:
        for name in [base_name, base_name.replace(".raw", "_native.raw")]:
            p = directory / name
            if p.exists():
                return p
        raise FileNotFoundError(f"'{base_name}' not found in {directory}")

    def _clear_workspace(self):
        for d in [self.in_dir, self.out_dir]:
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


# ══════════════════════════════════════════════════════════════════════════════
# Standalone test
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    import time

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    SDK  = r"C:\\Qualcomm\\AIStack\\QAIRT\\2.45.0.260326"
    LANG = "vi"

    MODELS = (
        r"models\\easyocr_qnn_vi\\easyocr-qnn_context_binary-float-qualcomm_snapdragon_x_elite"
        if LANG == "vi" else
        r"models\\easyocr_qnn\\easyocr-qnn_context_binary-float-qualcomm_snapdragon_x_elite"
    )

    engine = QAIRTEasyOCREngine(
        workspace_dir  = str(Path(".").resolve()),
        qairt_sdk_path = SDK,
        model_dir      = MODELS,
        lang           = LANG,
    )

    img_path = sys.argv[1] if len(sys.argv) > 1 else "test_line_3.png"
    mode     = sys.argv[2] if len(sys.argv) > 2 else "full"   # "full" or "region"

    if not Path(img_path).exists():
        print(f"[ERROR] Image not found: {img_path}")
        sys.exit(1)

    img = Image.open(img_path).convert("RGB")
    print(f"Input: {img.size[0]}×{img.size[1]} px | lang={LANG} | vocab={engine.rec_vocab} | mode={mode}")

    t0 = time.time()

    if mode == "region":
        # Test the layout-centric path (no CRAFT detection)
        text, conf = engine.recognize_region(img)
        elapsed = time.time() - t0
        print(f"\n✅ recognize_region in {elapsed:.2f}s:")
        print(f"  '{text}'  (conf={conf:.3f})")
    else:
        # Test the full standalone path
        results = engine.recognize(img)
        elapsed = time.time() - t0
        print(f"\n✅ {len(results)} region(s) in {elapsed:.2f}s:")
        for i, r in enumerate(results):
            print(f"  [{i+1}] '{r.text}'  (conf={r.confidence:.3f})")