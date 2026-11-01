"""
core/md_ocr_translator.py
==========================

Translate an OCR markdown page while keeping its exact structure.

Handles TWO shapes of input, transparently:

1. Raw grounding-mode markdown (debug dumps, ``*_ocr_raw.md``) — still
   has DeepSeek-OCR-2's own annotation tags:

       <|ref|>text<|/ref|><|det|>[[123, 141, 356, 160]]<|/det|>
       Luật số: 36/2024/QH15

2. Clean markdown (what ``end2end``'s ``ocr_full_page_markdown()``
   actually returns in production — ``strip_grounding_tags()`` already
   removed the tags before the pipeline ever sees it):

       ## Page 1

       Luật số: 36/2024/QH15

       | Cột 1 | Cột 2 |
       |---|---|
       | a | b |

       ![Figure](p003_b002.png)

Everything except actual sentence content is left byte-for-byte
untouched:

    * ``<|ref|>LABEL<|/ref|><|det|>[[x0,y0,x1,y1]]<|/det|>`` tag lines
    * ``## Page N`` headers
    * ``---`` separators (page breaks AND markdown table header rules —
      a rule row like ``|---|---|`` is detected by content, not just
      position, so it's never mis-translated)
    * blank lines
    * ``![alt](path)`` image references / bare ``[Figure]`` placeholders
    * leading markdown/list prefixes on a content line
      (``# ``, ``## ``, ``1. ``, ``- ``, ``* ``) — kept as-is, only the
      text *after* the prefix is sent to the LLM
    * pipe-table rows — each ``| cell |`` is translated individually and
      re-assembled with the same pipe structure, so columns can't drift

Why this is separate from ``TranslationEngine``
------------------------------------------------
``TranslationEngine`` (in ``document_pipeline.py``) translates
``DocumentPage`` IR objects — it needs the structured ``IRBlock`` /
``TableCell`` data that only the "blocks" pipeline (YOLO layout + OCR
per block) produces. The ``end2end`` engine has no IR — DeepSeek-OCR-2
hands back one markdown blob per page — so this module translates that
blob directly, text-content-only, structure-preserving.

Usage
-----
    python core/md_ocr_translator.py input_ocr_raw.md vi->en [output.md]

Or from Python:

    from md_ocr_translator import translate_ocr_markdown
    translated = translate_ocr_markdown(raw_text, direction="vi->en")
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

_log = logging.getLogger(__name__)

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from translator import translate_text  # noqa: E402


# ── Line classifiers ─────────────────────────────────────────────────────────

# A grounding tag line: <|ref|>LABEL<|/ref|><|det|>[[...]]<|/det|>
_TAG_LINE_RE = re.compile(r"^\s*<\|ref\|>.*?<\|/ref\|>\s*<\|det\|>.*?<\|/det\|>\s*$")

# "## Page 3" style page headers — structural, not translated content.
_PAGE_HEADER_RE = re.compile(r"^\s*##\s*Page\s+\d+\s*$", re.IGNORECASE)

# Page separator (three-or-more dashes on their own line).
_HR_RE = re.compile(r"^\s*-{3,}\s*$")

# Markdown image reference: ![alt](path)
_IMAGE_RE = re.compile(r"^\s*!\[[^\]]*\]\([^)]+\)\s*$")

# Bare figure placeholder emitted when there's no crop to embed.
_FIGURE_PLACEHOLDER_RE = re.compile(r"^\s*\[Figure\]\s*$")

# Pipe-table row: starts and ends with '|' (markdown table convention).
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")

# A table separator row like "|---|---|" or "|:--|--:|" — cell content
# is only dashes/colons/spaces.
_TABLE_SEP_CELL_RE = re.compile(r"^[\s:\-]+$")

# Leading structural prefix on a content line: markdown heading marks,
# numbered-list markers, or bullet markers. Captured so it survives
# untouched; only the remainder is translated.
_PREFIX_RE = re.compile(r"^(\s*(?:#{1,6}\s+|\d+[\.\)]\s+|[-*]\s+)?)(.*)$")


def _classify(line: str) -> str:
    """One of: blank, tag, page_header, hr, image, table_row, content."""
    stripped = line.strip()
    if not stripped:
        return "blank"
    if _TAG_LINE_RE.match(line):
        return "tag"
    if _PAGE_HEADER_RE.match(line):
        return "page_header"
    if _HR_RE.match(line):
        return "hr"
    if _IMAGE_RE.match(line) or _FIGURE_PLACEHOLDER_RE.match(line):
        return "image"
    if _TABLE_ROW_RE.match(line):
        return "table_row"
    return "content"


def _split_eol(line: str) -> tuple[str, str]:
    """Split a line into (body, eol) where eol is '', '\\n', '\\r' or '\\r\\n'."""
    m = re.search(r"(\r\n|\r|\n)?$", line)
    eol = m.group(1) or ""
    body = line[: len(line) - len(eol)] if eol else line
    return body, eol


def _translate_content_line(line: str, direction: str, cache: dict[str, str]) -> str:
    """
    Translate one content line, preserving its trailing newline and any
    leading structural prefix (#, ##, "1. ", "- ", ...) exactly.
    """
    body, eol = _split_eol(line)

    m = _PREFIX_RE.match(body)
    prefix, text = (m.group(1), m.group(2)) if m else ("", body)

    if not text.strip():
        return line  # prefix-only or empty after stripping

    if text in cache:
        translated = cache[text]
    else:
        translated = translate_text(text, direction=direction)
        cache[text] = translated

    return f"{prefix}{translated}{eol}"


def _translate_table_row(line: str, direction: str, cache: dict[str, str]) -> str:
    """
    Translate a markdown pipe-table row cell-by-cell. Separator rows
    (``|---|---|``) are detected and returned unchanged. Leading/
    trailing empty cells (from the row starting/ending with '|') are
    preserved as empty so the column count never shifts.
    """
    body, eol = _split_eol(line)
    stripped = body.strip()
    leading_ws = body[: len(body) - len(body.lstrip())]

    cells = stripped.split("|")
    # A well-formed row is "| a | b |" -> split gives ['', ' a ', ' b ', ''].
    if all(_TABLE_SEP_CELL_RE.match(c) for c in cells if c != ""):
        return line  # separator row — never translate dashes/colons

    out_cells: list[str] = []
    for cell in cells:
        if cell == "":
            out_cells.append("")
            continue
        text = cell.strip()
        if not text:
            out_cells.append(cell)
            continue
        if "<br>" in text:
            # Multi-line cell content (already using the writer's <br>
            # convention) — translate each visual line independently so
            # line breaks survive.
            parts = text.split("<br>")
            trans_parts = []
            for p in parts:
                p_strip = p.strip()
                if not p_strip:
                    trans_parts.append(p)
                    continue
                if p_strip in cache:
                    trans_parts.append(cache[p_strip])
                else:
                    t = translate_text(p_strip, direction=direction)
                    cache[p_strip] = t
                    trans_parts.append(t)
            translated = "<br>".join(trans_parts)
        else:
            if text in cache:
                translated = cache[text]
            else:
                translated = translate_text(text, direction=direction)
                cache[text] = translated
        out_cells.append(f" {translated} ")

    return leading_ws + "|".join(out_cells) + eol


def translate_ocr_markdown(
    text: str,
    *,
    direction: str = "vi->en",
    progress_every: int = 20,
    page_label: str | None = None,
) -> str:
    """
    Translate the sentence content of an OCR markdown string while
    leaving every tag, page header, separator, image reference, and
    table pipe-structure exactly as it was.

    Works on both raw grounding-tagged markdown and already-clean
    markdown (tags stripped) — tag lines simply won't be found in the
    latter, everything else behaves the same.

    Returns the translated markdown, same overall shape as the input.
    """
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    cache: dict[str, str] = {}

    n_content = sum(1 for ln in lines if _classify(ln) in ("content", "table_row"))

    if page_label:
        print(f"  {page_label} …")
    _log.info(f"translate_ocr_markdown: {len(lines)} lines, "
              f"{n_content} item(s) to translate ({direction})")
    print(f"  Translating {n_content} items ({direction})")

    done = 0
    for ln in lines:
        kind = _classify(ln)
        if kind in ("blank", "tag", "page_header", "hr", "image"):
            out.append(ln)
            continue
        if kind == "table_row":
            out.append(_translate_table_row(ln, direction, cache))
        else:
            out.append(_translate_content_line(ln, direction, cache))
        done += 1
        if progress_every and done % progress_every == 0:
            print(f"  {done}/{n_content}")

    return "".join(out)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    parser = argparse.ArgumentParser(
        description=(
            "Translate a DeepSeek-OCR-2 end2end markdown page (raw "
            "grounding-tagged or already-clean) while preserving its "
            "exact tag/page/table/structure layout."
        ),
    )
    parser.add_argument("source", help="Path to the OCR markdown file")
    parser.add_argument("direction", nargs="?", default="vi->en",
                        choices=["vi->en", "en->vi"])
    parser.add_argument("output", nargs="?", default=None,
                        help="Output path (default: <source>_translated.md)")
    args = parser.parse_args()

    src = Path(args.source)
    if not src.exists():
        print(f"[ERROR] File not found: {src}")
        sys.exit(1)

    raw_text = src.read_text(encoding="utf-8")
    translated = translate_ocr_markdown(raw_text, direction=args.direction)

    out_path = Path(args.output) if args.output else src.with_name(
        src.stem.replace("_raw", "") + "_translated.md"
    )
    out_path.write_text(translated, encoding="utf-8")
    print(f"\nDone → {out_path}  ({out_path.stat().st_size // 1024} KB)")
