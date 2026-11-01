"""
core/table_docx_writer.py
=========================

Span-aware DOCX rendering for tables and pages — replaces the lossy
``markdown → docx`` round-trip in the default pipeline.

Why this exists
---------------
``_table_to_markdown`` in the old code emits pipe rows that cannot encode
rowspan / colspan; the DOCX writer then re-parses them into a flat 1x1 grid.
Spans painstakingly detected by ``infer_rowspans_by_border`` and TATR were
silently lost.

Here we render straight from the ``TableRow`` / ``TableCell`` IR (defined in
:mod:`core.table_engines.base`) and apply ``cell.merge()`` for every detected
span. Markdown is still produced — but **only** as a side artefact for the
``*_translated.md`` debug output, not as the DOCX intermediate.
"""

from __future__ import annotations

import logging
from pathlib import Path

_log = logging.getLogger(__name__)


# ── Public: render a single table IR into the open Document ──────────────────

def render_table_to_docx(
    rows,
    doc,
    *,
    use_translated: bool  = True,
    text_width_in:  float = 7.0,
):
    """
    Render a ``list[TableRow]`` into ``doc`` (a ``docx.Document``) with cell
    merges and **content-proportional column widths** (longest text in each
    column drives its share of the available width, with a floor of 0.5"
    and a ceiling of 60% of the page width so a single fat column doesn't
    starve its neighbours). Returns the Word ``Table``.
    """
    from docx.shared import Pt, Inches
    from docx.oxml.ns import qn as _qn
    from docx.oxml import OxmlElement as _OXE

    if not rows:
        return None

    # Compute total columns from anchor cells only.
    ncols = 0
    for r in rows:
        for c in r.cells:
            if c.col_span <= 0:
                continue
            ncols = max(ncols, c.col + c.col_span)
    if ncols == 0:
        return None

    # ── Content-proportional column widths ──────────────────────────────────
    # Estimate each column's "weight" from the longest line of text in any
    # of its anchor (non-merged) cells. Spanned cells contribute their
    # length divided by their col_span so a 3-column merged header doesn't
    # inflate one column. Result is in EMU via Inches() so python-docx
    # layout matches the source visually.
    col_weights = _estimate_column_weights(
        rows, ncols, use_translated=use_translated,
    )
    col_widths_in = _weights_to_widths(
        col_weights,
        total_width_in = text_width_in,
        min_col_in     = 0.50,
        max_col_in     = text_width_in * 0.60,
    )

    tbl = doc.add_table(rows=len(rows), cols=ncols)
    tbl.style       = "Table Grid"
    tbl.autofit     = False
    tbl.allow_autofit = False

    # Fixed table width.
    tblW = _OXE("w:tblW")
    tblW.set(_qn("w:w"),    str(int(text_width_in * 1440)))
    tblW.set(_qn("w:type"), "dxa")
    tbl._tblPr.append(tblW)

    # Apply per-column widths to every cell (Word honours <w:tcW/> per cell).
    for r_idx in range(len(rows)):
        for c_idx in range(ncols):
            tbl.cell(r_idx, c_idx).width = Inches(col_widths_in[c_idx])

    # Pass 1 — write anchor text. If we asked for translation but it's
    # missing/empty (LLM server down, item skipped, ...), gracefully fall
    # back to the original OCR text instead of producing a blank cell.
    #
    # Cell text may contain "\n" (multi-line cells like "Tên thiết bị\n
    # Mã hiệu\n Số lượng"). python-docx renders a "\n" inside
    # ``add_paragraph(text)`` as a literal character, NOT a line break.
    # We split on "\n" and emit a real <w:br/> between each line so the
    # cell renders the same vertical structure as the source.
    for r_idx, row in enumerate(rows):
        for cell in row.cells:
            if cell.row_span == 0 or cell.col_span == 0:
                continue
            target_col = max(0, min(ncols - 1, cell.col))
            wc = tbl.cell(r_idx, target_col)
            if use_translated:
                txt = (cell.translated or "").strip() or (cell.text or "").strip()
            else:
                txt = (cell.text or "").strip()
            # Wipe default empty paragraph so we don't render a blank line.
            for p in list(wc.paragraphs):
                p._element.getparent().remove(p._element)

            # Split on \n into visual lines; an empty cell falls through
            # as a single empty paragraph.
            lines = txt.split("\n") if "\n" in txt else [txt]
            p = wc.add_paragraph()
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after  = Pt(0)
            for i, line in enumerate(lines):
                if i > 0:
                    # add_break() inserts <w:br/> — the real Word
                    # line-break, equivalent to Shift+Enter.
                    p.add_run().add_break()
                run = p.add_run(line)
                run.font.size = Pt(9 if r_idx > 0 else 10)
                if r_idx == 0:
                    run.bold = True

    # Pass 2 — rectangular merges.
    for r_idx, row in enumerate(rows):
        for cell in row.cells:
            if cell.row_span <= 1 and cell.col_span <= 1:
                continue
            r2 = min(len(rows) - 1, r_idx + cell.row_span - 1)
            c2 = min(ncols - 1, cell.col + cell.col_span - 1)
            try:
                tbl.cell(r_idx, cell.col).merge(tbl.cell(r2, c2))
            except Exception as e:
                # Out-of-range or overlapping merge — degrade gracefully.
                _log.debug(f"merge skipped ({r_idx},{cell.col})→({r2},{c2}): {e}")

    return tbl


# ── Column-width helpers ─────────────────────────────────────────────────────

def _estimate_column_weights(
    rows, ncols: int, *, use_translated: bool = True,
) -> list[float]:
    """
    Per-column 'weight' = max longest-line length seen in any anchor cell
    of that column. Spanned cells distribute their length across the
    columns they cover. Used by ``render_table_to_docx`` to size columns
    proportionally to content.
    """
    weights = [1.0] * ncols  # min floor so a fully-empty col still gets width
    for row in rows:
        for c in row.cells:
            if c.row_span == 0 or c.col_span == 0:
                continue
            if use_translated:
                txt = (c.translated or "").strip() or (c.text or "").strip()
            else:
                txt = (c.text or "").strip()
            if not txt:
                continue
            # Longest physical line drives column width; \n in cells is
            # rendered as a line break by render_table_to_docx, so each
            # line is laid out independently.
            longest = max((len(ln) for ln in txt.split("\n")), default=0)
            if longest <= 0:
                continue
            # Distribute the weight across spanned columns.
            share = float(longest) / float(max(1, c.col_span))
            for k in range(c.col_span):
                col_idx = c.col + k
                if 0 <= col_idx < ncols:
                    weights[col_idx] = max(weights[col_idx], share)
    return weights


def _weights_to_widths(
    weights: list[float],
    *,
    total_width_in: float,
    min_col_in:     float,
    max_col_in:     float,
) -> list[float]:
    """
    Normalise ``weights`` into per-column widths (inches) that sum to
    ``total_width_in``, while respecting ``min_col_in`` / ``max_col_in``
    on every column. Excess from clamping is redistributed across the
    unclamped columns proportionally to their weight.
    """
    n = len(weights)
    if n == 0:
        return []
    total_w = sum(weights) or 1.0
    raw = [(w / total_w) * total_width_in for w in weights]

    # Two-pass clamp: anything outside [min,max] is pinned, then the
    # remaining width is re-distributed across un-pinned columns.
    pinned = [None] * n  # type: ignore[var-annotated]
    for i, w in enumerate(raw):
        if w < min_col_in:
            pinned[i] = min_col_in
        elif w > max_col_in:
            pinned[i] = max_col_in

    free_idx = [i for i in range(n) if pinned[i] is None]
    pinned_total = sum(p for p in pinned if p is not None)  # type: ignore[arg-type]
    free_budget  = max(0.0, total_width_in - pinned_total)
    if free_idx:
        free_weight_sum = sum(weights[i] for i in free_idx) or 1.0
        for i in free_idx:
            pinned[i] = (weights[i] / free_weight_sum) * free_budget
            pinned[i] = max(min_col_in, min(max_col_in, pinned[i]))

    # Final scale to compensate for residual clamp drift.
    out = [float(p) for p in pinned]  # type: ignore[arg-type]
    s   = sum(out) or 1.0
    if abs(s - total_width_in) > 0.01:
        out = [w * total_width_in / s for w in out]
    return out


# ── Public: render a full DocumentPage directly into the Document ────────────

def render_page_to_docx(page, doc, *, use_translated: bool = True) -> None:
    """
    Walk a ``DocumentPage`` in reading order and emit the appropriate Word
    elements (heading / paragraph / italic / table / figure placeholder).

    Tables are emitted via :func:`render_table_to_docx` honouring detected
    spans — no markdown round-trip.
    """
    from docx.shared import Pt
    from docx.enum.text import WD_ALIGN_PARAGRAPH

    for blk in sorted(page.blocks, key=lambda b: b.reading_order):

        # ── Figure ───────────────────────────────────────────────────────────
        # If the OCR runner extracted text from inside the figure (banner /
        # logo / header captions), emit that as a centred bold paragraph
        # ABOVE the embedded image so the translated content survives.
        # Then embed the YOLO crop. Falls back to a "[Figure]" placeholder
        # only when no image and no text exist.
        if blk.type == "figure":
            from pathlib import Path as _Path
            from docx.shared import Inches as _Inches

            if use_translated:
                fig_text = (blk.translated_text or "").strip() \
                           or (blk.original_text or "").strip()
            else:
                fig_text = (blk.original_text or "").strip()
            if fig_text:
                # Multi-line figure text (institution + sub-name + doc number)
                # must keep each visible line on its own row. python-docx
                # renders a literal "\n" in add_run(text) as a U+000A
                # character, NOT a Word line break — we need real <w:br/>
                # elements via add_break() between runs.
                fig_lines = [ln.strip() for ln in fig_text.split("\n")
                             if ln.strip()]
                p = doc.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for i, ln in enumerate(fig_lines):
                    if i > 0:
                        p.add_run().add_break()
                    run = p.add_run(ln)
                    run.bold      = True
                    run.font.size = Pt(10)

            fig_path = getattr(blk, "figure_path", "") or ""
            if fig_path and _Path(fig_path).exists():
                p = doc.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                try:
                    # 6-inch cap keeps wide figures inside A4/Letter margins.
                    p.add_run().add_picture(fig_path, width=_Inches(6.0))
                except Exception as e:
                    import logging
                    logging.getLogger(__name__).warning(
                        f"DOCX figure embed failed for {fig_path}: {e}"
                    )
                    if not fig_text:
                        p.add_run("[Figure]")
            elif not fig_text:
                p = doc.add_paragraph("[Figure]")
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            continue

        if blk.skip:
            continue

        # ── Table ────────────────────────────────────────────────────────────
        if blk.type == "table" and blk.table_rows:
            render_table_to_docx(
                blk.table_rows, doc,
                use_translated=use_translated,
            )
            doc.add_paragraph()
            continue

        # ── Text-bearing blocks ──────────────────────────────────────────────
        # Same graceful fallback as cells: if translation is missing, use
        # the original OCR text so the document is not blank.
        if use_translated:
            text = (blk.translated_text or "").strip() \
                or (blk.original_text or "").strip()
        else:
            text = (blk.original_text or "").strip()
        if not text:
            continue

        if blk.type in ("title", "section_header"):
            p   = doc.add_paragraph()
            # Assign a real built-in Word Heading style — not just bold
            # text — so the document has actual structure: it shows up
            # in Word's Navigation Pane and any auto-generated Table of
            # Contents picks it up. "title" (Chương ...) gets the outer
            # level, "section_header" (Điều N. ...) the inner one.
            p.style = doc.styles["Heading 1"] if blk.type == "title" \
                else doc.styles["Heading 2"]
            run = p.add_run(text)
            run.bold      = True
            run.font.size = Pt(11)
            p.paragraph_format.space_before = Pt(4)
            p.paragraph_format.space_after  = Pt(2)
        elif blk.type == "caption":
            p   = doc.add_paragraph()
            run = p.add_run(text)
            run.italic    = True
            run.font.size = Pt(9)
        elif blk.type in ("page_footer", "page_header"):
            # Running header/footer lines (e.g. "CÔNG BÁO/Số .../Ngày
            # ..." + page number) — render small, italic, centred, like
            # the source document's own page furniture. Previously
            # "page_header" was bundled with the heading branch above,
            # which rendered it as a bold heading; it isn't one.
            p   = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = p.add_run(text)
            run.italic    = True
            run.font.size = Pt(9)
        else:
            p   = doc.add_paragraph()
            run = p.add_run(text)
            run.font.size = Pt(10)
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after  = Pt(2)


# ── Public: full pipeline-IR → DOCX file ─────────────────────────────────────

def pages_to_docx(
    pages,
    output_path: str | Path,
    *,
    use_translated: bool = True,
) -> Path:
    """
    Render an ordered list of ``DocumentPage`` IR objects into a single DOCX,
    preserving table spans. Used by the default pipeline; the legacy
    ``markdown_to_docx`` path remains available for the ``--table-engine
    docling`` flag.
    """
    from docx import Document
    from docx.shared import Inches
    from docx.enum.text import WD_BREAK_TYPE
    from docx.oxml.ns import qn as _qn

    doc = Document()
    for section in doc.sections:
        section.top_margin    = Inches(0.75)
        section.bottom_margin = Inches(0.75)
        section.left_margin   = Inches(0.75)
        section.right_margin  = Inches(0.75)

    body = doc.element.body
    for elem in list(body):
        if elem.tag != _qn("w:sectPr"):
            body.remove(elem)

    for page_idx, page in enumerate(pages):
        if page_idx > 0:
            para = doc.add_paragraph()
            para.add_run().add_break(WD_BREAK_TYPE.PAGE)
        render_page_to_docx(page, doc, use_translated=use_translated)

    output_path = Path(output_path)
    doc.save(str(output_path))
    _log.info(f"DOCX saved (span-aware): {output_path}")
    return output_path


__all__ = ["render_table_to_docx", "render_page_to_docx", "pages_to_docx"]