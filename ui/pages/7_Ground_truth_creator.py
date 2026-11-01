"""
ui/pages/5_ground_truth_creator_ui.py
======================================

Công cụ tạo Ground Truth (GT) theo block — dùng để đánh giá chất lượng
OCR/dịch của pipeline `ocr-docs-recovery`.

    * Dò REPO ROOT ngược lên từ vị trí file cho tới khi thấy `core/` +
      `output/` cùng cấp — không phụ thuộc file này nằm sâu bao nhiêu cấp.
    * Một bộ chọn phiên duy nhất. Mỗi phiên = 1 thư mục con trong `output/`
      (vd `output/vanban1_20260722_104034/`) chứa `<stem>_ocr.md` +
      `<stem>_ocr.pdf` / `<stem>_translated.pdf`. Quét ĐỆ QUY toàn bộ
      `output/` (rglob) để không bỏ sót phiên nằm sâu hơn 1 cấp.
    * Hiển thị PDF THẬT đã được pipeline xuất sẵn (ưu tiên bản dịch, rồi
      tới bản OCR gốc) qua iframe base64.
    * GT lưu THEO TỪNG THƯ MỤC PHIÊN (`<run_dir>/ground_truth/page_XXXX.json`)
      thay vì gộp chung theo `stem` — tránh đụng độ khi 2 lần chạy trùng
      tên nguồn nhưng khác thư mục.
    * Nếu phiên có bật `--debug-dump`, vẫn tự nạp sẵn block từ `ir.json`
      (đỡ gõ tay); nếu không có thì bắt đầu từ 1 block rỗng.
    * BẢNG (table_html): editor dạng LƯỚI (giống Excel — thêm/xoá hàng,
      gõ trực tiếp vào ô) qua `st.data_editor`, KHÔNG bắt người dùng gõ
      HTML tay. Hỗ trợ Ô GỘP (rowspan/colspan): chọn vùng hàng/cột rồi
      bấm "Gộp"; nội dung của vùng gộp lấy từ ô góc trên-trái. HTML được
      tự sinh khi lưu — JSON xuất ra CHỈ có `content` (HTML) như schema
      gốc, không lưu thêm trường phụ nào; khi mở lại trang, hệ thống tự
      PARSE NGƯỢC từ `content` (đọc rowspan/colspan) để dựng lại lưới +
      ô gộp cho người dùng sửa tiếp.
"""

from __future__ import annotations

import base64
import datetime
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st


# ── Dò REPO ROOT bằng cách đi ngược lên cho tới khi thấy core/ + output/ ────
def _find_repo_root(start: Path) -> Path:
    cur = start
    for _ in range(6):
        if (cur / "core").is_dir() or (cur / "output").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return start.parent.parent


_THIS_DIR  = Path(__file__).resolve().parent
_REPO_ROOT = _find_repo_root(_THIS_DIR)
_CORE_DIR  = _REPO_ROOT / "core"
_UI_ROOT   = _THIS_DIR.parent  # ui/
for _p in (_CORE_DIR, _REPO_ROOT, _UI_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from theme import apply_page_config, inject_base_css, render_sidebar, render_page_header, icon

apply_page_config("GT Creator", _UI_ROOT)
inject_base_css()
render_sidebar(active="gt")

st.markdown("""
    <style>
    .vnpt-debug-box {
        font-size: 11px; opacity: 0.6; line-height: 1.7; margin-top: 4px;
    }
    .vnpt-debug-box code { font-size: 10.5px; }

    /* ── Themed alert boxes (replace bare st.info/warning/error) ─── */
    .vnpt-alert {
        display: flex; gap: 11px; align-items: flex-start;
        border-radius: var(--vnpt-radius-md);
        border: 1px solid var(--border-color);
        padding: 13px 15px; margin: 6px 0 4px;
        font-size: 13.3px; line-height: 1.55;
    }
    .vnpt-alert svg { flex-shrink: 0; margin-top: 1px; }
    .vnpt-alert b { font-weight: 700; }
    .vnpt-alert-info    { background: var(--vnpt-blue-tint);  color: var(--vnpt-blue); }
    .vnpt-alert-info    b { color: var(--text-color); }
    .vnpt-alert-success { background: var(--vnpt-green-tint); color: var(--vnpt-green); }
    .vnpt-alert-success b { color: var(--text-color); }
    .vnpt-alert-warn    { background: var(--vnpt-amber-tint); color: var(--vnpt-amber); }
    .vnpt-alert-warn    b { color: var(--text-color); }
    .vnpt-alert-error   { background: var(--vnpt-red-tint);   color: var(--vnpt-red); }
    .vnpt-alert-error   b { color: var(--text-color); }
    .vnpt-alert-body { color: var(--text-color); opacity: 0.82; }

    /* ── Empty-state panel (no sessions yet) ──────────────────────── */
    .vnpt-empty-state {
        text-align: center; padding: 46px 24px 40px;
        border: 1.5px dashed var(--border-color);
        border-radius: var(--vnpt-radius-lg);
        background: rgba(255,255,255,0.015);
    }
    .vnpt-empty-state .e-icon {
        width: 52px; height: 52px; border-radius: 50%;
        display: inline-flex; align-items: center; justify-content: center;
        background: var(--vnpt-blue-tint); color: var(--vnpt-blue);
        margin-bottom: 16px;
    }
    .vnpt-empty-state .e-title {
        font-size: 16.5px; font-weight: 800; color: var(--text-color); margin-bottom: 8px;
    }
    .vnpt-empty-state .e-body {
        font-size: 13.3px; color: var(--text-color); opacity: 0.65;
        max-width: 480px; margin: 0 auto; line-height: 1.65;
    }
    .vnpt-empty-state code {
        font-size: 12px; background: var(--vnpt-blue-tint); padding: 1px 6px; border-radius: 5px;
    }

    /* ── Control-bar labels above the session/page selectors ──────── */
    .vnpt-ctrl-label {
        display: flex; align-items: center; gap: 7px;
        font-size: 11px; font-weight: 800; letter-spacing: 0.9px;
        text-transform: uppercase; color: var(--text-color); opacity: 0.55;
        margin-bottom: 6px;
    }
    .vnpt-ctrl-label svg { color: var(--vnpt-blue); opacity: 1; }

    /* ── Block-type badges inside each GT block card ──────────────── */
    .vnpt-type-badge {
        display: inline-flex; align-items: center; gap: 6px;
        font-size: 11px; font-weight: 800; letter-spacing: 0.4px;
        padding: 4px 10px; border-radius: 999px; text-transform: uppercase;
    }
    .vnpt-type-text        { background: var(--vnpt-blue-tint);  color: var(--vnpt-blue); }
    .vnpt-type-image       { background: var(--vnpt-amber-tint); color: var(--vnpt-amber); }
    .vnpt-type-figure_title{ background: var(--vnpt-amber-tint); color: var(--vnpt-amber); }
    .vnpt-type-table       { background: var(--vnpt-green-tint); color: var(--vnpt-green); }
    .vnpt-type-title       { background: rgba(148,99,255,0.14); color: #8B5CF6; }
    .vnpt-type-sub_title   { background: rgba(148,99,255,0.14); color: #8B5CF6; }

    /* ── Block card header row ─────────────────────────────────────── */
    .vnpt-block-head {
        display: flex; align-items: center; gap: 10px; margin-bottom: 4px;
    }
    .vnpt-block-head .vnpt-block-order { margin-right: 0; }

    /* ── "Added missing" flag panel ───────────────────────────────── */
    .vnpt-flag-panel {
        border-radius: var(--vnpt-radius-sm);
        border: 1px solid var(--border-color);
        padding: 2px 4px; margin-top: 10px;
    }

    /* ── Live pulse dot for the JSON preview panel ────────────────── */
    .vnpt-live-dot {
        width: 7px; height: 7px; border-radius: 50%; display: inline-block;
        background: var(--vnpt-green); box-shadow: 0 0 0 3px var(--vnpt-green-tint);
        animation: vnpt-pulse 1.6s ease-in-out infinite;
    }
    @keyframes vnpt-pulse {
        0%, 100% { opacity: 1; } 50% { opacity: 0.4; }
    }
    .vnpt-panel-head {
        display: flex; align-items: center; justify-content: space-between;
        margin-bottom: 2px;
    }
    .vnpt-panel-head .ph-title {
        display: flex; align-items: center; gap: 8px;
        font-size: 16.5px; font-weight: 800; color: var(--text-color);
    }
    .vnpt-panel-head .ph-live {
        display: inline-flex; align-items: center; gap: 6px;
        font-size: 10.5px; font-weight: 800; letter-spacing: 1px;
        color: var(--vnpt-green); text-transform: uppercase;
    }
    </style>
""", unsafe_allow_html=True)


def alert(kind: str, title: str, body: str = "") -> None:
    """Themed replacement for st.info/success/warning/error — matches the
    VNPT palette instead of Streamlit's default alert colors."""
    icon_map = {"info": "info", "success": "check", "warn": "alert", "error": "cross"}
    body_html = f'<div class="vnpt-alert-body">{body}</div>' if body else ""
    st.markdown(
        f'<div class="vnpt-alert vnpt-alert-{kind}">'
        f'{icon(icon_map.get(kind, "info"), 18)}'
        f'<div><b>{title}</b>{body_html}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

OUTPUT_ROOT = _REPO_ROOT / "output"
PAGE_SEP    = "\n\n---\n\n"

with st.sidebar:
    st.markdown(
        f"""
        <div class="vnpt-debug-box">
            Repo root dò được:<br><code>{_REPO_ROOT}</code><br><br>
            Output root: <code>{OUTPUT_ROOT}</code><br>
            {"Tồn tại" if OUTPUT_ROOT.exists() else "Không tồn tại"}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════
# Data-access
# ══════════════════════════════════════════════════════════════════════════

def discover_sessions() -> list[dict]:
    """Quét output/ (đệ quy) tìm các phiên đã chạy. Mỗi phiên = 1 file
    <stem>_ocr.md; thư mục cha của file đó là 'run_dir' riêng của phiên,
    dùng làm gốc để tìm PDF/ir.json/lưu GT — không giả định output phẳng."""
    sessions = []
    if not OUTPUT_ROOT.exists():
        return sessions
    for md_path in OUTPUT_ROOT.rglob("*_ocr.md"):
        run_dir = md_path.parent
        stem    = md_path.name[: -len("_ocr.md")]

        original_pdf = OUTPUT_ROOT / "_uploads" / f"{stem}.pdf"
        ocr_pdf      = run_dir / f"{stem}_ocr.pdf"

        if original_pdf.exists():
            pdf_file = original_pdf
        elif ocr_pdf.exists():
            pdf_file = ocr_pdf
        else:
            pdf_file = next((p for p in run_dir.glob("*.pdf") if "translated" not in p.name), None)

        sessions.append({
            "run_dir":       run_dir,
            "stem":          stem,
            "ocr_md":        md_path,
            "translated_md": run_dir / f"{stem}_translated.md",
            "pdf_file":      pdf_file,
            "mtime":         md_path.stat().st_mtime,
            "label": (run_dir.name if run_dir != OUTPUT_ROOT else stem),
        })
    sessions.sort(key=lambda s: s["mtime"], reverse=True)
    return sessions


@st.cache_data(show_spinner=False)
def load_pages_from_md(md_path_str: str, _mtime: float) -> list[str]:
    text = Path(md_path_str).read_text(encoding="utf-8")
    return text.split(PAGE_SEP)


def display_pdf(file_path: Path, height: int = 480, key: str | None = None) -> None:
    b64 = base64.b64encode(file_path.read_bytes()).decode("utf-8")
    st.markdown(
        f'<iframe src="data:application/pdf;base64,{b64}" '
        f'width="100%" height="{height}" type="application/pdf" '
        f'style="border:1px solid var(--border-color); border-radius:var(--vnpt-radius-md);">'
        f'</iframe>',
        unsafe_allow_html=True,
    )


def blocks_from_ir(run_dir: Path, page_idx: int) -> list[dict] | None:
    for debug_root in (run_dir / "debug", run_dir):
        ir_path = debug_root / f"page_{page_idx:02d}" / "ir.json"
        if ir_path.exists():
            break
    else:
        return None
    try:
        ir = json.loads(ir_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    blocks: list[dict] = []
    for b in sorted(ir.get("blocks", []), key=lambda x: x.get("reading_order", 0)):
        if b.get("skip"):
            continue
        ir_type = b.get("type", "text")
        if ir_type == "table":
            btype, ckind = "table", "table_html"
        elif ir_type == "figure":
            btype, ckind = "image", "text"
        else:
            btype, ckind = "text", "text"
        blocks.append({
            "type":          btype,
            "content_kind":  ckind,
            "content":       b.get("original_text", ""),
            "added_missing": False,
        })
    return blocks or None


# ══════════════════════════════════════════════════════════════════════════
# Bảng: chuyển đổi qua lại giữa lưới editor (+ ô gộp) và HTML lưu trong JSON
# ══════════════════════════════════════════════════════════════════════════
# Người dùng KHÔNG cần biết HTML tồn tại — chỉ thao tác trên lưới + nút
# "Gộp"/"Bỏ gộp"; hệ thống tự sinh/parse HTML ở phía sau.
#
# Một "merge" (ô gộp) là dict {"r0","c0","r1","c1"} — inclusive, 0-based —
# mô tả 1 vùng chữ nhật hàng r0..r1, cột c0..c1 được gộp thành 1 ô, nội
# dung hiển thị lấy từ ô góc trên-trái (r0, c0). Danh sách merge được lưu
# kèm grid trong JSON (khoá "table_grid" / "table_merges") để mở lại trang
# vẫn sửa tiếp được, không chỉ có mỗi HTML đã dựng sẵn.

def _escape_html(s: str) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
                  .replace(">", "&gt;"))


def _html_table_to_grid_and_merges(html: str) -> tuple[list[list[str]], list[dict]]:
    """Parse 1 bảng HTML (có thể có rowspan/colspan) thành lưới đặc
    (dense grid) + danh sách ô gộp. Dùng thuật toán "pending theo cột"
    giống hệt cách project đã dùng ở mineru_engine.py /
    grounding_docx_builder.py để đảm bảo nhất quán."""
    if not html or "<table" not in html.lower():
        return [], []
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return [], []
    try:
        soup  = BeautifulSoup(html, "html.parser")
        table = soup.find("table")
        if table is None:
            return [], []
        trs = table.find_all("tr")
        pending: dict[int, int] = {}
        grid:    list[list[str]] = []
        merges:  list[dict] = []

        for r_idx, tr in enumerate(trs):
            row: list[str] = []
            col_idx = 0
            tag_iter = iter(tr.find_all(["td", "th"]))
            while True:
                if pending.get(col_idx, 0) > 0:
                    row.append("")
                    pending[col_idx] -= 1
                    if pending[col_idx] <= 0:
                        del pending[col_idx]
                    col_idx += 1
                    continue
                try:
                    tag = next(tag_iter)
                except StopIteration:
                    break
                text       = tag.get_text(strip=True)
                colspan    = int(tag.get("colspan", 1) or 1)
                rowspan    = int(tag.get("rowspan", 1) or 1)
                anchor_col = col_idx
                row.append(text)
                for _e in range(1, colspan):
                    row.append("")
                if colspan > 1 or rowspan > 1:
                    merges.append({
                        "r0": r_idx, "c0": anchor_col,
                        "r1": r_idx + rowspan - 1,
                        "c1": anchor_col + colspan - 1,
                    })
                if rowspan > 1:
                    for extra_c in range(colspan):
                        pending[anchor_col + extra_c] = pending.get(
                            anchor_col + extra_c, 0) + (rowspan - 1)
                col_idx += colspan
            grid.append(row)

        max_len = max((len(r) for r in grid), default=0)
        grid = [r + [""] * (max_len - len(r)) for r in grid]
        return grid, merges
    except Exception:
        return [], []


def _markdown_pipe_to_grid(text: str) -> list[list[str]]:
    """Parse bảng Markdown dạng '| a | b |' (OCR từ docling/mineru/deepseek
    thường ra dạng này) thành lưới text — không có thông tin ô gộp."""
    lines = [ln for ln in text.splitlines() if ln.strip().startswith("|")]
    data_lines = [ln for ln in lines
                  if not all(c in "|-: \t" for c in ln.strip())]
    grid = []
    for ln in data_lines:
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        grid.append(cells)
    return grid


def _parse_pasted_table_with_merges(text: str) -> tuple[list[list[str]], list[dict]]:
    text = (text or "").strip()
    if not text:
        return [], []
    if "<table" in text.lower():
        return _html_table_to_grid_and_merges(text)
    grid = _markdown_pipe_to_grid(text)
    if grid:
        return grid, []
    # Fallback cuối: mỗi dòng là 1 hàng, tách cột bằng Tab nếu có.
    fallback = [[c.strip() for c in ln.split("\t")]
                for ln in text.splitlines() if ln.strip()]
    return fallback, []


def _pad_grid(grid: list[list[str]], n_cols: int) -> list[list[str]]:
    out = []
    for row in grid:
        row = list(row)
        if len(row) < n_cols:
            row = row + [""] * (n_cols - len(row))
        else:
            row = row[:n_cols]
        out.append(row)
    return out


def _grid_to_dataframe(grid: list[list[str]], n_cols: int) -> pd.DataFrame:
    if not grid:
        grid = [["" for _ in range(n_cols)] for _ in range(3)]
    grid = _pad_grid(grid, max(1, n_cols))
    cols = [f"Cột {c + 1}" for c in range(max(1, n_cols))]
    return pd.DataFrame(grid, columns=cols)


def _grid_and_merges_to_html(
    df: pd.DataFrame, merges: list[dict], *, header_row: bool,
) -> str:
    records = df.fillna("").astype(str).values.tolist()
    n_rows  = len(records)
    n_cols  = len(records[0]) if records else 0

    anchor_span: dict[tuple[int, int], dict] = {}
    covered:     set[tuple[int, int]] = set()
    for m in merges:
        r0, c0 = m["r0"], m["c0"]
        r1 = min(m["r1"], n_rows - 1)
        c1 = min(m["c1"], n_cols - 1)
        if r1 < r0 or c1 < c0 or r0 >= n_rows or c0 >= n_cols:
            continue
        anchor_span[(r0, c0)] = {"rowspan": r1 - r0 + 1, "colspan": c1 - c0 + 1}
        for rr in range(r0, r1 + 1):
            for cc in range(c0, c1 + 1):
                if (rr, cc) != (r0, c0):
                    covered.add((rr, cc))

    rows_html = []
    for r_idx, row in enumerate(records):
        cells_html = []
        for c_idx, val in enumerate(row):
            if (r_idx, c_idx) in covered:
                continue
            tag        = "th" if (header_row and r_idx == 0) else "td"
            span_attrs = ""
            info = anchor_span.get((r_idx, c_idx))
            if info:
                if info["rowspan"] > 1:
                    span_attrs += f' rowspan="{info["rowspan"]}"'
                if info["colspan"] > 1:
                    span_attrs += f' colspan="{info["colspan"]}"'
            cells_html.append(f"<{tag}{span_attrs}>{_escape_html(val)}</{tag}>")
        rows_html.append("  <tr>" + "".join(cells_html) + "</tr>")
    return "<table>\n" + "\n".join(rows_html) + "\n</table>"


def _render_table_editor(block: dict, cache_key, i: int) -> tuple[str, list[list[str]], list[dict]]:
    """Lưới editor cho block bảng — trả về (html, grid, merges).
    Không có ô nào bắt người dùng gõ HTML tay."""
    df_key     = f"table_df_{cache_key}_{i}"
    ncols_key  = f"table_ncols_{cache_key}_{i}"
    header_key = f"table_header_{cache_key}_{i}"
    merges_key = f"table_merges_{cache_key}_{i}"
    paste_key  = f"table_paste_{cache_key}_{i}"

    # Khởi tạo state lần đầu: ưu tiên grid/merges đã lưu trong block (từ
    # GT đã lưu trước đó), rồi mới tới parse ngược từ HTML cũ, cuối cùng
    # là lưới trống.
    if df_key not in st.session_state:
        saved_grid   = block.get("table_grid")
        saved_merges = block.get("table_merges")
        if saved_grid:
            grid, merges = saved_grid, (saved_merges or [])
        else:
            grid, merges = _html_table_to_grid_and_merges(block.get("content", ""))
        n_cols = max((len(r) for r in grid), default=3) or 3
        st.session_state[ncols_key]  = n_cols
        st.session_state[df_key]     = _grid_to_dataframe(grid, n_cols)
        st.session_state[merges_key] = merges

    with st.expander(
        "Dán nội dung OCR gốc của bảng (Markdown hoặc HTML) để nạp tự động",
        icon=":material/content_paste:",
    ):
        pasted = st.text_area(
            "Dán nguyên văn phần bảng copy từ ô OCR ở khu vực đối chiếu bên trên:",
            key=paste_key, height=80,
        )
        if st.button("Nạp vào lưới", key=f"parse_{cache_key}_{i}",
                     icon=":material/download:"):
            grid, merges = _parse_pasted_table_with_merges(pasted)
            if grid:
                n_cols = max(len(r) for r in grid)
                st.session_state[ncols_key]  = n_cols
                st.session_state[df_key]     = _grid_to_dataframe(grid, n_cols)
                st.session_state[merges_key] = merges
                # Xoá state cũ của widget data_editor — nếu không, lần
                # render sau vẫn đọc dữ liệu CŨ từ session_state[key] của
                # chính widget thay vì dataframe mới vừa gán ở trên.
                st.session_state.pop(f"editor_{cache_key}_{i}", None)
                msg = f"Đã nạp {len(grid)} hàng × {n_cols} cột"
                if merges:
                    msg += f", {len(merges)} ô gộp"
                alert("success", msg + ".")
                st.rerun()
            else:
                alert("warn", "Không nhận diện được bảng",
                      "Không tìm thấy cấu trúc bảng trong nội dung đã dán.")

    ctrl1, ctrl2 = st.columns([1, 2])
    with ctrl1:
        n_cols = st.number_input(
            "Số cột", min_value=1, max_value=20,
            key=ncols_key,
        )
    with ctrl2:
        header_row = st.checkbox(
            "Hàng đầu tiên là tiêu đề (in đậm)", value=True, key=header_key,
        )

    current_df = st.session_state[df_key]
    if current_df.shape[1] != n_cols:
        current_df = _grid_to_dataframe(current_df.values.tolist(), n_cols)
        st.session_state[df_key] = current_df
        st.session_state.pop(f"editor_{cache_key}_{i}", None)
        # Số cột đổi -> vị trí các ô gộp cũ có thể lệch, xoá cho an toàn
        # thay vì để merge trỏ sai ô.
        st.session_state[merges_key] = []

    st.caption(
        "Sửa trực tiếp vào ô như Excel. Với vùng sẽ gộp: chỉ gõ nội dung "
        "vào ô góc trên-trái, các ô còn lại trong vùng để trống."
    )
    edited_df = st.data_editor(
        current_df,
        num_rows="dynamic",
        width='stretch',
        key=f"editor_{cache_key}_{i}",
    )
    st.session_state[df_key] = edited_df

    n_rows_now = len(edited_df)
    n_cols_now = edited_df.shape[1]

    # ── Quản lý ô gộp ────────────────────────────────────────────────
    with st.expander(
        f"Quản lý ô gộp ({len(st.session_state[merges_key])} vùng)",
        icon=":material/grid_view:",
    ):
        st.caption(
            "Chọn vùng theo số thứ tự hàng/cột (bắt đầu từ 1) rồi bấm Gộp. "
            "Nội dung hiển thị lấy từ ô góc trên-trái của vùng."
        )
        mc1, mc2, mc3, mc4, mc5 = st.columns([1, 1, 1, 1, 1.2])
        with mc1:
            m_r0 = st.number_input("Hàng từ", min_value=1,
                                   max_value=max(1, n_rows_now), value=1,
                                   key=f"mr0_{cache_key}_{i}")
        with mc2:
            m_r1 = st.number_input("Hàng đến", min_value=1,
                                   max_value=max(1, n_rows_now), value=1,
                                   key=f"mr1_{cache_key}_{i}")
        with mc3:
            m_c0 = st.number_input("Cột từ", min_value=1,
                                   max_value=n_cols_now, value=1,
                                   key=f"mc0_{cache_key}_{i}")
        with mc4:
            m_c1 = st.number_input("Cột đến", min_value=1,
                                   max_value=n_cols_now, value=1,
                                   key=f"mc1_{cache_key}_{i}")
        with mc5:
            st.write("")
            st.write("")
            if st.button("Gộp", key=f"domerge_{cache_key}_{i}",
                         width='stretch', icon=":material/call_merge:"):
                r0, r1 = sorted((int(m_r0) - 1, int(m_r1) - 1))
                c0, c1 = sorted((int(m_c0) - 1, int(m_c1) - 1))
                if r0 == r1 and c0 == c1:
                    alert("warn", "Vùng chỉ có 1 ô — không cần gộp.")
                else:
                    def _overlap(mm: dict) -> bool:
                        return not (mm["r1"] < r0 or mm["r0"] > r1
                                   or mm["c1"] < c0 or mm["c0"] > c1)
                    new_merges = [m for m in st.session_state[merges_key]
                                 if not _overlap(m)]
                    new_merges.append({"r0": r0, "c0": c0, "r1": r1, "c1": c1})
                    st.session_state[merges_key] = new_merges
                    st.rerun()

        if st.session_state[merges_key]:
            for mi, m in enumerate(st.session_state[merges_key]):
                mcol1, mcol2 = st.columns([4, 1])
                with mcol1:
                    st.caption(
                        f"Hàng {m['r0'] + 1}–{m['r1'] + 1}, "
                        f"Cột {m['c0'] + 1}–{m['c1'] + 1}"
                    )
                with mcol2:
                    if st.button("Bỏ gộp", key=f"unmerge_{cache_key}_{i}_{mi}",
                                 width='stretch',
                                 icon=":material/call_split:"):
                        st.session_state[merges_key].pop(mi)
                        st.rerun()
        else:
            st.caption("Chưa có ô gộp nào.")

    html = _grid_and_merges_to_html(
        edited_df, st.session_state[merges_key], header_row=header_row,
    )
    with st.expander("Xem trước bảng đã dựng", icon=":material/table_chart:"):
        st.markdown(html, unsafe_allow_html=True)

    return html, edited_df.values.tolist(), st.session_state[merges_key]


# ══════════════════════════════════════════════════════════════════════════
# Lưu / đọc GT
# ══════════════════════════════════════════════════════════════════════════

def gt_file_path(run_dir: Path, page_num: int) -> Path:
    return run_dir / "ground_truth" / f"page_{page_num:04d}.json"


def load_saved_gt(run_dir: Path, page_num: int) -> list[dict] | None:
    p = gt_file_path(run_dir, page_num)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data["pages"][0]["blocks"]
    except Exception:
        return None


def _block_to_entry(idx: int, blk: dict) -> dict:
    """Chuyển 1 block editor -> 1 entry trong JSON xuất ra. Dùng chung
    cho cả bản xem trước (master_json) và lúc lưu thật (save_gt) để 2 nơi
    không lệch nhau.

    Chỉ xuất đúng 5 trường theo schema gốc (order/type/content_kind/
    content/verified/added_missing). Với block bảng, cấu trúc ô gộp đã
    được mã hoá đầy đủ trong `content` (HTML có rowspan/colspan) nên
    KHÔNG cần lưu thêm table_grid/table_merges — lúc mở lại trang,
    _render_table_editor() tự parse ngược từ `content` để dựng lại lưới
    + ô gộp (xem _html_table_to_grid_and_merges)."""
    return {
        "order":         idx + 1,
        "type":          blk["type"],
        "content_kind":  blk["content_kind"],
        "content":       blk["content"],
        "verified":      True,
        "added_missing": blk["added_missing"],
    }


def save_gt(run_dir: Path, stem: str, page_num: int, blocks: list[dict]) -> Path:
    final_blocks = [_block_to_entry(idx, blk) for idx, blk in enumerate(blocks)]
    payload = {"stem": stem, "pages": [{
        "page_num": page_num,
        "n_blocks": len(final_blocks),
        "blocks":   final_blocks,
    }]}
    p = gt_file_path(run_dir, page_num)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def merge_all_saved_pages(run_dir: Path, stem: str) -> dict:
    gt_dir = run_dir / "ground_truth"
    pages_out = []
    if gt_dir.exists():
        for p in sorted(gt_dir.glob("page_*.json")):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                pages_out.extend(data.get("pages", []))
            except Exception:
                continue
    pages_out.sort(key=lambda p: p.get("page_num", 0))
    return {"stem": stem, "pages": pages_out}


# ══════════════════════════════════════════════════════════════════════════
# UI — Bộ chọn PHIÊN + TRANG (trên cùng)
# ══════════════════════════════════════════════════════════════════════════

render_page_header(
    "GT CREATOR",
    "Tạo và quản lý Ground Truth theo từng block để đánh giá chất lượng OCR.",
    eyebrow="Ground truth annotation",
)
st.write("")

sessions = discover_sessions()
if not sessions:
    st.markdown(
        f"""
        <div class="vnpt-empty-state">
            <div class="e-icon">{icon('folder', 26, 1.7)}</div>
            <div class="e-title">Chưa có phiên xử lý nào</div>
            <div class="e-body">
                Chưa tìm thấy phiên nào trong <code>output/</code> (cần ít nhất
                một file <code>&lt;tên&gt;_ocr.md</code>, kể cả nằm trong thư
                mục con). Hãy chạy trang <b>OCR &amp; Dịch thuật</b> trước rồi
                quay lại đây.<br><br>
                Đường dẫn đang quét: <code>{OUTPUT_ROOT}</code>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

with st.container(border=True):
    sel_col1, sel_col2, sel_col3 = st.columns([3, 1, 1])

    session_labels = [
        f"{s['label']}  ·  "
        + datetime.datetime.fromtimestamp(s["mtime"]).strftime("%d/%m/%Y %H:%M")
        for s in sessions
    ]
    with sel_col1:
        st.markdown(
            f'<div class="vnpt-ctrl-label">{icon("folder", 13)} PHIÊN XỬ LÝ</div>',
            unsafe_allow_html=True,
        )
        sel_idx = st.selectbox(
            "Phiên xử lý", options=list(range(len(sessions))),
            format_func=lambda i: session_labels[i],
            key="gt_session_select", label_visibility="collapsed",
        )
    session = sessions[sel_idx]
    stem    = session["stem"]
    run_dir = session["run_dir"]

    pages   = load_pages_from_md(str(session["ocr_md"]), session["mtime"])
    n_pages = max(1, len(pages))

    with sel_col2:
        st.markdown(
            f'<div class="vnpt-ctrl-label">{icon("grid", 13)} TRANG</div>',
            unsafe_allow_html=True,
        )
        page_num = st.number_input(
            "Trang", min_value=1, max_value=n_pages, value=1, step=1,
            key=f"gt_page_num_{run_dir}", label_visibility="collapsed",
        )
    page_idx = page_num - 1  # 0-based, khớp document_pipeline.py

    with sel_col3:
        st.markdown(
            f'<div class="vnpt-ctrl-label">{icon("layers", 13)} TỔNG SỐ TRANG</div>',
            unsafe_allow_html=True,
        )
        st.metric("Tổng số trang", n_pages, label_visibility="collapsed")

    st.markdown(
        f"""
        <div style="margin-top:2px">
            <span class="vnpt-tag">{run_dir.name}</span>
            <span class="vnpt-tag" style="margin-left:6px">{stem}</span>
            <span class="vnpt-badge vnpt-badge-info" style="margin-left:6px">
                {icon('scan', 11)} Trang {page_num} / {n_pages}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")

# ── Nạp block cho đúng (phiên, trang) hiện tại — cache theo cache_key ───────
cache_key = (str(run_dir), page_num)
if "gt_cache" not in st.session_state:
    st.session_state.gt_cache = {}

if cache_key not in st.session_state.gt_cache:
    loaded = (
        load_saved_gt(run_dir, page_num)
        or blocks_from_ir(run_dir, page_idx)
        or [{"type": "text", "content_kind": "text",
             "content": "", "added_missing": False}]
    )
    st.session_state.gt_cache[cache_key] = loaded


# ══════════════════════════════════════════════════════════════════════════
# KHU VỰC 1: ĐỐI CHIẾU DỮ LIỆU THẬT
# ══════════════════════════════════════════════════════════════════════════

st.markdown(
    f"""
    <div class="vnpt-section-title" style="margin-bottom:2px">
        {icon("compare", 19)} Khu vực Đối chiếu
    </div>
    <div style="font-size:13px; opacity:0.6; margin-bottom:16px;">
        So sánh PDF gốc với bản OCR để phát hiện sai sót trước khi tạo Ground Truth.
    </div>
    """,
    unsafe_allow_html=True,
)
top_col1, top_col2 = st.columns(2)

with top_col1:
    with st.container(border=True):
        st.markdown(
            f'<span class="vnpt-badge vnpt-badge-info">{icon("docx", 12)} File PDF của phiên này</span>',
            unsafe_allow_html=True,
        )
        st.write("")
        if session["pdf_file"] and session["pdf_file"].exists():
            display_pdf(session["pdf_file"], key=f"pdf_{cache_key}")
            st.caption(f"Đang xem: `{session['pdf_file'].name}` (cuộn để tới đúng trang cần đối chiếu)")
        else:
            alert(
                "warn", "Không tìm thấy file PDF",
                f"Không thấy file PDF trong <code>{run_dir}</code>. Kiểm tra "
                "pipeline đã xuất PDF thành công chưa "
                "(<code>docx_to_pdf</code> cần LibreOffice hoặc Word cài sẵn).",
            )

with top_col2:
    with st.container(border=True):
        st.markdown(
            f'<span class="vnpt-badge vnpt-badge-on">{icon("check", 12)} Bản OCR của đúng trang này (copy sang block bên dưới)</span>',
            unsafe_allow_html=True,
        )
        st.write("")
        st.text_area(
            "Kết quả OCR trang hiện tại:",
            value=pages[page_idx] if page_idx < len(pages) else "",
            height=440, disabled=True, key=f"ocr_preview_{cache_key}",
            label_visibility="collapsed",
        )

st.write("")


# ══════════════════════════════════════════════════════════════════════════
# KHU VỰC 2: QUẢN LÝ BLOCK & JSON PREVIEW
# ══════════════════════════════════════════════════════════════════════════

st.markdown(
    f'<div class="vnpt-section-title">{icon("layers", 19)} Trình tạo Ground Truth JSON (Theo Block)</div>',
    unsafe_allow_html=True,
)
bot_col1, bot_col2 = st.columns([1.3, 1])


def add_block():
    st.session_state.gt_cache[cache_key].append({
        "type": "text", "content_kind": "text", "content": "",
        "added_missing": False,
    })


def delete_block(index):
    st.session_state.gt_cache[cache_key].pop(index)


def move_up(index):
    lst = st.session_state.gt_cache[cache_key]
    if index > 0:
        lst[index], lst[index - 1] = lst[index - 1], lst[index]


def move_down(index):
    lst = st.session_state.gt_cache[cache_key]
    if index < len(lst) - 1:
        lst[index], lst[index + 1] = lst[index + 1], lst[index]


# Danh sách type — "sub_title" dùng gạch dưới để khớp đúng quy ước
# _LABEL_TYPE_MAP trong grounding_docx_builder.py ("sub_title" ->
# "section_header"); trước đây file này ghi "sub-title" (gạch nối) nên
# nếu JSON GT từng được dùng để đối chiếu với luồng grounding sẽ không
# khớp nhãn.
_BLOCK_TYPES = ["text", "image", "table", "title", "sub_title", "figure_title"]

with bot_col1:
    with st.container(border=True):
        st.markdown(
            f'<div class="vnpt-section-title" style="margin:0 0 10px">'
            f'{icon("package", 18)} Quản lý danh sách Block</div>',
            unsafe_allow_html=True,
        )

        # Hiển thị trạng thái lưu GT (đã bỏ nút Thêm Block ở đây)
        existing_path = gt_file_path(run_dir, page_num)
        if existing_path.exists():
            st.markdown(
                f'<span class="vnpt-badge vnpt-badge-on">{icon("check", 11)} '
                f'Đã có GT đã lưu · <code>{existing_path.name}</code></span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<span class="vnpt-badge vnpt-badge-neutral">{icon("info", 11)} '
                f'Chưa lưu GT cho trang này</span>',
                unsafe_allow_html=True,
            )

        st.divider()

        blocks = st.session_state.gt_cache[cache_key]

        if not blocks:
            st.markdown(
                f"""
                <div class="vnpt-empty-state" style="padding:32px 20px">
                    <div class="e-icon">{icon('layers', 22, 1.7)}</div>
                    <div class="e-title">Chưa có block nào</div>
                    <div class="e-body">Bấm "Thêm Block Mới" bên dưới để bắt đầu.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        _TYPE_BADGE_ICON = {
            "text": "layers", "image": "image", "figure_title": "image",
            "table": "table", "title": "flag", "sub_title": "flag",
        }

        for i, block in enumerate(blocks):
            _btype = block["type"] if block["type"] in _TYPE_BADGE_ICON else "text"
            st.markdown(
                f'<div class="vnpt-block-card">'
                f'<div class="vnpt-block-head">'
                f'<span class="vnpt-block-order">{i + 1}</span>'
                f'<span class="vnpt-type-badge vnpt-type-{_btype}">'
                f'{icon(_TYPE_BADGE_ICON[_btype], 12)} {block["type"]}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

            head_col2, head_col3 = st.columns([1.5, 0.8])
            with head_col2:
                btn_up, btn_down = st.columns(2)
                with btn_up:
                    st.button("Lên", key=f"up_{cache_key}_{i}",
                              on_click=move_up, args=(i,),
                              icon=":material/keyboard_arrow_up:",
                              disabled=(i == 0), width='stretch')
                with btn_down:
                    st.button("Xuống", key=f"down_{cache_key}_{i}",
                              on_click=move_down, args=(i,),
                              icon=":material/keyboard_arrow_down:",
                              disabled=(i == len(blocks) - 1),
                              width='stretch')
            with head_col3:
                st.button("Xóa", key=f"del_{cache_key}_{i}",
                          on_click=delete_block, args=(i,),
                          icon=":material/delete:",
                          type="secondary", width='stretch')

            # Nếu type hiện tại không còn nằm trong _BLOCK_TYPES (vd dữ
            # liệu cũ có "sub-title" gạch nối) thì chuẩn hoá lại để
            # .index() không crash.
            if block["type"] == "sub-title":
                block["type"] = "sub_title"
            if block["type"] not in _BLOCK_TYPES:
                block["type"] = "text"

            block["type"] = st.selectbox(
                "Định dạng (Type)", _BLOCK_TYPES,
                index=_BLOCK_TYPES.index(block["type"]),
                key=f"type_{cache_key}_{i}",
            )
            # content_kind được suy ra tự động từ type — người dùng KHÔNG
            # cần biết khái niệm HTML/table_html, chỉ cần chọn "table".
            block["content_kind"] = (
                "table_html" if block["type"] == "table" else "text"
            )

            if block["type"] == "table":
                html, grid, merges = _render_table_editor(block, cache_key, i)
                block["content"]      = html
                block["table_grid"]   = grid
                block["table_merges"] = merges
            else:
                block["content"] = st.text_area(
                    "Nội dung (Copy từ OCR sang và sửa lại chữ sai):",
                    value=block["content"], key=f"content_{cache_key}_{i}",
                    height=90,
                )

            _flag_bg = "var(--vnpt-amber-tint)" if block["added_missing"] else "transparent"
            st.markdown(
                f'<div class="vnpt-flag-panel" style="background:{_flag_bg}">',
                unsafe_allow_html=True,
            )
            block["added_missing"] = st.checkbox(
                "Đánh dấu: Block này có trên PDF nhưng bị OCR bỏ sót "
                "hoàn toàn (Added Missing)",
                value=block["added_missing"], key=f"missing_{cache_key}_{i}",
            )
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)
            
        # NÚT THÊM BLOCK MỚI ĐÃ ĐƯỢC CHUYỂN XUỐNG DƯỚI CÙNG
        st.write("") # Thêm một chút khoảng trống
        st.button("Thêm Block Mới", on_click=add_block,
                  type="primary", width='stretch',
                  icon=":material/add:",
                  key=f"add_{cache_key}")

with bot_col2:
    with st.container(border=True):
        st.markdown(
            f"""
            <div class="vnpt-panel-head">
                <div class="ph-title">{icon('cpu', 18)} Master JSON Preview</div>
                <div class="ph-live"><span class="vnpt-live-dot"></span> LIVE</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        alert("info", "Tự động cập nhật",
              "JSON bên dưới cập nhật ngay khi bạn thay đổi dữ liệu ở cột bên trái.")

        final_blocks = [_block_to_entry(idx, blk) for idx, blk in enumerate(blocks)]

        master_json = {
            "stem": stem,
            "pages": [{
                "page_num": page_num,
                "n_blocks": len(final_blocks),
                "blocks":   final_blocks,
            }],
        }

        _n_missing = sum(1 for b in final_blocks if b["added_missing"])
        st.markdown(
            f"""
            <div style="margin:2px 0 12px">
                <span class="vnpt-badge vnpt-badge-info">{icon('layers', 11)} {len(final_blocks)} block</span>
                <span class="vnpt-badge {'vnpt-badge-off' if _n_missing else 'vnpt-badge-neutral'}" style="margin-left:6px">
                    {icon('flag', 11)} {_n_missing} added-missing
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.json(master_json, expanded=True)

        st.write("")
        save_col1, save_col2 = st.columns(2)
        with save_col1:
            if st.button("Lưu trang này", type="primary",
                         width='stretch', icon=":material/save:",
                         key=f"save_{cache_key}"):
                out_path = save_gt(run_dir, stem, page_num, blocks)
                alert("success", "Đã lưu thành công",
                      f"{len(final_blocks)} block → <code>{out_path}</code>")
        with save_col2:
            if st.button("Gộp toàn bộ trang đã lưu", width='stretch',
                         icon=":material/inventory_2:",
                         key=f"merge_{run_dir}"):
                merged = merge_all_saved_pages(run_dir, stem)
                out_dir = run_dir / "ground_truth"
                out_dir.mkdir(parents=True, exist_ok=True)
                master_path = out_dir / "_master.json"
                master_path.write_text(
                    json.dumps(merged, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                alert("success", "Đã gộp toàn bộ trang",
                      f"{len(merged['pages'])} trang → <code>{master_path}</code>")
                st.download_button(
                    "Tải file master JSON",
                    data=json.dumps(merged, ensure_ascii=False, indent=2),
                    file_name=f"{stem}_ground_truth.json",
                    mime="application/json",
                    icon=":material/download:",
                    width='stretch',
                )