"""
ui/pages/_ocr_body.py
=====================

Streamlit front-end for the OCR document pipeline.
Focused strictly on high-accuracy Document OCR (YOLO DocLayNet + EasyOCR/VietOCR/DeepSeek + TATR + python-docx).
"""

from __future__ import annotations

import base64
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
UI_ROOT   = Path(__file__).resolve().parent.parent
for _p in (REPO_ROOT, UI_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from core.grounding_docx_builder import build_docx_from_grounding
from core.document_pipeline import docx_to_pdf
from theme import render_page_header, icon
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


def render_ocr_mode():
    """Full page body — invoked by the Workspace router."""

    # ── PDF / DOCX preview helpers ───────────────────────────────────────────────
    try:
        from streamlit_pdf_viewer import pdf_viewer as _pdf_viewer
        _HAS_PDF_VIEWER = True
    except Exception:
        _HAS_PDF_VIEWER = False

    def render_pdf(pdf_path: Path, *, height: int = 720, key: str | None = None) -> None:
        if not pdf_path.exists():
            st.caption(f"Không tìm thấy file: `{pdf_path.name}`")
            return
        if _HAS_PDF_VIEWER:
            try:
                _pdf_viewer(input=pdf_path.read_bytes(), height=height, key=key, render_text=True)
                return
            except Exception as e:
                st.warning(f"PDF viewer lỗi ({e}); chuyển sang dùng iframe.")

        b64 = base64.b64encode(pdf_path.read_bytes()).decode("ascii")
        st.markdown(
            f"""<iframe src="data:application/pdf;base64,{b64}" width="100%" height="{height}px" 
            style="border: 1px solid var(--border-color); border-radius: 8px; background: var(--secondary-background-color);"></iframe>""",
            unsafe_allow_html=True,
        )

    def render_source(source_path: Path | None, *, height: int = 720, key: str | None = None) -> None:
        if source_path is None or not source_path.exists():
            st.caption("Chưa có file gốc hiển thị.")
            return
        suffix = source_path.suffix.lower()
        if suffix == ".pdf":
            render_pdf(source_path, height=height, key=key)
            return
        if suffix in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
            b64 = base64.b64encode(source_path.read_bytes()).decode("ascii")
            mime = "image/png" if suffix == ".png" else "image/jpeg"
            st.markdown(
                f"""<div style="height:{height}px; overflow:auto; border:1px solid var(--border-color);
                border-radius:8px; padding:8px; background:var(--secondary-background-color);"><img src="data:{mime};base64,{b64}" style="width:100%; display:block;" /></div>""",
                unsafe_allow_html=True,
            )
            return
        st.caption(f"Định dạng `{suffix}` không hỗ trợ xem trước.")

    # ── Repo layout & Python path ────────────────────────────────────────────────
    CORE_ENTRY   = REPO_ROOT / "core" / "document_pipeline.py"
    DEFAULT_OUT  = REPO_ROOT / "output"
    UPLOADS_DIR  = DEFAULT_OUT / "_uploads"
    ALLOWED_EXTS = (".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff")

    _default_ocr_py = r"C:\Users\cunam\anaconda3\envs\uc3_ocr\python.exe"
    UC3_OCR_PYTHON  = os.getenv("UC3_OCR_PYTHON", _default_ocr_py)
    if not os.path.exists(UC3_OCR_PYTHON):
        UC3_OCR_PYTHON = sys.executable

    # ── Page CSS ────────────────────────────────────────────────────────────────
    st.markdown("""
    <style>
        .vnpt-stat-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px 24px; }
        .vnpt-stat-field .vnpt-info-label {
            font-size: 11px; font-weight: 800; letter-spacing: 0.4px; text-transform: uppercase;
            color: var(--text-color); opacity: 0.55; margin-bottom: 3px;
        }
        .vnpt-stat-field .vnpt-info-value { font-size: 13.5px; font-weight: 600; word-break: break-word; }

        .vnpt-uploader-empty {
            border: 1.5px dashed var(--border-color); border-radius: var(--vnpt-radius-md);
            padding: 18px; text-align: center; color: var(--text-color); opacity: 0.55;
            font-size: 12.5px; margin-top: 6px;
        }

        /* Stepper styles */
        .stepper-container { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; padding: 16px 30px; background: var(--secondary-background-color); border-radius: var(--vnpt-radius-lg); border: 1px solid var(--border-color); box-shadow: var(--vnpt-shadow-sm); }
        .step { display: flex; align-items: center; gap: 11px; width: 140px; text-align: center; }
        .step-icon { min-width: 32px; height: 32px; border-radius: 50%; display: flex; justify-content: center; align-items: center; background: var(--background-color); color: var(--text-color); border: 2px solid var(--border-color); transition: 0.25s; }
        .step-icon.active { background: var(--vnpt-green); border-color: var(--vnpt-green); box-shadow: 0 0 0 4px var(--vnpt-green-tint); }
        .step-label { font-size: 14px; font-weight: 700; color: var(--text-color); white-space: nowrap; }
        .step-line { flex-grow: 1; height: 3px; background: var(--border-color); margin: 0 20px; border-radius: 2px; transition: 0.3s; }
        .step-line.active { background: var(--vnpt-green); }
    </style>
    """, unsafe_allow_html=True)

    # ── Lịch sử phiên từ đĩa ────────────────────────────────────────────────────
    def load_session_history():
        history = []
        if DEFAULT_OUT.exists():
            for d in DEFAULT_OUT.iterdir():
                if d.is_dir() and (d / "metadata.json").exists():
                    try:
                        with open(d / "metadata.json", "r", encoding="utf-8") as f:
                            history.append(json.load(f))
                    except Exception:
                        pass
        history.sort(key=lambda x: x.get("id", ""), reverse=True)
        return history

    # ── Session state defaults ───────────────────────────────────────────────────
    _DEFAULT_STATE = {
        "is_running": False,
        "ocr_progress": 0.0,
        "last_run_dir": None,
        "last_run_stem": None,
        "last_source_path": None,
        "last_exit_code": None,
        "last_log_lines": [],
        "active_session": None,
        "is_layout_done": False,
        "uploader_version": 0,
    }
    for k, v in _DEFAULT_STATE.items():
        st.session_state.setdefault(k, v)

    if "session_history" not in st.session_state:
        st.session_state.session_history = load_session_history()

    # ── Helpers ──────────────────────────────────────────────────────────────────
    _PAGE_RE  = re.compile(r"PAGE\s+(\d+)\s*/\s*(\d+)", re.IGNORECASE)
    _TOTAL_RE = re.compile(r"\(\s*(\d+)\s+pages?\s*\)", re.IGNORECASE)
    _BUILD_RE = re.compile(r"BUILD\s+OUTPUT", re.IGNORECASE)

    def save_upload(uploaded_file) -> Path:
        UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        dest = UPLOADS_DIR / uploaded_file.name
        with open(dest, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return dest

    def file_info(path: Path) -> dict:
        info = {
            "Tên file": path.name,
            "Dung lượng": f"{path.stat().st_size / 1024:.1f} KB",
            "Định dạng": path.suffix.upper().lstrip("."),
        }
        if path.suffix.lower() == ".pdf":
            try:
                import fitz
                with fitz.open(str(path)) as doc:
                    info["Số trang"] = str(len(doc))
            except Exception:
                try:
                    import pypdf
                    reader = pypdf.PdfReader(str(path))
                    info["Số trang"] = str(len(reader.pages))
                except Exception:
                    pass
        return info

    def build_cli(source: Path, output_dir: Path) -> list[str]:
        cmd = [UC3_OCR_PYTHON, str(CORE_ENTRY), str(source), "vi->en", str(output_dir), "--no-translate"]

        config_path = DEFAULT_OUT / "config" / "pipeline_settings.json"
        cfg = {}
        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
            except Exception:
                pass

        engine_mode    = cfg.get("engine_mode", "end2end")
        ocr_engine     = cfg.get("ocr_engine", "deepseek")
        table_engine   = cfg.get("table_engine", "img2table")
        layout_backend = cfg.get("layout_backend", "auto")
        yolo_model     = cfg.get("yolo_model", "")

        cmd += [
            "--engine-mode", engine_mode,
            "--ocr-engine", ocr_engine,
            "--table-engine", table_engine,
            "--layout-backend", layout_backend,
        ]

        if yolo_model:
            yolo_path = str(REPO_ROOT / "models" / "yolo" / yolo_model)
            cmd += ["--yolo-model", yolo_path]

        return cmd

    def parse_progress(line: str, state: dict) -> None:
        m = _TOTAL_RE.search(line)
        if m:
            state["ocr_total"] = int(m.group(1))

        m = _PAGE_RE.search(line)
        if m:
            cur = int(m.group(1))
            tot = int(m.group(2))
            state["ocr_cur"] = cur
            state["ocr_total"] = tot
            state["ocr_progress"] = max(state.get("ocr_progress", 0.0), min(0.97, (cur - 0.5) / max(1, tot)))

        if _BUILD_RE.search(line) or ("TOTAL " in line and "%" in line):
            state["ocr_progress"] = 1.0
            state["build_done"] = True

    def stream_subprocess(cmd: list[str], extra_env: dict[str, str], ocr_bar, page_bar, stepper_placeholder=None) -> tuple[int, list[str]]:
        env = os.environ.copy()
        env.update({"PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8", **(extra_env or {})})
        proc = subprocess.Popen(
            cmd, cwd=str(REPO_ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1, env=env,
        )
        q: "Queue[str]" = Queue()

        def _reader():
            for line in iter(proc.stdout.readline, ""):
                q.put(line.rstrip("\n"))
                print(line.rstrip("\n"), flush=True)
            proc.stdout.close()

        t = threading.Thread(target=_reader, daemon=True)
        t.start()
        log_lines: list[str] = []
        pstate = {"ocr_cur": 0, "ocr_total": 0, "ocr_progress": 0.0, "build_done": False}
        last_paint = 0.0

        while True:
            try:
                line = q.get(timeout=0.15)
                log_lines.append(line)
                parse_progress(line, pstate)
            except Empty:
                pass

            now = time.time()
            if now - last_paint > 0.2:
                if ocr_bar:
                    ocr_bar.progress(pstate["ocr_progress"], text=f"OCR ({int(pstate['ocr_progress']*100)}%)")
                if page_bar:
                    ptot = max(1, pstate["ocr_total"])
                    pcur = pstate["ocr_cur"]
                    completed = ptot if (pstate.get("build_done") or pstate.get("build_started")) else max(0, pcur - 1)
                    page_bar.progress(completed / ptot, text=f"Số trang hoàn thành: {completed}/{ptot}")
                if stepper_placeholder is not None:
                    ocr_step_done = pstate.get("ocr_progress", 0.0) >= 0.999
                    export_step_done = pstate.get("build_done", False)
                    with stepper_placeholder:
                        render_stepper(True, ocr_step_done, export_step_done)
                last_paint = now

            if proc.poll() is not None and q.empty():
                break

        t.join(timeout=2)
        if ocr_bar:
            ocr_bar.progress(1.0, text="OCR (100%)")
        if page_bar and pstate["ocr_total"] > 0:
            page_bar.progress(1.0, text=f"Số trang hoàn thành: {pstate['ocr_total']}/{pstate['ocr_total']}")
        if stepper_placeholder is not None:
            ok = proc.returncode == 0
            with stepper_placeholder:
                render_stepper(True, ok, ok)
        return proc.returncode, log_lines

    def find_outputs(output_dir: Path, stem: str) -> dict[str, Path]:
        candidates = {
            "ocr_md":   output_dir / f"{stem}_ocr.md",
            "docx_ocr": output_dir / f"{stem}_ocr.docx",
            "pdf_ocr":  output_dir / f"{stem}_ocr.pdf",
        }
        return {k: v for k, v in candidates.items() if v.exists()}

    def finalize_document(output_dir: Path, stem: str):
        raw_md_path     = output_dir / f"{stem}_ocr_raw.md"
        pages_dir       = output_dir / "pages"
        output_docx_ocr = output_dir / f"{stem}_ocr.docx"

        if not raw_md_path.exists():
            st.error("Không tìm thấy dữ liệu thô. Vui lòng chạy OCR trước.")
            return

        with st.spinner("Đang nội suy tọa độ và dựng lại Layout DOCX / PDF..."):
            try:
                try:
                    import pythoncom
                    pythoncom.CoInitialize()
                except ImportError:
                    pass

                build_docx_from_grounding(
                    raw_md_path=str(raw_md_path),
                    pages_dir=str(pages_dir),
                    stem=stem,
                    output_docx=str(output_docx_ocr),
                )
                docx_to_pdf(output_docx_ocr, output_dir)

                st.session_state.is_layout_done = True
                st.toast("✅ Đã hoàn thiện cấu trúc Layout DOCX / PDF!", icon=":material/check_circle:")
                st.rerun()
            except Exception as e:
                st.error(f"Đã xảy ra lỗi khi hoàn thiện Layout: {e}")

    # ── Stepper ──────────────────────────────────────────────────────────────────
    def render_stepper(file_sel: bool, ocr_done: bool, export_done: bool):
        def step_icon_html(done):
            return ("step-icon active", icon("check", 15, 2.2, "white")) if done else ("step-icon", "")

        f_c, f_t = step_icon_html(file_sel)
        o_c, o_t = step_icon_html(ocr_done)
        e_c, e_t = step_icon_html(export_done)

        l1 = "active" if ocr_done else ""
        l2 = "active" if export_done else ""

        st.markdown(f"""
        <div class="stepper-container">
            <div class="step"><div class="{f_c}">{f_t}</div><div class="step-label">Chọn file</div></div>
            <div class="step-line {l1}"></div>
            <div class="step"><div class="{o_c}">{o_t}</div><div class="step-label">OCR xử lý</div></div>
            <div class="step-line {l2}"></div>
            <div class="step"><div class="{e_c}">{e_t}</div><div class="step-label">Xuất file</div></div>
        </div>
        """, unsafe_allow_html=True)

    # Trạng thái toàn cục
    is_file_sel   = st.session_state.last_source_path is not None
    is_ocr_done   = st.session_state.last_exit_code == 0
    outputs_exist = st.session_state.last_run_dir is not None

    # ── Header ───────────────────────────────────────────────────────────────────
    CONFIG_PATH    = DEFAULT_OUT / "config" / "pipeline_settings.json"
    current_engine = "end2end"
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                saved_cfg = json.load(f)
                current_engine = saved_cfg.get("engine_mode", "end2end")
        except Exception:
            pass

    render_page_header(
        "OCR TÀI LIỆU",
        f"Engine nhận diện: <b style=\'color:var(--vnpt-blue);text-transform:uppercase;\'>{current_engine}</b>",
        eyebrow="Document OCR processing",
    )
    st.write("")

    stepper_placeholder = st.empty()
    st.write("")

    # ── Layout 2 Cột cân đối (50% Trái - 50% Phải) ──────────────────────────────
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        # 1. Tải file lên
        with st.container(border=True):
            st.subheader("Tải file tài liệu")
            uploaded = st.file_uploader(
                "Kéo và thả file",
                type=[e.lstrip(".") for e in ALLOWED_EXTS],
                label_visibility="collapsed",
                key=f"file_uploader_{st.session_state.uploader_version}",
                help="Định dạng hỗ trợ: PDF, PNG, JPG, TIFF",
            )
            if uploaded:
                src_path = save_upload(uploaded)
                info = file_info(src_path)
                is_file_sel = True
                st.session_state.last_source_path = str(src_path)
            else:
                info = {}
                is_file_sel = False

        # 2. Thông tin cơ bản & Điều khiển OCR
        with st.container(border=True):
            st.subheader("Thông tin & Xử lý")

            fields = [
                ("Tên file", info.get("Tên file", "--")),
                ("Định dạng", info.get("Định dạng", "--")),
                ("Dung lượng", info.get("Dung lượng", "--")),
                ("Số trang", info.get("Số trang", "--")),
            ]
            fields_html = "".join(
                f'<div class="vnpt-stat-field"><div class="vnpt-info-label">{label}</div>'
                f'<div class="vnpt-info-value">{value}</div></div>'
                for label, value in fields
            )
            st.markdown(f'<div class="vnpt-stat-row">{fields_html}</div>', unsafe_allow_html=True)
            st.write("")

            if st.session_state.is_running:
                st.button("Đang thực hiện OCR...", type="primary", disabled=True, use_container_width=True, icon=":material/hourglass_top:")
                run_clicked = False
            else:
                run_clicked = st.button("Bắt đầu xử lý OCR", type="primary", disabled=(uploaded is None), use_container_width=True, icon=":material/play_arrow:")

            st.write("")
            st.markdown("**Tiến trình OCR:**")
            ocr_bar_slot  = st.empty()
            page_bar_slot = st.empty()

            if not st.session_state.is_running:
                ocr_bar_slot.progress(0.0, text="OCR (0%)")
                page_bar_slot.progress(0.0, text="Số trang hoàn thành: 0/0")

            if st.session_state.last_exit_code not in (None, 0):
                st.error(f"❌ Tiến trình kết thúc với mã lỗi ({st.session_state.last_exit_code}).")
                if st.session_state.get("last_log_lines"):
                    with st.expander("📋 Xem nhật ký lỗi (Console Log)", expanded=False):
                        st.code("\n".join(st.session_state.last_log_lines), language="log")

    with col_right:
        # 1. Hoàn thiện Layout & Quản lý phiên làm việc
        with st.container(border=True):
            st.subheader("Hoàn thiện & Lưu trữ")
            st.caption("Dựng lại cấu trúc Layout theo tọa độ gốc và quản lý lưu trữ.")

            finalize_clicked = st.button(
                "✨ Xử lý Layout DOCX / PDF", 
                type="primary", 
                use_container_width=True, 
                disabled=not is_ocr_done, 
                icon=":material/auto_fix_high:",
                help="Nội suy tọa độ thực tế để dựng file DOCX và PDF chuẩn bố cục văn bản gốc"
            )

            st.write("")
            col_act1, col_act2 = st.columns([1, 1], gap="small")
            with col_act1:
                save_history_clicked = st.button(
                    "💾 Lưu lịch sử", 
                    type="secondary", 
                    use_container_width=True, 
                    disabled=not st.session_state.is_layout_done, 
                    icon=":material/save:",
                    help="Lưu phiên vào cơ sở dữ liệu để xem lại ở tab Review Sessions"
                )
            with col_act2:
                new_session_clicked = st.button(
                    "🆕 Phiên mới", 
                    type="secondary", 
                    use_container_width=True, 
                    disabled=st.session_state.is_running, 
                    icon=":material/refresh:",
                    help="Dọn sạch file hiện tại để nạp tài liệu mới"
                )

        # 2. Xuất kết quả OCR
        with st.container(border=True):
            st.markdown(f'<span class="vnpt-pill-label">{icon("download", 14)} Export kết quả OCR</span>', unsafe_allow_html=True)

            if outputs_exist and not st.session_state.is_running:
                run_dir = Path(st.session_state.last_run_dir)
                stem    = st.session_state.last_run_stem
                outs    = find_outputs(run_dir, stem)
            else:
                outs = {}

            def _fmt_size(path: Path) -> str:
                kb = path.stat().st_size / 1024
                return f"{kb:.0f} KB" if kb < 1024 else f"{kb / 1024:.1f} MB"

            def _export_row(row_key: str, file_path, file_kind: str, title: str,
                             subtitle: str, mime: str, primary: bool = False) -> None:
                ready = file_path is not None
                status_html = (
                    f'<span class="vnpt-export-status ready">{icon("check", 11)} {_fmt_size(file_path)}</span>'
                    if ready else
                    f'<span class="vnpt-export-status empty">{icon("cross", 11)} Chưa có</span>'
                )
                row_cls = "vnpt-export-row ready" if ready else "vnpt-export-row empty"
                row_col, btn_col = st.columns([4.4, 1.3], vertical_alignment="center")
                with row_col:
                    st.markdown(
                        f'<div class="{row_cls}">'
                        f'<div class="vnpt-export-icon {file_kind}">{icon("docx" if file_kind == "docx" else "scan", 17)}</div>'
                        f'<div class="vnpt-export-meta">'
                        f'<div class="vnpt-export-title">{title}</div>'
                        f'<div class="vnpt-export-sub">{subtitle}</div>'
                        f'</div>{status_html}</div>',
                        unsafe_allow_html=True,
                    )
                with btn_col:
                    if ready:
                        st.download_button(
                            "Tải", data=file_path.read_bytes(), file_name=file_path.name,
                            mime=mime, width="stretch",
                            type="primary" if primary else "secondary",
                            icon=":material/download:", key=f"dl_{row_key}",
                        )
                    else:
                        st.button("Tải", disabled=True, width="stretch", key=f"dl_disabled_{row_key}")

            _export_row(
                "o_docx", outs.get("docx_ocr"), "docx", "Văn bản Word (.docx)",
                "Đã căn chỉnh bố cục · Sẵn sàng biên tập",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                primary=True
            )
            st.write("")
            _export_row(
                "o_pdf", outs.get("pdf_ocr"), "pdf", "Tài liệu PDF (.pdf)",
                "Bản in điện tử / Chia sẻ nhanh",
                "application/pdf",
                primary=True
            )

    # Cập nhật Stepper trên đầu
    with stepper_placeholder:
        render_stepper(is_file_sel, is_ocr_done, outputs_exist)

    # ── Pipeline Execution ───────────────────────────────────────────────────────
    if run_clicked and uploaded is not None and not st.session_state.is_running:
        st.session_state.is_running = True
        st.session_state.is_layout_done = False
        st.session_state.last_exit_code = None
        st.rerun()

    if st.session_state.is_running and uploaded is not None:
        config_path = DEFAULT_OUT / "config" / "pipeline_settings.json"
        custom_out_dir = str(DEFAULT_OUT)
        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    custom_out_dir = json.load(f).get("output_dir", str(DEFAULT_OUT))
            except Exception:
                pass

        output_dir = Path(custom_out_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = src_path.stem

        cmd = build_cli(src_path, output_dir)
        extra_env: dict[str, str] = {}

        exit_code, log_lines = stream_subprocess(
            cmd, extra_env, 
            ocr_bar=ocr_bar_slot, page_bar=page_bar_slot,
            stepper_placeholder=stepper_placeholder,
        )

        st.session_state.is_running       = False
        st.session_state.last_run_dir     = str(output_dir)
        st.session_state.last_run_stem    = stem
        st.session_state.last_source_path = str(src_path)
        st.session_state.last_exit_code   = exit_code
        st.session_state.last_log_lines   = log_lines[-30:] if log_lines else []
        st.rerun()

    if finalize_clicked:
        finalize_document(Path(st.session_state.last_run_dir), st.session_state.last_run_stem)

    if new_session_clicked:
        keep_keys = {"session_history", "workspace_mode"}
        next_uploader_version = st.session_state.get("uploader_version", 0) + 1
        for k in list(st.session_state.keys()):
            if k not in keep_keys:
                del st.session_state[k]
        st.session_state.uploader_version = next_uploader_version
        st.toast("Đã tạo phiên mới — hãy tải lên tài liệu tiếp theo.", icon=":material/refresh:")
        st.rerun()

    if save_history_clicked:
        old_dir = Path(st.session_state.last_run_dir)

        config_path  = DEFAULT_OUT / "config" / "pipeline_settings.json"
        base_out_dir = DEFAULT_OUT
        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    base_out_dir = Path(json.load(f).get("output_dir", str(DEFAULT_OUT)))
            except Exception:
                pass

        if old_dir == base_out_dir:
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            time_str   = datetime.now().strftime("%H:%M:%S")
            stem       = st.session_state.last_run_stem

            session_folder_name = f"{stem}_{session_id}"
            new_session_dir     = base_out_dir / session_folder_name
            new_session_dir.mkdir(parents=True, exist_ok=True)

            for p in old_dir.glob(f"{stem}*.*"):
                if p.is_file():
                    shutil.move(str(p), str(new_session_dir / p.name))

            old_pages = old_dir / "pages"
            if old_pages.exists():
                new_pages = new_session_dir / "pages"
                shutil.copytree(old_pages, new_pages, dirs_exist_ok=True)

            src_info = file_info(Path(st.session_state.last_source_path))
            entry = {
                "id":          session_id,
                "label":       f"{stem} · {time_str}",
                "stem":        stem,
                "source_path": st.session_state.last_source_path,
                "run_dir":     str(new_session_dir), 
                "translate":   False, 
                "direction":   "vi->en",
                "exit_code":   st.session_state.last_exit_code,
                "started_at":  datetime.now().isoformat(),
                "pages":       src_info.get("Số trang", "--"),
            }

            with open(new_session_dir / "metadata.json", "w", encoding="utf-8") as f:
                json.dump(entry, f, ensure_ascii=False, indent=4)

            st.session_state.session_history.insert(0, entry)
            st.session_state.last_run_dir = str(new_session_dir)

            st.toast("Đã lưu phiên OCR vào lịch sử thành công!", icon=":material/check_circle:")
            time.sleep(0.5)
            st.rerun()
        else:
            st.toast("Phiên xử lý này đã được lưu vào lịch sử từ trước.", icon=":material/info:")

    # ── Previews (Bottom Area) ───────────────────────────────────────────────────
    if is_file_sel and st.session_state.last_source_path and not st.session_state.is_running:
        source_p = Path(st.session_state.last_source_path)
        stem     = source_p.stem
        outs     = find_outputs(Path(st.session_state.last_run_dir), stem) if outputs_exist else {}

        st.write("")
        st.markdown(
            f'<div style="font-family:IBM Plex Mono,monospace;font-size:11px;font-weight:700;'
            f'letter-spacing:1px;text-transform:uppercase;color:#0060B5;margin:24px 0 12px;'
            f'display:flex;align-items:center;gap:6px;">'
            f'{icon("scan", 14)} XEM TRƯỚC TÀI LIỆU (PREVIEWS)</div>',
            unsafe_allow_html=True,
        )
        sc1, sc2 = st.columns(2, gap="large")
        with sc1:
            st.subheader("📄 File tài liệu gốc")
            with st.container(border=True):
                render_source(source_p, key=f"src_{stem}")
        with sc2:
            st.subheader("📑 File kết quả OCR (PDF)")
            with st.container(border=True):
                if not outputs_exist:
                    st.caption("ℹ️ Kết quả xử lý sẽ xuất hiện tại đây sau khi tiến trình hoàn tất.")
                elif outs.get("pdf_ocr"):
                    render_pdf(outs.get("pdf_ocr"), key=f"dst_{stem}")
                else:
                    st.caption("PDF chưa sẵn sàng. (Bấm ✨ Xử lý Layout để tạo file PDF)")
