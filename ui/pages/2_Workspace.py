"""
pages/2_Workspace.py
=====================

Modern Unified Workspace:
  • "Cuộc họp thông minh"  (Smart Meeting / STT)  → core body: _live_mode_body.render_live_mode()
  • "OCR Tài liệu"         (Document OCR)         → core body: _ocr_body.render_ocr_mode()
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PAGES_DIR = Path(__file__).resolve().parent
REPO_ROOT = PAGES_DIR.parent.parent
UI_ROOT   = PAGES_DIR.parent
for _p in (REPO_ROOT, UI_ROOT, PAGES_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from theme import apply_page_config, inject_base_css, render_sidebar, icon

# Body modules
from _live_mode_body import render_live_mode
from _ocr_body import render_ocr_mode

MODE_MEETING = "meeting"
MODE_OCR     = "ocr"

_MODE_META = {
    MODE_MEETING: {
        "label":     "🎙️ Cuộc họp thông minh",
        "page_title": "Không gian làm việc — Cuộc họp",
        "page_icon": "🎙️",
    },
    MODE_OCR: {
        "label":     "📄 OCR Tài liệu",
        "page_title": "Không gian làm việc — OCR",
        "page_icon": "📄",
    },
}

# ── Resolve initial mode ──────────────────────────────────────────────────────
_qp_mode = st.query_params.get("mode")
if "workspace_mode" not in st.session_state:
    st.session_state["workspace_mode"] = (
        _qp_mode if _qp_mode in _MODE_META else MODE_MEETING
    )
elif _qp_mode in _MODE_META and _qp_mode != st.session_state["workspace_mode"]:
    st.session_state["workspace_mode"] = _qp_mode

active_mode = st.session_state["workspace_mode"]
meta        = _MODE_META[active_mode]

apply_page_config(meta["page_title"], UI_ROOT)
inject_base_css()
render_sidebar(active="workspace")

st.markdown(f"""
<div style="font-family:'JetBrains Mono',monospace;font-size:11px;letter-spacing:2px;text-transform:uppercase;
            color:#2563EB;padding-bottom:12px;border-bottom:1px solid rgba(148,163,184,0.22);margin-bottom:18px;
            display:flex;align-items:center;gap:8px;">
    {icon('layers', 13, 2.0, '#2563EB')}&nbsp; KHÔNG GIAN LÀM VIỆC — CHỌN CHẾ ĐỘ VẬN HÀNH
</div>
""", unsafe_allow_html=True)

# ── Mode switcher ────────────────────────────────────────────────────────────
st.markdown("""
<style>
.app-mode-switch [data-testid="stButton"] button {
    height: 3.2rem;
    font-size: 14.5px;
    font-weight: 700;
    border-radius: 12px !important;
    transition: all 0.2s ease;
}
</style>
""", unsafe_allow_html=True)

sw_col1, sw_col2, sw_col3 = st.columns([1, 1, 2])
with sw_col1:
    if st.button(
        _MODE_META[MODE_MEETING]["label"], width="stretch",
        type=("primary" if active_mode == MODE_MEETING else "secondary"),
        key="wm_btn_meeting",
    ):
        if active_mode != MODE_MEETING:
            st.session_state["workspace_mode"] = MODE_MEETING
            st.query_params["mode"] = MODE_MEETING
            st.rerun()
with sw_col2:
    if st.button(
        _MODE_META[MODE_OCR]["label"], width="stretch",
        type=("primary" if active_mode == MODE_OCR else "secondary"),
        key="wm_btn_ocr",
    ):
        if active_mode != MODE_OCR:
            st.session_state["workspace_mode"] = MODE_OCR
            st.query_params["mode"] = MODE_OCR
            st.rerun()

st.write("")

# ── Delegate to selected mode ────────────────────────────────────────────────
if active_mode == MODE_MEETING:
    render_live_mode()
else:
    render_ocr_mode()
