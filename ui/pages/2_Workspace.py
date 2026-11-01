"""
pages/2_Workspace.py
=====================

Merged workspace: ONE page, TWO modes.

  • "Cuộc họp thông minh"  (Smart Meeting / STT)  → core body: _live_mode_body.render_live_mode()
  • "OCR & Dịch thuật"     (Document OCR)         → core body: _ocr_body.render_ocr_mode()

Rationale
---------
Both original pages (`2_Live_Mode.py`, `7_OCR&Translate.py`) are full,
independent Streamlit scripts — each called `st.set_page_config`,
`apply_page_config`, `inject_base_css`, `render_sidebar` on its own.
Streamlit only allows `st.set_page_config` ONCE per run, so they can't be
imported side-by-side unmodified.

Fix: both scripts were mechanically split into a thin "chrome" header
(imports / env setup, unchanged) and a `render_*()` function containing
everything that used to run at module scope (unchanged, just indented).
The page-config / sidebar / css calls were removed from the bodies —
this router owns them exactly once — everything else is byte-for-byte
the same logic as before.

This router just:
  1. Sets page config + injects the shared design system ONCE.
  2. Renders one sidebar (shared nav — see theme.py).
  3. Shows a mode switcher (Cuộc họp / OCR), persisted in
     `st.session_state["workspace_mode"]` and deep-linkable via
     `?mode=meeting` / `?mode=ocr` query params (used by Dashboard /
     Review / Benchmark to jump straight into the right mode).
  4. Delegates to whichever body module matches the selected mode.
     Only the SELECTED mode's heavy imports/models are ever touched —
     the other mode's body module is imported (cheap: just function
     defs) but its `render_*()` is not called, so e.g. picking OCR mode
     never touches sherpa_onnx / diart / torch.
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

# Body modules — importing them only defines functions (cheap); the heavy
# model/engine imports happen lazily *inside* render_live_mode() /
# render_ocr_mode(), so only the mode actually selected pays that cost.
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

# ── Resolve initial mode: query param takes priority (deep link from
#    Dashboard/Review/Benchmark), then session_state, then default. ─────────
_qp_mode = st.query_params.get("mode")
if "workspace_mode" not in st.session_state:
    st.session_state["workspace_mode"] = (
        _qp_mode if _qp_mode in _MODE_META else MODE_MEETING
    )
elif _qp_mode in _MODE_META and _qp_mode != st.session_state["workspace_mode"]:
    # A fresh deep link with a different mode was clicked — honour it.
    st.session_state["workspace_mode"] = _qp_mode

active_mode = st.session_state["workspace_mode"]
meta        = _MODE_META[active_mode]

apply_page_config(meta["page_title"], UI_ROOT)
inject_base_css()
render_sidebar(active="workspace")

st.markdown(f"""
<div style="font-size:11px;letter-spacing:3px;text-transform:uppercase;
            color:var(--text-color);opacity:0.7;padding-bottom:12px;
            border-bottom:1px solid rgba(128,128,128,0.2);margin-bottom:18px;">
    {icon('layers', 13)}&nbsp; KHÔNG GIAN LÀM VIỆC — chọn chế độ bên dưới
</div>
""", unsafe_allow_html=True)

# ── Mode switcher ────────────────────────────────────────────────────────────
st.markdown("""
<style>
.vnpt-mode-switch [data-testid="stButton"] button {
    height: 3.1rem; font-size: 14.5px; font-weight: 700; border-radius: 12px !important;
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

# ── Delegate to the selected mode's full body ───────────────────────────────
if active_mode == MODE_MEETING:
    render_live_mode()
else:
    render_ocr_mode()
