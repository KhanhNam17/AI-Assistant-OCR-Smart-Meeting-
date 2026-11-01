# core/design_system.py
#
# Nạp 1 lần duy nhất file assets/design_system.css và inject vào trang hiện tại.
# Dùng chung cho app.py + toàn bộ pages/*.py để đồng bộ Typography, Spacing,
# Color tokens, Hover states theo Design Audit.
#
# Cách dùng — thêm 2 dòng NGAY SAU st.set_page_config() ở mỗi trang:
#
#     from core.design_system import inject_design_system
#     inject_design_system()
#
from __future__ import annotations
 
import os
import streamlit as st
 
_CSS_PATH = os.path.join(os.path.dirname(__file__), "..", "assets", "design_system.css")
 
 
@st.cache_resource(show_spinner=False)
def _load_css(path: str) -> str:
    """Đọc file CSS 1 lần/process, cache lại — tránh đọc đĩa mỗi rerun."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
 
 
def inject_design_system() -> None:
    """
    Inject assets/design_system.css vào trang hiện tại.
 
    Luôn gọi hàm này TRƯỚC mọi CSS riêng của từng trang (nếu trang vẫn còn
    CSS cục bộ), để các token/class dùng chung (--brand, .card, .card-title,
    .data-value, .transcript-row, ...) có sẵn cho phần CSS phía sau override
    nếu cần.
    """
    try:
        css = _load_css(_CSS_PATH)
    except FileNotFoundError:
        st.warning(
            "⚠️ Không tìm thấy assets/design_system.css — "
            "kiểm tra lại đường dẫn thư mục project.",
            icon="⚠️",
        )
        return
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)