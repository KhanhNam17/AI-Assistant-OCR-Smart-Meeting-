import base64
from pathlib import Path
import streamlit as st
from theme import apply_page_config, inject_base_css, render_sidebar

# ══════════════════════════════════════════════════════════════════════════════
# 1. PAGE CONFIG & DESIGN SYSTEM — VNPT Blue Corporate Theme
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Trợ lý Cuộc họp Thông minh",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Nạp logo công ty (assets/logo_vnpt.png) và mã hoá base64 để nhúng trực tiếp vào HTML
@st.cache_data(show_spinner=False)
def _load_logo_b64() -> str | None:
    logo_path = Path(__file__).resolve().parent.parent / "assets" / "logo_vnpt.png"
    if not logo_path.exists():
        logo_path = Path(__file__).resolve().parent / "assets" / "logo_vnpt.png"
    if not logo_path.exists():
        return None
    return base64.b64encode(logo_path.read_bytes()).decode()

LOGO_B64 = _load_logo_b64()

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
    footer { visibility: hidden; }

    .stButton > button[kind="primary"],
    .stDownloadButton > button[kind="primary"] {
        background: #0067B1 !important; border: none !important;
        font-weight: 600 !important; color: white !important;
    }
    .stButton > button[kind="primary"]:hover,
    .stDownloadButton > button[kind="primary"]:hover { background: #004C87 !important; }

    .stDownloadButton > button:not([kind="primary"]) {
        background: var(--secondary-background-color) !important;
        color: var(--text-color) !important;
        border: 1px solid rgba(128,128,128,0.2) !important;
    }

    /* Hero */
    .hero-eyebrow {
        font-family: 'IBM Plex Mono', monospace; font-size: 11px; letter-spacing: 4px;
        text-transform: uppercase; color: #0067B1; font-weight: 600; margin-bottom: 14px;
    }
    .hero-card {
        background: linear-gradient(135deg, #0B2540 0%, #0067B1 100%);
        border-radius: 12px; padding: 32px; margin: 20px 0 16px;
        box-shadow: 0 12px 28px rgba(0,103,177,0.25);
        color: white;
    }
    .hero-feature { font-size: 13px; color: rgba(255,255,255,0.9); }
    .hero-tag {
        font-family: 'IBM Plex Mono', monospace; font-size: 10px; padding: 3px 10px;
        border-radius: 3px; background: rgba(255,255,255,0.12);
        border: 1px solid rgba(255,255,255,0.25); color: #fff; opacity: 0.95;
    }
    .status-bar {
        display:flex; gap:32px; padding:14px 20px;
        background: var(--secondary-background-color);
        border: 1px solid rgba(128,128,128,0.2); border-radius: 8px;
    }
    .status-bar .lbl { font-family:'IBM Plex Mono',monospace; font-size:10px; color:var(--text-color); }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# 2. ĐIỀU HƯỚNG SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
UI_ROOT = Path(__file__).resolve().parent
apply_page_config("Trợ lý AI công việc", UI_ROOT)
inject_base_css()
render_sidebar(active="home")

# ══════════════════════════════════════════════════════════════════════════════
# 3. GIAO DIỆN TRANG CHỦ (MAIN LAYOUT)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="text-align:center;padding:40px 0 20px;">
    <div class="hero-eyebrow">—— AI ASSISTANT ——</div>
    <h1 style="font-size:40px;font-weight:700;margin:0 0 16px;color:var(--text-color);">
        Trợ lý AI thông minh<br><span style="color:#0067B1;">Smart Meeting & OCR Documents</span>
    </h1>
    <p style="font-size:14px;color:var(--text-color);opacity:0.7;max-width:600px;margin:0 auto;line-height:1.6;">
        Trợ lý cuộc họp thông minh · Nhận diện giọng nói VoiceID · OCR Tài liệu & Bố cục<br>
        Xuất biên bản DOCX theo mẫu chuẩn · Xử lý cục bộ, không cần internet
    </p>
</div>
""", unsafe_allow_html=True)

# Khối chức năng trung tâm
c1, c2, c3 = st.columns([1, 2, 1])
with c2:
    st.markdown(
        '<div class="hero-card">'
        '<div style="display:flex;align-items:center;gap:12px;margin-bottom:20px;">'
        '<span style="font-size:24px;">🎙️</span>'
        '<div style="font-size:18px;font-weight:600;color:#fff;">Trợ lý AI công việc — STT & OCR văn bản</div>'
        '</div>'
        '<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:28px;">'
        '<div class="hero-feature">• Ghi âm realtime từ micro</div>'
        '<div class="hero-feature">• Phiên âm tiếng Việt streaming</div>'
        '<div class="hero-feature">• Phân tách người nói (Diart + Pyannote)</div>'
        '<div class="hero-feature">• Nhận diện định danh (Cam++ VoiceID)</div>'
        '<div class="hero-feature">• Tái tạo Layout & Xuất DOCX/PDF</div>'
        '<div class="hero-feature">• OCR Văn bản thông minh</div>'
        '<div class="hero-feature">• Xuất biên bản DOCX</div>'
        '</div>'
        '<div style="display:flex;gap:8px;flex-wrap:wrap;border-top:1px solid rgba(255,255,255,0.18);padding-top:16px;margin-bottom:8px;">'
        '<span class="hero-tag">.mic</span>'
        '<span class="hero-tag">realtime</span>'
        '<span class="hero-tag">zipformer & Deepseek</span>'
        '<span class="hero-tag">diart</span>'
        '<span class="hero-tag">VoiceID</span>'
        '<span class="hero-tag">Cam++ VoiceID</span>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    _cta1, _cta2 = st.columns(2)
    with _cta1:
        if st.button("🎙️  Bắt đầu cuộc họp", width="stretch", key="home_cta_meeting"):
            st.session_state["workspace_mode"] = "meeting"
            st.switch_page("pages/2_Workspace.py")
    with _cta2:
        if st.button("📄  Bắt đầu xử lý văn bản", width="stretch", key="home_cta_ocr"):
            st.session_state["workspace_mode"] = "ocr"
            st.switch_page("pages/2_Workspace.py")

# ── Thanh trạng thái phía dưới (Status Bar) ───────────────────────────────────
st.markdown("""
<div style="max-width:800px;margin:40px auto 0;padding:0 8px;">
    <div class="status-bar">
        <div class="lbl"><span style="opacity:0.7;">STATUS</span> &nbsp;<span style="color:#16A34A;font-weight:bold;">● READY</span></div>
        <div class="lbl"><span style="opacity:0.7;">MODE</span> &nbsp;<b>OFFLINE / LOCAL</b></div>
        <div class="lbl"><span style="opacity:0.7;">ENGINE</span> &nbsp;<b>Zipformer + Diart + Pyannote 3.0 + Cam++ VoiceID + DeepseekOCR</b></div>
    </div>
</div>
""", unsafe_allow_html=True)