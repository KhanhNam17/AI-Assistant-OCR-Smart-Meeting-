from pathlib import Path
import streamlit as st
from theme import apply_page_config, inject_base_css, render_sidebar, icon

# ══════════════════════════════════════════════════════════════════════════════
# 1. SETUP PAGE CONFIG & DESIGN SYSTEM
# ══════════════════════════════════════════════════════════════════════════════
UI_ROOT = Path(__file__).resolve().parent
apply_page_config("Trợ lý AI Công Việc", UI_ROOT)
inject_base_css()
render_sidebar(active="home")

# ══════════════════════════════════════════════════════════════════════════════
# 2. LOCAL CSS CHO TRANG CHỦ (HỖ TRỢ CẢ LIGHT & DARK)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    .home-hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #2563EB;
        background: rgba(37, 99, 235, 0.12);
        border: 1px solid rgba(37, 99, 235, 0.3);
        padding: 6px 16px;
        border-radius: 9999px;
        font-weight: 700;
        margin-bottom: 18px;
    }
    .home-title {
        font-size: 38px;
        font-weight: 800;
        letter-spacing: -0.6px;
        margin: 0 0 16px;
        color: var(--text-color);
        line-height: 1.25;
    }
    .home-title-accent {
        background: linear-gradient(135deg, #2563EB 0%, #06B6D4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: inline-block;
    }
    .home-desc {
        font-size: 14.5px;
        color: var(--text-color);
        opacity: 0.75;
        max-width: 680px;
        margin: 0 auto 30px;
        line-height: 1.6;
    }
    .hero-card-modern {
        background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 100%);
        border-radius: 18px;
        padding: 32px 34px;
        box-shadow: 0 12px 30px rgba(37, 99, 235, 0.28);
        color: #FFFFFF;
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin-bottom: 24px;
        position: relative;
        overflow: hidden;
    }
    .hero-feature-item {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 13.5px;
        color: rgba(255, 255, 255, 0.92);
        font-weight: 500;
    }
    .hero-tag-modern {
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        padding: 4px 12px;
        border-radius: 6px;
        background: rgba(255, 255, 255, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.22);
        color: #FFFFFF;
        font-weight: 600;
    }
    .action-card {
        background: var(--secondary-background-color);
        border: 1px solid rgba(148, 163, 184, 0.22);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.04);
        transition: all 0.2s ease;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .action-card:hover {
        border-color: rgba(37, 99, 235, 0.5);
        box-shadow: 0 8px 24px rgba(37, 99, 235, 0.12);
        transform: translateY(-2px);
    }
    .action-card-icon {
        width: 48px;
        height: 48px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 24px;
        margin-bottom: 14px;
    }
    .action-card-title {
        font-size: 18px;
        font-weight: 700;
        color: var(--text-color);
        margin-bottom: 8px;
    }
    .action-card-desc {
        font-size: 13.5px;
        color: var(--text-color);
        opacity: 0.72;
        line-height: 1.5;
        margin-bottom: 20px;
    }
    .status-bar-modern {
        display: flex;
        align-items: center;
        justify-content: space-around;
        gap: 20px;
        padding: 14px 24px;
        background: var(--secondary-background-color);
        border: 1px solid rgba(148, 163, 184, 0.22);
        border-radius: 14px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        flex-wrap: wrap;
    }
    .status-bar-item {
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        color: var(--text-color);
        opacity: 0.75;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .status-bar-item b {
        color: var(--text-color);
        opacity: 1;
    }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# 3. GIAO DIỆN TRANG CHỦ (MAIN LAYOUT)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="text-align:center;padding:10px 0 20px;">
    <div class="home-hero-badge">✦ SMART AI WORKSPACE</div>
    <div class="home-title">
        Trợ lý AI Cuộc họp & OCR Văn bản<br>
        <span class="home-title-accent">High-Performance Offline Intelligence</span>
    </div>
    <div class="home-desc">
        Hệ thống AI xử lý cuộc họp realtime, bóc tách người nói đa luồng VoiceID kết hợp OCR tài liệu phục hồi bố cục chuyên nghiệp. Hoạt động offline 100%, bảo mật tuyệt đối.
    </div>
</div>
""", unsafe_allow_html=True)

# Khối trung tâm
col_center = st.columns([1, 4, 1])[1]
with col_center:
    st.markdown(
        f"""
        <div class="hero-card-modern">
            <div style="display:flex;align-items:center;gap:14px;margin-bottom:20px;">
                <div style="width:44px;height:44px;border-radius:12px;background:rgba(255,255,255,0.18);display:flex;align-items:center;justify-content:center;font-size:22px;">
                    🎙️
                </div>
                <div>
                    <div style="font-size:20px;font-weight:700;color:#fff;">Smart Meeting & Document Intelligence</div>
                    <div style="font-size:12px;color:rgba(255,255,255,0.8);">Zipformer STT · Diart Speaker Diarization · Cam++ VoiceID · Deepseek OCR</div>
                </div>
            </div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:26px;">
                <div class="hero-feature-item">{icon('check', 16, 2.2, '#34D399')} Ghi âm và streaming realtime từ micro</div>
                <div class="hero-feature-item">{icon('check', 16, 2.2, '#34D399')} Phân tách người nói chính xác (Diart + Pyannote 3.0)</div>
                <div class="hero-feature-item">{icon('check', 16, 2.2, '#34D399')} Nhận dạng giọng nói theo hồ sơ VoiceID 192d</div>
                <div class="hero-feature-item">{icon('check', 16, 2.2, '#34D399')} OCR tài liệu đa trang & tái tạo layout chuẩn DOCX</div>
                <div class="hero-feature-item">{icon('check', 16, 2.2, '#34D399')} Hỗ trợ từ điển chuyên ngành (Domain Terms)</div>
                <div class="hero-feature-item">{icon('check', 16, 2.2, '#34D399')} Chạy hoàn toàn cục bộ, bảo mật dữ liệu tuyệt đối</div>
            </div>
            <div style="display:flex;gap:8px;flex-wrap:wrap;border-top:1px solid rgba(255,255,255,0.18);padding-top:18px;">
                <span class="hero-tag-modern">.mic-realtime</span>
                <span class="hero-tag-modern">zipformer-vi</span>
                <span class="hero-tag-modern">pyannote-3.0</span>
                <span class="hero-tag-modern">cam++-voiceid</span>
                <span class="hero-tag-modern">deepseek-ocr</span>
                <span class="hero-tag-modern">docx-exporter</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        st.markdown(
            f"""
            <div class="action-card">
                <div>
                    <div class="action-card-icon" style="background:rgba(37,99,235,0.15);color:#2563EB;">🎙️</div>
                    <div class="action-card-title">Cuộc họp thông minh</div>
                    <div class="action-card-desc">Bắt đầu phiên họp mới, thu âm từ micro hoặc file audio, phân tách người nói realtime và xuất biên bản cuộc họp.</div>
                </div>
            </div>
            """, unsafe_allow_html=True
        )
        if st.button("🎙️ Bắt đầu cuộc họp", type="primary", width="stretch", key="home_btn_meeting"):
            st.session_state["workspace_mode"] = "meeting"
            st.switch_page("pages/2_Workspace.py")

    with c2:
        st.markdown(
            f"""
            <div class="action-card">
                <div>
                    <div class="action-card-icon" style="background:rgba(16,185,129,0.15);color:#10B981;">📄</div>
                    <div class="action-card-title">OCR & Xử lý văn bản</div>
                    <div class="action-card-desc">Tải lên hình ảnh hoặc PDF tài liệu, nhận diện ký tự, phân tích bố cục văn bản và xuất DOCX giữ nguyên cấu trúc.</div>
                </div>
            </div>
            """, unsafe_allow_html=True
        )
        if st.button("📄 Bắt đầu OCR tài liệu", type="primary", width="stretch", key="home_btn_ocr"):
            st.session_state["workspace_mode"] = "ocr"
            st.switch_page("pages/2_Workspace.py")

    st.markdown("<div style='margin-top:28px;'></div>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="status-bar-modern">
            <div class="status-bar-item">
                <span style="opacity:0.7;">STATUS</span>
                <span style="color:#10B981;font-weight:700;">● SYSTEM READY</span>
            </div>
            <div class="status-bar-item">
                <span style="opacity:0.7;">MODE</span>
                <b>OFFLINE LOCALHOST</b>
            </div>
            <div class="status-bar-item">
                <span style="opacity:0.7;">PIPELINE</span>
                <b>Zipformer + Diart + Cam++ + DeepseekOCR</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
