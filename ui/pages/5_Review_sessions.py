"""
ui/pages/5_Review_sessions.py
==============================
Trang xem lại (Review) — gộp 2 hệ thống vào chung 1 trang, chia theo tab:
  • Tab 1: Cuộc họp (Smart Meeting)  — review transcript, đổi tên người nói, xuất DOCX/MP3
  • Tab 2: Tài liệu (OCR & Dịch thuật) — đối chiếu file gốc vs kết quả OCR/Dịch

Cấu trúc theo đúng mô hình đã dùng ở ui/pages/1_Dashboard.py.
"""

import sys
import os
import json
import time
import base64
import html as _html
from pathlib import Path
import streamlit as st

# ══════════════════════════════════════════════════════════════════════════════
# 1. SETUP ĐƯỜNG DẪN & THEME CHUNG (giống Dashboard)
# ══════════════════════════════════════════════════════════════════════════════
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
UI_ROOT = Path(__file__).resolve().parent.parent
for p in (REPO_ROOT, UI_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from theme import apply_page_config, inject_base_css, render_sidebar, render_page_header, icon, render_workspace_jump

apply_page_config("Xem lại phiên xử lý — Review Sessions", UI_ROOT)
inject_base_css()
render_sidebar(active="review")

DEFAULT_OUT = REPO_ROOT / "output"                      # dữ liệu OCR
SESSIONS_DIR = os.path.join("data", "sessions")          # dữ liệu Smart Meeting

# ══════════════════════════════════════════════════════════════════════════════
# 2. GỘP CSS CỦA CẢ 2 TRANG REVIEW
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    /* ── CSS Smart Meeting Review ── */
    .rv-card-title {
        font-family: 'IBM Plex Mono', monospace; font-size: 11px;
        letter-spacing: 2px; text-transform: uppercase; color: #0067B1;
        margin-bottom: 14px; font-weight: 600; display: flex; align-items: center; gap: 8px;
    }
    .stButton > button[kind="primary"],
    .stDownloadButton > button[kind="primary"] {
        background: #0067B1 !important; border: none !important;
        font-weight: 600 !important; color: white !important;
    }
    .stButton > button[kind="primary"]:hover,
    .stDownloadButton > button[kind="primary"]:hover { background: #004C87 !important; }

    /* ── CSS OCR Review ── */
    .stSelectbox label { font-size: 1.05rem !important; font-weight: 700 !important; }
    .vnpt-translate-card {
        border-radius: var(--vnpt-radius-md);
        padding: 16px 20px; text-align: center; margin-bottom: 14px;
    }
    .vnpt-translate-card.on {
        background: var(--vnpt-blue-tint-2); border: 1px solid var(--vnpt-blue-light);
    }
    .vnpt-translate-card.off {
        background: rgba(148,163,184,0.10); border: 1px solid var(--border-color);
    }
    .vnpt-translate-card .badge-label {
        font-weight: 800; font-size: 13.5px; display: flex; align-items: center;
        justify-content: center; gap: 8px; margin-bottom: 4px;
    }
    .vnpt-translate-card.on .badge-label { color: var(--vnpt-blue); }
    .vnpt-translate-card.off .badge-label { color: var(--text-color); opacity: 0.7; }
    .vnpt-translate-direction { font-weight: 700; font-size: 14px; opacity: 0.85; }
    .vnpt-info-field { }
    .vnpt-info-label {
        font-size: 11.5px; font-weight: 800; letter-spacing: 0.4px;
        color: var(--text-color); opacity: 0.55; text-transform: uppercase; margin-bottom: 4px;
    }
    .vnpt-info-value { font-size: 14px; font-weight: 600; word-wrap: break-word; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# 3. TIÊU ĐỀ CHÍNH
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="font-size:11px;letter-spacing:3px;text-transform:uppercase;
            color:var(--text-color);opacity:0.7;padding-bottom:12px;
            border-bottom:1px solid rgba(128,128,128,0.2);margin-bottom:16px;">
    📂  Xem lại phiên xử lý — Review Sessions
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# 4. HELPER DÙNG CHUNG CHO TAB SMART MEETING
# ══════════════════════════════════════════════════════════════════════════════
SPEAKER_COLORS = ["#0067B1", "#0D9488", "#7C3AED", "#D97706", "#DB2777", "#475569"]


def _speaker_color(idx: int) -> str:
    return SPEAKER_COLORS[idx % len(SPEAKER_COLORS)]


class FileTurn:
    def __init__(self, speaker, start, end, text, voice_confirmed=False, confidence=1.0):
        self.speaker = speaker
        self.start = float(start)
        self.end = float(end)
        self.text = text
        self.voice_confirmed = voice_confirmed
        self.confidence = confidence


_WAVEFORM_MAX_BYTES = 25 * 1024 * 1024


def render_waveform_player(audio_bytes: bytes, height: int = 70) -> None:
    if not audio_bytes:
        st.warning("⚠️ Không tìm thấy file âm thanh gốc (audio.wav) trong thư mục lưu trữ.")
        return

    if len(audio_bytes) > _WAVEFORM_MAX_BYTES:
        st.caption(f"ℹ️ File âm thanh khá lớn ({len(audio_bytes)/1_048_576:.0f} MB) — "
                    f"dùng trình phát mặc định để tránh giật lag trình duyệt.")
        st.audio(audio_bytes, format="audio/wav")
        return

    b64 = base64.b64encode(audio_bytes).decode()
    html = f"""
    <div id="waveform" style="cursor:pointer;"></div>
    <div style="display:flex;align-items:center;gap:10px;margin-top:10px;">
        <button id="playBtn" style="background:#0067B1;border:none;color:white;
            width:34px;height:34px;border-radius:50%;cursor:pointer;font-size:13px;
            flex-shrink:0;">▶</button>
        <span id="timeLabel" style="font-family:monospace;font-size:12px;color:#8b949e;">0:00 / 0:00</span>
    </div>
    <script src="https://unpkg.com/wavesurfer.js@7/dist/wavesurfer.min.js"></script>
    <script>
        const wavesurfer = WaveSurfer.create({{
            container: '#waveform',
            waveColor: '#4da6e8',
            progressColor: '#0067B1',
            cursorColor: '#ffffff',
            barWidth: 2,
            barGap: 1,
            height: {height},
            url: 'data:audio/wav;base64,{b64}'
        }});
        const playBtn = document.getElementById('playBtn');
        playBtn.onclick = () => {{
            wavesurfer.playPause();
            playBtn.innerText = wavesurfer.isPlaying() ? '⏸' : '▶';
        }};
        wavesurfer.on('finish', () => {{ playBtn.innerText = '▶'; }});
        wavesurfer.on('audioprocess', () => {{
            const cur = wavesurfer.getCurrentTime();
            const dur = wavesurfer.getDuration();
            const fmt = s => `${{Math.floor(s/60)}}:${{String(Math.floor(s%60)).padStart(2,'0')}}`;
            document.getElementById('timeLabel').innerText = `${{fmt(cur)}} / ${{fmt(dur||0)}}`;
        }});
        wavesurfer.on('ready', () => {{
            const dur = wavesurfer.getDuration();
            const fmt = s => `${{Math.floor(s/60)}}:${{String(Math.floor(s%60)).padStart(2,'0')}}`;
            document.getElementById('timeLabel').innerText = `0:00 / ${{fmt(dur)}}`;
        }});
        wavesurfer.on('interaction', () => {{
            if (!wavesurfer.isPlaying()) {{ /* seek only, không tự play */ }}
        }});
    </script>
    """
    st.iframe(html, height=height + 60)


def _ring_html(pct: float, color: str, size: int = 92) -> str:
    pct_clamped = max(0.0, min(100.0, pct))
    deg = pct_clamped * 3.6
    return (
        f'<div style="width:{size}px;height:{size}px;border-radius:50%;'
        f'background:conic-gradient({color} {deg}deg, rgba(128,128,128,0.15) {deg}deg);'
        f'display:flex;align-items:center;justify-content:center;margin:0 auto;">'
        f'<div style="width:{size-14}px;height:{size-14}px;border-radius:50%;'
        f'background:var(--background-color, #ffffff);display:flex;'
        f'align-items:center;justify-content:center;flex-direction:column;">'
        f'<span style="font-family:monospace;font-size:15px;font-weight:700;'
        f'color:{color};">{pct_clamped:.1f}%</span>'
        f'</div></div>'
    )


def speaker_editor_ring(stats: dict, key_prefix: str) -> dict:
    if not stats:
        st.info("Chưa phát hiện người nói")
        return {}

    default_names = ["Người số 1", "Người số 2", "Người số 3", "Người số 4", "Khác"]
    name_map = {}
    items = list(stats.items())
    n_cols = min(4, len(items)) or 1
    cols = st.columns(n_cols)

    for i, (spk, info) in enumerate(items):
        color = _speaker_color(i)
        default = default_names[i] if i < len(default_names) else f"Người {i+1}"
        pct = info.get("percent", 0)
        dur = f"{info.get('duration', 0):.0f}s"
        n_turns = info.get("turns", 0)

        with cols[i % n_cols]:
            st.markdown(
                f'<div style="text-align:center;font-size:10px;font-weight:700;color:{color};'
                f'letter-spacing:1px;text-transform:uppercase;margin-bottom:8px;'
                f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{_html.escape(spk)}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(_ring_html(pct, color), unsafe_allow_html=True)
            st.markdown(
                f'<div style="text-align:center;font-size:11px;color:#8b949e;'
                f'margin-top:6px;margin-bottom:10px;">{dur} · {n_turns} lượt</div>',
                unsafe_allow_html=True,
            )
            name_map[spk] = st.text_input(
                label=f"Tên cho {spk}",
                value=spk,
                key=f"{key_prefix}_{spk}",
                placeholder=f"VD: {default}",
                label_visibility="collapsed",
            )

    return name_map


def render_voiceids_loaded(voiceids_loaded: list) -> None:
    if not voiceids_loaded:
        st.caption("ℹ️ Không có dữ liệu — phiên này lưu trước khi tính năng ghi lại "
                    "danh sách VoiceID được thêm, hoặc phiên không dùng VoiceID.")
        return

    html_content = "<div style='display: flex; flex-wrap: wrap; gap: 12px;'>"
    for i, name in enumerate(voiceids_loaded):
        color = _speaker_color(i)
        html_content += (
            f'<div style="background: rgba(128,128,128,0.06); '
            f'border-left: 3px solid {color}; border-radius: 0 6px 6px 0; '
            f'padding: 8px 16px; display: flex; align-items: center; gap: 8px;">'
            f'<span style="font-size: 14px;">🪪</span>'
            f'<span style="font-weight: 600; color: var(--text-color); font-size: 13px;">'
            f'{_html.escape(name)}</span>'
            f'</div>'
        )
    html_content += "</div>"
    st.markdown(html_content, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# 5. HELPER DÙNG CHUNG CHO TAB OCR
# ══════════════════════════════════════════════════════════════════════════════
try:
    from streamlit_pdf_viewer import pdf_viewer as _pdf_viewer
    _HAS_PDF_VIEWER = True
except Exception:
    _HAS_PDF_VIEWER = False


def load_ocr_session_history():
    history = []
    if DEFAULT_OUT.exists():
        for d in DEFAULT_OUT.iterdir():
            if d.is_dir() and (d / "metadata.json").exists():
                try:
                    with open(d / "metadata.json", "r", encoding="utf-8") as f:
                        history.append(json.load(f))
                except Exception:
                    pass
    history.sort(key=lambda x: x["id"], reverse=True)
    return history


def render_pdf(pdf_path: Path, *, height: int = 720, key: str | None = None) -> None:
    if not pdf_path.exists():
        st.caption(f"Không tìm thấy file: `{pdf_path.name}`")
        return

    b64 = base64.b64encode(pdf_path.read_bytes()).decode("ascii")
    st.markdown(
        f'<iframe src="data:application/pdf;base64,{b64}" '
        f'width="100%" height="{height}" type="application/pdf" '
        f'style="border:1px solid var(--border-color); border-radius:8px; background:var(--secondary-background-color);">'
        f'</iframe>',
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


def ocr_file_info(path: Path) -> dict:
    info = {"Tên file": path.name, "Dung lượng": f"{path.stat().st_size / 1024:.1f} KB", "Định dạng": path.suffix.upper().lstrip(".")}
    if path.suffix.lower() == ".pdf":
        try:
            import fitz
            with fitz.open(str(path)) as doc:
                info["Số trang"] = str(len(doc))
        except Exception:
            pass
    return info


# ══════════════════════════════════════════════════════════════════════════════
# 6. KHỞI TẠO TABS (giống Dashboard)
# ══════════════════════════════════════════════════════════════════════════════
tab_meeting, tab_ocr = st.tabs(["🎙️ Cuộc họp thông minh", "📄 OCR Tài liệu"])

# ==============================================================================
# TAB 1: REVIEW CUỘC HỌP (SMART MEETING)
# ==============================================================================
with tab_meeting:
    render_workspace_jump("meeting")
    os.makedirs(SESSIONS_DIR, exist_ok=True)

    saved_sessions = []
    if os.path.exists(SESSIONS_DIR):
        saved_sessions = sorted(
            [d for d in os.listdir(SESSIONS_DIR)
             if os.path.isdir(os.path.join(SESSIONS_DIR, d))],
            key=lambda d: os.path.getmtime(os.path.join(SESSIONS_DIR, d)),
            reverse=True,
        )

    if not saved_sessions:
        st.markdown("""
        <div style="background:var(--secondary-background-color);border:1px solid rgba(128,128,128,0.2);
                    border-radius:10px;padding:60px 20px; text-align:center; color:var(--text-color); opacity:0.6;">
            <span style="font-size:36px; display:block; margin-bottom:12px;">📂</span>
            Chưa có cuộc họp nào được lưu vào lịch sử.<br>
            Hãy thực hiện cuộc họp tại mục 🎙️ <b>Live Mode</b> và bấm <b>Lưu vào Lịch sử</b> để hiển thị ở đây.
        </div>
        """, unsafe_allow_html=True)
    else:
        if "rv_selectbox_session" not in st.session_state:
            st.session_state["rv_selectbox_session"] = saved_sessions[0]
        if st.session_state["rv_selectbox_session"] not in saved_sessions:
            st.session_state["rv_selectbox_session"] = saved_sessions[0]

        st.markdown('<div style="font-size:10px;letter-spacing:2px;text-transform:uppercase;color:var(--text-color);opacity:0.7;margin-bottom:8px;">① Chọn phiên cuộc họp</div>', unsafe_allow_html=True)
        selected_session = st.selectbox(
            "Danh sách cuộc họp đã lưu",
            options=saved_sessions,
            index=saved_sessions.index(st.session_state["rv_selectbox_session"]),
            label_visibility="collapsed",
            key="rv_selectbox_widget",
        )

        # ── Phát hiện thay đổi session và xóa toàn bộ state cũ ──
        if selected_session != st.session_state.get("rv_loaded_session"):
            stale_prefixes = ("tv_", "rev_spk_")
            for key in list(st.session_state.keys()):
                if any(key.startswith(p) for p in stale_prefixes):
                    del st.session_state[key]

            sdir_new = os.path.join(SESSIONS_DIR, selected_session)
            transcript_path_new = os.path.join(sdir_new, "transcript.json")
            metadata_path_new = os.path.join(sdir_new, "metadata.json")

            turns = []
            if os.path.isfile(transcript_path_new):
                with open(transcript_path_new, "r", encoding="utf-8") as f:
                    raw_turns = json.load(f)
                turns = [
                    FileTurn(
                        speaker=t["speaker"],
                        start=t["start"],
                        end=t["end"],
                        text=t["text"],
                        voice_confirmed=t.get("voice_confirmed", False),
                        confidence=t.get("confidence", 1.0),
                    ) for t in raw_turns
                ]

            meta = {}
            if os.path.isfile(metadata_path_new):
                with open(metadata_path_new, "r", encoding="utf-8") as f:
                    meta = json.load(f)

            st.session_state["rv_turns"] = turns
            st.session_state["rv_meta"] = meta
            st.session_state["rv_loaded_session"] = selected_session
            st.session_state["rv_selectbox_session"] = selected_session
            st.rerun()

        sdir = os.path.join(SESSIONS_DIR, selected_session)
        audio_path = os.path.join(sdir, "audio.wav")
        transcript_path = os.path.join(sdir, "transcript.json")
        metadata_path = os.path.join(sdir, "metadata.json")

        turns_list = st.session_state["rv_turns"]
        meta_data = st.session_state["rv_meta"]

        st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)

        stats_current = meta_data.get("stats", {})
        if not stats_current and turns_list:
            spk_dur, spk_turns = {}, {}
            for t in turns_list:
                dur = t.end - t.start
                spk_dur[t.speaker] = spk_dur.get(t.speaker, 0.0) + dur
                spk_turns[t.speaker] = spk_turns.get(t.speaker, 0) + 1
            total_dur = sum(spk_dur.values()) or 1.0
            stats_current = {
                spk: {
                    "duration": spk_dur[spk],
                    "turns": spk_turns[spk],
                    "percent": round(spk_dur[spk] / total_dur * 100, 1),
                }
                for spk in spk_dur
            }

        row1_c1, row1_c2 = st.columns(2, gap="medium")

        with row1_c1:
            with st.container(border=True):
                st.markdown(
                    f'<div class="rv-card-title">📋 Thông tin phiên</div>'
                    f'<div style="font-size:16px;font-weight:700;color:var(--text-color);'
                    f'margin-bottom:4px;">{_html.escape(selected_session.replace("_", " "))}</div>'
                    f'<div style="font-size:12px;color:#8b949e;margin-bottom:16px;">'
                    f'📅 Ngày tạo: {meta_data.get("date", "Không rõ")}'
                    f'&nbsp;·&nbsp;💬 Tổng số: {len(turns_list)} lượt phát biểu</div>',
                    unsafe_allow_html=True,
                )

                audio_bytes = b""
                if os.path.isfile(audio_path):
                    with open(audio_path, "rb") as f:
                        audio_bytes = f.read()
                render_waveform_player(audio_bytes)

        with row1_c2:
            with st.container(border=True):
                st.markdown('<div class="rv-card-title">② Gán lại tên người nói</div>', unsafe_allow_html=True)

                if stats_current:
                    from core.aligner import rename_turns

                    name_map = speaker_editor_ring(stats_current, key_prefix=f"rev_spk_{selected_session}")

                    if st.button("🔄 Xác nhận đổi tên người nói", key="rev_btn_rename", type="primary", use_container_width=True):
                        st.session_state["rv_turns"] = rename_turns(turns_list, name_map)

                        new_stats = {}
                        for old_spk, s_val in stats_current.items():
                            new_name = name_map.get(old_spk, old_spk)
                            new_stats[new_name] = s_val
                        st.session_state["rv_meta"]["stats"] = new_stats

                        updated_turns_raw = [
                            {
                                "start": t.start,
                                "end": t.end,
                                "speaker": t.speaker,
                                "text": t.text,
                                "voice_confirmed": getattr(t, "voice_confirmed", False),
                                "confidence": getattr(t, "confidence", 1.0),
                            }
                            for t in st.session_state["rv_turns"]
                        ]
                        with open(transcript_path, "w", encoding="utf-8") as f:
                            json.dump(updated_turns_raw, f, ensure_ascii=False, indent=2)
                        with open(metadata_path, "w", encoding="utf-8") as f:
                            json.dump(st.session_state["rv_meta"], f, ensure_ascii=False, indent=2)

                        st.success("🎉 Đã đồng bộ tên mới xuống cơ sở dữ liệu!")
                        st.rerun()
                else:
                    st.caption("Chưa có dữ liệu thống kê người nói cho phiên này.")

        st.markdown("<div style='margin:10px 0;'></div>", unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown('<div class="rv-card-title">🪪 VoiceID đã nạp trong phiên</div>', unsafe_allow_html=True)
            voiceids_loaded = meta_data.get("voiceids_loaded", [])
            render_voiceids_loaded(voiceids_loaded)

        st.markdown("<div style='margin:10px 0;'></div>", unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown('<div class="rv-card-title">③ Nội dung hội thoại & hiệu chỉnh</div>', unsafe_allow_html=True)

            if turns_list:
                from components.transcript_viewer import full as show_full
                show_full(turns_list, editable=True, key_prefix=f"tv_{selected_session}")

                if st.button("💾 Lưu các chỉnh sửa chữ", type="primary",
                             use_container_width=True, key="rev_btn_save_text"):
                    updated_turns_raw = [
                        {
                            "start": t.start,
                            "end": t.end,
                            "speaker": t.speaker,
                            "text": t.text,
                            "voice_confirmed": getattr(t, "voice_confirmed", False),
                            "confidence": getattr(t, "confidence", 1.0),
                        }
                        for t in turns_list
                    ]
                    with open(transcript_path, "w", encoding="utf-8") as f:
                        json.dump(updated_turns_raw, f, ensure_ascii=False, indent=2)
                    st.success("✅ Đã lưu mọi chỉnh sửa nội dung văn bản!")
            else:
                st.info("Không có dữ liệu lượt nói trong file transcript.json.")

        st.markdown("<hr style='border-color:rgba(128,128,128,0.2);margin:32px 0 16px;'>", unsafe_allow_html=True)
        st.markdown('<div style="font-size:10px;letter-spacing:2px;text-transform:uppercase;color:var(--text-color);opacity:0.7;margin-bottom:14px;">④ Xuất văn bản biên bản</div>', unsafe_allow_html=True)

        export_data = b""
        if turns_list:
            try:
                from components.export_docx import export_to_docx
                export_data = export_to_docx(turns=turns_list, session_name=selected_session)
            except Exception as e:
                st.error(f"Lỗi chuẩn bị xuất file DOCX: {e}")

        ex_c1, ex_c2 = st.columns(2)
        with ex_c1:
            st.download_button(
                label="📄 Xuất biên bản cuộc họp (.docx)",
                data=export_data,
                file_name=f"Bien_ban_{selected_session}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                type="primary",
                disabled=not bool(export_data),
                use_container_width=True,
                key="rev_dl_docx",
            )

        with ex_c2:
            mp3_bytes = b""
            if os.path.isfile(audio_path):
                try:
                    import subprocess, tempfile
                    ffmpeg_cmd = os.getenv("FFMPEG_PATH", "ffmpeg").strip().strip('"').strip("'")
                    tmp_mp3 = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
                    tmp_mp3.close()
                    subprocess.run([
                        ffmpeg_cmd, "-y", "-i", audio_path,
                        "-codec:a", "libmp3lame", "-qscale:a", "2", tmp_mp3.name
                    ], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
                    with open(tmp_mp3.name, "rb") as f:
                        mp3_bytes = f.read()
                    os.unlink(tmp_mp3.name)
                except Exception:
                    with open(audio_path, "rb") as f:
                        mp3_bytes = f.read()

            st.download_button(
                label="🎵 Tải file âm thanh cuộc họp",
                data=mp3_bytes,
                file_name=f"Audio_{selected_session}.mp3" if ".mp3" in str(mp3_bytes[:10]) else f"Audio_{selected_session}.wav",
                mime="audio/mpeg" if ".mp3" in str(mp3_bytes[:10]) else "audio/wav",
                disabled=not bool(mp3_bytes),
                use_container_width=True,
                key="rev_dl_audio",
            )

# ==============================================================================
# TAB 2: REVIEW TÀI LIỆU (OCR)
# ==============================================================================
with tab_ocr:
    render_workspace_jump("ocr")
    if "session_history" not in st.session_state:
        st.session_state.session_history = load_ocr_session_history()

    ocr_history = st.session_state.get("session_history", [])

    if not ocr_history:
        with st.container(border=True):
            st.info("Chưa có phiên xử lý nào trong phiên làm việc hiện tại. Vui lòng chạy OCR ở trang 📄 OCR Tài liệu trước.")
    else:
        session_dict = {entry["label"]: entry for entry in ocr_history}

        selected_label = st.selectbox(
            "Chọn phiên tài liệu",
            options=list(session_dict.keys()),
            index=0,
            key="ocr_review_select",
        )

        session = session_dict[selected_label]
        src_path = Path(session["source_path"])
        run_dir = Path(session["run_dir"])
        stem = session["stem"]
        exit_code = session.get("exit_code", 0)
        target_pdf = run_dir / f"{stem}_ocr.pdf"
        if not target_pdf.exists():
            target_pdf_alt = run_dir / f"{stem}_translated.pdf"
            if target_pdf_alt.exists():
                target_pdf = target_pdf_alt

        col_info, col_status = st.columns([6, 4], gap="large")

        with col_info:
            with st.container(border=True):
                st.subheader("Thông tin file gốc")
                if src_path.exists():
                    info = ocr_file_info(src_path)
                    ic1, ic2, ic3, ic4 = st.columns(4)
                    fields = [
                        (ic1, "Tên file", info.get("Tên file", "--")),
                        (ic2, "Định dạng", info.get("Định dạng", "--")),
                        (ic3, "Dung lượng", info.get("Dung lượng", "--")),
                        (ic4, "Số trang", info.get("Số trang", "--")),
                    ]
                    for col, label, value in fields:
                        col.markdown(
                            f'<div class="vnpt-info-field"><div class="vnpt-info-label">{label}</div>'
                            f'<div class="vnpt-info-value">{value}</div></div>',
                            unsafe_allow_html=True,
                        )
                else:
                    st.error("Không tìm thấy file gốc trên hệ thống.")

        with col_status:
            with st.container(border=True):
                st.subheader("Trạng thái xử lý")
                status_badge = (
                    f'<span class="vnpt-badge vnpt-badge-on">{icon("check", 12)} Thành công</span>'
                    if exit_code == 0 else
                    f'<span class="vnpt-badge vnpt-badge-off">{icon("cross", 12)} Lỗi ({exit_code})</span>'
                )
                pdf_ready = target_pdf.exists()
                pdf_badge = (
                    f'<span class="vnpt-badge vnpt-badge-on">{icon("check", 12)} Đã sinh PDF</span>'
                    if pdf_ready else
                    f'<span class="vnpt-badge vnpt-badge-neutral">{icon("clock", 12)} Chưa có PDF</span>'
                )
                st.markdown(
                    f"""
                    <div style="display:flex;gap:12px;align-items:center;margin-top:6px;margin-bottom:8px;">
                        {status_badge} {pdf_badge}
                    </div>
                    <div style="font-size:12px;opacity:0.75;margin-top:4px;">
                        Thư mục: <code>{run_dir.name}</code>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.divider()

        sc1, sc2 = st.columns(2, gap="large")

        with sc1:
            st.subheader("Hiển thị file nạp vào")
            with st.container(border=True):
                render_source(src_path, key=f"review_src_{session['id']}")

        with sc2:
            st.subheader("Hiển thị file kết quả (Đã căn Layout)")
            with st.container(border=True):
                if target_pdf.exists():
                    render_pdf(target_pdf, key=f"review_dst_{session['id']}")
                else:
                    st.caption("File PDF kết quả chưa sẵn sàng. Bạn có thể cần quay lại trang 📄 OCR Tài liệu và bấm **Xử lý Layout**.")