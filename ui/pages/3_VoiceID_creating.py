"""
pages/3_VoiceID.py  — v2
------------------------
THAY ĐỔI so với v1:
    - Bỏ nút ✏️ chỉnh sửa (và popup edit) — chỉ giữ nút xoá
    - Thêm nút "Add data to exist VoiceID" nằm cạnh "Add new VoiceID"
      → chọn người từ dropdown + upload thêm audio → gọi add_audio_to_voiceid()
    - Đơn giản hóa bảng: bỏ cột "Vân tay AI", gộp trạng thái vào checkbox RAM
    - load_vectors_for_entry() dùng thay vì đọc vector_path đơn lẻ
"""

import sys
import os
import time
import queue
import io
import wave
import numpy as np

import base64
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

# ══════════════════════════════════════════════════════════════════════════════
# SETUP ĐƯỜNG DẪN & THEME CHUNG (đồng bộ với các page khác)
# ══════════════════════════════════════════════════════════════════════════════
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
UI_ROOT   = Path(__file__).resolve().parent.parent
for _p in (REPO_ROOT, UI_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from theme import apply_page_config, inject_base_css, render_sidebar

try:
    import sounddevice as sd
except ImportError:
    sd = None

TARGET_SR = 16000
CHUNK_SIZE = 1024

try:
    from core.voiceid_database import (
        add_voiceid, add_audio_to_voiceid, delete_voiceid, update_voiceid,
        get_all_entries, get_db_stats, get_entry,
        get_preview_paths, get_vector_paths,
        get_groups, add_group,
    )
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False

try:
    from components.voiceid_popup import voiceid_popup
    POPUP_AVAILABLE = True
except ImportError:
    POPUP_AVAILABLE = False



# ══════════════════════════════════════════════════════════════════════════════
apply_page_config("VoiceID — Quản lý giọng nói", UI_ROOT)
inject_base_css()
render_sidebar(active="voiceid")

# CSS đặc thù riêng cho trang VoiceID (stat-card, empty-state, nút primary...)
st.markdown("""
<style>
    .stButton > button[kind="primary"],
    .stDownloadButton > button[kind="primary"] {
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%) !important;
        border: none !important;
        font-weight: 600 !important;
        color: white !important;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.35) !important;
        border-radius: 10px !important;
    }
    .stButton > button[kind="primary"]:hover {
        filter: brightness(1.08) !important;
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.45) !important;
    }
    hr { border-color: rgba(148, 163, 184, 0.25) !important; }
    .stat-card {
        background: var(--secondary-background-color);
        border: 1px solid rgba(148, 163, 184, 0.22);
        border-radius: 14px;
        padding: 18px 20px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        transition: all 0.2s ease;
    }
    .stat-card:hover {
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.12);
        border-color: rgba(37, 99, 235, 0.4);
    }
    .stat-num {
        font-family: 'JetBrains Mono', monospace;
        font-size: 28px;
        font-weight: 800;
        color: #2563EB;
        display: block;
    }
    .stat-lbl {
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: var(--text-color);
        opacity: 0.65;
        margin-top: 4px;
        display: block;
        font-weight: 600;
    }
    .empty-state {
        padding: 60px 20px;
        text-align: center;
        color: var(--text-color);
        opacity: 0.65;
        font-size: 14px;
    }
    .empty-icon { font-size: 36px; margin-bottom: 12px; display: block; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="font-family:'JetBrains Mono',monospace;font-size:11px;letter-spacing:2px;text-transform:uppercase;
            color:#2563EB;padding-bottom:12px;border-bottom:1px solid rgba(148,163,184,0.22);margin-bottom:24px;">
    🪪  QUẢN LÝ VOICEID — CƠ SỞ DỮ LIỆU NHẬN DIỆN GIỌNG NÓI
</div>
""", unsafe_allow_html=True)

if not DB_AVAILABLE:
    st.error("⚠️ Không tìm thấy `core/voiceid_database.py`.")
    st.stop()

# ── Session state ──────────────────────────────────────────────────────────────
for k, v in {
    "vid_show_add_popup"    : False,
    "vid_show_adddata_popup": False,   # popup mới: add data to exist
    "vid_edit_target"       : None,
    "vid_show_creategroup_popup": False,
    "vid_show_move_popup"   : False,
    "vid_delete_confirm"    : None,
    "vid_refresh_ts"        : 0.0,
    # Các biến quản lý bộ thu âm SoundDevice cho Add Data
    "adddata_recording"     : False,
    "adddata_rec_start"     : 0.0,
    "adddata_frames"        : [],
    "adddata_wav"           : None,
    "adddata_stream"        : None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

if "adddata_audio_q" not in st.session_state:
    st.session_state["adddata_audio_q"] = queue.Queue()

# Bổ sung thêm CSS cho dấu chấm đỏ nhấp nháy
st.markdown("<style>@keyframes rec-blink { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } } .rec-dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; background: #EF4444; animation: rec-blink 1s ease-in-out infinite; margin-right: 8px; }</style>", unsafe_allow_html=True)

def _toast(kind, msg):
    if kind == "success": st.toast(f"✅ {msg}", icon="✅")
    elif kind == "error": st.toast(f"❌ {msg}", icon="❌")
    else: st.toast(msg)

# ── Stats ──────────────────────────────────────────────────────────────────────
stats = get_db_stats()

# Định nghĩa danh sách nhóm ở ngay đây để dùng cho cả việc đếm Stats và vẽ UI phía dưới
VOICEID_GROUPS = get_groups()

# Chia làm 3 cột
s1, s2, s3 = st.columns(3)

def _stat(num, lbl):
    return (f'<div class="stat-card"><span class="stat-num">{num}</span>'
            f'<span class="stat-lbl">{lbl}</span></div>')

s1.markdown(_stat(stats["total"],                                   "TỔNG VOICEID"),     unsafe_allow_html=True)
s2.markdown(_stat(len(VOICEID_GROUPS),                              "SỐ NHÓM"),          unsafe_allow_html=True)
s3.markdown(_stat(stats.get("total_files", stats["has_vector"]),    "TỔNG FILE AUDIO"),  unsafe_allow_html=True)

st.markdown("<div style='margin:20px 0 0;'></div>", unsafe_allow_html=True)

# ── Toolbar: tiêu đề + 2 nút ──────────────────────────────────────────────────
tb_left, tb_move, tb_edit, tb_mid, tb_right = st.columns([1.5, 1, 1, 1.2, 1.2])
with tb_left:
    st.markdown(
        '<div style="font-size:10px;letter-spacing:2px;text-transform:uppercase;'
        'color:var(--text-color);opacity:0.7;padding-top:6px;">① Database VoiceID</div>',
        unsafe_allow_html=True)
with tb_move:
    if st.button("🔄 Move VoiceID", width="stretch", help="Chuyển VoiceID sang nhóm khác"):
        st.session_state.vid_show_move_popup    = True
        st.session_state.vid_show_edit_popup    = False
        st.session_state.vid_show_adddata_popup = False
        st.session_state.vid_show_add_popup     = False
        st.session_state.vid_delete_confirm     = None
        st.rerun()
with tb_edit:
    if st.button("➕ Tạo nhóm mới", use_container_width=True,
                 help="Tạo thêm 1 nhóm VoiceID mới"):
        st.session_state.vid_show_creategroup_popup = True
        st.session_state.vid_show_adddata_popup     = False
        st.session_state.vid_show_add_popup         = False
        st.session_state.vid_show_move_popup        = False
        st.session_state.vid_delete_confirm         = None
        st.rerun()
with tb_mid:
    if st.button("➕ Add data to exist", width="stretch",
                 help="Thêm file audio vào VoiceID đã có"):
        st.session_state.vid_show_adddata_popup = True
        st.session_state.vid_show_edit_popup    = False
        st.session_state.vid_show_add_popup     = False
        st.session_state.vid_delete_confirm     = None
        st.rerun()
with tb_right:
    if st.button("＋ Add new VoiceID", type="primary", width="stretch",
                 help="Tạo lập 1 voiceid mới vào cơ sở dữ liệu",
                 key="btn_open_add"):
        st.session_state.vid_show_add_popup     = True
        st.session_state.vid_show_adddata_popup = False
        st.session_state.vid_show_edit_popup    = False
        st.session_state.vid_delete_confirm     = None
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# POPUP: MOVE VOICEID GROUP (CHUYỂN NHÓM)
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.get("vid_show_move_popup"):
    entries_all = get_all_entries()
    if not entries_all:
        st.warning("Chưa có VoiceID nào để chuyển. Hãy thêm mới trước.")
        if st.button("Đóng##move_close"):
            st.session_state.vid_show_move_popup = False; st.rerun()
    else:
        with st.container(border=True):
            st.markdown(
                '<div style="font-family:\'IBM Plex Mono\',monospace;font-size:11px;'
                'letter-spacing:2px;text-transform:uppercase;color:#2563EB;'
                'margin-bottom:14px;">// CHUYỂN NHÓM VOICEID</div>',
                unsafe_allow_html=True)

            # Dropdown 1: Chọn VoiceID cần chuyển (hiển thị luôn tên nhóm hiện tại cho dễ nhìn)
            name_to_vid_move = {f"{e['name']} (Đang ở: {e.get('group', 'Nhóm 1: Toạ đàm')})": e["id"] for e in entries_all}
            chosen_label_move = st.selectbox(
                "Chọn VoiceID cần chuyển nhóm",
                options=list(name_to_vid_move.keys()),
                key="move_select_name",
            )
            chosen_vid_move = name_to_vid_move[chosen_label_move]

            # Dropdown 2: Chọn nhóm đích muốn chuyển tới
            VOICEID_GROUPS = get_groups()
            target_group = st.selectbox("Chọn nhóm đích", options=VOICEID_GROUPS, key="move_target_group")

            st.divider()
            c1, c2, c3 = st.columns([3, 1, 1])
            cancel_move = c2.button("Huỷ", key="move_cancel", width="stretch")
            submit_move = c3.button("Xác nhận", key="move_submit", type="primary", width="stretch")

            if cancel_move:
                st.session_state.vid_show_move_popup = False
                st.rerun()

            if submit_move:
                ok, msg = update_voiceid(chosen_vid_move, group=target_group)
                st.session_state.vid_show_move_popup = False
                if ok:
                    _toast("success", f"Đã chuyển sang {target_group} thành công!")
                else:
                    _toast("error", msg)
                st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# POPUP: ADD NEW
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.vid_show_add_popup and POPUP_AVAILABLE:
    with st.container(border=True):
        done, result = voiceid_popup(mode="add", key="popup_add")
        if done:
            st.session_state.vid_show_add_popup = False
            if result:
                success_count = 0; last_error = ""
                for wav_bytes in (result.get("wav_bytes_list") or []):
                    ok, msg = add_voiceid(result["name"], wav_bytes, result["source"], result.get("group", "Nhóm 1: Toạ đàm"))
                    if ok: success_count += 1
                    else:  last_error = msg
                if success_count > 0:
                    _toast("success", f"Đã thêm {success_count} file cho '{result['name']}'")
                else:
                    _toast("error", last_error or "Có lỗi xảy ra")
            st.rerun()
elif st.session_state.vid_show_add_popup and not POPUP_AVAILABLE:
    st.warning("Không tìm thấy `components/voiceid_popup.py`.")
    if st.button("Đóng"): st.session_state.vid_show_add_popup = False; st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# POPUP: TẠO NHÓM VOICEID MỚI
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.get("vid_show_creategroup_popup"):
    with st.container(border=True):
        st.markdown(
            '<div style="font-family:\'IBM Plex Mono\',monospace;font-size:11px;'
            'letter-spacing:2px;text-transform:uppercase;color:#2563EB;'
            'margin-bottom:14px;">// TẠO NHÓM VOICEID MỚI</div>',
            unsafe_allow_html=True)

        new_group_name = st.text_input(
            "Tên nhóm mới", placeholder="VD: Nhóm 4: Hội thảo",
            key="new_group_name_input",
        )

        st.divider()
        c1, c2, c3 = st.columns([3, 1, 1])
        cancel_grp = c2.button("Huỷ", key="creategroup_cancel", use_container_width=True)
        submit_grp = c3.button("Tạo nhóm", key="creategroup_submit",
                                type="primary", use_container_width=True)

        if cancel_grp:
            st.session_state.vid_show_creategroup_popup = False
            st.rerun()

        if submit_grp:
            ok, msg = add_group(new_group_name)
            st.session_state.vid_show_creategroup_popup = False
            if ok:
                st.session_state.vid_active_group = new_group_name.strip()
                _toast("success", msg)
            else:
                _toast("error", msg)
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# POPUP: EDIT VOICEID NAME (mở từ nút ✏️ trong bảng, theo đúng hàng)
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.get("vid_edit_target"):
    _vid_edit = st.session_state.vid_edit_target
    _entry_edit = get_entry(_vid_edit)

    if not _entry_edit:
        st.session_state.vid_edit_target = None
    else:
        with st.container(border=True):
            st.markdown(
                '<div style="font-family:\'IBM Plex Mono\',monospace;font-size:11px;'
                'letter-spacing:2px;text-transform:uppercase;color:#2563EB;'
                'margin-bottom:14px;">// CHỈNH SỬA TÊN VOICEID</div>',
                unsafe_allow_html=True)
            st.caption(f"ID: `{_vid_edit}`")

            _old_name = _entry_edit.get("name", "")
            new_name = st.text_input("Tên mới", value=_old_name,
                                      key=f"edit_new_name_{_vid_edit}")

            st.divider()
            c1, c2, c3 = st.columns([3, 1, 1])
            cancel_edit = c2.button("Huỷ", key="edit_cancel", use_container_width=True)
            submit_edit = c3.button("Lưu thay đổi", key="edit_submit",
                                     type="primary", use_container_width=True)

            if cancel_edit:
                st.session_state.vid_edit_target = None
                st.rerun()

            if submit_edit:
                new_name_clean = new_name.strip()
                if not new_name_clean:
                    _toast("error", "Tên không được để trống!")
                elif new_name_clean == _old_name:
                    st.session_state.vid_edit_target = None
                    st.rerun()
                else:
                    ok, msg = update_voiceid(_vid_edit, name=new_name_clean)
                    st.session_state.vid_edit_target = None
                    _toast("success" if ok else "error",
                           msg if not ok else f"Đã đổi tên thành '{new_name_clean}'")
                    st.rerun()
# ══════════════════════════════════════════════════════════════════════════════
# POPUP: ADD DATA TO EXIST VOICEID
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.vid_show_adddata_popup:
    entries_all = get_all_entries()
    if not entries_all:
        st.warning("Chưa có VoiceID nào. Hãy thêm mới trước.")
        if st.button("Đóng##adddata_close"):
            st.session_state.vid_show_adddata_popup = False; st.rerun()
    else:
        with st.container(border=True):
            st.markdown(
                '<div style="font-family:\'IBM Plex Mono\',monospace;font-size:11px;'
                'letter-spacing:2px;text-transform:uppercase;color:#2563EB;'
                'margin-bottom:14px;">// THÊM AUDIO VÀO VOICEID ĐÃ CÓ</div>',
                unsafe_allow_html=True)

            # Dropdown chọn người
            name_to_vid = {e["name"]: e["id"] for e in entries_all}
            chosen_name = st.selectbox(
                "Chọn người nhận audio",
                options=list(name_to_vid.keys()),
                key="adddata_select_name",
            )
            chosen_vid = name_to_vid[chosen_name]
            n_existing = len(get_preview_paths(get_entry(chosen_vid) or {}))
            st.caption(f"ID: `{chosen_vid}` · Hiện có **{n_existing}** file audio")

            # Upload
            uploaded = st.file_uploader(
                "Chọn file WAV để thêm (có thể chọn nhiều)",
                type=["wav"],
                accept_multiple_files=True,
                key="adddata_uploader",
            )
            if uploaded:
                total_kb = sum(len(f.getbuffer()) for f in uploaded) / 1024
                st.info(f"✅ Đã chọn {len(uploaded)} file ({total_kb:.1f} KB)")

            # Thu âm mic bằng SoundDevice
            st.caption("— hoặc thu âm trực tiếp —")
            if sd is None:
                st.error("Thư viện `sounddevice` chưa được cài đặt. Không thể thu âm phần cứng.")
            else:
                is_recording = st.session_state.adddata_recording
                
                c_rec1, c_rec2 = st.columns(2)
                with c_rec1:
                    if not is_recording:
                        if st.button("🔴 BẮT ĐẦU THU", width="stretch", type="primary", key="adddata_btn_rec"):
                            st.session_state.adddata_audio_q = queue.Queue()
                            st.session_state.adddata_frames = []
                            st.session_state.adddata_wav = None
                            st.session_state.adddata_rec_start = time.time()

                            # Trích xuất tham chiếu Queue vào RAM tĩnh
                            active_queue = st.session_state.adddata_audio_q

                            def _mic_callback(indata, frames, time_info, status):
                                active_queue.put(indata.copy())

                            try:
                                stream = sd.InputStream(
                                    samplerate=TARGET_SR, channels=1, dtype="int16", blocksize=CHUNK_SIZE,
                                    callback=_mic_callback
                                )
                                stream.start()
                                st.session_state.adddata_stream = stream
                                st.session_state.adddata_recording = True
                                st.rerun()
                            except Exception as e:
                                st.error(f"Không kết nối được Micro: {e}")
                    else:
                        if st.button("⏹️ DỪNG THU ÂM", width="stretch", key="adddata_btn_stop"):
                            if st.session_state.adddata_stream:
                                try:
                                    st.session_state.adddata_stream.stop()
                                    st.session_state.adddata_stream.close()
                                except: pass
                            
                            q = st.session_state.adddata_audio_q
                            frames_list = st.session_state.adddata_frames
                            while not q.empty():
                                frames_list.append(q.get())

                            st.session_state.adddata_recording = False

                            if len(frames_list) > 0:
                                audio_data = np.concatenate(frames_list)
                                buf = io.BytesIO()
                                with wave.open(buf, 'wb') as wf:
                                    wf.setnchannels(1)
                                    wf.setsampwidth(2)
                                    wf.setframerate(TARGET_SR)
                                    wf.writeframes(audio_data.tobytes())
                                
                                st.session_state.adddata_wav = buf.getvalue()
                            st.rerun()

                if is_recording:
                    dur = time.time() - st.session_state.adddata_rec_start
                    st.markdown(f"<div style='margin-top:8px; display:flex; align-items:center;'>"
                                f"<span class='rec-dot'></span>"
                                f"<span style='color:#EF4444; font-weight:bold; font-family:monospace;'>ĐANG THU ÂM TRỰC TIẾP... {dur:.1f}s</span>"
                                f"</div>", unsafe_allow_html=True)
                    
                    while not st.session_state.adddata_audio_q.empty():
                        st.session_state.adddata_frames.append(st.session_state.adddata_audio_q.get())
                        
                    time.sleep(0.25)
                    st.rerun()
                
                elif st.session_state.adddata_wav:
                    st.success("✅ Đã ghi âm thành công mẫu giọng nói!")
                    st.audio(st.session_state.adddata_wav, format="audio/wav")

            st.divider()
            c1, c2, c3 = st.columns([3, 1, 1])
            cancel = c2.button("Huỷ", key="adddata_cancel", width="stretch")
            
            # Khóa nút bấm nếu chưa có file/ghi âm nào HOẶC hệ thống đang ghi âm
            can_submit = (uploaded or st.session_state.adddata_wav) and not st.session_state.adddata_recording
            submit = c3.button("Thêm", key="adddata_submit", type="primary",
                               disabled=not can_submit,
                               width="stretch")

            if cancel:
                if st.session_state.adddata_stream:
                    try: st.session_state.adddata_stream.stop(); st.session_state.adddata_stream.close()
                    except: pass
                st.session_state.adddata_wav = None
                st.session_state.vid_show_adddata_popup = False
                st.rerun()

            if submit:
                success_count = 0; last_error = ""
                # Xử lý file upload
                for f in (uploaded or []):
                    ok, msg = add_audio_to_voiceid(chosen_vid, f.getbuffer().tobytes())
                    if ok: success_count += 1
                    else:  last_error = msg
                
                # Xử lý mic recording từ SoundDevice
                if st.session_state.adddata_wav is not None:
                    ok, msg = add_audio_to_voiceid(chosen_vid, st.session_state.adddata_wav, source="record")
                    if ok: success_count += 1
                    else:  last_error = msg
                    st.session_state.adddata_wav = None # Dọn dẹp RAM

                st.session_state.vid_show_adddata_popup = False
                if success_count > 0:
                    _toast("success", f"Đã thêm {success_count} audio vào '{chosen_name}'")
                else:
                    _toast("error", last_error or "Có lỗi xảy ra")
                st.rerun()
# ══════════════════════════════════════════════════════════════════════════════
# DATABASE TABLE & GROUP TOGGLE
# ══════════════════════════════════════════════════════════════════════════════
entries_all = get_all_entries()
st.markdown("<div style='margin:12px 0 8px;'></div>", unsafe_allow_html=True)

# ── Khai báo và xử lý UI chọn nhóm ──
VOICEID_GROUPS = get_groups()
if "vid_active_group" not in st.session_state:
    st.session_state.vid_active_group = VOICEID_GROUPS[0]

# Hiển thị các nút Toggle ngang
g_cols = st.columns(len(VOICEID_GROUPS))
for i, grp in enumerate(VOICEID_GROUPS):
    with g_cols[i]:
        # Nút đang chọn sẽ sáng màu (primary), nút khác sẽ tắt màu (secondary)
        is_active = (st.session_state.vid_active_group == grp)
        btn_type = "primary" if is_active else "secondary"
        
        if st.button(grp, width="stretch", type=btn_type, key=f"btn_grp_{i}"):
            st.session_state.vid_active_group = grp
            st.rerun()

st.markdown("<div style='margin:12px 0 8px;'></div>", unsafe_allow_html=True)

# ── Lọc danh sách hiển thị theo nhóm đã chọn ──
# Nếu entry cũ chưa có key "group", mặc định xếp vào Nhóm 1
entries = [
    e for e in entries_all 
    if e.get("group", VOICEID_GROUPS[0]) == st.session_state.vid_active_group
]

if not entries:
    st.markdown("""
    <div style="background:var(--secondary-background-color);border:1px solid rgba(128,128,128,0.2);
                border-radius:10px;">
        <div class="empty-state">
            <span class="empty-icon">🪪</span>
            Chưa có VoiceID nào trong database.<br>
            Bấm <b>＋ Add new VoiceID</b> để thêm người nói đầu tiên.
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    # Cột: Index | VoiceID | Tên | Trạng thái (RAM) | Nghe thử | File audio | Ngày tạo | Action(xoá)
    col_ratios = [0.5, 1.3, 1.8, 1.2, 2.5, 0.9, 1.1, 1.3]

    with st.container(border=True):
        # Header
        h_cols = st.columns(col_ratios, gap="small")
        
        # Cấu hình Tên cột và Căn lề (alignment) tương ứng
        headers = [
            ("#", "left"), 
            ("VoiceID", "left"), 
            ("Tên", "left"), 
            ("Vector", "left"), 
            ("Nghe thử", "center"), 
            ("Files", "center"),   # Căn giữa để thẳng hàng với con số bên dưới
            ("Ngày tạo", "left"), 
            ("Action", "center")      # Căn giữa để thẳng với nút bấm thùng rác
        ]
        
        for i, (h, align) in enumerate(headers):
            h_cols[i].markdown(
                f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:10px;'
                f'letter-spacing:1.5px;text-transform:uppercase;color:#2563EB;'
                f'padding-bottom:8px;font-weight:600;text-align:{align};">{h}</div>',
                unsafe_allow_html=True)

        for idx, entry in enumerate(entries, 1):
            vid      = entry["id"]
            name     = entry.get("name", "—")
            
            # Lấy chuỗi ngày gốc (YYYY-MM-DD)
            raw_date = entry.get("created_at", "—")[:10]
            
            # Đảo ngược định dạng sang DD-MM-YYYY
            if raw_date != "—" and "-" in raw_date:
                y, m, d = raw_date.split("-")
                created = f"{d}-{m}-{y}"
            else:
                created = raw_date

            # Đếm số file audio thực tế trên disk
            prev_paths = get_preview_paths(entry)
            n_files    = len(prev_paths)
            # Preview: dùng file đầu tiên tồn tại
            valid_previews = [p for p in prev_paths if os.path.exists(p)]

            row_c = st.columns(col_ratios, gap="small")

            with row_c[0]:
                st.markdown(f'<div style="font-weight:600;padding-top:12px;">{idx}</div>',
                            unsafe_allow_html=True)
            with row_c[1]:
                st.markdown(f'<div style="font-weight:600;padding-top:12px;">{vid}</div>',
                            unsafe_allow_html=True)
            with row_c[2]:
                st.markdown(f'<div style="font-weight:600;padding-top:12px;">{name}</div>',
                            unsafe_allow_html=True)

            # Badge vector (thay checkbox RAM)
            with row_c[3]:
                has_v = entry.get("has_vector", False)
                badge_html = (
                    '<span style="font-family:\'IBM Plex Mono\',monospace;font-size:11px;'
                    'color:#4da6e8;padding-top:12px;display:block;">✦ Có</span>'
                    if has_v else
                    '<span style="font-family:\'IBM Plex Mono\',monospace;font-size:11px;'
                    'color:#888;padding-top:12px;display:block;">— Chưa</span>'
                )
                st.markdown(badge_html, unsafe_allow_html=True)

            # Preview audio (file đầu tiên)
            with row_c[4]:
                if valid_previews:
                    # 1. Tính toán chiều cao động (Mỗi thanh audio tốn ~45px, cộng thêm khoảng đệm)
                    # Nếu có 1-2 file: Khung tự động giãn vừa khít (None)
                    # Nếu từ 3 file trở lên: Tính chiều cao vừa đủ hoặc giới hạn max là 180px
                    num_audios = len(valid_previews)
                    
                    if num_audios > 2:
                        # Công thức dynamic height dựa trên số lượng file audio
                        dynamic_height = min(180, num_audios * 45 + 10) 
                        audio_container = st.container(height=dynamic_height, border=False)
                    else:
                        audio_container = st.container(border=False)
                        
                    # 2. Render danh sách audio vào container tuỳ biến chiều cao
                    with audio_container:
                        for p in valid_previews:
                            with open(p, "rb") as f:
                                st.audio(f.read(), format="audio/wav")
                else:
                    st.markdown('<div style="padding-top:12px;color:#888;">—</div>',
                                unsafe_allow_html=True)

            # Số file audio
            with row_c[5]:
                st.markdown(
                    f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:13px;'
                    f'font-weight:700;color:#4da6e8;padding-top:12px;text-align:center;">'
                    f'{n_files}</div>', unsafe_allow_html=True)

            # Ngày tạo
            with row_c[6]:
                st.markdown(f'<div style="font-weight:600;padding-top:12px;">{created}</div>',
                            unsafe_allow_html=True)

            # Nút xoá (chỉ xoá, không còn sửa)
            with row_c[7]:
                ac1, ac2 = st.columns(2)
                with ac1:
                    if st.button("✏️", key=f"edit_vid_{vid}", help=f"Đổi tên '{name}'",
                                use_container_width=True):
                        st.session_state.vid_edit_target    = vid
                        st.session_state.vid_delete_confirm = None
                        st.rerun()
                with ac2:
                    if st.session_state.vid_delete_confirm == vid:
                        if st.button("✓", key=f"del_confirm_{vid}", type="primary",
                                    help="Xác nhận xoá", use_container_width=True):
                            ok, msg = delete_voiceid(vid)
                            st.session_state.vid_delete_confirm = None
                            _toast("success" if ok else "error", msg); st.rerun()
                    else:
                        if st.button("🗑️", key=f"del_{vid}", help=f"Xoá '{name}'",
                                    use_container_width=True):
                            st.session_state.vid_delete_confirm = vid; st.rerun()

            if st.session_state.vid_delete_confirm == vid:
                st.warning(f"⚠️ Sắp xoá **{name}** (`{vid}`) cùng {n_files} file audio. "
                           f"Bấm **✓** để xác nhận.", icon="⚠️")

            st.markdown("<hr style='margin:12px 0 8px 0;border-color:rgba(128,128,128,0.1);'>",
                        unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("<div style='margin-top:32px;'></div>", unsafe_allow_html=True)
st.markdown("<hr style='border-color:rgba(128,128,128,0.2);'>", unsafe_allow_html=True)
enroll_dir = os.getenv("VOICEID_ENROLL_DIR", "enrollment_data")
st.markdown(
    f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:10px;'
    f'color:var(--text-color);opacity:0.45;text-align:center;padding:8px 0;">'
    f'📂 Thư mục: <code>{enroll_dir}/</code>'
    f' &nbsp;·&nbsp; Model: <code>{os.getenv("VOICEID_MODEL","iic/speech_campplus_sv_zh-cn_16k-common")}</code>'
    f'</div>', unsafe_allow_html=True)