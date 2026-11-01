"""
components/voiceid_popup.py
---------------------------
Popup component dùng để Thêm mới / Chỉnh sửa VoiceID.
Sử dụng SoundDevice backend để thu âm trực tiếp, tránh rào cản bảo mật của trình duyệt.
"""

from __future__ import annotations

import time
import queue
import io
import wave
import numpy as np
from typing import Optional

import streamlit as st

try:
    from core.voiceid_database import get_groups
except ImportError:
    get_groups = None

# Cố gắng import sounddevice an toàn
try:
    import sounddevice as sd
except ImportError:
    sd = None

# ── Hằng số cấu hình Audio ─────────────────────────────────────────────────────
TARGET_SR = 16000
CHUNK_SIZE = 1024

# ── Giao diện CSS tùy biến cho Popup ───────────────────────────────────────────
_POPUP_CSS = """
<style>
/* Overlay backdrop */
.popup-backdrop { position:fixed; inset:0; background:rgba(0,0,0,0.55); backdrop-filter:blur(3px); z-index:900; }
/* Panel trung tâm */
.popup-panel { background:var(--background-color); border:1px solid rgba(232,82,10,0.4); border-radius:10px; padding:28px 32px 24px; max-width:560px; margin:0 auto; }
.popup-title { font-family:'IBM Plex Mono',monospace; font-size:11px; letter-spacing:2px; text-transform:uppercase; color:#e8520a; margin-bottom:18px; }
.popup-section-label { font-family:'IBM Plex Mono',monospace; font-size:10px; letter-spacing:1px; text-transform:uppercase; color:var(--text-color); opacity:0.6; margin:14px 0 6px; }
@keyframes rec-blink { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }
.rec-dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; background: #e8520a; animation: rec-blink 1s ease-in-out infinite; margin-right: 8px; }
</style>
"""

def _inject_css() -> None:
    st.markdown(_POPUP_CSS, unsafe_allow_html=True)

def _section(label: str) -> None:
    st.markdown(f'<div class="popup-section-label">{label}</div>', unsafe_allow_html=True)


# ── Component chính ──────────────────────────────────────────────────────────────
def voiceid_popup(
    mode     : str               = "add",    # "add" | "edit"
    entry    : Optional[dict]    = None,     # bắt buộc khi mode="edit"
    key      : str               = "vp",
) -> tuple[bool, Optional[dict]]:
    """ Render giao diện popup quản lý VoiceID """
    
    _inject_css()

    # ── Khởi tạo Session State nội bộ cho Popup ──────────────────────────────
    pfx        = f"_vp_{key}"
    src_key    = f"{pfx}_src"       
    name_key   = f"{pfx}_name"
    group_key  = f"{pfx}_group"
    wav_key    = f"{pfx}_wav"
    
    # Các khóa trạng thái dành cho bộ thu âm SoundDevice
    rec_key    = f"{pfx}_recording"
    rec_t_key  = f"{pfx}_rec_start"
    q_key      = f"{pfx}_audio_q"
    frames_key = f"{pfx}_frames"
    stream_key = f"{pfx}_stream"

    for k, default in [
        (src_key,   "upload"),
        (name_key,  entry["name"] if entry else ""),
        (group_key, entry.get("group", "Nhóm 1: Toạ đàm") if entry else "Nhóm 1: Toạ đàm"),
        (wav_key,   None),
        (rec_key,   False),
        (rec_t_key, 0.0),
        (frames_key, []),
    ]:
        if k not in st.session_state:
            st.session_state[k] = default

    if q_key not in st.session_state:
        st.session_state[q_key] = queue.Queue()

    # ── Tiêu đề Popup ───────────────────────────────────────────────────────
    title = "THÊM VOICEID MỚI" if mode == "add" else "CHỈNH SỬA VOICEID"
    st.markdown(f'<div class="popup-title">// {title}</div>', unsafe_allow_html=True)
    if mode == "edit" and entry:
        st.caption(f"ID: `{entry['id']}`  ·  Tạo lúc: {entry.get('created_at', '—')}")

    st.divider()

    # ── ① Bộ nhập tên người nói ───────────────────────────────────────────────
    _section("① THÔNG TIN CƠ BẢN")
    
    c_name, c_grp = st.columns([2, 1], gap="small")
    with c_name:
        name_val = st.text_input(
            "Tên VoiceID",
            value       = st.session_state[name_key],
            placeholder = "Ví dụ: Nguyễn Văn A...",
            label_visibility = "collapsed",
            key         = f"{pfx}_name_input",
        )
        st.session_state[name_key] = name_val
        
    with c_grp:
        group_opts = get_groups() if get_groups else ["Nhóm 1: Toạ đàm", "Nhóm 2: Podcast", "Nhóm 3: Meeting"]
        current_grp = st.session_state[group_key]
        if current_grp not in group_opts: current_grp = group_opts[0]
        
        grp_val = st.selectbox(
            "Chọn Nhóm",
            options=group_opts,
            index=group_opts.index(current_grp),
            label_visibility = "collapsed",
            key=f"{pfx}_grp_select",
        )
        st.session_state[group_key] = grp_val

    # ── ② Bộ chọn nguồn đầu vào Audio ─────────────────────────────────────────
    _section("② NGUỒN ÂM THANH")

    src_col1, src_col2 = st.columns(2)
    with src_col1:
        if st.button("📂  Tải file WAV", key=f"{pfx}_btn_src_upload", width="stretch", type="primary" if st.session_state[src_key] == "upload" else "secondary"):
            st.session_state[src_key] = "upload"
            st.session_state[wav_key] = None
            st.rerun()

    with src_col2:
        if st.button("🎤  Thu âm từ mic", key=f"{pfx}_btn_src_record", width="stretch", type="primary" if st.session_state[src_key] == "record" else "secondary"):
            st.session_state[src_key] = "record"
            st.session_state[wav_key] = None
            st.rerun()

    st.markdown("<div style='margin-top:10px;'></div>", unsafe_allow_html=True)

    # =========================================================================
    # ── PHÂN ĐOẠN 1: GIAO DIỆN TẢI FILE UPLOAD ───────────────────────────────
    # =========================================================================
    if st.session_state[src_key] == "upload":
        is_multiple = (mode == "add")
        up = st.file_uploader("Chọn file WAV (16kHz mono khuyến nghị)", type=["wav"], key=f"{pfx}_uploader", accept_multiple_files=is_multiple, label_visibility="visible")
        
        if up:
            if is_multiple:
                wav_bytes_list = [f.getbuffer().tobytes() for f in up]
                st.session_state[wav_key] = wav_bytes_list
                st.success(f"✅ Đã nạp {len(up)} file")
            else:
                # Ép bọc trong mảng danh sách để tránh lỗi lặp biến đơn lẻ ngoài vòng lặp chính
                st.session_state[wav_key] = [up.getbuffer().tobytes()]
                st.success(f"✅ Đã nạp: **{up.name}**")

    # =========================================================================
    # ── PHÂN ĐOẠN 2: GIAO DIỆN THU ÂM TRỰC TIẾP (SOUNDDEVICE BACKEND) ────────
    # =========================================================================
    else:
        if sd is None:
            st.error("Thư viện `sounddevice` chưa được cài đặt trong môi trường này. Không thể thu âm phần cứng.")
        else:
            st.caption("ℹ️ Hệ thống ghi âm bằng nhân SoundDevice (16kHz Mono). Hãy nhấn nút đỏ, đọc 2-3 câu kỹ thuật dài.")

            is_recording = st.session_state[rec_key]
            
            c_rec1, c_rec2 = st.columns(2)
            with c_rec1:
                if not is_recording:
                    if st.button("🔴 BẮT ĐẦU THU", width="stretch", type="primary"):
                        st.session_state[q_key] = queue.Queue()
                        st.session_state[frames_key] = []
                        st.session_state[wav_key] = None
                        st.session_state[rec_t_key] = time.time()

                        # GIẢI PHÁP VÁ LỖI CFFI: Trích xuất thẳng tham chiếu Queue vào RAM tĩnh độc lập
                        # Tránh việc luồng chạy ngầm của ngôn ngữ C gọi trực tiếp st.session_state gây xung đột luồng
                        active_queue = st.session_state[q_key]

                        def _mic_callback(indata, frames, time_info, status):
                            active_queue.put(indata.copy())

                        try:
                            stream = sd.InputStream(
                                samplerate=TARGET_SR, channels=1, dtype="int16", blocksize=CHUNK_SIZE,
                                callback=_mic_callback
                            )
                            stream.start()
                            st.session_state[stream_key] = stream
                            st.session_state[rec_key] = True
                            st.rerun()
                        except Exception as e:
                            st.error(f"Không kết nối được tới thiết bị phần cứng Micro: {e}")
                else:
                    if st.button("⏹️ DỪNG THU ÂM", width="stretch"):
                        # 1. Tắt luồng InputStream của thiết bị phần cứng
                        if stream_key in st.session_state and st.session_state[stream_key]:
                            try:
                                st.session_state[stream_key].stop()
                                st.session_state[stream_key].close()
                            except:
                                pass
                        
                        # 2. Thu gom toàn bộ dữ liệu còn sót lại trong Hàng đợi (Queue)
                        q = st.session_state[q_key]
                        frames_list = st.session_state[frames_key]
                        while not q.empty():
                            frames_list.append(q.get())

                        st.session_state[rec_key] = False

                        # 3. Đóng gói dữ liệu mảng thô (Numpy arrays) thành file WAV nhị phân hoàn chỉnh
                        if len(frames_list) > 0:
                            audio_data = np.concatenate(frames_list)
                            buf = io.BytesIO()
                            with wave.open(buf, 'wb') as wf:
                                wf.setnchannels(1)
                                wf.setsampwidth(2)
                                wf.setframerate(TARGET_SR)
                                wf.writeframes(audio_data.tobytes())
                            
                            # CỰC KỲ QUAN TRỌNG: Bọc kết quả nhị phân vào List [] để đồng bộ với cơ chế tạo vòng lặp DB
                            st.session_state[wav_key] = [buf.getvalue()]
                        
                        st.rerun()

            # Vẽ màn hình hiển thị trạng thái động khi đang ghi âm
            if is_recording:
                dur = time.time() - st.session_state[rec_t_key]
                st.markdown(f"<div style='margin-top:8px; display:flex; align-items:center;'>"
                            f"<span class='rec-dot'></span>"
                            f"<span style='color:#e8520a; font-weight:bold; font-family:monospace;'>ĐANG THU ÂM TRỰC TIẾP... {dur:.1f}s</span>"
                            f"</div>", unsafe_allow_html=True)
                
                # Rút dọn phần tử queue liên tục đẩy vào bộ nhớ đệm tránh tràn RAM
                while not st.session_state[q_key].empty():
                    st.session_state[frames_key].append(st.session_state[q_key].get())
                    
                time.sleep(0.25)
                st.rerun() # Kích hoạt cơ chế Polling giao diện liên tục
            
            # Khung nghe thử (Audio Player) xuất hiện sau khi dừng thu thành công
            elif st.session_state[wav_key]:
                st.success("✅ Đã ghi âm thành công mẫu giọng nói!")
                st.audio(st.session_state[wav_key][0], format="audio/wav")

    st.divider()

    # ── Kiểm tra điều kiện đầu vào (Validation) ─────────────────────────────
    has_name   = bool(st.session_state[name_key].strip())
    has_wav    = st.session_state[wav_key] is not None
    # Khóa nút bấm gửi dữ liệu nếu đang trong quá trình ghi âm
    can_submit = has_name and (has_wav or (mode == "edit")) and not st.session_state[rec_key]

    if not has_name:
        st.caption("❌ Hệ thống yêu cầu nhập tên danh tính người nói")
    elif not has_wav and mode == "add":
        st.caption("❌ Hệ thống yêu cầu dữ liệu âm thanh (Vui lòng tải file hoặc thực hiện thu âm)")

    # ── Các nút Thao tác Cuối (Hủy / Xác nhận) ──────────────────────────────────
    submit_label = "Thêm VoiceID" if mode == "add" else "Lưu thay đổi"
    b_col1, b_col2, b_col3 = st.columns([3, 1, 1])

    cancel_clicked = False
    submit_clicked = False

    with b_col2:
        if st.button("Huỷ", key=f"{pfx}_cancel", width="stretch"):
            cancel_clicked = True

    with b_col3:
        if st.button(
            submit_label,
            key             = f"{pfx}_submit",
            type            = "primary",
            disabled        = not can_submit,
            use_container_width = True,
        ):
            submit_clicked = True

    # ── Điều phối Kết quả đầu ra ─────────────────────────────────────────────
    if cancel_clicked:
        # Giải phóng Micro khẩn cấp nếu người dùng đột ngột ấn hủy lúc đang thu
        if stream_key in st.session_state and st.session_state[stream_key]:
            try: 
                st.session_state[stream_key].stop()
                st.session_state[stream_key].close()
            except: 
                pass
        _reset_state(pfx)
        return True, None

    if submit_clicked and can_submit:
        result = {
            "name"      : st.session_state[name_key].strip(),
            "group"     : st.session_state[group_key],
            "wav_bytes_list" : st.session_state[wav_key], # Đã được bọc List phần tử an toàn tuyệt đối
            "source"    : st.session_state[src_key],
        }
        _reset_state(pfx)
        return True, result

    return False, None


def _reset_state(pfx: str) -> None:
    """Xoá toàn bộ cấu trúc trạng thái session của popup để giải phóng bộ nhớ."""
    for suffix in ["_src", "_name", "_group", "_wav", "_recording", "_rec_start", "_audio_q", "_frames", "_stream"]:
        k = f"{pfx}{suffix}"
        if k in st.session_state:
            del st.session_state[k]