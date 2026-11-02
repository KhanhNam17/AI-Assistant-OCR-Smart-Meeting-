import os
import sys
import time
import queue as _queue_module
import threading
import warnings
import tempfile
import html as _html
import wave

import numpy as np
import base64
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
from streamlit.runtime.scriptrunner import add_script_run_ctx

load_dotenv()

os.environ.setdefault("SPEECHBRAIN_DISABLE_K2", "1")
warnings.filterwarnings("ignore")


def render_live_mode():
    """Full page body — invoked by the Workspace router below."""

    # ══════════════════════════════════════════════════════════════════════════════
    # 0. LAZY IMPORT
    # ══════════════════════════════════════════════════════════════════════════════
    def _try_import(name):
        try:
            import importlib
            return importlib.import_module(name)
        except ImportError:
            return None

    sherpa_onnx    = _try_import("sherpa_onnx")
    sd             = _try_import("sounddevice")
    LIVE_ENV_READY = all([sherpa_onnx, sd])

    # ══════════════════════════════════════════════════════════════════════════════
    # 1. CONFIG
    # ══════════════════════════════════════════════════════════════════════════════
    MODEL_DIR    = os.getenv("LIVE_MODEL_DIR",   "models/zipformer")
    TOKENS_PATH  = os.getenv("LIVE_TOKENS",      f"{MODEL_DIR}/config.json")
    ENCODER_PATH = os.getenv("LIVE_ENCODER",     f"{MODEL_DIR}/encoder-epoch-31-avg-11-chunk-32-left-128.fp16.onnx")
    DECODER_PATH = os.getenv("LIVE_DECODER",     f"{MODEL_DIR}/decoder-epoch-31-avg-11-chunk-32-left-128.fp16.onnx")
    JOINER_PATH  = os.getenv("LIVE_JOINER",      f"{MODEL_DIR}/joiner-epoch-31-avg-11-chunk-32-left-128.fp16.onnx")

    SAMPLE_RATE  = int(os.getenv("LIVE_SAMPLE_RATE",  "16000"))
    CHUNK_FRAMES = int(os.getenv("LIVE_CHUNK_FRAMES", "3200"))   # 200ms/chunk

    DIART_DEVICE  = os.getenv("DIART_DEVICE",  "cpu").lower()
    DIART_STEP    = float(os.getenv("DIART_STEP",    "0.5"))
    DIART_LATENCY = float(os.getenv("DIART_LATENCY", "5.0"))
    DIART_PRESET  = os.getenv("DIART_PRESET",  "ami").lower()   # ami | dihard | voxconverse | default
    DIART_EMBEDDING = os.getenv("DIART_EMBEDDING", "pyannote/embedding")  # hoặc speechbrain/spkrec-ecapa-voxceleb
    DIART_SEGMENTATION = os.getenv("DIART_SEGMENTATION", "Khanh17/toadam-segmentation-model")

    # ══════════════════════════════════════════════════════════════════════════════
    # 2. LOG
    # ══════════════════════════════════════════════════════════════════════════════
    def _log(msg: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    # ══════════════════════════════════════════════════════════════════════════════
    # 3. SHARED RESOURCES (thread-safe singleton)
    # ══════════════════════════════════════════════════════════════════════════════
    @st.cache_resource
    def _get_shared():
        return {
            "audio_q":     _queue_module.Queue(),
            "audio_peak":  0.0,
            "wav_writer":  None,
            "wav_path":    None,
            "wav_lock":    threading.Lock(),
            "file_streaming": False,
            "file_done":   False,
            "finalize_state": {
                "running": False, "pct": 0, "msg": "",
                "result":  None, "error": None,
            },
            "finalize_lock": threading.Lock(),
            # Rolling buffer cho visualizer sóng âm + phổ (giữ VIZ_SECONDS gần nhất).
            "viz_buf":   np.zeros(0, dtype=np.int16),
            "viz_lock":  threading.Lock(),
        }

    _shared       = _get_shared()
    _AUDIO_QUEUE  = _shared["audio_q"]

    VIZ_SECONDS = 6   # độ dài cửa sổ hiển thị sóng âm/phổ


    def _viz_push(chunk_i16: np.ndarray) -> None:
        """Nối chunk vào rolling buffer, cắt còn VIZ_SECONDS cuối. Gọi từ producer."""
        keep = VIZ_SECONDS * SAMPLE_RATE
        with _shared["viz_lock"]:
            buf = np.concatenate([_shared["viz_buf"], chunk_i16])
            if len(buf) > keep:
                buf = buf[-keep:]
            _shared["viz_buf"] = buf


    def _viz_clear() -> None:
        with _shared["viz_lock"]:
            _shared["viz_buf"] = np.zeros(0, dtype=np.int16)


    def _render_spectrum():
        """Vẽ sóng âm (trái) + spectrogram (phải) kiểu màn hình demo. Trả Figure hoặc None.
        Lazy-import matplotlib để app không crash nếu chưa cài."""
        with _shared["viz_lock"]:
            buf = _shared["viz_buf"]
            samples = buf.astype(np.float32) / 32768.0 if len(buf) else None
        if samples is None or len(samples) < int(SAMPLE_RATE * 0.25):
            return None
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception:
            return None

        bg = "#0e1117"
        fig, (ax_w, ax_s) = plt.subplots(1, 2, figsize=(11, 2.5), facecolor=bg)
        t = np.arange(len(samples)) / SAMPLE_RATE

        # Sóng âm
        ax_w.plot(t, samples, color="#22d3ee", linewidth=0.6)
        ax_w.set_facecolor(bg)
        ax_w.set_ylim(-1.0, 1.0)
        ax_w.set_xlim(0, max(VIZ_SECONDS, t[-1]))
        # ax_w.set_title("Sóng âm (amplitude)", color="#8b949e", fontsize=9, loc="left")
        ax_w.tick_params(colors="#484f58", labelsize=7)
        for s in ax_w.spines.values():
            s.set_color("#21262d")

        # Spectrogram
        try:
            ax_s.specgram(samples, NFFT=256, Fs=SAMPLE_RATE, noverlap=128, cmap="viridis")
        except Exception:
            pass
        ax_s.set_facecolor(bg)
        # ax_s.set_title("Phổ tần số (spectrogram)", color="#8b949e", fontsize=9, loc="left")
        ax_s.tick_params(colors="#484f58", labelsize=7)
        for s in ax_s.spines.values():
            s.set_color("#21262d")

        fig.tight_layout(pad=0.6)
        return fig

    # ══════════════════════════════════════════════════════════════════════════════
    # 3b. HELPERS — friendly speaker labels, mic devices
    # ══════════════════════════════════════════════════════════════════════════════
    def _friendly_speaker(raw_label: str) -> str:
        """SPEAKER_00 → 'Người nói 1'. Tên thật gán sau ở mục ② Gán tên người nói."""
        if not raw_label or raw_label == "...":
            return "…"
        if raw_label.upper().startswith("SPEAKER_"):
            try:
                idx = int(raw_label.split("_", 1)[1])
            except (ValueError, IndexError):
                return raw_label
            return f"Người nói {idx + 1}"
        return raw_label


    def _spk_css_class(label: str) -> str:
        seen = st.session_state.setdefault("l_spk_color_map", {})
        if label not in seen:
            seen[label] = f"spk-{len(seen) % 6}" if label != "…" else "spk-x"
        return seen[label]

    def _render_html(html_str: str) -> None:
        if hasattr(st, "html"):
            st.html(html_str)
        else:
            clean_html = "\n".join(line.strip() for line in html_str.splitlines() if line.strip())
            st.markdown(clean_html, unsafe_allow_html=True)



    @st.cache_data(show_spinner=False)
    def _list_input_devices():
        if sd is None:
            return []
        try:
            devices = sd.query_devices()
        except Exception:
            return []
        out = []
        for i, d in enumerate(devices):
            if d.get("max_input_channels", 0) > 0:
                out.append((i, f'{d["name"]}  ({int(d.get("default_samplerate", 0))}Hz)'))
        return out


    # ── Streaming WAV writer (disk, not RAM) ──────────────────────────────────────
    def _wav_open(sample_rate: int = SAMPLE_RATE) -> str:
        _wav_close()
        tmp = tempfile.NamedTemporaryFile(suffix="_live.wav", delete=False)
        tmp.close()
        wf = wave.open(tmp.name, "wb")
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sample_rate)
        with _shared["wav_lock"]:
            _shared["wav_writer"] = wf
            _shared["wav_path"]   = tmp.name
        _log(f"[WAV] mở stream → {tmp.name}")
        return tmp.name


    def _wav_append(chunk_i16: np.ndarray) -> None:
        with _shared["wav_lock"]:
            wf = _shared.get("wav_writer")
            if wf is None:
                return
            try:
                wf.writeframes(chunk_i16.tobytes())
            except Exception as e:
                _log(f"[WAV] writeframes lỗi: {e}")


    def _wav_close():
        with _shared["wav_lock"]:
            wf   = _shared.get("wav_writer")
            path = _shared.get("wav_path")
            if wf is None:
                return None
            try: wf.close()
            except Exception: pass
            _shared["wav_writer"] = None
        _log(f"[WAV] đóng stream ← {path}")
        return path


    def _wav_path():
        return _shared.get("wav_path")


    # ── WAV file validation + format adjustment ──────────────────────────────────
    def _read_wav_validated(path: str, target_sr: int = SAMPLE_RATE):
        """
        Đọc WAV và đảm bảo: 16-bit PCM, mono, target_sr.
        Stereo → downmix mono (mean). Sample-rate khác → linear resample (numpy).
        Trả về (audio_int16, info_str) hoặc raise ValueError nếu file không hợp lệ.
        """
        with wave.open(path, "rb") as wf:
            n_ch = wf.getnchannels()
            sw   = wf.getsampwidth()
            sr   = wf.getframerate()
            n    = wf.getnframes()
            raw  = wf.readframes(n)

        if sw != 2:
            raise ValueError(
                f"File phải là WAV 16-bit PCM (file hiện tại: {sw*8}-bit). "
                f"Convert: ffmpeg -i in.wav -acodec pcm_s16le out.wav"
            )

        audio = np.frombuffer(raw, dtype=np.int16)
        if n_ch == 2:
            audio = audio.reshape(-1, 2).mean(axis=1).astype(np.int16)
        elif n_ch != 1:
            raise ValueError(f"Số kênh không hỗ trợ ({n_ch}). Cần mono hoặc stereo.")

        info = f"{n_ch}ch · {sr}Hz · {sw*8}-bit · {len(audio)/sr:.1f}s"

        if sr != target_sr:
            # Linear resample (đủ tốt cho test pipeline, không thêm dependency).
            ratio    = target_sr / float(sr)
            n_target = int(round(len(audio) * ratio))
            x_old    = np.linspace(0, 1, num=len(audio), endpoint=False, dtype=np.float64)
            x_new    = np.linspace(0, 1, num=n_target,   endpoint=False, dtype=np.float64)
            audio    = np.interp(x_new, x_old, audio.astype(np.float64)).astype(np.int16)
            info    += f"  →  resample 16kHz"

        return audio, info


    def _stream_audio_thread(audio_i16: np.ndarray, realtime: bool):
        """Bơm mảng int16 (đã chuẩn hoá 16kHz mono) vào _AUDIO_QUEUE giống mic."""
        _log(f"[FileStream] ▶ {len(audio_i16)/SAMPLE_RATE:.1f}s · {'realtime' if realtime else 'fast'}")
        chunk_dt = CHUNK_FRAMES / SAMPLE_RATE
        pos = 0
        while _shared.get("file_streaming") and pos < len(audio_i16):
            chunk = audio_i16[pos: pos + CHUNK_FRAMES]
            if len(chunk) == 0:
                break
            pos += len(chunk)

            _AUDIO_QUEUE.put(chunk.copy())
            _viz_push(chunk)
            # Diart cũng cần audio để phân tách trong file-test mode
            try:
                diarizer_push_chunk(_diarizer, chunk)
            except Exception:
                pass

            if realtime:
                time.sleep(chunk_dt)

        _shared["file_done"] = True
        _log("[FileStream] ✅ Đã đọc hết file")


    # ══════════════════════════════════════════════════════════════════════════════
    # 4. PAGE CONFIG & CSS
    # ══════════════════════════════════════════════════════════════════════════════

    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700&family=IBM+Plex+Sans:wght@300;400;500;600;700&display=swap');
        html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
        footer { visibility: hidden; }

        /* Modern Primary Buttons & Downloads */
        .stButton > button[kind="primary"],
        .stDownloadButton > button[kind="primary"] {
            background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%) !important;
            border: none !important;
            border-radius: 10px !important;
            font-weight: 600 !important;
            color: white !important;
            box-shadow: 0 4px 14px rgba(37, 99, 235, 0.35) !important;
            transition: all 0.2s ease !important;
        }
        .stButton > button[kind="primary"]:hover,
        .stDownloadButton > button[kind="primary"]:hover {
            filter: brightness(1.08) !important;
            transform: translateY(-1px) !important;
            box-shadow: 0 6px 20px rgba(37, 99, 235, 0.45) !important;
        }
        .stButton > button:not([kind="primary"]),
        .stDownloadButton > button:not([kind="primary"]) {
            border-radius: 10px !important;
            font-weight: 500 !important;
        }

        .stProgress > div > div > div { background: #2563EB !important; border-radius: 4px; }
        hr { border-color: rgba(128,128,128,0.15) !important; }

        /* Studio Hero Status Card */
        .studio-status-card {
            background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 100%);
            border-radius: 16px;
            padding: 20px 22px;
            color: #ffffff;
            box-shadow: 0 10px 25px rgba(37, 99, 235, 0.25);
            margin-bottom: 16px;
            border: 1px solid rgba(255, 255, 255, 0.18);
        }
        .studio-timer {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 32px;
            font-weight: 700;
            letter-spacing: 2px;
            color: #ffffff;
            margin: 6px 0 10px 0;
            line-height: 1;
        }
        .studio-badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            font-family: 'IBM Plex Mono', monospace;
            font-size: 10px;
            font-weight: 700;
            letter-spacing: 1px;
            padding: 4px 10px;
            border-radius: 20px;
            text-transform: uppercase;
        }
        .studio-badge.rec {
            background: rgba(239, 68, 68, 0.25);
            color: #FCA5A5;
            border: 1px solid rgba(239, 68, 68, 0.5);
        }
        .studio-badge.paused {
            background: rgba(245, 158, 11, 0.25);
            color: #FDE68A;
            border: 1px solid rgba(245, 158, 11, 0.5);
        }
        .studio-badge.done {
            background: rgba(16, 185, 129, 0.25);
            color: #6EE7B7;
            border: 1px solid rgba(16, 185, 129, 0.5);
        }
        .studio-badge.ready {
            background: rgba(255, 255, 255, 0.12);
            color: rgba(255, 255, 255, 0.85);
            border: 1px solid rgba(255, 255, 255, 0.25);
        }

        @keyframes pulse-dot { 0%, 100% { opacity: 1; transform: scale(1); } 50% { opacity: 0.3; transform: scale(0.85); } }
        .live-dot {
            width: 7px; height: 7px; border-radius: 50%;
            background: #EF4444; display: inline-block;
            animation: pulse-dot 1.2s infinite ease-in-out;
        }

        /* Modern Settings Card */
        .panel-card {
            background: var(--secondary-background-color);
            border: 1px solid rgba(128, 128, 128, 0.18);
            border-radius: 12px;
            padding: 16px 18px;
            margin-bottom: 16px;
        }
        .panel-title {
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 1.5px;
            text-transform: uppercase;
            color: #2563EB;
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 14px;
        }

        /* Chat Timeline / Conversation Bubble */
        .chat-bubble {
            background: var(--secondary-background-color);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 14px 18px;
            margin-bottom: 10px;
            transition: all 0.15s ease;
            position: relative;
            box-shadow: 0 1px 3px rgba(0,0,0,0.02);
        }
        .chat-bubble:hover {
            border-color: rgba(37, 99, 235, 0.4);
            box-shadow: 0 4px 12px rgba(37, 99, 235, 0.08);
        }
        .chat-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 6px;
        }
        .chat-ts {
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
            color: var(--text-color);
            opacity: 0.65;
        }
        .chat-text {
            font-size: 14px;
            line-height: 1.6;
            color: var(--text-color);
            margin: 0;
        }
        .streaming-box {
            background: rgba(37, 99, 235, 0.06);
            border: 1px dashed rgba(37, 99, 235, 0.35);
            border-radius: 12px;
            padding: 14px 18px;
            margin-bottom: 10px;
        }

        /* Speaker Pills */
        .spk-pill {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            font-family: 'Plus Jakarta Sans', sans-serif;
            font-size: 12px;
            font-weight: 700;
            padding: 2px 10px;
            border-radius: 9999px;
        }
        .spk-0 { background: rgba(37, 99, 235, 0.12); color: #2563EB; border: 1px solid rgba(37, 99, 235, 0.25); }
        .spk-1 { background: rgba(18, 180, 196, 0.12); color: #0E9FAD; border: 1px solid rgba(18, 180, 196, 0.25); }
        .spk-2 { background: rgba(31, 169, 113, 0.12); color: #1FA971; border: 1px solid rgba(31, 169, 113, 0.25); }
        .spk-3 { background: rgba(124, 58, 237, 0.12); color: #7C3AED; border: 1px solid rgba(124, 58, 237, 0.25); }
        .spk-4 { background: rgba(225, 150, 43, 0.12); color: #D97706; border: 1px solid rgba(225, 150, 43, 0.25); }
        .spk-5 { background: rgba(219, 39, 119, 0.12); color: #DB2777; border: 1px solid rgba(219, 39, 119, 0.25); }
        .spk-x { background: rgba(128, 128, 128, 0.12); color: #888; border: 1px solid rgba(128, 128, 128, 0.25); }

        /* VU Meter */
        .vu-track {
            background: rgba(255, 255, 255, 0.15);
            border-radius: 4px;
            height: 6px;
            flex: 1;
            overflow: hidden;
        }
        .vu-fill {
            background: linear-gradient(90deg, #34D399 0%, #FBBF24 70%, #EF4444 95%);
            height: 100%;
            transition: width 80ms ease;
        }
    </style>
    """, unsafe_allow_html=True)

    REPO_ROOT = Path(__file__).resolve().parent.parent.parent
    UI_ROOT = Path(__file__).resolve().parent.parent
    for p in (REPO_ROOT, UI_ROOT):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))

    from theme import icon

    st.markdown(f"""
    <div style="display:flex;justify-content:space-between;align-items:center;padding-bottom:12px;margin-bottom:18px;border-bottom:1px solid var(--border-color);">
        <div style="display:flex;align-items:center;gap:10px;">
            <span style="font-size:16px;font-weight:800;color:var(--text-color);letter-spacing:-0.2px;">Trợ lý Cuộc họp Trực tiếp</span>
            <span style="font-family:'JetBrains Mono',monospace;font-size:10px;font-weight:700;padding:3px 10px;border-radius:9999px;background:var(--app-blue-tint, rgba(37,99,235,0.12));color:#2563EB;border:1px solid rgba(37,99,235,0.25);letter-spacing:1px;">AI PIPELINE</span>
        </div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--text-color);opacity:0.65;">
            Sherpa Zipformer · Diart ({DIART_PRESET.upper()}) · Cam++ VoiceID
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not LIVE_ENV_READY:
        missing = [n for n, m in [("sherpa_onnx", sherpa_onnx), ("sounddevice", sd)] if not m]
        st.error(f"⚠️ Thiếu thư viện: {', '.join(missing)}")
        st.stop()

    # ══════════════════════════════════════════════════════════════════════════════
    # 5. CACHE MODELS
    # ══════════════════════════════════════════════════════════════════════════════
    @st.cache_resource(show_spinner=False)
    def _load_zipformer():
        _log("⏳ Đang nạp Zipformer STT...")
        r = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens                     = TOKENS_PATH,
            encoder                    = ENCODER_PATH,
            decoder                    = DECODER_PATH,
            joiner                     = JOINER_PATH,
            num_threads                = 2,
            sample_rate                = SAMPLE_RATE,
            feature_dim                = 80,
            enable_endpoint_detection  = True,
            rule1_min_trailing_silence = 0.7,
            rule2_min_trailing_silence = 0.4,
            rule3_min_utterance_length = 10,
        )
        _log("✅ Zipformer nạp xong")
        return r

    @st.cache_resource(show_spinner=False)
    def _load_diarizer():
        from core.diart_diarizer import DiartDiarizer
        return DiartDiarizer(
            sample_rate=SAMPLE_RATE,
            step=DIART_STEP,
            latency=DIART_LATENCY,
            device=DIART_DEVICE,
            preset=DIART_PRESET,
            embedding_model=DIART_EMBEDDING,
            segmentation_model=DIART_SEGMENTATION,
        )


    # Nạp SONG SONG: ONNX build của Zipformer (native, nhả GIL) chạy đồng thời
    # với import chain nặng của diart/pyannote → tổng ≈ max(19s, 28s) thay vì cộng.
    # @st.cache_resource đảm bảo chỉ nạp 1 lần/process; rerun sau là tức thì.
    with st.spinner("Đang nạp mô hình (song song)…"):
        import concurrent.futures as _cf
        from streamlit.runtime.scriptrunner import get_script_run_ctx as _get_ctx
        _ctx = _get_ctx()

        def _init_thread():
            # Gắn ScriptRunContext để st.cache_resource không cảnh báo trong thread con.
            add_script_run_ctx(threading.current_thread(), _ctx)

        with _cf.ThreadPoolExecutor(max_workers=2, initializer=_init_thread) as _ex:
            _f_zip  = _ex.submit(_load_zipformer)
            _f_diar = _ex.submit(_load_diarizer)
            recognizer = _f_zip.result()
            _diarizer  = _f_diar.result()

    # ══════════════════════════════════════════════════════════════════════════════
    # 6. DIARIZER API (thin wrappers over core.diart_diarizer.DiartDiarizer)
    # ══════════════════════════════════════════════════════════════════════════════
    def diarizer_push_chunk(d, audio_i16: np.ndarray) -> None: d.push(audio_i16)
    def diarizer_assign_speaker(d, t_start: float, t_end: float) -> str: return d.assign_speaker(t_start, t_end)
    def diarizer_get_windows(d) -> list: return d.get_windows()
    def diarizer_get_stats(d) -> dict:   return d.get_stats()
    def diarizer_reset(d) -> None:       d.reset()
    def diarizer_start(d) -> None:       d.start()
    def diarizer_stop(d) -> None:        d.stop()
    def diarizer_wait_drain(d, progress_cb=None) -> None: d.wait_drain(progress_cb=progress_cb)


    # ══════════════════════════════════════════════════════════════════════════════
    # 7. AUDIO CALLBACK (sounddevice C thread) — VU peak + fan-out
    # ══════════════════════════════════════════════════════════════════════════════
    def _audio_callback(indata, frames, time_info, status):
        if status:
            _log(f"[Mic] {status}")
        chunk_i16 = (indata[:, 0] * 32767).astype(np.int16)
        # VU meter peak (smoothed)
        try:
            peak = float(np.max(np.abs(indata[:, 0]))) if indata.size else 0.0
            prev = _shared.get("audio_peak", 0.0)
            _shared["audio_peak"] = max(peak, prev * 0.75)
        except Exception:
            pass
        _AUDIO_QUEUE.put(chunk_i16.copy())
        diarizer_push_chunk(_diarizer, chunk_i16)
        _viz_push(chunk_i16)


    # ══════════════════════════════════════════════════════════════════════════════
    # 8. STT helpers (real timestamps, prettify, flush on stop)
    # ══════════════════════════════════════════════════════════════════════════════
    def _prettify_live(text: str) -> str:
        if not text:
            return text
        try:
            from core.punctuation_restorer import restore
            return restore(text.lower())
        except Exception:
            return text.lower().capitalize()


    def _zipformer_result_with_ts(z_stream):
        """Trả (text, [per-token-seconds])."""
        try:
            import json
            s = recognizer.get_result_as_json_string(z_stream)
            if s:
                data = json.loads(s)
                return (data.get("text") or "").strip(), list(data.get("timestamps") or [])
        except Exception:
            pass
        try:
            r = recognizer.get_result(z_stream)
            text = r if isinstance(r, str) else getattr(r, "text", str(r))
            return text.strip(), []
        except Exception:
            return "", []


    # ══════════════════════════════════════════════════════════════════════════════
    # 9. STT DRAIN (làn 1) + flush trên dừng
    # ══════════════════════════════════════════════════════════════════════════════
    import subprocess

    def _extract_live_chunk(wav_path, start_sec, end_sec, out_path):
        """Cắt nhanh audio từ file wav (hỗ trợ cả file đang ghi dở và file hoàn chỉnh)."""
        dur = max(0.0, end_sec - start_sec)
        if dur <= 0:
            return

        # ── Cách 1: Cắt trực tiếp bằng Python (nhanh, không cần process ngoài, không bị WinError) ──
        try:
            with _shared.get("wav_lock", threading.Lock()):
                wf = _shared.get("wav_writer")
                if wf is not None and hasattr(wf, "_file"):
                    try:
                        wf._file.flush()
                    except Exception:
                        pass

            if os.path.exists(wav_path):
                raw_data = None
                sr = SAMPLE_RATE
                sw = 2
                ch = 1
                try:
                    with wave.open(wav_path, "rb") as wf_in:
                        sr = wf_in.getframerate()
                        sw = wf_in.getsampwidth()
                        ch = wf_in.getnchannels()
                        s_fr = int(start_sec * sr)
                        n_fr = int(dur * sr)
                        wf_in.setpos(min(s_fr, wf_in.getnframes()))
                        raw_data = wf_in.readframes(n_fr)
                except Exception:
                    # Nếu file đang ghi dở (header chưa chốt), đọc raw PCM 16-bit từ byte 44
                    bytes_per_sec = sr * sw * ch
                    start_byte = 44 + int(start_sec * bytes_per_sec)
                    byte_len = int(dur * bytes_per_sec)
                    with open(wav_path, "rb") as f:
                        f.seek(start_byte)
                        raw_data = f.read(byte_len)

                if raw_data:
                    with wave.open(out_path, "wb") as out_wf:
                        out_wf.setnchannels(ch)
                        out_wf.setsampwidth(sw)
                        out_wf.setframerate(sr)
                        out_wf.writeframes(raw_data)
                    return
        except Exception as e:
            _log(f"[VoiceID] Python slice error, falling back to ffmpeg: {e}")

        # ── Cách 2: Fallback FFmpeg ──────────────────────────────────────────
        import shutil
        ffmpeg_cmd = os.getenv("FFMPEG_PATH", "").strip().strip('"').strip("'")
        if not ffmpeg_cmd or not os.path.exists(ffmpeg_cmd):
            found = shutil.which("ffmpeg")
            if found:
                ffmpeg_cmd = found
            else:
                alt = r"E:\SOFTWARE\ffmpeg-2026-04-09-git-d3d0b7a5ee-essentials_build\ffmpeg-2026-04-09-git-d3d0b7a5ee-essentials_build\bin\ffmpeg.exe"
                if os.path.exists(alt):
                    ffmpeg_cmd = alt
                else:
                    ffmpeg_cmd = "ffmpeg"

        cmd = [
            ffmpeg_cmd, "-y", "-v", "quiet",
            "-ss", str(start_sec),
            "-t", str(dur),
            "-i", wav_path,
            "-c", "copy", out_path
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)

    def _voiceid_decide(best_score: float, margin: float) -> bool:
        """
        Quyết định có chấp nhận kết quả VoiceID hay không.

        Vấn đề: CamPPlus đôi khi cho absolute score không cao (0.65-0.74) dù
        margin (khoảng cách với người thứ 2) rất lớn — đây vẫn là tín hiệu
        đáng tin vì model phân biệt rõ người này khác hẳn người còn lại,
        chỉ là độ tương đồng tuyệt đối của giọng nói/điều kiện ghi âm chưa cao.

        Ngược lại, score cao (0.85+) nhưng margin nhỏ (<0.05) tức 2 người
        đang rất giống nhau theo model → không nên tin tưởng dù score cao.

        Logic 2 nhánh (OR), một trong hai đạt là chấp nhận:
          A. Tiêu chuẩn thường: score >= 0.75 và margin >= 0.06
          B. Tiêu chuẩn margin cao: score >= 0.60 và margin >= 0.20
             (margin rất lớn bù cho absolute score thấp hơn)
        """
        STRICT_SCORE,  STRICT_MARGIN  = 0.75, 0.06
        RELAXED_SCORE, RELAXED_MARGIN = 0.60, 0.20

        if best_score >= STRICT_SCORE and margin >= STRICT_MARGIN:
            return True
        if best_score >= RELAXED_SCORE and margin >= RELAXED_MARGIN:
            return True
        return False


    def _nearest_confirmed_speaker(
        t_start   : float,
        t_end     : float,
        diart_lbl : str,
        turns     : list,
        id_map    : dict,
        max_gap   : float = 8.0,
    ) -> str:
        """
        Tìm tên người nói phù hợp nhất cho turn ngắn hoặc turn chưa đủ tin cậy.

        Thứ tự ưu tiên:
            1. Turn đã voice_confirmed gần nhất CÓ CÙNG diart label (khoảng cách nhỏ nhất).
            2. Turn đã voice_confirmed gần nhất BẤT KỲ label, chỉ nếu cách < 3s.
            3. Cache live_id_map[diart_label].
            4. Raw diart label (SPEAKER_XX) — last resort.

        Tại sao ưu tiên cùng label:
            Diart gán cùng SPEAKER_XX cho cùng 1 người trong phần lớn thời gian.
            Turn ngắn của A kề cận turn dài đã confirmed của A → cùng label →
            được gán đúng tên A dù không qua VoiceID.

        Tại sao có fallback "bất kỳ label, < 3s":
            Khi diart re-label (SPEAKER_00 → SPEAKER_01 cùng người), label mới
            chưa có confirmed turn → fallback sang confirmed gần nhất bất kỳ label,
            nhưng giới hạn 3s tránh kéo tên người đã nói lâu trước đó.
        """
        t_mid = (t_start + t_end) / 2.0

        best_same = None   # (gap, tên) cùng diart label
        best_any  = None   # (gap, tên) bất kỳ label

        cached_name = id_map.get(diart_lbl)   # tên cache của label này (nếu có)

        for old_t in turns:
            if not getattr(old_t, "voice_confirmed", False):
                continue
            old_mid = (old_t.start + old_t.end) / 2.0
            gap = abs(t_mid - old_mid)
            if gap > max_gap:
                continue

            entry = (gap, old_t.speaker)
            # "Cùng label" khi tên của confirmed turn trùng với cache của diart_lbl
            is_same = (cached_name is not None and old_t.speaker == cached_name)

            if is_same:
                if best_same is None or gap < best_same[0]:
                    best_same = entry
            else:
                if best_any is None or gap < best_any[0]:
                    best_any = entry

        if best_same is not None:
            return best_same[1]
        if best_any is not None and best_any[0] <= 3.0:
            return best_any[1]
        return cached_name if cached_name else diart_lbl


    def _drain_queue():
        from core.aligner import AlignedTurn

        if st.session_state.l_asr_stream is None:
            st.session_state.l_asr_stream      = recognizer.create_stream()
            st.session_state.l_utterance_start = 0.0

        z_stream  = st.session_state.l_asr_stream
        completed = []

        while not _AUDIO_QUEUE.empty():
            try:
                chunk = _AUDIO_QUEUE.get_nowait()
            except _queue_module.Empty:
                break

            _wav_append(chunk)
            st.session_state.l_n_chunks += 1

            samples = chunk.astype(np.float32) / 32768.0
            z_stream.accept_waveform(SAMPLE_RATE, samples)

            while recognizer.is_ready(z_stream):
                recognizer.decode_stream(z_stream)

            if recognizer.is_endpoint(z_stream):
                text, ts = _zipformer_result_with_ts(z_stream)
                now = round((st.session_state.l_n_chunks * CHUNK_FRAMES) / SAMPLE_RATE, 2)
                if text:
                    # ts của sherpa-onnx reset về 0 sau mỗi reset() → là tương đối
                    # với utterance hiện tại. Phải cộng base = mốc tuyệt đối của
                    # endpoint trước (l_utterance_start) để ra thời gian thật.
                    base = st.session_state.l_utterance_start
                    if ts:
                        t_start = round(base + float(ts[0]), 2)
                        t_end   = round(base + float(ts[-1]), 2)
                        if t_end <= t_start:
                            t_end = now
                    else:
                        t_end   = now
                        est_sec = max(0.5, len(text.split()) / 2.5)
                        t_start = round(max(base, t_end - est_sec), 2)

                    speaker = diarizer_assign_speaker(_diarizer, t_start, t_end)

                    # =====================================================================
                    # TRẠM GÁC VOICEID — Cách 2: mỗi turn tự nhận diện theo embedding
                    # =====================================================================
                    # Thiết kế:
                    #   - Turn đủ dài (≥3s) → chạy embedding trực tiếp, gán tên từ audio.
                    #     Quyết định chấp nhận qua _voiceid_decide() (strict hoặc relaxed).
                    #   - Turn ngắn (<3s) hoặc score không đủ tin cậy:
                    #     → _nearest_confirmed_speaker(): tìm turn confirmed gần nhất
                    #        cùng diart label, rồi bất kỳ label (trong 3s), rồi cache.
                    #   - voice_confirmed=True → _retro_update và _finalize KHÔNG ghi đè.
                    # =====================================================================

                    voice_confirmed = False
                    final_speaker   = speaker  # default: raw diart label

                    dur = t_end - t_start
                    if dur >= 3.0:
                        temp_wav = os.path.join(
                            tempfile.gettempdir(),
                            f"vid_{int(t_start*100)}_{int(t_end*100)}.wav"
                        )
                        try:
                            from core.voiceid_database import compute_embedding
                            from scipy.spatial.distance import cosine

                            _extract_live_chunk(_wav_path(), t_start, t_end, temp_wav)

                            vec = compute_embedding(temp_wav)
                            if vec is not None:
                                # Tính score với tất cả người trong DB
                                scores = {}
                                for db_name, vec_list in st.session_state.db_vectors.items():
                                    scores[db_name] = max(
                                        1.0 - cosine(vec, db_vec)
                                        for db_vec in vec_list
                                    )

                                if scores:
                                    sorted_scores = sorted(scores.items(), key=lambda x: -x[1])
                                    best_name,  best_score = sorted_scores[0]
                                    second_score = sorted_scores[1][1] if len(sorted_scores) > 1 else 0.0
                                    margin = best_score - second_score

                                    _log(f"[VoiceID] [{t_start:.1f}-{t_end:.1f}s] {speaker} "
                                         f"top1={best_name}({best_score:.2f}) "
                                         f"top2={sorted_scores[1][0] if len(sorted_scores)>1 else '-'}"
                                         f"({second_score:.2f}) margin={margin:.2f}")

                                    if _voiceid_decide(best_score, margin):
                                        final_speaker   = best_name
                                        voice_confirmed = True

                                        # Cập nhật cache: ghi đè chỉ khi score cao hơn lần trước
                                        prev_score = st.session_state.live_id_score.get(speaker, 0.0)
                                        if best_score > prev_score:
                                            st.session_state.live_id_map[speaker]   = best_name
                                            st.session_state.live_id_score[speaker] = best_score

                                        # Backfill: vá turn cũ cùng diart label CHƯA confirmed
                                        backfill_count = 0
                                        for old_t in st.session_state.l_turns:
                                            if (old_t.speaker == speaker
                                                    and not getattr(old_t, "voice_confirmed", False)):
                                                old_t.speaker = best_name
                                                old_t.voice_confirmed = True
                                                backfill_count += 1
                                        if backfill_count:
                                            _log(f"[VoiceID] Backfill {backfill_count} turn "
                                                 f"({speaker} → {best_name})")
                                        _log(f"[VoiceID] ✅ {speaker} → {best_name} "
                                             f"({best_score:.2f}, margin={margin:.2f})")
                                    else:
                                        # Turn ngắn (<3s): dùng cache live_id_map nếu có, không thì giữ raw diart label
                                        final_speaker = st.session_state.live_id_map.get(speaker, speaker)
                                        _log(f"[VoiceID] ⚠️ {speaker} [{t_start:.1f}-{t_end:.1f}s] "
                                             f"score={best_score:.2f} margin={margin:.2f} "
                                             f"→ nearest: {final_speaker}")

                        except Exception as e:
                            _log(f"[VoiceID] ⚠️ Lỗi: {e}")
                            final_speaker = st.session_state.live_id_map.get(speaker, speaker)
                        finally:
                            if os.path.exists(temp_wav):
                                try:
                                    os.remove(temp_wav)
                                except Exception:
                                    pass

                    else:
                        # Turn ngắn (<3s): dùng cache live_id_map nếu có, không thì giữ raw diart label
                        final_speaker = st.session_state.live_id_map.get(speaker, speaker)

                    completed.append(AlignedTurn(
                        speaker         = final_speaker,
                        start           = t_start,
                        end             = t_end,
                        text            = _prettify_live(text),
                        voice_confirmed = voice_confirmed,
                    ))
                    _log(f"[STT] [{t_start:.1f}→{t_end:.1f}s] {speaker} | \"{text[:60]}\"")
                recognizer.reset(z_stream)
                st.session_state.l_utterance_start = now

        # Re-tag turns cũ với label mới của diart (online relabeling)
        cur_wins = len(diarizer_get_windows(_diarizer))
        if cur_wins > st.session_state.l_prev_win_count:
            _retro_update_speakers()
            st.session_state.l_prev_win_count = cur_wins

        partial, _ = _zipformer_result_with_ts(z_stream)
        partial = partial.strip().lower()
        if partial:
            partial = partial[:1].upper() + partial[1:]
        return partial, completed


    def _flush_final_asr() -> None:
        """Khi DỪNG: ép Zipformer commit phần partial cuối cùng."""
        from core.aligner import AlignedTurn

        z = st.session_state.get("l_asr_stream")
        if z is None:
            return

        while not _AUDIO_QUEUE.empty():
            try:
                chunk = _AUDIO_QUEUE.get_nowait()
            except _queue_module.Empty:
                break
            _wav_append(chunk)
            st.session_state.l_n_chunks += 1
            z.accept_waveform(SAMPLE_RATE, chunk.astype(np.float32) / 32768.0)

        try: z.input_finished()
        except Exception: pass
        while recognizer.is_ready(z):
            recognizer.decode_stream(z)

        text, ts = _zipformer_result_with_ts(z)
        if text:
            now  = round((st.session_state.l_n_chunks * CHUNK_FRAMES) / SAMPLE_RATE, 2)
            base = st.session_state.l_utterance_start
            if ts:
                t_start = round(base + float(ts[0]), 2)
                t_end   = round(base + float(ts[-1]), 2)
                if t_end <= t_start:
                    t_end = now
            else:
                t_end   = now
                est_sec = max(0.5, len(text.split()) / 2.5)
                t_start = round(max(base, t_end - est_sec), 2)
            speaker = diarizer_assign_speaker(_diarizer, t_start, t_end)
            st.session_state.l_turns.append(AlignedTurn(
                speaker = speaker,
                start   = t_start,
                end     = t_end,
                text    = _prettify_live(text),
            ))
            _log(f"[Flush] Commit cuối ({len(text.split())} từ): {text[:60]}…")

        st.session_state.l_partial = ""


    def _retro_update_speakers() -> int:
        turns   = st.session_state.get("l_turns", [])
        updated = 0
        for t in turns:
            # Turn đã được VoiceID embedding xác nhận → KHÔNG cho diart đè lên.
            # Diart label không ổn định (re-labeling), VoiceID embedding là nguồn
            # sự thật đáng tin hơn.
            if getattr(t, "voice_confirmed", False):
                continue

            spk = diarizer_assign_speaker(_diarizer, t.start, t.end)
            # Dùng cache gợi ý live_id_map thay vì live_spk_map cũ.
            spk = st.session_state.live_id_map.get(spk, spk)
            if spk != "..." and spk != t.speaker:
                t.speaker = spk
                updated  += 1
        if updated:
            _log(f"[Retro] Vá {updated} câu với nhãn speaker mới")
        return updated


    def _live_normalize(segments, min_duration=0.3, merge_gap=2.5):
        from core.diarizer import SpeakerSegment
        if not segments:
            return []
        filtered = [s for s in segments if (s.end - s.start) >= min_duration]
        if not filtered:
            return []
        merged = [filtered[0]]
        for cur in filtered[1:]:
            prev = merged[-1]
            if cur.speaker == prev.speaker and (cur.start - prev.end) <= merge_gap:
                merged[-1] = SpeakerSegment(
                    speaker=prev.speaker, start=prev.start,
                    end=cur.end, text=prev.text,
                )
            else:
                merged.append(cur)
        return merged


    # ══════════════════════════════════════════════════════════════════════════════
    # 10. ASYNC FINALIZE WORKER (background, không block UI)
    # ══════════════════════════════════════════════════════════════════════════════
    def _voiceid_relabel_offline(
        labeled    : list,
        wav_path   : str,
        db_vectors : dict,
        prog_cb,
        min_dur    : float = 4.0,
    ) -> list:
        """
        Re-assign speaker cho toàn bộ turns bằng VoiceID offline sau khi dừng ghi.

        Khác với live VoiceID:
          - Dùng toàn bộ duration của turn (không bị cắt bởi Zipformer endpoint).
            Turn dài hơn → embedding chất lượng cao hơn.
          - Không bị race condition với audio đang ghi.
          - Chạy tuần tự, không bị giới hạn CPU real-time.

        Quyết định chấp nhận dùng _voiceid_decide() — CÙNG logic với live, để
        2 giai đoạn nhất quán (live confirmed sẽ không bị offline đảo ngược
        bởi tiêu chuẩn khác nhau).

        Chiến lược gán tên:
          1. Turn đã voice_confirmed từ live (đã đủ tin cậy) → GIỮ NGUYÊN.
          2. Turn chưa confirmed VÀ dur >= min_dur → chạy embedding offline.
             - score >= threshold VÀ margin >= margin → gán tên VoiceID, set confirmed.
             - Không đủ → giữ nguyên diart label (sẽ thành "Người nói N" sau).
          3. Turn ngắn (< min_dur) chưa confirmed → propagate từ turn confirmed
             gần nhất cùng diart label (nearest-confirmed trong labeled list).

        Sau khi hàm này chạy, rename_map CHỈ áp dụng cho các turn vẫn còn
        SPEAKER_XX (không ai trong DB khớp) → không đổi tên VoiceID đã gán.

        Returns: labeled list đã được cập nhật speaker inplace.
        """
        if not wav_path or not db_vectors:
            return labeled

        import tempfile
        from scipy.spatial.distance import cosine

        try:
            from core.voiceid_database import compute_embedding
        except ImportError:
            _log("[Finalize] VoiceID offline: không import được compute_embedding")
            return labeled

        if not os.path.exists(wav_path):
            _log(f"[Finalize] VoiceID offline: WAV không tồn tại: {wav_path}")
            return labeled

        total = len(labeled)
        confirmed_count = 0
        skipped_confirmed = 0
        short_propagated = 0

        # ── Pass 1: Embed tất cả turn đủ dài chưa confirmed ─────────────────────
        for i, t in enumerate(labeled):
            pct = 30 + int(30 * i / total)
            prog_cb(pct, f"🎙️ VoiceID offline {i+1}/{total}: {t.speaker} [{t.start:.1f}-{t.end:.1f}s]")

            # Turn đã confirmed từ live → bỏ qua, tin tưởng kết quả live
            if getattr(t, "voice_confirmed", False):
                skipped_confirmed += 1
                continue

            dur = t.end - t.start
            if dur < min_dur:
                # Turn ngắn → xử lý ở Pass 2
                continue

            # Cắt audio đoạn tương ứng
            tmp = os.path.join(
                tempfile.gettempdir(),
                f"fin_vid_{int(t.start*100)}_{int(t.end*100)}.wav"
            )
            try:
                _extract_live_chunk(wav_path, t.start, t.end, tmp)
                vec = compute_embedding(tmp)
            except Exception as e:
                _log(f"[Finalize] VoiceID offline lỗi turn [{t.start:.1f}-{t.end:.1f}s]: {e}")
                vec = None
            finally:
                if os.path.exists(tmp):
                    try: os.remove(tmp)
                    except: pass

            if vec is None:
                continue

            # Tính score top-2
            scores = {}
            for db_name, vec_list in db_vectors.items():
                scores[db_name] = max(
                    1.0 - cosine(vec, db_vec)
                    for db_vec in vec_list
                )

            if not scores:
                continue

            sorted_s = sorted(scores.items(), key=lambda x: -x[1])
            best_name,  best_score  = sorted_s[0]
            second_score = sorted_s[1][1] if len(sorted_s) > 1 else 0.0
            gap = best_score - second_score

            _log(f"[Finalize] VoiceID [{t.start:.1f}-{t.end:.1f}s] "
                 f"top1={best_name}({best_score:.2f}) "
                 f"top2={sorted_s[1][0] if len(sorted_s)>1 else '-'}({second_score:.2f}) "
                 f"margin={gap:.2f}")

            if _voiceid_decide(best_score, gap):
                old_spk = t.speaker
                t.speaker = best_name
                t.voice_confirmed = True
                confirmed_count += 1
                _log(f"[Finalize] ✅ {old_spk} → {best_name} ({best_score:.2f})")
            else:
                _log(f"[Finalize] ⚠️  {t.speaker} không đủ tin cậy "
                     f"(score={best_score:.2f}, margin={gap:.2f}) → giữ diart label")
        _log(f"[Finalize] VoiceID offline xong: "
             f"confirmed={confirmed_count}, "
             f"live_kept={skipped_confirmed}, total={total}")
        return labeled


    def _finalize_worker(wav_path: str, turns_snapshot: list,
                         windows_snapshot: list, domain_files: list = None) -> None: # Đổi thành domain_files: list
        state = _shared["finalize_state"]
        lock  = _shared["finalize_lock"]

        def _prog(pct: int, msg: str):
            with lock:
                state["pct"] = max(state["pct"], int(pct))
                state["msg"] = msg
            _log(f"[Finalize {pct:3d}%] {msg}")

        try:
            # Đợi diart xử lý nốt backlog (nếu CPU chậm hơn realtime) rồi mới lấy
            # windows đầy đủ. Hiện tiến độ backlog cho user.
            def _drain_prog(pending):
                _prog(10, f"⏳ Đợi diart xử lý nốt… còn {pending} đoạn audio")
            diarizer_wait_drain(_diarizer, progress_cb=_drain_prog)
            windows_snapshot = diarizer_get_windows(_diarizer)
            from core.diart_diarizer import _postprocess_windows
            windows_snapshot = _postprocess_windows(windows_snapshot)

            _prog(15, "🧭 Gán người nói cho từng lượt (overlap với diart)…")
            from core.diarizer import SpeakerSegment, get_speaker_stats
            from core.aligner import AlignedTurn, merge_consecutive, rename_turns
            from core.punctuation_restorer import restore

            if not windows_snapshot:
                raise RuntimeError(
                    "Diart không trả ra segment nào. Kiểm tra audio có speech và "
                    "HF_TOKEN + license cho pyannote/segmentation-3.0 và pyannote/embedding."
                )

            # ── Gán speaker theo overlap, GIỮ NGUYÊN ranh giới lượt của STT ──────
            # KHÔNG dùng forced-align/word-level (gây xé câu ở ranh giới chuyển
            # turn do diart boundary lệch khỏi ranh giới câu nói thật — mỗi từ
            # gần điểm chuyển có thể bị gán nhầm sang speaker liền kề).
            # Mỗi turn STT (đã có timestamp thật từ Zipformer endpoint) chỉ cần
            # map 1 lần vào diart window có overlap nhiều nhất — giữ trọn câu.
            def _assign(t_start: float, t_end: float) -> str:
                best, best_ov = None, 0.0
                for w in windows_snapshot:
                    ov = max(0.0, min(w["end"], t_end) - max(w["start"], t_start))
                    if ov > best_ov:
                        best_ov, best = ov, w["label"]
                return best or "SPEAKER_00"

            turns = [t for t in turns_snapshot if t.text and t.text.strip()]
            labeled = []
            for t in turns:
                if getattr(t, "voice_confirmed", False):
                    # Đã confirmed từ live → giữ nguyên tên
                    spk = t.speaker
                else:
                    # Chưa confirmed → dùng diart windows finalize (chính xác hơn live)
                    spk = _assign(float(t.start), float(t.end))
                    spk = st.session_state.get("live_id_map", {}).get(spk, spk)
                labeled.append(AlignedTurn(
                    speaker         = spk,
                    start           = float(t.start),
                    end             = float(t.end),
                    text            = t.text,
                    voice_confirmed = getattr(t, "voice_confirmed", False),
                ))

            # ── VOICEID OFFLINE: Nhận diện lại từng câu (Sửa Label Bleeding) ────
            _prog(20, "🔬 VoiceID offline: nhận diện lại toàn bộ turns…")
            db_vectors = st.session_state.get("db_vectors", {})
            if db_vectors and wav_path and __import__("os").path.exists(wav_path):
                labeled = _voiceid_relabel_offline(
                    labeled    = labeled,
                    wav_path   = wav_path,
                    db_vectors = db_vectors,
                    prog_cb    = _prog,
                    min_dur    = 2.0,   # Chỉ cần câu dài 2 giây là đủ trích xuất vân tay
                )
            else:
                _log("[Finalize] Bỏ qua VoiceID offline (không có DB hoặc WAV)")

            # ── Khôi phục dấu câu ────────────────────────────────────────────────
            _prog(65, "🪡 Gộp lượt liên tiếp cùng người nói…")
            merged = merge_consecutive(labeled, gap_limit=2.5)

            # ── 2. Khôi phục dấu câu BẰNG MÔ HÌNH AI OFFLINE ──
            _prog(75, "🧠 Khôi phục dấu câu bằng AI (XLM-RoBERTa)…")
            try:
                from core.ai_punctuation import restore_text
                for t in merged:
                    t.text = restore_text(t.text)
            except Exception as e:
                _log(f"[Finalize Lỗi AI] {e}. Dự phòng về Rule-based!")
                from core.punctuation_restorer import restore
                for t in merged:
                    t.text = restore(t.text)

            # ── 3. CHUẨN HÓA THUẬT NGỮ & SỐ LIỆU (TERM NORMALIZER) ──
            _prog(85, "⚙️ Chuẩn hóa thuật ngữ kỹ thuật và số liệu…")
            try:
                from core.term_normalizer import normalize_terms
                # Nếu người dùng chọn nhiều file, hệ thống sẽ tuần tự chạy thực thi đè qua từng từ điển
                if domain_files:
                    for domain_f in domain_files:
                        for t in merged:
                            t.text = normalize_terms(t.text, domain_file=domain_f)
            except Exception as e:
                _log(f"⚠️ Lỗi chuẩn hóa thuật ngữ: {e}")

            # ── Đánh số người nói theo thứ tự xuất hiện ──────────────────────────
            # CHỈ đánh số SPEAKER_XX chưa được VoiceID gán tên.
            # Turn đã có tên thật (Nguyen_Lac_Huy, Hoang…) không match SPEAKER_XX
            # → rename_turns() giữ nguyên tên đó, không đổi thành "Người nói N".
            first_seen = {}
            for t in merged:
                if (t.speaker.upper().startswith("SPEAKER_")
                        and not getattr(t, "voice_confirmed", False)
                        and t.speaker not in first_seen):
                    first_seen[t.speaker] = float(t.start)
            raw_speakers_sorted = sorted(first_seen, key=lambda s: first_seen[s])
            rename_map = {raw_s: f"Người nói {i + 1}"
                          for i, raw_s in enumerate(raw_speakers_sorted)}
            if rename_map:
                merged = rename_turns(merged, rename_map)

            n_spk = len({t.speaker for t in merged})
            stats_segs = [
                SpeakerSegment(speaker=t.speaker, start=t.start, end=t.end)
                for t in merged
            ]

            # Gọi hàm _prog để nó in mốc 100% ra Terminal 
            _prog(100, f"✅ Hoàn tất! {len(merged)} lượt · {n_spk} người nói")

            with lock:
                state["result"] = {
                    "merged":     merged,
                    "stats":      get_speaker_stats(stats_segs),
                    "n_speakers": n_spk,
                    "backend":    "diart + pyannote/segmentation-3.0 + VoiceID offline",
                }
                state["running"] = False
        except Exception as e:
            import traceback; traceback.print_exc()
            with lock:
                state["error"]   = f"{type(e).__name__}: {e}"
                state["running"] = False


    # ══════════════════════════════════════════════════════════════════════════════
    # 11. SESSION STATE
    # ══════════════════════════════════════════════════════════════════════════════
    _defaults = {
        "l_recording":        False,
        "l_finished":         False,
        "l_diar_done":        False,
        "l_renamed":          False,
        "l_stream_ref":       None,
        "l_asr_stream":       None,
        "l_utterance_start":  0.0,
        "l_turns":            [],
        "l_stats":            {},
        "l_name_map":         {},
        "l_n_chunks":         0,
        "l_partial":          "",
        "l_session_name":     "",
        "l_show_full":        False,
        "l_prev_win_count":   0,
        "l_mic_device":       None,
        "l_paused":           False,
        "l_file_mode":        False,
        "l_finalize_count":   0,
        "l_spk_color_map":    {},
        "l_ram_warned":       False,
        "l_show_spectrum":    True,
        "l_show_live_text":   True,
        "l_selected_voices":  [],      # danh sách tên VoiceID được chọn cho phiên này
        "l_domain_files_list": [],
    }
    for k, v in _defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # ══════════════════════════════════════════════════════════════════════════════
    # 12. COCKPIT WORKSPACE (2-COLUMN MODERN STUDIO LAYOUT)
    # ══════════════════════════════════════════════════════════════════════════════
    col_ctrl, col_stage = st.columns([1, 1.85], gap="large")

    # ──────────────────────────────────────────────────────────────────────────────
    # CỘT TRÁI (35%): ĐIỀU KHIỂN COCKPIT, STATUS CARD, CẤU HÌNH & MONITOR
    # ──────────────────────────────────────────────────────────────────────────────
    with col_ctrl:
        # 1. STUDIO HERO STATUS CARD
        dur_sec = (st.session_state.l_n_chunks * CHUNK_FRAMES) / SAMPLE_RATE if (st.session_state.l_recording or st.session_state.l_finished) else 0.0
        peak = float(_shared.get("audio_peak", 0.0))
        peak_pct = min(100, int(peak * 130))

        if dur_sec >= 3600:
            timer_text = f"{int(dur_sec//3600):02d}:{int((dur_sec%3600)//60):02d}:{int(dur_sec%60):02d}"
        else:
            timer_text = f"{int(dur_sec//60):02d}:{int(dur_sec%60):02d}"

        if st.session_state.l_recording:
            if st.session_state.l_paused:
                badge_html = '<span class="studio-badge paused">⏸ TẠM DỪNG</span>'
                status_sub = "Đang tạm dừng thu âm"
            else:
                tag = "FILE TEST" if st.session_state.l_file_mode else "MICRO LIVE"
                badge_html = f'<span class="studio-badge rec"><span class="live-dot"></span> ĐANG GHI ({tag})</span>'
                status_sub = "Hệ thống đang lắng nghe & bóc băng..."
        elif st.session_state.l_diar_done:
            badge_html = '<span class="studio-badge done">✓ ĐÃ HOÀN TẤT</span>'
            status_sub = f"Đã chuẩn hoá ({len(st.session_state.l_stats)} người nói)"
        elif st.session_state.l_finished:
            badge_html = '<span class="studio-badge paused">⏹ ĐÃ DỪNG</span>'
            status_sub = "Sẵn sàng chạy Hoàn thiện Transcript"
        else:
            badge_html = '<span class="studio-badge ready">○ SẴN SÀNG</span>'
            status_sub = "Sẵn sàng bắt đầu cuộc họp mới"

        n_turns_count = len(st.session_state.l_turns)
        n_vid_count = len(st.session_state.get("db_vectors", {}))

        hero_html = (
            f'<div class="studio-status-card">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">'
            f'<div>{badge_html}</div>'
            f'<div style="font-family:IBM Plex Mono,monospace;font-size:11px;opacity:0.75;letter-spacing:1px;">{time.strftime("%H:%M:%S")}</div>'
            f'</div>'
            f'<div class="studio-timer">{timer_text}</div>'
            f'<div style="font-size:12px;opacity:0.8;margin-bottom:14px;">{status_sub}</div>'
            f'<div style="background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.08);border-radius:8px;padding:10px 12px;margin-bottom:12px;">'
            f'<div style="display:flex;justify-content:space-between;font-family:IBM Plex Mono,monospace;font-size:10px;opacity:0.85;margin-bottom:6px;">'
            f'<span>TÍN HIỆU ÂM THANH</span>'
            f'<span>{peak_pct}%</span>'
            f'</div>'
            f'<div class="vu-track">'
            f'<div class="vu-fill" style="width:{peak_pct}%;"></div>'
            f'</div>'
            f'</div>'
            f'<div style="display:flex;justify-content:space-between;padding-top:8px;border-top:1px solid rgba(255,255,255,0.12);font-size:11px;opacity:0.9;">'
            f'<span>Lượt nói: <b>{n_turns_count}</b></span>'
            f'<span>VoiceID: <b>{n_vid_count} người</b></span>'
            f'</div>'
            f'</div>'
        )
        _render_html(hero_html)

        # 2. ACTION CONTROLS BUTTON CLUSTER
        _fs_running = bool(_shared["finalize_state"].get("running"))

        if not st.session_state.l_recording:
            if not st.session_state.l_finished:
                # State: Ready to start
                if st.button("🔴 Bắt đầu ghi âm", type="primary", use_container_width=True, key="btn_start_rec"):
                    for k in ["l_turns"]:
                        st.session_state[k] = []
                    for k in ["l_stats", "l_name_map", "l_spk_color_map"]:
                        st.session_state[k] = {}
                    for k in ["l_n_chunks", "l_prev_win_count", "l_finalize_count"]:
                        st.session_state[k] = 0
                    st.session_state.l_partial          = ""
                    st.session_state.l_finished         = False
                    st.session_state.l_diar_done        = False
                    st.session_state.l_renamed          = False
                    st.session_state.l_asr_stream       = None
                    st.session_state.l_utterance_start  = 0.0
                    st.session_state.l_show_full        = False
                    st.session_state.l_paused           = False
                    st.session_state.l_file_mode        = False
                    st.session_state.l_ram_warned       = False
                    st.session_state.live_id_map        = {}
                    st.session_state.live_id_score      = {}
                    st.session_state.db_vectors         = {}

                    try:
                        from core.voiceid_database import get_all_entries, load_vectors_for_entry
                        selected = set(st.session_state.get("l_selected_voices") or [])
                        for entry in get_all_entries():
                            if entry["name"] in selected and entry.get("has_vector"):
                                vecs = load_vectors_for_entry(entry)
                                if vecs:
                                    st.session_state.db_vectors[entry["name"]] = vecs
                        print(f"[VoiceID] 🚀 Đã nạp {len(st.session_state.db_vectors)} người vào RAM!")
                    except Exception as e:
                        print(f"[VoiceID] ⚠️ Lỗi nạp dữ liệu: {e}")

                    with _AUDIO_QUEUE.mutex: _AUDIO_QUEUE.queue.clear()
                    old = _wav_close()
                    if old:
                        try: os.unlink(old)
                        except: pass
                    _wav_open(SAMPLE_RATE)

                    _viz_clear()
                    diarizer_reset(_diarizer)
                    diarizer_start(_diarizer)

                    _log(f"🟢 Bắt đầu: {st.session_state.l_session_name} · SR={SAMPLE_RATE}Hz · chunk={CHUNK_FRAMES}f")

                    stream = sd.InputStream(
                        samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                        blocksize=CHUNK_FRAMES, callback=_audio_callback,
                        device=st.session_state.l_mic_device,
                    )
                    stream.start()
                    st.session_state.l_stream_ref = stream
                    st.session_state.l_recording  = True
                    st.rerun()

            elif not st.session_state.l_diar_done:
                # State: Stopped recording, ready to finalize
                btn_fin_label = "⏳ Đang hoàn thiện..." if _fs_running else "✅ Hoàn thiện Transcript"
                if st.button(btn_fin_label, type="primary", use_container_width=True, disabled=_fs_running, key="btn_fin_primary"):
                    st.session_state.l_diar_done = False
                    st.session_state.l_renamed = False
                    st.session_state.l_finalize_count += 1
                    with _shared["finalize_lock"]:
                        _shared["finalize_state"].update({
                            "running": True, "pct": 0, "msg": "Khởi động pipeline...",
                            "result": None, "error": None,
                        })
                    _worker = threading.Thread(
                        target=_finalize_worker,
                        kwargs={
                            "wav_path": _wav_path(),
                            "turns_snapshot": list(st.session_state.l_turns),
                            "windows_snapshot": diarizer_get_windows(_diarizer),
                            "domain_files": list(st.session_state.l_domain_files_list),
                        },
                        daemon=True, name="finalize-worker",
                    )
                    add_script_run_ctx(_worker); _worker.start()
                    st.rerun()

            else:
                # State: Diarization done! Show re-run finalize and new session buttons
                col_b1, col_b2 = st.columns([1, 1], gap="small")
                with col_b1:
                    btn_refin_label = "⏳ Đang chạy..." if _fs_running else "🔄 Chạy lại"
                    if st.button(btn_refin_label, use_container_width=True, disabled=_fs_running, key="btn_fin_rerun"):
                        st.session_state.l_diar_done = False
                        st.session_state.l_renamed = False
                        st.session_state.l_finalize_count += 1
                        with _shared["finalize_lock"]:
                            _shared["finalize_state"].update({
                                "running": True, "pct": 0, "msg": "Khởi động pipeline...",
                                "result": None, "error": None,
                            })
                        _worker = threading.Thread(
                            target=_finalize_worker,
                            kwargs={
                                "wav_path": _wav_path(),
                                "turns_snapshot": list(st.session_state.l_turns),
                                "windows_snapshot": diarizer_get_windows(_diarizer),
                                "domain_files": list(st.session_state.l_domain_files_list),
                            },
                            daemon=True, name="finalize-worker",
                        )
                        add_script_run_ctx(_worker); _worker.start()
                        st.rerun()

                with col_b2:
                    if st.button("🆕 Phiên mới", type="primary", use_container_width=True, key="btn_new_session_main",
                                 help="Xoá transcript hiện tại, dọn RAM/file tạm, sẵn sàng ghi phiên mới"):
                        old_wav = _wav_close()
                        if old_wav:
                            try: os.unlink(old_wav)
                            except Exception: pass
                        with _AUDIO_QUEUE.mutex:
                            _AUDIO_QUEUE.queue.clear()
                        _viz_clear()
                        try:
                            diarizer_reset(_diarizer)
                        except Exception:
                            pass
                        with _shared["finalize_lock"]:
                            _shared["finalize_state"].update({
                                "running": False, "pct": 0, "msg": "",
                                "result": None, "error": None,
                            })
                        _KEEP_ON_RESET = {"l_mic_device", "l_domain_files_list",
                                           "l_selected_voices", "l_active_group"}
                        for k, v in _defaults.items():
                            if k not in _KEEP_ON_RESET:
                                st.session_state[k] = v
                        st.session_state.l_session_name = f"MEETING-{time.strftime('%Y%m%d-%H%M')}"
                        st.session_state.live_id_map   = {}
                        st.session_state.live_id_score = {}
                        st.session_state.db_vectors    = {}
                        _log("🆕 Đã tạo phiên mới — reset toàn bộ state")
                        st.rerun()

        else:
            # Currently recording: Pause / Resume + Stop
            col_r1, col_r2 = st.columns([1, 1], gap="small")
            with col_r1:
                if st.session_state.l_paused:
                    if st.button("▶️ Tiếp tục", type="primary", use_container_width=True, key="btn_resume"):
                        try:
                            stream = sd.InputStream(
                                samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                                blocksize=CHUNK_FRAMES, callback=_audio_callback,
                                device=st.session_state.l_mic_device,
                            )
                            stream.start()
                            st.session_state.l_stream_ref = stream
                            st.session_state.l_paused = False
                            st.rerun()
                        except Exception as e:
                            st.error(f"Không mở được mic: {e}")
                else:
                    if st.button("⏸️ Tạm dừng", use_container_width=True, disabled=st.session_state.l_file_mode, key="btn_pause"):
                        if st.session_state.l_stream_ref:
                            try: st.session_state.l_stream_ref.stop()
                            except: pass
                            try: st.session_state.l_stream_ref.close()
                            except: pass
                            st.session_state.l_stream_ref = None
                        st.session_state.l_paused = True
                        st.rerun()

            with col_r2:
                if st.button("⏹️ Dừng ghi", use_container_width=True, key="btn_stop_live"):
                    _shared["file_streaming"] = False
                    if st.session_state.l_stream_ref:
                        st.session_state.l_stream_ref.stop()
                        st.session_state.l_stream_ref.close()
                        st.session_state.l_stream_ref = None
                    _flush_final_asr()
                    diarizer_stop(_diarizer)
                    dur = (st.session_state.l_n_chunks * CHUNK_FRAMES) / SAMPLE_RATE
                    s = diarizer_get_stats(_diarizer)
                    _log(f"🔴 Dừng | dur={dur:.1f}s | turns={len(st.session_state.l_turns)} | active={s['handler_active']}")
                    st.session_state.l_asr_stream = None
                    st.session_state.l_recording  = False
                    st.session_state.l_finished   = True
                    st.session_state.l_paused     = False
                    st.session_state.l_file_mode  = False
                    _wav_close()
                    st.rerun()

        # 3. SESSION CONFIGURATION PANEL
        st.markdown(
            '<div style="font-family:\'JetBrains Mono\',monospace;font-size:11px;font-weight:700;'
            'letter-spacing:1px;text-transform:uppercase;color:#2563EB;margin:20px 0 10px;'
            'display:flex;align-items:center;gap:6px;">'
            '⚙️ THIẾT LẬP PHIÊN HỌP</div>',
            unsafe_allow_html=True,
        )

        session_name = st.text_input(
            "Tên cuộc họp",
            value    = st.session_state.l_session_name or f"MEETING-{time.strftime('%Y%m%d-%H%M')}",
            disabled = st.session_state.l_recording,
            key      = "input_session_name",
        )
        if session_name != st.session_state.l_session_name:
            st.session_state.l_session_name = session_name

        devices = _list_input_devices()
        if devices:
            device_labels = ["🎙️ Mặc định hệ thống"] + [f"#{i} · {name}" for i, name in devices]
            device_ids    = [None] + [i for i, _ in devices]
            try:
                current_idx = device_ids.index(st.session_state.l_mic_device) if st.session_state.l_mic_device in device_ids else 0
            except ValueError:
                current_idx = 0
            sel = st.selectbox("Thiết bị micro", device_labels, index=current_idx,
                               disabled=st.session_state.l_recording, key="sel_mic_device")
            st.session_state.l_mic_device = device_ids[device_labels.index(sel)]
        else:
            st.caption("🎙️ Mic: Mặc định hệ thống")

        domain_dir = os.path.join("core", "data", "domains")
        os.makedirs(domain_dir, exist_ok=True)
        domain_options = [f for f in os.listdir(domain_dir) if f.endswith(".json")]

        if not st.session_state.get("l_domain_files_list"):
            st.session_state.l_domain_files_list = []

        selected_domains = st.multiselect(
            "📚 Chủ đề từ điển chuẩn hoá",
            options  = domain_options,
            default  = [f for f in st.session_state.l_domain_files_list if f in domain_options],
            disabled = st.session_state.l_recording,
            key      = "l_domains_multiselect",
        )
        st.session_state.l_domain_files_list = selected_domains

        try:
            from core.voiceid_database import get_all_entries as _get_all_vid, get_groups as _get_groups_vid
            _vid_entries = [e for e in _get_all_vid() if e.get("has_vector")]
        except Exception:
            _vid_entries = []
            _get_groups_vid = None

        try:
            VOICEID_GROUPS = _get_groups_vid() if _get_groups_vid else ["Nhóm 1: Toạ đàm", "Nhóm 2: Podcast", "Nhóm 3: Meeting"]
        except Exception:
            VOICEID_GROUPS = ["Nhóm 1: Toạ đàm", "Nhóm 2: Podcast", "Nhóm 3: Meeting"]
        if "l_active_group" not in st.session_state:
            st.session_state.l_active_group = VOICEID_GROUPS[0]

        if _vid_entries:
            selected_group = st.selectbox(
                "📁 Nhóm VoiceID",
                options=VOICEID_GROUPS,
                index=VOICEID_GROUPS.index(st.session_state.l_active_group) if st.session_state.l_active_group in VOICEID_GROUPS else 0,
                disabled=st.session_state.l_recording,
                key="l_group_select"
            )
            if selected_group != st.session_state.l_active_group:
                st.session_state.l_active_group = selected_group
                st.session_state.l_selected_voices = []
                st.rerun()

            _filtered_names = [
                e["name"] for e in _vid_entries 
                if e.get("group", VOICEID_GROUPS[0]) == st.session_state.l_active_group
            ]

            if "l_selected_voices" not in st.session_state or st.session_state.l_selected_voices is None:
                st.session_state.l_selected_voices = []

            selected_voices = st.multiselect(
                "👥 Người tham gia nhận diện giọng nói",
                options  = _filtered_names,
                default  = [n for n in st.session_state.l_selected_voices if n in _filtered_names],
                disabled = st.session_state.l_recording,
                key      = "l_voices_select",
            )
            st.session_state.l_selected_voices = selected_voices
        else:
            st.caption("ℹ️ Chưa có vector VoiceID — thêm tại tab VoiceID.")

        st.caption("💡 Tự động gán tên thật từ VoiceID. Lượt nói chưa định danh sẽ tạm đánh số (Người nói 1, 2…).")

        # 4. REAL-TIME AUDIO MONITOR EXPANDER
        with st.expander("📡 Sóng âm & Phổ tần số", expanded=st.session_state.l_recording):
            st.session_state.l_show_spectrum = st.checkbox(
                "Bật hiển thị sóng âm/phổ", value=st.session_state.get("l_show_spectrum", True),
                key="chk_show_spectrum",
            )
            if st.session_state.l_recording and st.session_state.get("l_show_spectrum", True):
                _fig = _render_spectrum()
                if _fig is not None:
                    st.pyplot(_fig, clear_figure=True)
                else:
                    st.caption("⏳ Đang chờ tín hiệu âm thanh…")
            elif not st.session_state.l_recording:
                st.caption("ℹ️ Sóng âm thời gian thực sẽ hiển thị khi bắt đầu ghi âm.")

        # 5. TEST AUDIO EXPANDER (WAV File)
        if not st.session_state.l_recording and not st.session_state.l_finished:
            with st.expander("🧪 Thử nghiệm bằng file WAV", expanded=False):
                st.caption("Bơm file WAV qua pipeline STT Zipformer + Diart như micro thật.")
                up = st.file_uploader("Chọn file WAV (16-bit PCM)", type=["wav"], key="l_test_file")
                realtime = st.checkbox("Phát tốc độ 1× (mô phỏng mic)", value=True,
                                       help="Bỏ chọn để bơm tốc độ cao (fast).", key="chk_realtime_file")
                if st.button("▶️ Chạy thử nghiệm file", disabled=(up is None), use_container_width=True, key="btn_run_wav_test"):
                    tmp = tempfile.NamedTemporaryFile(suffix="_test.wav", delete=False)
                    tmp.write(up.getbuffer()); tmp.flush(); tmp.close()
                    try:
                        audio_i16, info = _read_wav_validated(tmp.name, target_sr=SAMPLE_RATE)
                    except ValueError as e:
                        st.error(f"❌ {e}")
                        try: os.unlink(tmp.name)
                        except: pass
                    else:
                        st.info(f"📁 {up.name} · {info}")
                        st.session_state.l_turns = []; st.session_state.l_stats = {}; st.session_state.l_name_map = {}
                        st.session_state.l_n_chunks = 0; st.session_state.l_partial = ""
                        st.session_state.l_finished = False; st.session_state.l_diar_done = False; st.session_state.l_renamed = False
                        st.session_state.l_asr_stream = None; st.session_state.l_paused = False
                        st.session_state.l_finalize_count = 0; st.session_state.l_ram_warned = False
                        st.session_state.l_spk_color_map = {}; st.session_state.l_prev_win_count = 0
                        st.session_state.live_id_map = {}; st.session_state.live_id_score = {}; st.session_state.db_vectors = {}
                        try:
                            from core.voiceid_database import get_all_entries, load_vectors_for_entry
                            selected = set(st.session_state.get("l_selected_voices") or [])
                            for entry in get_all_entries():
                                if entry["name"] in selected and entry.get("has_vector"):
                                    vecs = load_vectors_for_entry(entry)
                                    if vecs: st.session_state.db_vectors[entry["name"]] = vecs
                            print(f"[VoiceID] 🚀 Đã nạp {len(st.session_state.db_vectors)} người vào RAM!")
                        except Exception as e:
                            print(f"[VoiceID] ⚠️ Lỗi nạp database: {e}")

                        with _AUDIO_QUEUE.mutex: _AUDIO_QUEUE.queue.clear()
                        old = _wav_close()
                        if old:
                            try: os.unlink(old)
                            except: pass
                        _wav_open(SAMPLE_RATE)
                        _viz_clear()
                        diarizer_reset(_diarizer)
                        diarizer_start(_diarizer)

                        _shared["file_streaming"] = True
                        _shared["file_done"]      = False
                        ft = threading.Thread(
                            target=_stream_audio_thread, args=(audio_i16, realtime),
                            daemon=True, name="file-stream",
                        )
                        add_script_run_ctx(ft); ft.start()

                        st.session_state.l_file_mode  = True
                        st.session_state.l_recording  = True
                        st.rerun()

    # ──────────────────────────────────────────────────────────────────────────────
    # CỘT PHẢI (65%): BIÊN BẢN HỘI THOẠI TRỰC TIẾP, TIMELINE, EDIT & EXPORT DOCK
    # ──────────────────────────────────────────────────────────────────────────────
    with col_stage:
        # 1. FINALIZE PROGRESS & STATUS
        _fs = _shared["finalize_state"]
        with _shared["finalize_lock"]:
            _fs_running = bool(_fs.get("running"))
            _fs_pct     = int(_fs.get("pct", 0))
            _fs_msg     = str(_fs.get("msg", ""))
            _fs_result  = _fs.get("result")
            _fs_error   = _fs.get("error")

        if _fs_running:
            st.progress(max(0, min(100, _fs_pct)), text=_fs_msg or "Đang xử lý phân đoạn & chuẩn hoá...")
            time.sleep(0.5)
            st.rerun()
        elif _fs_result is not None and not st.session_state.l_diar_done:
            res = _fs_result
            st.session_state.l_turns     = res["merged"]
            st.session_state.l_stats     = res["stats"]
            st.session_state.l_diar_done = True
            st.success(f"✅ Hoàn tất tinh chỉnh — {len(res['merged'])} lượt nói · {res['n_speakers']} người nói ({res['backend']})")
            with _shared["finalize_lock"]:
                _fs["result"] = None
            st.rerun()
        elif _fs_error and not st.session_state.l_diar_done:
            st.error(f"❌ Lỗi khi hoàn thiện: {_fs_error}")
            with _shared["finalize_lock"]:
                _fs["error"] = None

        # 2. STAGE HEADER BAR (TITLE + STATS + ACTIONS)
        turns = st.session_state.l_turns
        total_words = sum(len(t.text.split()) for t in turns) if turns else 0
        n_speakers_count = len(st.session_state.l_stats) if st.session_state.l_stats else (len(set(t.speaker for t in turns)) if turns else 0)

        stage_h1, stage_h2 = st.columns([1.6, 1], gap="medium")
        with stage_h1:
            st.markdown(
                '<div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">'
                '<span style="font-family:\'JetBrains Mono\',monospace;font-size:12px;font-weight:700;'
                'letter-spacing:1.5px;text-transform:uppercase;color:#2563EB;">'
                '🎙️ BIÊN BẢN HỘI THOẠI TRỰC TIẾP</span>'
                '</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div style="font-family:IBM Plex Mono,monospace;font-size:11px;color:var(--text-color);opacity:0.65;margin-bottom:8px;">'
                f'<b>{len(turns)}</b> lượt nói &nbsp;·&nbsp; <b>~{total_words}</b> từ &nbsp;·&nbsp; <b>{n_speakers_count}</b> người nói'
                f'</div>',
                unsafe_allow_html=True,
            )

        with stage_h2:
            st_btn_col1, st_btn_col2 = st.columns([1, 1], gap="small")
            with st_btn_col1:
                if turns:
                    full_transcript_text = "\n".join(
                        f"[{int(t.start//60):02d}:{int(t.start%60):02d}] {t.speaker}: {t.text}"
                        for t in turns
                    )
                    with st.popover("📋 Sao chép", use_container_width=True):
                        st.caption("Bấm Ctrl+A rồi Ctrl+C để sao chép toàn bộ:")
                        st.code(full_transcript_text, language=None)
            with st_btn_col2:
                if st.session_state.l_recording:
                    st.session_state.l_show_live_text = st.checkbox(
                        "Hiện live", value=st.session_state.get("l_show_live_text", True),
                        key="chk_show_live_text",
                    )
                elif turns:
                    lbl = "👁️ Xem nhanh" if st.session_state.l_show_full else "✏️ Chỉnh sửa"
                    if st.button(lbl, use_container_width=True, key="btn_toggle_view_mode"):
                        st.session_state.l_show_full = not st.session_state.l_show_full
                        st.rerun()

        # 3. SPEAKER EDITOR EXPANDER (when diarization is done)
        if st.session_state.l_diar_done and st.session_state.l_stats:
            with st.expander("👥 Chuẩn hoá danh tính người nói", expanded=not st.session_state.l_renamed):
                from components.speaker_editor import speaker_editor
                name_map = speaker_editor(st.session_state.l_stats, key_prefix="l_spk")
                if st.button("✅ Xác nhận tên người nói", type="primary", use_container_width=True, key="btn_confirm_spk_names"):
                    from core.aligner import rename_turns
                    st.session_state.l_turns    = rename_turns(st.session_state.l_turns, name_map)
                    st.session_state.l_name_map = name_map
                    st.session_state.l_renamed  = True
                    st.rerun()

        # 4. CONVERSATION TIMELINE STREAM
        if st.session_state.l_recording:
            if st.session_state.get("l_show_live_text", True):
                if not turns and not st.session_state.l_partial:
                    st.markdown(
                        '<div style="background:var(--secondary-background-color);border:1px dashed rgba(37,99,235,0.35);'
                        'border-radius:14px;padding:48px 20px;text-align:center;margin-bottom:12px;box-shadow:0 1px 3px rgba(0,0,0,0.02);">'
                        '<div style="font-size:36px;margin-bottom:10px;">🎙️</div>'
                        '<div style="font-weight:600;font-size:14px;color:var(--text-color);margin-bottom:4px;">'
                        'Đang lắng nghe âm thanh từ Micro...</div>'
                        '<div style="font-size:12px;color:var(--text-color);opacity:0.6;">'
                        'Hãy phát biểu vào micro. Lời nói sẽ tự động chuyển thành văn bản và xuất hiện theo thời gian thực.</div>'
                        '</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    visible_turns = turns[-15:] if len(turns) > 15 else turns
                    cards_html = ""
                    for t in visible_turns:
                        spk_friendly = _friendly_speaker(t.speaker)
                        spk_class    = _spk_css_class(spk_friendly)
                        ts_str       = f"{int(t.start//60):02d}:{int(t.start%60):02d}"
                        if hasattr(t, "end") and t.end > t.start:
                            ts_str  += f" → {int(t.end//60):02d}:{int(t.end%60):02d}"

                        cards_html += (
                            f'<div class="chat-bubble">'
                            f'<div class="chat-header">'
                            f'<span class="spk-pill {spk_class}">{_html.escape(spk_friendly)}</span>'
                            f'<span class="chat-ts">[{ts_str}]</span>'
                            f'</div>'
                            f'<div class="chat-text">{_html.escape(t.text)}</div>'
                            f'</div>'
                        )

                    partial = st.session_state.l_partial
                    if partial:
                        cards_html += (
                            f'<div class="streaming-box">'
                            f'<div class="chat-header">'
                            f'<span class="spk-pill spk-x"><span class="live-dot"></span> Đang nhận diện...</span>'
                            f'</div>'
                            f'<div class="chat-text" style="font-style:italic;opacity:0.85;">{_html.escape(partial)}</div>'
                            f'</div>'
                        )

                    st.markdown(cards_html, unsafe_allow_html=True)
                    st.caption("⏺ Phiên âm trực tiếp · Nhãn người nói sẽ được tinh chỉnh chính xác nhất khi bấm **Hoàn thiện Transcript**.")
            else:
                st.info("👁️ Đã ẩn hiển thị trực tiếp để tối ưu hiển thị. Hệ thống vẫn đang tiếp tục ghi nhận âm thanh ngầm...")

        elif turns:
            if st.session_state.l_show_full:
                from components.transcript_viewer import full as show_full
                show_full(turns, editable=True)
                if st.button("💾 Lưu thay đổi", type="primary", use_container_width=True, key="btn_save_edits"):
                    st.session_state.l_turns = turns
                    st.success("✅ Đã lưu các chỉnh sửa của bạn.")
            else:
                cards_html = ""
                for t in turns:
                    spk_friendly = _friendly_speaker(t.speaker)
                    spk_class    = _spk_css_class(spk_friendly)
                    ts_str       = f"{int(t.start//60):02d}:{int(t.start%60):02d}"
                    if hasattr(t, "end") and t.end > t.start:
                        ts_str  += f" → {int(t.end//60):02d}:{int(t.end%60):02d}"

                    cards_html += (
                        f'<div class="chat-bubble">'
                        f'<div class="chat-header">'
                        f'<span class="spk-pill {spk_class}">{_html.escape(spk_friendly)}</span>'
                        f'<span class="chat-ts">[{ts_str}]</span>'
                        f'</div>'
                        f'<div class="chat-text">{_html.escape(t.text)}</div>'
                        f'</div>'
                    )
                st.markdown(cards_html, unsafe_allow_html=True)

        else:
            st.markdown(
                '<div style="background:var(--secondary-background-color);border:1px dashed rgba(128,128,128,0.22);'
                'border-radius:12px;padding:60px 20px;text-align:center;margin-top:8px;">'
                '<div style="font-size:38px;margin-bottom:12px;">📑</div>'
                '<div style="font-weight:600;font-size:15px;color:var(--text-color);margin-bottom:6px;">'
                'Chưa có dữ liệu hội thoại</div>'
                '<div style="font-size:13px;color:var(--text-color);opacity:0.65;max-width:420px;margin:0 auto;line-height:1.5;">'
                'Thiết lập thông tin ở bảng điều khiển bên trái, sau đó nhấn <b>🔴 Bắt đầu ghi âm</b> hoặc tải file WAV để bắt đầu cuộc họp.'
                '</div>'
                '</div>',
                unsafe_allow_html=True,
            )

        # 5. EXPORT & PERSISTENCE DOCK
        if turns and (st.session_state.l_finished or st.session_state.l_diar_done):
            st.markdown(
                '<div style="font-family:\'JetBrains Mono\',monospace;font-size:11px;font-weight:700;'
                'letter-spacing:1px;text-transform:uppercase;color:#2563EB;margin:28px 0 12px;'
                'display:flex;align-items:center;gap:6px;">'
                '📥 XUẤT BIÊN BẢN & LƯU TRỮ PHIÊN HỌP</div>',
                unsafe_allow_html=True,
            )

            export_data = b""
            try:
                from components.export_docx import export_to_docx
                export_data = export_to_docx(
                    turns        = st.session_state.l_turns,
                    session_name = st.session_state.l_session_name,
                )
            except Exception as e:
                st.error(f"Lỗi tạo file DOCX: {e}")

            current_wav_path = _wav_path()
            wav_bytes = b""
            if current_wav_path and os.path.exists(current_wav_path):
                try:
                    with open(current_wav_path, "rb") as f:
                        wav_bytes = f.read()
                except Exception:
                    pass

            d_col1, d_col2, d_col3 = st.columns(3, gap="small")
            with d_col1:
                st.download_button(
                    label    = "📄 Xuất Word (.docx)",
                    data     = export_data,
                    file_name= f"Bien_ban_{st.session_state.l_session_name.replace(' ','_')}.docx",
                    mime     = "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    type     = "primary",
                    disabled = not bool(export_data),
                    use_container_width = True,
                    key      = "dl_btn_docx",
                )
            with d_col2:
                st.download_button(
                    label    = "🎵 Tải file ghi âm (.wav)",
                    data     = wav_bytes,
                    file_name= f"Audio_{st.session_state.l_session_name.replace(' ','_')}.wav",
                    mime     = "audio/wav",
                    disabled = not bool(wav_bytes),
                    use_container_width = True,
                    key      = "dl_btn_wav",
                )
            with d_col3:
                if st.button("💾 Lưu vào Lịch sử", use_container_width=True, key="btn_save_session_history"):
                    import json, shutil
                    session_id = st.session_state.l_session_name.replace(' ','_')
                    save_dir = os.path.join("data", "sessions", session_id)
                    os.makedirs(save_dir, exist_ok=True)
                    try:
                        if current_wav_path and os.path.exists(current_wav_path):
                            shutil.copy(current_wav_path, os.path.join(save_dir, "audio.wav"))

                        turns_data = [
                            {"start": t.start, "end": t.end, "speaker": t.speaker, "text": t.text}
                            for t in st.session_state.l_turns
                        ]
                        with open(os.path.join(save_dir, "transcript.json"), "w", encoding="utf-8") as f:
                            json.dump(turns_data, f, ensure_ascii=False, indent=2)

                        meta = {
                            "session_name": session_id,
                            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "stats": st.session_state.l_stats,
                            "voiceids_loaded": list(st.session_state.get("db_vectors", {}).keys()),
                        }
                        with open(os.path.join(save_dir, "metadata.json"), "w", encoding="utf-8") as f:
                            json.dump(meta, f, ensure_ascii=False, indent=2)

                        st.success(f"✅ Đã lưu cuộc họp '{session_id}' vào Lịch sử!")
                    except Exception as e:
                        st.error(f"❌ Lỗi khi lưu: {e}")

    # ══════════════════════════════════════════════════════════════════════════════
    # 16. POLL 200ms (khi đang ghi / file mode)
    # ══════════════════════════════════════════════════════════════════════════════
    if st.session_state.l_recording and not st.session_state.l_paused:
        partial, completed = _drain_queue()
        st.session_state.l_partial = partial
        if completed:
            st.session_state.l_turns.extend(completed)

        dur_min = (st.session_state.l_n_chunks * CHUNK_FRAMES / SAMPLE_RATE) / 60.0
        if dur_min > 60 and not st.session_state.l_ram_warned:
            st.session_state.l_ram_warned = True
            st.toast(f"ℹ️ Phiên đã chạy {dur_min:.0f} phút. Finalize sẽ tốn thời gian.", icon="ℹ️")

        # File mode: khi file đọc xong + queue cạn → auto dừng
        if (st.session_state.l_file_mode
                and _shared.get("file_done")
                and _AUDIO_QUEUE.empty()):
            _shared["file_streaming"] = False
            _flush_final_asr()
            diarizer_stop(_diarizer)
            st.session_state.l_asr_stream = None
            st.session_state.l_recording  = False
            st.session_state.l_finished   = True
            st.session_state.l_file_mode  = False
            _wav_close()
            st.rerun()

        time.sleep(0.2)
        st.rerun()