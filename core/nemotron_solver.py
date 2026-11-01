import os
import sys
import io

# =====================================================================
# 0. BẮT BUỘC UTF-8 CHO STDOUT/STDERR TRÊN WINDOWS
#    Phải đặt TRƯỚC mọi thứ khác, kể cả import argparse
# =====================================================================
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer,
        encoding="utf-8",
        errors="replace",
        line_buffering=True,
    )
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(
        sys.stderr.buffer,
        encoding="utf-8",
        errors="replace",
        line_buffering=True,
    )

import argparse
import json
import time
import re
import shutil
import tempfile

# =====================================================================
# 1. CẤU HÌNH ĐƯỜNG DẪN FFMPEG DLL CHO WINDOWS
# =====================================================================
ffmpeg_dll_path = r"E:\SOFTWARE\ffmpeg-2026-04-09-git-d3d0b7a5ee-essentials_build\ffmpeg-2026-04-09-git-d3d0b7a5ee-essentials_build\bin"

if os.path.exists(ffmpeg_dll_path):
    try:
        os.add_dll_directory(ffmpeg_dll_path)
    except Exception as e:
        sys.stderr.write(f"⚠️ Không thể add DLL directory: {e}\n")

import torch
import soundfile as sf
import nemo.collections.asr as nemo_asr
from nemo.utils import logging as nemo_logging

nemo_logging.setLevel(nemo_logging.ERROR)

# =====================================================================
# KHÔNG CÒN MONKEY PATCH — đã fix bằng "lang" đúng trong manifest
# =====================================================================

# =====================================================================
# 2. MODEL SINGLETON — chỉ load 1 lần duy nhất trong suốt vòng đời process
# =====================================================================
_model_instance = None
_model_device = None

def get_model():
    """Trả về model đã load sẵn. Nếu chưa có thì load lần đầu và cache lại."""
    global _model_instance, _model_device

    if _model_instance is not None:
        return _model_instance, _model_device

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sys.stderr.write(f"⏳ Đang nạp Nemotron 0.6B lên {device.upper()}...\n")

    model = nemo_asr.models.ASRModel.from_pretrained(
        model_name="nvidia/nemotron-3.5-asr-streaming-0.6b"
    )
    model = model.to(device)
    model.eval()

    _model_instance = model
    _model_device = device
    sys.stderr.write(f"✅ Đã nạp xong mô hình!\n")
    return _model_instance, _model_device


# =====================================================================
# 3. HÀM CHIA CHUNK VÀ TRANSCRIBE
# =====================================================================
def split_audio_to_chunks(wav_path, chunk_duration_sec=20):
    audio, sr = sf.read(wav_path, dtype='float32')
    chunk_size = int(chunk_duration_sec * sr)
    chunks = []
    for start in range(0, len(audio), chunk_size):
        chunk = audio[start:start + chunk_size]
        duration = len(chunk) / sr
        if duration < 0.5:
            continue
        chunks.append((chunk, sr, duration))
    return chunks


def transcribe_wav(wav_path, chunk_sec=20, lang="vi-VN"):
    """
    Transcribe file WAV. Model chỉ load 1 lần nhờ singleton.
    
    Args:
        wav_path:  Đường dẫn file .wav
        chunk_sec: Độ dài mỗi chunk (giây)
        lang:      Mã ngôn ngữ hợp lệ: vi-VN | en-US | en-GB | zh-CN | es-ES
    """
    if not os.path.exists(wav_path):
        return {
            "error": f"Không tìm thấy file audio: {wav_path}",
            "text": ""
        }

    try:
        # Lấy model từ singleton — không load lại nếu đã có
        model, device = get_model()

        chunks = split_audio_to_chunks(wav_path, chunk_duration_sec=chunk_sec)
        if not chunks:
            return {"error": "File audio quá ngắn hoặc không đọc được", "text": ""}

        results = []
        tmp_dir = tempfile.mkdtemp()

        try:
            for i, (audio, sr, duration) in enumerate(chunks):
                chunk_wav = os.path.join(tmp_dir, f"chunk_{i:04d}.wav")
                sf.write(chunk_wav, audio, sr)

                manifest_path = os.path.join(tmp_dir, f"manifest_{i:04d}.json")
                manifest_data = {
                    "audio_filepath": chunk_wav,
                    "duration": duration,
                    "text": "",
                    "lang": lang   # ← Key đúng, Lhotse đọc được supervision.language
                }
                with open(manifest_path, 'w', encoding='utf-8') as f:
                    f.write(json.dumps(manifest_data, ensure_ascii=False) + '\n')

                with torch.no_grad():
                    transcriptions = model.transcribe([manifest_path], verbose=False)

                if isinstance(transcriptions, tuple):
                    text = transcriptions[0][0]
                else:
                    text = transcriptions[0]

                if hasattr(text, 'text'):
                    text = text.text

                # Strip language tags VÀ lọc ký tự không phải Việt/Latin
                text = re.sub(r'<[^>]+>', '', text)
                text = _filter_non_vietnamese(text)
                text = text.strip()
                results.append(text)

                # Progress ra stderr để không lẫn vào JSON output
                done = i + 1
                total = len(chunks)
                bar = "█" * int(20 * done / total) + "░" * (20 - int(20 * done / total))
                sys.stderr.write(f"\r  [{bar}] {done}/{total} chunks")
                sys.stderr.flush()

            sys.stderr.write("\n")

        finally:
            try:
                shutil.rmtree(tmp_dir)
            except Exception:
                pass

        full_text = " ".join(results)
        return {
            "text": full_text,
            "raw": full_text,
            "error": None
        }

    except Exception as e:
        return {
            "error": f"Lỗi xử lý Nemotron: {str(e)}",
            "text": ""
        }


def _filter_non_vietnamese(text: str) -> str:
    """
    Lọc bỏ ký tự không thuộc Latin/tiếng Việt/dấu câu thông thường.
    Loại bỏ CJK và các block Unicode không liên quan.
    Giữ lại: ASCII, Latin Extended, khối tiếng Việt (U+00C0–U+1EF9), dấu câu.
    """
    def _keep(ch: str) -> bool:
        cp = ord(ch)
        # ASCII cơ bản (bao gồm dấu câu, chữ số, Latin)
        if cp < 0x0100:
            return True
        # Latin Extended + dấu tiếng Việt precomposed (U+00C0 – U+1EFF)
        if 0x00C0 <= cp <= 0x1EFF:
            return True
        # Combining diacritical marks (U+0300 – U+036F) — dấu rời
        if 0x0300 <= cp <= 0x036F:
            return True
        # Dấu câu Unicode phổ thông (…, –, —, ", ")
        if cp in (0x2013, 0x2014, 0x2018, 0x2019, 0x201C, 0x201D, 0x2026):
            return True
        # Ký tự lạ ⁇ thường xuất hiện trong output Nemotron — giữ lại
        if cp == 0x2047:
            return True
        return False

    return "".join(ch for ch in text if _keep(ch))


# =====================================================================
# 4. CLI
# =====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nemotron ASR Solver")
    parser.add_argument("--wav",   type=str, help="Đường dẫn file .wav")
    parser.add_argument("--lang",  type=str, default="vi-VN",
                        help="vi-VN | en-US | en-GB | zh-CN | es-ES")
    parser.add_argument("--chunk", type=int, default=20,
                        help="Độ dài mỗi chunk (giây), mặc định 20s")
    parser.add_argument("--test",  action="store_true",
                        help="Chạy test với file mặc định")
    args = parser.parse_args()

    test_file = args.wav or "Audio_MEETING-20260605-1059.wav"

    if args.test or not args.wav:
        sys.stderr.write("="*60 + "\n")
        sys.stderr.write("🔬 CHẾ ĐỘ TEST: NEMOTRON_SOLVER.PY\n")
        sys.stderr.write("="*60 + "\n")
        sys.stderr.write(f"🎙️  File: {test_file}\n")
        sys.stderr.write(f"✂️  Chunk: {args.chunk}s | 🌐 Lang: {args.lang}\n")

    if not os.path.exists(test_file):
        sys.stderr.write(f"❌ Không tìm thấy file '{test_file}'\n")
        sys.exit(1)

    start_time = time.time()
    output_data = transcribe_wav(test_file, chunk_sec=args.chunk, lang=args.lang)
    elapsed = time.time() - start_time

    if args.test or not args.wav:
        sys.stderr.write(f"⏱️  Tổng thời gian: {elapsed:.2f}s\n")
        sys.stderr.write("="*60 + "\n")

    # Output JSON ra stdout — sạch, không lẫn log
    print(json.dumps(output_data, ensure_ascii=False, indent=4 if args.test else None))