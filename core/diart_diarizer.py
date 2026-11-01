"""
Online speaker diarization wrapper around diart + pyannote/segmentation-3.0.

Exposes a dict-shaped API compatible with the previous Silero+WeSpeaker
worker in 2_Live_Mode.py so the rest of the page (STT lane, assign,
retro-update, finalize) keeps working unchanged.
"""
from __future__ import annotations

import os
import queue as _queue
import threading
import time
from typing import Optional

import numpy as np
import torch

# ──────────────────────────────────────────────────────────────────────────────
# torch>=2.6 đặt weights_only=True làm mặc định cho torch.load → block các
# class pickle trong checkpoint pyannote (TorchVersion, pyannote.audio.*.Specifications,
# omegaconf containers, lightning state...). Allowlist từng class là vô tận vì
# pyannote pickle hàng chục class khác nhau.
#
# Giải pháp: monkey-patch torch.load → weights_only=False mặc định.
# AN TOÀN trong trường hợp này vì:
#   - Chỉ load checkpoint pyannote chính chủ từ HuggingFace
#   - User đã accept license, model đã pin trong code (segmentation-3.0, embedding)
#   - Không load checkpoint từ nguồn không tin cậy
# ──────────────────────────────────────────────────────────────────────────────
_orig_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    # FORCE override: lightning_fabric.utilities.cloud_io._load truyền
    # weights_only=True explicit → setdefault sẽ no-op. Phải ép = False.
    kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from diart import SpeakerDiarization, SpeakerDiarizationConfig
from diart.models import SegmentationModel, EmbeddingModel
from diart.sources import AudioSource
from diart.inference import StreamingInference

from rx.subject import Subject, ReplaySubject


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ──────────────────────────────────────────────────────────────────────────────
# Custom AudioSource: pushed from the existing sounddevice callback
# ──────────────────────────────────────────────────────────────────────────────
class SoundDeviceAudioSource(AudioSource):
    """
    Pull-style AudioSource backed by an rx Subject.

    Audio frames are pushed in via push() from the sounddevice C-thread
    callback. diart's StreamingInference subscribes to .stream and slices
    the incoming buffer into segmentation windows.
    """

    def __init__(self, sample_rate: int = 16000, uri: str = "mic"):
        # diart.sources.AudioSource.__init__ tự tạo self.stream = Subject().
        # KHÔNG override stream thành @property (sẽ raise "can't set attribute").
        super().__init__(uri, sample_rate)
        # Subject() là HOT observable: chunks push trước khi inference subscribe
        # bị mất → "ran 0 times". Đổi sang ReplaySubject để buffer & replay khi
        # subscribe muộn (200 chunks ≈ 40s buffer, đủ thời gian khởi tạo).
        self.stream = ReplaySubject(buffer_size=200)
        self._closed = False
        self._stop_event = threading.Event()

    @property
    def duration(self) -> Optional[float]:
        return None

    def push(self, audio_f32: np.ndarray) -> None:
        """audio_f32: shape (samples,) or (channels, samples), dtype float32."""
        if self._closed:
            return
        if audio_f32.ndim == 1:
            audio_f32 = audio_f32[np.newaxis, :]
        self.stream.on_next(audio_f32)

    def read(self) -> None:
        # Streaming source — block until close() is called. Trả sớm khiến
        # StreamingInference nghĩ stream đã hết → exit ngay lập tức.
        self._stop_event.wait()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.stream.on_completed()
        except Exception:
            pass
        self._stop_event.set()   # unblock read()


# ──────────────────────────────────────────────────────────────────────────────
# Diarizer
# ──────────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────
# Dataset-tuned hyperparameter presets (from diart paper, juanmc2005/diart)
# https://github.com/juanmc2005/diart  — tuned with diart.tune on each corpus.
# Pick via env: DIART_PRESET=ami | dihard | voxconverse | default
# ──────────────────────────────────────────────────────────────────────────────
PRESETS = {
    "ami":         dict(tau_active=0.507, rho_update=0.006, delta_new=1.057),
    "dihard":      dict(tau_active=0.555, rho_update=0.422, delta_new=1.517),
    "voxconverse": dict(tau_active=0.576, rho_update=0.915, delta_new=0.648),
    "default":     dict(tau_active=0.5,   rho_update=0.3,   delta_new=1.0),
}


class DiartDiarizer:
    """
    Online diarization runner.

    State exposed for compatibility with the previous code:
        .windows  list[dict]   — [{"start", "end", "label"}, ...]
        .lock     threading.Lock
        .running  bool
        .stats    dict
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        step: float = 0.5,
        latency: float = 5.0,
        device: Optional[str] = None,
        preset: Optional[str] = None,
        max_speakers: Optional[int] = None,
        embedding_model: Optional[str] = None,
        segmentation_model="Khanh17/toadam-segmentation-model",
    ):
        self.sample_rate = sample_rate
        self.step = step
        self.latency = latency
        self.device = torch.device(device or os.getenv("DIART_DEVICE", "cpu"))

        # Hyperparameter preset — AMI is the best match for meeting/interrogation.
        preset_name = (preset or os.getenv("DIART_PRESET", "ami")).lower()
        if preset_name not in PRESETS:
            _log(f"[Diart] Unknown preset '{preset_name}', dùng 'ami'")
            preset_name = "ami"
        hp = PRESETS[preset_name]
        _log(f"[Diart] preset={preset_name}  τ={hp['tau_active']:.3f}  ρ={hp['rho_update']:.3f}  δ={hp['delta_new']:.3f}")

        # max_speakers: ưu tiên tham số > env > None (unlimited).
        if max_speakers is None:
            env_max = os.getenv("DIART_MAX_SPEAKERS", "").strip()
            if env_max.isdigit():
                max_speakers = int(env_max)
        if max_speakers:
            _log(f"[Diart] max_speakers={max_speakers}")

        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")

        _log(f"⏳ Dang nap {segmentation_model} (diart)...")
        segmentation = SegmentationModel.from_pretrained(
            segmentation_model, use_hf_token=hf_token
        )
        _log(f"✅ {segmentation_model} nap xong")

        # Embedding model — đổi qua env DIART_EMBEDDING.
        #   pyannote/embedding              : nhẹ, mặc định diart (model 2020, yếu)
        #   speechbrain/spkrec-ecapa-voxceleb : ECAPA-TDNN, tách giọng tốt hơn
        #                                       nhiều (khuyến nghị, ~80MB tải lần đầu)
        emb_name = embedding_model or os.getenv("DIART_EMBEDDING", "pyannote/embedding")
        _log(f"⏳ Dang nap embedding: {emb_name} ...")
        embedding = EmbeddingModel.from_pretrained(emb_name, use_hf_token=hf_token)
        self.embedding_name = emb_name
        _log(f"✅ embedding nap xong ({emb_name})")

        cfg_kwargs = dict(
            segmentation=segmentation,
            embedding=embedding,
            step=step,
            latency=latency,
            tau_active=hp["tau_active"],
            rho_update=hp["rho_update"],
            delta_new=hp["delta_new"],
            device=self.device,
        )
        if max_speakers:
            cfg_kwargs["max_speakers"] = max_speakers
        self._config = SpeakerDiarizationConfig(**cfg_kwargs)
        self.max_speakers = max_speakers
        self._pipeline: Optional[SpeakerDiarization] = None
        self._source: Optional[SoundDeviceAudioSource] = None
        self._inference: Optional[StreamingInference] = None
        self._thread: Optional[threading.Thread] = None
        self._feeder_thread: Optional[threading.Thread] = None
        # Hàng đợi tách diart khỏi producer (mic/file). push() chỉ bỏ vào queue
        # (tức thì); feeder thread mới chạy pipeline diart (chậm) → KHÔNG block
        # mic callback / file stream / STT.
        self._feed_q: "_queue.Queue" = _queue.Queue()

        self.windows: list = []
        self.lock = threading.Lock()
        self.running = False
        self.stats = self._fresh_stats()

        # Accumulated annotation across tất cả prediction (KHÔNG replace mỗi hook).
        from pyannote.core import Annotation
        self._annotation = Annotation(uri="live")
        self._track_counter = 0

        _log("✅ Diart Diarizer san sang")

    @staticmethod
    def _fresh_stats() -> dict:
        return {
            "total_chunks": 0,
            "annotations": 0,
            "active_speakers": set(),
        }

    # ── Lifecycle ────────────────────────────────────────────────────────────
    def start(self) -> None:
        if self.running:
            return

        self._pipeline = SpeakerDiarization(self._config)
        self._source = SoundDeviceAudioSource(sample_rate=self.sample_rate)
        self._inference = StreamingInference(
            self._pipeline,
            self._source,
            do_plot=False,
            show_progress=False,
        )
        self._inference.attach_hooks(self._on_prediction)

        # Reset queue mỗi lần start
        self._feed_q = _queue.Queue()
        self.running = True

        # Inference thread: chỉ giữ subscription sống (read() block tới close()).
        def _run():
            try:
                final = self._inference()
                if final is not None:
                    self._finalize_annotation(final)
            except Exception as e:
                _log(f"[Diart] inference error: {e}")

        # Feeder thread: drain queue → source.push() → pipeline diart chạy Ở ĐÂY.
        # Đây là nơi duy nhất gọi on_next nên toàn bộ tính toán nặng nằm trong
        # thread này, không đụng tới producer.
        def _feed():
            while True:
                try:
                    audio = self._feed_q.get(timeout=0.5)
                except _queue.Empty:
                    if not self.running:
                        break
                    continue
                if audio is None:        # poison pill
                    break
                try:
                    self._source.push(audio)
                except Exception as e:
                    _log(f"[Diart] feed error: {e}")

        self._thread = threading.Thread(target=_run, name="diart-inference", daemon=True)
        self._feeder_thread = threading.Thread(target=_feed, name="diart-feeder", daemon=True)
        self._thread.start()
        self._feeder_thread.start()
        _log("🟢 Diart inference bat dau (feeder tách luồng)")

    def stop(self) -> None:
        """Báo dừng — KHÔNG block. Feeder tiếp tục drain backlog ở nền.
        Gọi wait_drain() (trong finalize worker) để đợi diart xử lý nốt."""
        if not self.running and self._thread is None:
            return
        self.running = False
        self._feed_q.put(None)   # poison pill (đứng cuối FIFO, sau toàn bộ audio)
        _log(f"🟡 Diart stop signal (còn {self._feed_q.qsize()} chunk backlog)")

    def wait_drain(self, timeout: float = 600.0, progress_cb=None) -> None:
        """Đợi diart xử lý hết backlog + finalize annotation. Gọi từ thread nền."""
        t0 = time.time()
        # 1) Đợi feeder drain hết queue
        while self._feeder_thread is not None and self._feeder_thread.is_alive():
            if progress_cb:
                progress_cb(self._feed_q.qsize())
            self._feeder_thread.join(timeout=0.5)
            if time.time() - t0 > timeout:
                _log("[Diart] wait_drain timeout (feeder)")
                break
        # 2) Đóng source → inference() trả về → _finalize_annotation chạy
        if self._source is not None:
            self._source.close()
        if self._thread is not None:
            self._thread.join(timeout=30.0)
        _log("🔴 Diart drain xong")

    def push(self, audio_i16: np.ndarray) -> None:
        """Bỏ audio vào queue (TỨC THÌ). Không block producer."""
        if not self.running:
            return
        audio_f32 = audio_i16.astype(np.float32) / 32768.0
        self._feed_q.put(audio_f32)
        self.stats["total_chunks"] += 1

    def pending_chunks(self) -> int:
        """Số chunk diart còn phải xử lý (để UI hiện độ trễ)."""
        return self._feed_q.qsize()

    def reset(self) -> None:
        from pyannote.core import Annotation
        with self.lock:
            self.windows.clear()
        self._annotation = Annotation(uri="live")
        self._track_counter = 0
        self.stats = self._fresh_stats()
        _log("🔄 Diart reset")

    # ── Diart hook ───────────────────────────────────────────────────────────
    def _on_prediction(self, result) -> None:
        """
        result: tuple (Annotation, SlidingWindowFeature).
        Mỗi prediction CHỈ chứa cửa sổ rolling-buffer hiện tại (≈latency),
        KHÔNG phải toàn bộ lịch sử. Phải TÍCH LŨY vào self._annotation,
        nếu replace thì cuối stream chỉ còn 1 speaker của chunk cuối.
        """
        try:
            annotation = result[0] if isinstance(result, tuple) else result
            for segment, _, label in annotation.itertracks(yield_label=True):
                # Track key unique để update() không ghi đè nhau giữa các step.
                self._annotation[segment, self._track_counter] = str(label)
                self._track_counter += 1
            self._rebuild_windows()
            self.stats["annotations"] += 1
        except Exception as e:
            _log(f"[Diart] hook error: {e}")

    def _rebuild_windows(self) -> None:
        """Dồn annotation tích lũy → support() (gộp segment cùng speaker liền kề)
        → list windows cho assign_speaker/finalize."""
        try:
            merged = self._annotation.support(collar=0.5)
            new_windows = []
            for segment, _, label in merged.itertracks(yield_label=True):
                norm = self._normalize_label(label)
                new_windows.append({
                    "start": round(float(segment.start), 3),
                    "end":   round(float(segment.end),   3),
                    "label": norm,
                })
                self.stats["active_speakers"].add(norm)
            new_windows.sort(key=lambda w: w["start"])
            with self.lock:
                self.windows = new_windows
        except Exception as e:
            _log(f"[Diart] rebuild error: {e}")

    def _finalize_annotation(self, final) -> None:
        """Thay annotation tích lũy bằng kết quả cuối của inference() (chuẩn nhất)."""
        try:
            from pyannote.core import Annotation
            ann = Annotation(uri="live")
            i = 0
            for segment, _, label in final.itertracks(yield_label=True):
                ann[segment, i] = str(label)
                i += 1
            self._annotation = ann
            self._track_counter = i
            self._rebuild_windows()
            n_spk = len({w["label"] for w in self.windows})
            _log(f"[Diart] finalize annotation: {len(self.windows)} windows, {n_spk} speakers")
        except Exception as e:
            _log(f"[Diart] finalize annotation error: {e}")

    @staticmethod
    def _normalize_label(label) -> str:
        """diart labels: 'speaker0', 'speaker1', ... → 'SPEAKER_00'."""
        s = str(label).lower().replace("speaker", "").strip()
        try:
            idx = int(s)
        except ValueError:
            idx = abs(hash(s)) % 100
        return f"SPEAKER_{idx:02d}"
    
    

    # ── Query API (used by 2_Live_Mode.py) ───────────────────────────────────
    def assign_speaker(self, t_start: float, t_end: float) -> str:
        best_label, best_ov = "...", 0.0
        with self.lock:
            for w in self.windows:
                ov = max(0.0, min(w["end"], t_end) - max(w["start"], t_start))
                if ov > best_ov:
                    best_ov = ov
                    best_label = w["label"]
        return best_label

    def get_windows(self) -> list:
        with self.lock:
            return list(self.windows)

    def get_stats(self) -> dict:
        return {
            "total_chunks": self.stats["total_chunks"],
            "annotations": self.stats["annotations"],
            "active_speakers": sorted(self.stats["active_speakers"]),
            "pending_count": 0,
            "handler_active": sorted(self.stats["active_speakers"]),
        }
def _postprocess_windows(
    windows: list,
    merge_gap: float = 0.3,
    min_duration: float = 0.2,
) -> list:
    """
    Làm sạch windows_snapshot sau khi diart finalize — chạy 1 lần duy nhất
    trong _finalize_worker, TRƯỚC khi gán overlap cho các lượt STT.

    Ba bước theo thứ tự:
    Bước A — DROP window quá ngắn (< min_duration giây)
    Bước B — MERGE window liên tiếp CÙNG speaker nếu gap < merge_gap
    Bước C — RESOLVE OVERLAP: Dọn dẹp chồng chéo ranh giới giữa 2 người nói khác nhau
    """
    if not windows:
        return []
        
    # ── ĐỌC THAM SỐ TỪ FILE .ENV (Có kèm giá trị mặc định dự phòng) ──
    full_overlap_short_threshold = float(os.getenv("OVERLAP_FULL_SHORT", "1.0"))
    partial_short_threshold      = float(os.getenv("OVERLAP_PARTIAL_SHORT", "2.0"))
    partial_overlap_threshold    = float(os.getenv("OVERLAP_PARTIAL_RATIO", "0.6"))

    # ── Bước A: Drop window quá ngắn ─────────────────────────────────────────
    before_drop = len(windows)
    filtered = [
        w for w in windows
        if (w["end"] - w["start"]) >= min_duration
    ]
    dropped = before_drop - len(filtered)
    if dropped:
        _log(f"[WindowPost] Drop {dropped} window < {min_duration}s "
             f"({before_drop} → {len(filtered)})")
    if not filtered:
        return []

    # Sort đảm bảo thứ tự thời gian trước khi merge
    filtered.sort(key=lambda w: w["start"])

    # ── Bước B: Merge window liên tiếp cùng speaker ──────────────────────────
    merged = [dict(filtered[0])]   # copy để không mutate input gốc
    merge_count = 0

    for cur in filtered[1:]:
        prev = merged[-1]
        gap  = cur["start"] - prev["end"]

        if cur["label"] == prev["label"] and gap <= merge_gap:
            # Kéo dài prev đến hết cur (bao gồm cả khoảng gap nhỏ)
            prev["end"] = cur["end"]
            merge_count += 1
        else:
            merged.append(dict(cur))

    if merge_count:
        _log(f"[WindowPost] Merge {merge_count} cặp window cùng speaker "
             f"(gap ≤ {merge_gap}s)  →  {len(merged)} windows")

    # ── Bước C: Giải quyết OVERLAP giữa các windows ──────────────────────────
    def _resolve_overlaps(windows_list: list) -> list:
        result = []
        i = 0
        while i < len(windows_list):
            cur = dict(windows_list[i])
            if not result:
                result.append(cur)
                i += 1
                continue
                
            prev = result[-1]
            
            # ── FIX: Chống lỗi thời gian âm (Chồng chéo >= 3 lớp) ──
            if cur["start"] < prev["start"]:
                cur["start"] = prev["start"]
                # Nếu ép start lên mà đuôi cũng hết → đoạn này bị đè mất tăm → Xóa bỏ
                if cur["end"] <= cur["start"]:
                    i += 1
                    continue
            # ───────────────────────────────────────────────────────

            overlap = prev["end"] - cur["start"]
            if overlap <= 0:
                result.append(cur)
                i += 1
                continue
                
            cur_dur = cur["end"] - cur["start"]

            # ── Phân loại: Full overlap (cur kết thúc trước prev) ────────────
            if cur["end"] <= prev["end"]:
                if cur_dur < full_overlap_short_threshold:
                    # B ngắn → bỏ qua
                    i += 1
                    continue
                else:
                    # B đủ dài → cắt prev thành 2 phần, B chèn vào giữa
                    prev_end_orig = prev["end"]
                    prev["end"] = cur["start"]   
                    result.append(cur)            
                    if prev_end_orig > cur["end"]:
                        tail = {
                            "start": cur["end"],
                            "end":   prev_end_orig,
                            "label": prev["label"],
                        }
                        result.append(tail)
                    i += 1
                    continue

            # ── Phân loại: Partial overlap (cur kéo dài qua prev.end) ────────
            else:
                if cur_dur < partial_short_threshold and overlap > partial_overlap_threshold:
                    # B ngắn + overlap lớn → ngắt lời không cần thiết → bỏ qua B
                    i += 1
                    continue
                else:
                    # Trim prev.end lùi về cur.start để vừa khít
                    prev["end"] = cur["start"]
                    result.append(cur)
                    i += 1
                    continue

        return result

    before_overlap = len(merged)
    merged = _resolve_overlaps(merged)
    overlap_removed = before_overlap - len(merged)
    if overlap_removed != 0:
        _log(f"[WindowPost] Bước C: {before_overlap} → {len(merged)} windows "
             f"(xóa/cắt {overlap_removed} overlap)")

    _log(f"[WindowPost] Kết quả cuối cùng: {len(merged)} windows  "
         f"({len({w['label'] for w in merged})} speakers)")

    return merged