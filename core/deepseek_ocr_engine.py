"""
deepseek_ocr_engine.py  (v2 — persistent worker)
==================================================

SỬA 2 LỖI của bản trước:
    1. CHẬM (~60s/block): model bị load lại từ đầu mỗi lần recognize_region()
       được gọi, vì mỗi lần gọi là 1 subprocess MỚI + `AutoModel.from_pretrained`
       chạy lại từ đầu. -> Sửa: spawn 1 worker subprocess DUY NHẤT khi
       DeepSeekOCREngine được khởi tạo, load model 1 lần, giữ sống suốt cả
       run, giao tiếp qua stdin/stdout cho từng crop (không spawn lại).

    2. "0 lines" toàn bộ: `conda run -n ...` qua `shell=True` dễ thất bại
       âm thầm trên Windows nếu conda chưa init đúng trong shell con được
       spawn -> raise CalledProcessError TRƯỚC KHI đọc được JSON kết quả.
       -> Sửa: gọi thẳng `python.exe` của env đích bằng đường dẫn tuyệt đối
       (không qua `conda run`), dùng subprocess dạng list (không `shell=True`,
       tránh lỗi quoting), và bắt lỗi rõ ràng ngay khi spawn thất bại.

Cấu hình đường dẫn python.exe của env `deepseek_ocr` — SỬA LẠI cho đúng máy bạn.
"""

import os
import sys
import json
import re
import time
import tempfile
import subprocess
import threading
import logging
from pathlib import Path
from PIL import Image

_log = logging.getLogger(__name__)

# ⚠ SỬA ĐƯỜNG DẪN NÀY CHO ĐÚNG MÁY BẠN (kiểm tra bằng:
#    conda activate deepseek_ocr && python -c "import sys; print(sys.executable)")
DEEPSEEK_ENV_PYTHON = os.getenv("DEEPSEEK_OCR_PYTHON", r"C:\Users\cunam\anaconda3\envs\deepseek_ocr\python.exe")


# ══════════════════════════════════════════════════════════════════════════════
# Hậu xử lý output chế độ <|grounding|>: loại bỏ thẻ <|ref|>...<|/ref|><|det|>...<|/det|>
# nhưng GIỮ NGUYÊN phần text/markdown thật sự (chúng đứng SAU thẻ, không phải bên
# trong thẻ — thẻ chỉ là annotation toạ độ đứng trước mỗi block).
# ══════════════════════════════════════════════════════════════════════════════

_GROUNDING_TAG_RE = re.compile(
    r"<\|ref\|>.*?<\|/ref\|>\s*<\|det\|>.*?<\|/det\|>",
    flags=re.DOTALL,
)


def strip_grounding_tags(text: str) -> str:
    """Xoá mọi annotation toạ độ khỏi output grounding."""
    if not text:
        return text
        
    # 1. Xoá độc lập các thẻ <|ref|>...<|/ref|>
    import re
    text = re.sub(r"<\|ref\|>.*?<\|/ref\|>", "", text, flags=re.DOTALL)
    
    # 2. Xoá độc lập các thẻ <|det|>...<|/det|> (Quét sạch toạ độ trong bảng)
    text = re.sub(r"<\|det\|>.*?<\|/det\|>", "", text, flags=re.DOTALL)
    
    # 3. Quét các toạ độ mồ côi dạng [[x, y, w, h]] nếu model lỡ sinh thiếu thẻ bọc
    text = re.sub(r"\[\[\d+,\s*\d+,\s*\d+,\s*\d+\]\]", "", text)
    
    # 4. Dọn dẹp khoảng trắng thừa (Đặc biệt quan trọng để Markdown Table không bị vỡ)
    text = re.sub(r"\s+\|", " |", text) 
    cleaned = re.sub(r"\n[ \t]*\n[ \t]*\n+", "\n\n", text)
    
    return cleaned.strip()


def extract_grounding_blocks(text: str) -> list[dict]:
    """Nếu về sau bạn CẦN giữ lại bbox (vd để tự dựng layout thay YOLO), dùng
    hàm này thay vì strip_grounding_tags() — trả về list các block kèm toạ độ,
    không làm mất thông tin vị trí."""
    pattern = re.compile(
        r"<\|ref\|>(?P<label>.*?)<\|/ref\|>\s*<\|det\|>(?P<bbox>.*?)<\|/det\|>"
        r"(?P<content>.*?)(?=<\|ref\|>|\Z)",
        flags=re.DOTALL,
    )
    blocks = []
    for m in pattern.finditer(text):
        blocks.append({
            "label": m.group("label").strip(),
            "bbox_raw": m.group("bbox").strip(),
            "content": m.group("content").strip(),
        })
    return blocks


# ══════════════════════════════════════════════════════════════════════════════
# PHẦN 1: ENGINE WRAPPER (chạy ở môi trường chính, vd ocr_env/uc3_ocr)
# ══════════════════════════════════════════════════════════════════════════════

class DeepSeekOCREngine:
    """
    Spawn ĐÚNG 1 worker subprocess khi khởi tạo (load model 1 lần), rồi gửi
    từng ảnh crop qua stdin của worker, đọc kết quả JSON qua stdout — không
    spawn subprocess mới cho mỗi block.
    """

    def __init__(self, python_bin: str = DEEPSEEK_ENV_PYTHON, startup_timeout: float = 180.0):
        self.python_bin = python_bin
        self.script_path = Path(__file__).resolve()
        self._lock = threading.Lock()  # phòng khi pipeline gọi đa luồng
        self.proc: subprocess.Popen | None = None
        self._start_worker(startup_timeout)

    def _start_worker(self, startup_timeout: float) -> None:
        if not Path(self.python_bin).exists():
            raise FileNotFoundError(
                f"Không tìm thấy python.exe của env deepseek_ocr tại: "
                f"{self.python_bin}. Sửa DEEPSEEK_ENV_PYTHON trong "
                f"{self.script_path.name} cho đúng máy bạn."
            )

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        # Tránh mỗi lần load phải ping Hugging Face Hub kiểm tra update
        # (network round-trip không cần thiết khi model đã cache local).
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"

        _log.info(f"[DeepSeekOCREngine] Đang khởi động worker persistent: {self.python_bin}")
        self.proc = subprocess.Popen(
            [self.python_bin, "-u", str(self.script_path), "--worker-server"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",   # tránh UnicodeDecodeError (cp1252) khi đọc log tiếng Việt của worker
            errors="replace",
            bufsize=1,
            env=env,
        )

        # Chờ worker báo "READY" (sau khi load xong model) trước khi dùng.
        t0 = time.time()
        ready_line = None
        while time.time() - t0 < startup_timeout:
            line = self.proc.stdout.readline()
            if not line:
                # Worker chết sớm -> đọc stderr để biết lý do
                err = self.proc.stderr.read()
                raise RuntimeError(
                    f"Worker DeepSeek-OCR-2 thoát trước khi sẵn sàng. Stderr:\n{err}"
                )
            line = line.strip()
            if line == "READY":
                ready_line = line
                break
            else:
                # Log các dòng khởi động khác (vd tiến trình load model) để debug
                _log.info(f"[deepseek-worker] {line}")

        if ready_line != "READY":
            raise TimeoutError(
                f"Worker DeepSeek-OCR-2 không sẵn sàng sau {startup_timeout}s."
            )
        _log.info(f"[DeepSeekOCREngine] Worker sẵn sàng ({time.time() - t0:.1f}s load model).")

    def _wait_for_response_with_heartbeat(self, heartbeat_sec=20.0):
        import concurrent.futures
        
        def _read_loop():
            for _ in range(50):
                response_line = self.proc.stdout.readline()
                if not response_line:
                    err = self.proc.stderr.read()
                    raise RuntimeError(f"Worker không trả kết quả (có thể đã crash). Stderr:\n{err}")
                response_line = response_line.strip()
                if not response_line:
                    continue
                try:
                    return json.loads(response_line)
                except json.JSONDecodeError:
                    _log.warning(f"[DeepSeekOCREngine] Bỏ qua dòng không phải JSON: {response_line!r}")
                    continue
            raise RuntimeError("Không nhận được dòng JSON hợp lệ nào từ worker sau nhiều lần thử.")

        t0 = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_read_loop)
            while True:
                try:
                    return future.result(timeout=heartbeat_sec)
                except concurrent.futures.TimeoutError:
                    elapsed = time.time() - t0
                    msg = f"  ⏳ DeepSeek-OCR-2 vẫn đang xử lý... đã {elapsed:.0f}s trôi qua"
                    # Print thẳng ra để Streamlit UI bắt được qua đường ống stdout
                    print(msg, flush=True)
                    _log.info(msg)

    def recognize_region(self, img: Image.Image, line_separator: str = "\n") -> tuple[str, float]:
        if self.proc is None or self.proc.poll() is not None:
            _log.error("[DeepSeekOCREngine] Worker đã chết — thử khởi động lại...")
            self._start_worker(startup_timeout=180.0)

        fd, temp_img_path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        img.save(temp_img_path)

        text, conf = "", 0.0
        with self._lock:  # 1 request/lần qua cùng 1 worker
            try:
                request = json.dumps({"image": temp_img_path}, ensure_ascii=False)
                self.proc.stdin.write(request + "\n")
                self.proc.stdin.flush()

                # Thay thế vòng lặp cũ bằng hàm nhịp tim
                data = self._wait_for_response_with_heartbeat(heartbeat_sec=20.0)
                
                if "error" in data:
                    _log.error(f"[DeepSeekOCREngine] Worker báo lỗi: {data['error']}")
                else:
                    text = (data.get("text") or "").strip()
                    conf = data.get("confidence", 0.99 if text else 0.0)

            except Exception as e:
                _log.error(f"[DeepSeekOCREngine] Lỗi khi giao tiếp với worker: {e}")
            finally:
                if os.path.exists(temp_img_path):
                    os.remove(temp_img_path)

        if text and line_separator != "\n":
            text = text.replace("\n", line_separator)
        return text, conf

    def ocr_full_page_markdown(self, img: Image.Image) -> tuple[str, str]:
        """Chế độ 'end-to-end': đưa CẢ ẢNH TRANG (không phải crop) vào model
        với prompt <|grounding|>, để model tự làm layout+OCR+dựng bảng trong
        1 lần gọi. Dùng chung 1 worker/lock với recognize_region() — chỉ
        khác field "mode" gửi qua giao thức.

        Trả về TUPLE (raw_grounding, clean_markdown):
          * raw_grounding  — văn bản GỐC, còn nguyên thẻ
            <|ref|>...<|/ref|><|det|>[[x1,y1,x2,y2]]<|/det|>. Đây là input
            cho bước "Hoàn thiện văn bản" (dựng lại DOCX/PDF theo đúng bbox
            từng vùng thay vì tin cấu trúc markdown model tự đoán — xem
            grounding_docx_builder.py).
          * clean_markdown — bản ĐÃ strip sạch thẻ, dùng cho xem nhanh /
            tương thích ngược với luồng cũ (markdown_to_docx tự động).

        Cả hai cùng lấy từ 1 lần gọi worker — không tốn thêm chi phí OCR."""
        if self.proc is None or self.proc.poll() is not None:
            _log.error("[DeepSeekOCREngine] Worker đã chết — thử khởi động lại...")
            self._start_worker(startup_timeout=180.0)

        fd, temp_img_path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        img.save(temp_img_path)

        raw_grounding  = ""
        clean_markdown = ""
        with self._lock:
            try:
                request = json.dumps(
                    {"image": temp_img_path, "mode": "grounding"}, ensure_ascii=False
                )
                self.proc.stdin.write(request + "\n")
                self.proc.stdin.flush()

                # Thay thế vòng lặp cũ bằng hàm nhịp tim
                data = self._wait_for_response_with_heartbeat(heartbeat_sec=20.0)

                if "error" in data:
                    _log.error(f"[DeepSeekOCREngine] Worker báo lỗi (grounding): {data['error']}")
                else:
                    clean_markdown = (data.get("text") or "").strip()
                    # Fallback: worker cũ (chưa cập nhật) không có field "raw"
                    # — dùng clean_markdown làm raw để không crash caller.
                    raw_grounding  = (data.get("raw") or clean_markdown).strip()

            except Exception as e:
                _log.error(f"[DeepSeekOCREngine] Lỗi khi giao tiếp với worker (grounding): {e}")
            finally:
                if os.path.exists(temp_img_path):
                    os.remove(temp_img_path)

        return raw_grounding, clean_markdown

    def close(self) -> None:
        """Gọi khi pipeline kết thúc để tắt worker gọn gàng."""
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.stdin.write(json.dumps({"cmd": "exit"}) + "\n")
                self.proc.stdin.flush()
                self.proc.wait(timeout=10)
            except Exception:
                self.proc.terminate()

    def __del__(self):
        self.close()


# ══════════════════════════════════════════════════════════════════════════════
# PHẦN 2: WORKER SERVER (chạy trong môi trường deepseek_ocr, SỐNG SUỐT cả run)
# ══════════════════════════════════════════════════════════════════════════════

def run_worker_server():
    """
    Load model 1 LẦN DUY NHẤT, sau đó vòng lặp: đọc 1 dòng JSON từ stdin
    ({"image": path}), OCR, in 1 dòng JSON kết quả ra stdout, lặp lại.
    In "READY" ra stdout ngay sau khi model load xong để parent biết có thể
    bắt đầu gửi request.
    """
    import torch
    import torch.nn as nn

    # --- Hack CPU (giữ nguyên) ---
    original_tensor_to = torch.Tensor.to
    original_module_to = nn.Module.to

    def cpu_only_tensor_to(self, *args, **kwargs):
        if args and (isinstance(args[0], str) and 'cuda' in args[0] or isinstance(args[0], torch.device) and args[0].type == 'cuda'):
            args = ('cpu',) + args[1:]
        if kwargs.get('device') and (isinstance(kwargs['device'], str) and 'cuda' in kwargs['device'] or isinstance(kwargs['device'], torch.device) and kwargs['device'].type == 'cuda'):
            kwargs['device'] = 'cpu'
        return original_tensor_to(self, *args, **kwargs)

    def cpu_only_module_to(self, *args, **kwargs):
        if args and (isinstance(args[0], str) and 'cuda' in args[0] or isinstance(args[0], torch.device) and args[0].type == 'cuda'):
            args = ('cpu',) + args[1:]
        if kwargs.get('device') and (isinstance(kwargs['device'], str) and 'cuda' in kwargs['device'] or isinstance(kwargs['device'], torch.device) and kwargs['device'].type == 'cuda'):
            kwargs['device'] = 'cpu'
        return original_module_to(self, *args, **kwargs)

    torch.Tensor.to = cpu_only_tensor_to
    nn.Module.to = cpu_only_module_to
    torch.Tensor.cuda = lambda self, *args, **kwargs: original_tensor_to(self, 'cpu')
    nn.Module.cuda = lambda self, *args, **kwargs: original_module_to(self, 'cpu')

    original_autocast = torch.autocast
    class cpu_autocast:
        def __init__(self, device_type, *args, **kwargs):
            self.ctx = original_autocast('cpu', *args, **kwargs)
        def __enter__(self): return self.ctx.__enter__()
        def __exit__(self, *args): return self.ctx.__exit__(*args)
    torch.autocast = cpu_autocast
    # -------------------------------

    # ── CÔ LẬP stdout thật ──────────────────────────────────────────────────
    # model.infer()/from_pretrained() có thể tự print() log/progress ra
    # stdout (transformers, tqdm, hoặc chính code custom của DeepSeek-OCR-2).
    # Những dòng đó sẽ LẪN vào giao thức "1 dòng = 1 JSON response" mà parent
    # đang readline() theo, gây lỗi `json.loads` "Expecting value...".
    #
    # Cách xử lý: dup lại file descriptor stdout gốc sang 1 kênh riêng
    # (real_stdout) CHỈ dùng để in JSON protocol, rồi redirect fd stdout
    # mặc định (fd 1) sang devnull — mọi print()/log của thư viện (kể cả
    # print ở tầng C, không chỉ Python-level) từ giờ đều rơi vào devnull,
    # không còn lẫn vào stream mà parent đang đọc.
    real_stdout_fd = os.dup(sys.stdout.fileno())
    real_stdout = os.fdopen(real_stdout_fd, "w", encoding="utf-8", errors="replace", buffering=1)

    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull_fd, sys.stdout.fileno())
    os.close(devnull_fd)

    def protocol_print(payload: str) -> None:
        """In đúng 1 dòng ra kênh stdout THẬT (không bị mọi print() khác chen vào)."""
        real_stdout.write(payload + "\n")
        real_stdout.flush()
    # ──────────────────────────────────────────────────────────────────────

    from transformers import AutoModel, AutoTokenizer

    model_name = 'deepseek-ai/DeepSeek-OCR-2'

    # 2 prompt phục vụ 2 mục đích khác nhau — chọn theo field "mode" trong
    # từng request, KHÔNG hardcode 1 prompt duy nhất cho cả worker nữa:
    #   - "free_ocr" (mặc định): dùng cho crop nhỏ đã có bbox sẵn từ YOLO —
    #     luồng hiện tại của document_pipeline.py, giữ nguyên không đổi.
    #   - "grounding": dùng cho ẢNH CẢ TRANG — chế độ "end-to-end", model tự
    #     detect layout + OCR + dựng bảng trong 1 lần gọi. Output có kèm thẻ
    #     <|ref|>/<|det|> nên cần strip_grounding_tags() trước khi trả về.
    PROMPTS = {
        "free_ocr":  "<image>\nFree OCR.",
        "grounding": "<image>\n<|grounding|>Convert the document to markdown.",
    }

    # In qua stderr để bạn vẫn thấy tiến trình load khi debug thủ công
    # (stderr không bị redirect, và parent không đọc stderr theo giao thức
    # nghiêm ngặt như stdout — chỉ dùng để log khi lỗi).
    print("[worker] Đang tải tokenizer + model (chỉ 1 lần)...", file=sys.stderr, flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name, trust_remote_code=True, use_safetensors=True,
    ).eval().to('cpu')

    protocol_print("READY")  # <-- tín hiệu cho parent, đi qua kênh sạch

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            protocol_print(json.dumps({"error": f"invalid request json: {line}"}))
            continue

        if request.get("cmd") == "exit":
            break

        image_path = request.get("image")
        mode = request.get("mode", "free_ocr")
        prompt = PROMPTS.get(mode)
        if prompt is None:
            protocol_print(json.dumps({
                "error": f"mode không hợp lệ: {mode!r}. Chỉ hỗ trợ: {list(PROMPTS)}"
            }))
            continue

        # crop_mode: False cho free_ocr (crop nhỏ, không cần chia tile động —
        # đã thống nhất trước đó), True cho grounding (ảnh cả trang, cần
        # Gundam dynamic tiling để giữ độ phân giải chi tiết).
        crop_mode = (mode == "grounding")

        try:
            res = model.infer(
                tokenizer,
                prompt=prompt,
                image_file=image_path,
                output_path=os.path.dirname(image_path) or ".",
                base_size=1024,
                image_size=768,
                crop_mode=crop_mode,
                save_results=False,
                eval_mode=True,
            )
            raw_text = (res or "").strip()
            # Chỉ strip thẻ grounding khi thật sự ở mode grounding — ở mode
            # free_ocr, model không sinh thẻ này nên strip cũng vô hại nhưng
            # tách rõ ràng theo mode để dễ debug hơn nếu cần xem raw output.
            final_text = strip_grounding_tags(raw_text) if mode == "grounding" else raw_text
            # "raw" LUÔN giữ nguyên văn bản gốc (kèm <|ref|>/<|det|> nếu có) —
            # đây là nguồn dữ liệu cho bước "Hoàn thiện văn bản" (stage 2:
            # dựng lại DOCX/PDF theo đúng bbox thay vì tin markdown model tự
            # sinh). Không tốn thêm lần gọi model nào — chỉ là không vứt bỏ
            # dữ liệu model đã trả về.
            result = {"text": final_text, "raw": raw_text, "confidence": 0.99}
        except Exception as e:
            result = {"error": str(e)}

        protocol_print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-server", action="store_true",
                         help="Chạy chế độ worker persistent (load model 1 lần, đọc request qua stdin)")
    args = parser.parse_args()

    if args.worker_server:
        run_worker_server()