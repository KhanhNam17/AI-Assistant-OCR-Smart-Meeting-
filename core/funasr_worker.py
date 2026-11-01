import sys
import os
import re

# ======================================================================
# THỦ THUẬT: Tạm thời tắt toàn bộ 'print' rác của thư viện FunASR 
# để kết quả trả về qua stdout được sạch sẽ 100%
# ======================================================================
original_stdout = sys.stdout
sys.stdout = open(os.devnull, 'w')

try:
    from funasr import AutoModel
    from asr_hallucination_filter import clean_asr_text
    
    def run_funasr(wav_path):
        # 1. Nạp mô hình
        model = AutoModel(
            model="models/Fun-ASR-MLT-Nano-2512",
            vad_model="fsmn-vad",
            punc_model="ct-punc",
            disable_update=True
        )
        
        # 2. Giải mã âm thanh
        res = model.generate(input=wav_path)
        
        full_text = ""
        if isinstance(res, list) and len(res) > 0:
            full_text = res[0].get("text", "")
            
        # 3. CHIA ĐỂ TRỊ: Tách thành từng câu ngắn để lọc ảo giác
        # Việc này giúp bộ lọc đánh giá tỷ lệ chữ chính xác hơn, không bị drop oan cả đoạn dài
        sentences = re.split(r'(?<=[.!?]) +', full_text)
        
        cleaned_sentences = []
        for sentence in sentences:
            clean_s = clean_asr_text(sentence)
            if clean_s:  # Nếu câu không bị đánh dấu là ảo giác (chuỗi rỗng)
                cleaned_sentences.append(clean_s)
                
        return " ".join(cleaned_sentences)

    if __name__ == "__main__":
        if len(sys.argv) > 1:
            wav_file = sys.argv[1]
            final_context = run_funasr(wav_file)
            
            # Khôi phục lại stdout để in kết quả cho Subprocess đọc
            sys.stdout = original_stdout
            sys.stdout.reconfigure(encoding='utf-8')
            print(final_context)
            
except Exception as e:
    sys.stdout = original_stdout
    sys.stdout.reconfigure(encoding='utf-8')
    print(f"ERROR:{str(e)}")