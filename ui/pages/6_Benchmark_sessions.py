"""
ui/pages/6_Benchmark_sessions.py
=================================
Trang đánh giá chất lượng (Benchmark) — gộp 2 hệ thống vào chung 1 trang, chia theo tab:
  • Tab 1: Đánh giá Cuộc họp (STT/Diarization) — WER/CER/DER, so khớp Ground Truth transcript
  • Tab 2: Đánh giá Tài liệu (OCR)            — CER/WER/Reading Order/TEDS bảng biểu

Cấu trúc theo đúng mô hình đã dùng ở ui/pages/1_Dashboard.py.
"""

import sys
import os
import json
import base64
import tempfile
import traceback
import warnings
from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore", message="'uem' was approximated")

# ══════════════════════════════════════════════════════════════════════════════
# 1. SETUP ĐƯỜNG DẪN & THEME CHUNG (giống Dashboard)
# ══════════════════════════════════════════════════════════════════════════════
# Bỏ bớt 1 đuôi .parent để REPO_ROOT trỏ đúng vào thư mục STT-offline-uc2
def _find_repo_root(start: Path) -> Path:
    cur = start
    for _ in range(6):
        if (cur / "core").is_dir() and (cur / "output").exists():
            return cur
        if cur.parent == cur: break
        cur = cur.parent
    return start.parent.parent

_THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = _find_repo_root(_THIS_DIR)
UI_ROOT = _THIS_DIR.parent

for p in (REPO_ROOT, UI_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Thư mục chứa run_evaluate.py (lõi Benchmark OCR)
OCR_TOOLS_DIR = REPO_ROOT / "benchmark" / "tools"
if str(OCR_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(OCR_TOOLS_DIR))

from theme import apply_page_config, inject_base_css, render_sidebar, render_benchmark_groups, icon, render_workspace_jump

apply_page_config("Đánh giá chất lượng — Benchmark", UI_ROOT)
inject_base_css()
render_sidebar(active="benchmark")

# ── Import lõi logic Benchmark Cuộc họp (STT/Diarization) ──────────────────
try:
    from benchmark import metrics, report
    from benchmark.parsers import parse_transcript, segments_to_text
    from benchmark.thresholds import evaluate_all, PASS, FAIL, NA, MANUAL
    _STT_BENCHMARK_OK = True
    _STT_IMPORT_ERR = ""
except ImportError as _e:
    _STT_BENCHMARK_OK = False
    _STT_IMPORT_ERR = str(_e)

# ── Import lõi logic Benchmark OCR ──────────────────────────────────────────
try:
    from run_evaluate import evaluate_document
    _OCR_BENCHMARK_OK = True
    _OCR_IMPORT_ERR = ""
except ImportError as _e:
    _OCR_BENCHMARK_OK = False
    _OCR_IMPORT_ERR = str(_e)

OUTPUT_ROOT = REPO_ROOT / "output"               # dữ liệu OCR
SESSIONS_DIR = os.path.join("data", "sessions")  # dữ liệu Smart Meeting

# ══════════════════════════════════════════════════════════════════════════════
# 2. CSS RIÊNG CHO TAB CUỘC HỌP (metric-card/score-card kiểu cũ)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    .stButton > button[kind="primary"] {
        background: #0067B1 !important; border: none !important;
        font-weight: 600 !important; color: white !important;
    }
    .metric-card {
        background: var(--secondary-background-color);
        padding: 20px; border-radius: 12px;
        border: 1px solid rgba(128,128,128,0.2);
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        height: 100%;
    }
    .metric-val { font-size: 32px; font-weight: 700; color: #0067B1; margin-bottom: 4px; }
    .metric-lbl { font-size: 12px; letter-spacing: 1px; opacity: 0.8; text-transform: uppercase; font-weight: 600; }
    .metric-sub { font-size: 11px; opacity: 0.6; margin-top: 8px; }
    .score-card {
        background: linear-gradient(135deg, #0B2540 0%, #0067B1 100%);
        color: white; border: 1px solid #00A0DC;
        box-shadow: 0 8px 20px rgba(0,103,177,0.25);
    }
    .score-card .metric-lbl { color: rgba(255,255,255,0.75); }
    .score-card .metric-sub { color: rgba(255,255,255,0.6); }
    .score-card .metric-val { color: #4ADE80; font-size: 48px; }
    .score-card-bad .metric-val { color: #F87171; font-size: 48px; }
    .score-card-avg .metric-val { color: #FBBF24; font-size: 48px; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# 3. TIÊU ĐỀ CHÍNH
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="font-size:11px;letter-spacing:3px;text-transform:uppercase;
            color:var(--text-color);opacity:0.7;padding-bottom:12px;
            border-bottom:1px solid rgba(128,128,128,0.2);margin-bottom:16px;">
    📊  Đánh giá chất lượng toàn diện — Benchmark
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# 4. KHỞI TẠO TABS (giống Dashboard)
# ══════════════════════════════════════════════════════════════════════════════
tab_meeting, tab_ocr = st.tabs(["🎙️ Cuộc họp thông minh (STT/Diarization)", "📄 OCR Tài liệu"])

# ==============================================================================
# TAB 1: BENCHMARK CUỘC HỌP (SMART MEETING)
# ==============================================================================
with tab_meeting:
    render_workspace_jump("meeting")
    if not _STT_BENCHMARK_OK:
        st.error(f"Không thể import gói benchmark: {_STT_IMPORT_ERR}. Vui lòng kiểm tra lại môi trường.")
    else:
        os.makedirs(SESSIONS_DIR, exist_ok=True)
        stt_saved_sessions = [d for d in os.listdir(SESSIONS_DIR) if os.path.isdir(os.path.join(SESSIONS_DIR, d))]

        if not stt_saved_sessions:
            st.info("💡 Chưa có phiên cuộc họp nào trong lịch sử (data/sessions) để chạy benchmark.")
        else:
            col_left, col_right = st.columns([1, 1.5], gap="large")

            with col_left:
                st.markdown("#### Phiên cần đánh giá (Hypothesis)")
                selected_session = st.selectbox("Chọn phiên từ lịch sử", options=stt_saved_sessions, key="stt_bench_select")
                sdir = os.path.join(SESSIONS_DIR, selected_session)

                hyp_path = os.path.join(sdir, "transcript.json")
                timings_path = os.path.join(sdir, "timings.json")
                wav_path = os.path.join(sdir, "audio.wav")

                hyp_ok = os.path.isfile(hyp_path)
                timings_ok = os.path.isfile(timings_path)
                wav_ok = os.path.isfile(wav_path)

                check = '<span style="color:#16A34A">&#10003;</span>'
                cross = '<span style="color:#c44">&#10007;</span>'

                st.markdown(
                    f"<small>"
                    f"{check if hyp_ok else cross} transcript.json &nbsp; "
                    f"{check if timings_ok else cross} timings.json &nbsp; "
                    f"{check if wav_ok else cross} audio.wav"
                    f"</small>",
                    unsafe_allow_html=True,
                )

                if not hyp_ok:
                    st.error("Phiên này chưa có transcript.json. Không thể đánh giá.")

                st.markdown("#### Điều kiện âm thanh")
                audio_condition = st.radio(
                    "Môi trường (ảnh hưởng tiêu chí 1.1 / 1.2)",
                    ["Phòng yên tĩnh (1.1, < 22%)", "Có nhiễu / phòng vang (1.2, < 38%)"],
                    label_visibility="collapsed",
                    key="stt_bench_condition",
                )
                st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
                st.markdown("#### Cấu hình kiểm thử (Preset)")
                tested_preset = st.selectbox(
                    "Preset đánh giá:",
                    options=["Không rõ", "ami", "voxconverse", "dihard", "default"],
                    index=0,
                    label_visibility="collapsed",
                    help="Lọc các tiêu chí theo hệ dữ liệu tương ứng.",
                    key="stt_bench_preset",
                )
                st.session_state.tested_preset = tested_preset

            with col_right:
                st.markdown("#### Transcript tham chiếu (Ground Truth)")
                ref_mode = st.radio("Nguồn", ["Tải file lên", "Nhập trực tiếp"], horizontal=True,
                                     label_visibility="collapsed", key="stt_bench_ref_mode")

                ref_content = None
                if ref_mode == "Tải file lên":
                    uploaded_gt = st.file_uploader("Chọn file Ground Truth .json hoặc .txt", type=["json", "txt"], key="stt_bench_gt_upload")
                    if uploaded_gt:
                        ref_content = uploaded_gt.read().decode("utf-8")
                else:
                    ref_content = st.text_area("Dán nội dung transcript chuẩn vào đây", height=120, key="stt_bench_ref_text").strip() or None

                with st.expander("Tuỳ chọn nâng cao — Từ khoá & Tóm tắt (Option từ app.py)"):
                    keywords_raw = st.text_area("Từ khoá cần kiểm tra (mỗi từ 1 dòng)", placeholder="ngân sách\nKPI\nđối tác", key="stt_bench_keywords")
                    ref_summary = st.text_area("Tóm tắt chuẩn (để tính Rouge-L)", height=80, key="stt_bench_ref_summary")

            run_disabled = not hyp_ok or not ref_content
            st.divider()
            run_clicked = st.button("🚀 Chạy Benchmark", type="primary", disabled=run_disabled, key="stt_bench_run")

            if run_clicked:
                with st.spinner("Đang tính toán các chỉ số chuyên sâu..."):
                    with open(hyp_path, "r", encoding="utf-8") as f:
                        hyp_content = f.read()

                    timings = {}
                    if timings_ok:
                        with open(timings_path, "r", encoding="utf-8") as f:
                            timings = json.load(f)

                    audio_dur = None
                    if wav_ok:
                        with open(wav_path, "rb") as f:
                            audio_dur = metrics.get_audio_duration_seconds(f.read())

                    try:
                        ref_segs = parse_transcript(ref_content)
                        hyp_segs = parse_transcript(hyp_content)
                        parse_ok = True
                    except ValueError as exc:
                        st.error(f"Lỗi phân tích transcript: {exc}")
                        parse_ok = False

                    if parse_ok:
                        ref_text = segments_to_text(ref_segs)
                        hyp_text = segments_to_text(hyp_segs)

                        results = {}
                        results["wer"] = metrics.compute_wer(ref_text, hyp_text)
                        results["cer"] = metrics.compute_cer(ref_text, hyp_text)

                        results.update(metrics.compute_diarization_metrics(ref_segs, hyp_segs))
                        results.update(metrics.compute_turn_metrics(ref_segs, hyp_segs))

                        stt_t = timings.get("stt", 0.0)
                        diar_t = timings.get("diarization", 0.0)
                        fin_t = timings.get("finalize", 0.0)

                        if stt_t > 0 and audio_dur: results["rtf_stt"] = metrics.compute_rtf(stt_t, audio_dur)
                        if diar_t > 0 and audio_dur: results["rtf_diarization"] = metrics.compute_rtf(diar_t, audio_dur)
                        if fin_t > 0 and audio_dur: results["finalize_ratio"] = metrics.compute_rtf(fin_t, audio_dur)

                        e2e_stages = [t for t in [stt_t, diar_t, fin_t] if t > 0]
                        if e2e_stages and audio_dur: results["rtf_e2e"] = metrics.compute_rtf(sum(e2e_stages), audio_dur)

                        keywords = [kw.strip() for kw in keywords_raw.splitlines() if kw.strip()] if keywords_raw else []
                        if keywords:
                            kw_res = metrics.compute_keyword_recognition(ref_text, hyp_text, keywords)
                            results["keyword_recognition"] = kw_res.get("keyword_recognition")

                        if ref_summary and ref_summary.strip():
                            summary_path = os.path.join(sdir, "summary.txt")
                            if os.path.isfile(summary_path):
                                with open(summary_path, "r", encoding="utf-8") as f:
                                    hyp_summary = f.read().strip()
                                if hyp_summary:
                                    results["rouge_l"] = metrics.compute_rouge_l(ref_summary.strip(), hyp_summary)

                        rows = evaluate_all(results)

                        if audio_condition.startswith("Phòng yên tĩnh"):
                            rows = [r for r in rows if r["ID"] != "1.2"]
                        else:
                            rows = [r for r in rows if r["ID"] != "1.1"]

                        n_pass = sum(1 for r in rows if r["Kết quả"] == PASS)
                        n_fail = sum(1 for r in rows if r["Kết quả"] == FAIL)
                        n_na = sum(1 for r in rows if r["Kết quả"] in [NA, MANUAL])

                        meta_info = {
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "session_id": selected_session,
                            "tested_preset": st.session_state.get("tested_preset", "Không rõ"),
                            "audio_duration": f"{audio_dur:.1f}s" if audio_dur else "N/A",
                            "audio_condition": audio_condition,
                            "n_hyp_segments": len(hyp_segs),
                            "n_ref_segments": len(ref_segs),
                            "n_keywords": len(keywords),
                            "has_summary": "rouge_l" in results,
                        }

                        json_report = report.build_json_report(meta_info, results, rows, n_pass, n_fail, n_na)
                        md_report = report.build_markdown_report(meta_info, results, rows, n_pass, n_fail, n_na)

                        bench_out = os.path.join(sdir, "benchmark.json")
                        with open(bench_out, "w", encoding="utf-8") as f:
                            f.write(json_report)

                        st.session_state["benchmark_run"] = {
                            "sid": selected_session,
                            "results": results,
                            "rows": rows,
                            "n_pass": n_pass, "n_fail": n_fail, "n_na": n_na,
                            "json_report": json_report,
                            "md_report": md_report,
                            "ref_segs": [s.__dict__ for s in ref_segs],
                            "hyp_segs": [s.__dict__ for s in hyp_segs],
                        }
                        st.success(f"Đã lưu kết quả thành công vào phiên {selected_session}")

            run_data = st.session_state.get("benchmark_run")

            if run_data and run_data.get("sid") == selected_session:
                results = run_data["results"]
                rows = run_data["rows"]

                wer = (results.get("wer", 0.0) or 0.0) * 100
                cer = (results.get("cer", 0.0) or 0.0) * 100
                der = (results.get("der", 0.0) or 0.0) * 100

                stt_score = max(0.0, 100.0 - wer)
                diar_score = max(0.0, 100.0 - der)
                overall_score = (stt_score * 0.7) + (diar_score * 0.3)

                score_class = "score-card"
                if overall_score < 60: score_class += " score-card-bad"
                elif overall_score < 80: score_class += " score-card-avg"
                grade = "Xuất sắc (A)" if overall_score >= 90 else "Tốt (B)" if overall_score >= 80 else "Đạt (C)" if overall_score >= 60 else "Kém (D)"

                st.markdown("<hr style='border-color:rgba(128,128,128,0.2);margin:24px 0;'>", unsafe_allow_html=True)

                st.markdown(f'''
                <div class="metric-card {score_class}" style="margin-bottom: 16px;">
                    <div class="metric-lbl" style="color:white;">TỔNG ĐIỂM HỆ THỐNG (OVERALL SCORE)</div>
                    <div class="metric-val">{overall_score:.1f} / 100</div>
                    <div class="metric-sub" style="color:white; font-size:14px;">Xếp loại dự kiến: <b>{grade}</b></div>
                </div>
                ''', unsafe_allow_html=True)

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown(f'<div class="metric-card"><div class="metric-val">{wer:.1f}%</div><div class="metric-lbl">WER (Lỗi từ)</div></div>', unsafe_allow_html=True)
                with c2:
                    st.markdown(f'<div class="metric-card"><div class="metric-val">{cer:.1f}%</div><div class="metric-lbl">CER (Lỗi ký tự)</div></div>', unsafe_allow_html=True)
                with c3:
                    st.markdown(f'<div class="metric-card"><div class="metric-val">{der:.1f}%</div><div class="metric-lbl">DER (Lỗi phân tách)</div></div>', unsafe_allow_html=True)

                st.markdown("<div style='margin-top:32px;'></div>", unsafe_allow_html=True)
                st.markdown('<div style="font-size:10px;letter-spacing:2px;text-transform:uppercase;color:var(--text-color);opacity:0.7;margin-bottom:14px;">Đánh giá theo bộ tiêu chuẩn Benchmark (KPIs)</div>', unsafe_allow_html=True)

                sm1, sm2, sm3 = st.columns(3)
                sm1.metric("Đạt (Pass)", run_data["n_pass"])
                sm2.metric("Không đạt (Fail)", run_data["n_fail"])
                sm3.metric("N/A / Thủ công", run_data["n_na"])

                df_all = pd.DataFrame(rows)
                cols_show = ["ID", "Tiêu chí", "Ngưỡng", "Giá trị đo", "Kết quả", "Ưu tiên"]

                def _style_row(row: pd.Series) -> list[str]:
                    val = row.get("Kết quả", "")
                    if val == PASS: return ["background-color: rgba(0,160,0,0.12)"] * len(row)
                    elif val == FAIL: return ["background-color: rgba(200,0,0,0.12)"] * len(row)
                    else: return ["color: #888888"] * len(row)

                for group_name in df_all["Nhóm"].unique():
                    st.markdown(f"**{group_name.strip()}**")
                    gdf = df_all[df_all["Nhóm"] == group_name][cols_show].copy()
                    styled = gdf.style.apply(_style_row, axis=1)
                    st.dataframe(styled, width="stretch", hide_index=True)
                    st.markdown("<div style='margin:4px 0;'></div>", unsafe_allow_html=True)

                st.markdown("<hr style='border-color:rgba(128,128,128,0.2);margin:32px 0 24px;'>", unsafe_allow_html=True)
                with st.expander("🔍 Đối chiếu chi tiết (Ground Truth vs Hypothesis)", expanded=False):
                    view_c1, view_c2 = st.columns(2)
                    ref_turns = run_data["ref_segs"]
                    hyp_turns = run_data["hyp_segs"]

                    with view_c1:
                        st.markdown("**📜 Văn bản chuẩn (Ground Truth)**")
                        for r in ref_turns[:20]:
                            st.markdown(f"**[{r.get('start', 0):.1f}s - {r.get('end', 0):.1f}s] {r.get('speaker', '')}:** {r.get('text', '')}")
                        if len(ref_turns) > 20: st.caption(f"... và {len(ref_turns)-20} lượt nói khác")

                    with view_c2:
                        st.markdown("**🎙️ Văn bản hệ thống dịch (Hypothesis)**")
                        for h in hyp_turns[:20]:
                            st.markdown(f"**[{h.get('start', 0):.1f}s - {h.get('end', 0):.1f}s] {h.get('speaker', '')}:** {h.get('text', '')}")
                        if len(hyp_turns) > 20: st.caption(f"... và {len(hyp_turns)-20} lượt nói khác")

                st.divider()
                st.markdown("#### Tải báo cáo")
                dl1, dl2, dl3 = st.columns(3)
                with dl1:
                    st.download_button("Tải JSON (.json)", data=run_data["json_report"], file_name=f"benchmark_{selected_session}.json", mime="application/json", width="stretch", key="stt_dl_json")
                with dl2:
                    st.download_button("Tải Markdown (.md)", data=run_data["md_report"], file_name=f"benchmark_{selected_session}.md", mime="text/markdown", width="stretch", key="stt_dl_md")
                with dl3:
                    st.download_button("Tải bảng CSV (.csv)", data=df_all.to_csv(index=False), file_name=f"benchmark_{selected_session}.csv", mime="text/csv", width="stretch", key="stt_dl_csv")

            elif os.path.isfile(os.path.join(SESSIONS_DIR, selected_session, "benchmark.json")):
                st.info("Phiên này đã có dữ liệu benchmark lưu trước đó. Vui lòng **Chạy lại Benchmark** để hiển thị giao diện phân tích chi tiết.")

# ==============================================================================
# TAB 2: BENCHMARK TÀI LIỆU (OCR)
# ==============================================================================
with tab_ocr:
    render_workspace_jump("ocr")
    if not _OCR_BENCHMARK_OK:
        st.error(f"Không thể import module benchmark: {_OCR_IMPORT_ERR}. Kiểm tra lại thư mục `benchmark/tools`.")
    elif not OUTPUT_ROOT.exists():
        st.info("Chưa có dữ liệu trong thư mục `output/`.")
    else:
        ocr_saved_sessions = [d for d in OUTPUT_ROOT.iterdir() if d.is_dir() and list(d.glob("*_ocr.md"))]

        if not ocr_saved_sessions:
            st.info("Chưa có phiên xử lý nào hoàn thành (không tìm thấy file `*_ocr.md`).")
        else:
            def display_pdf(file_path: Path, height: int = 700) -> None:
                b64 = base64.b64encode(file_path.read_bytes()).decode("utf-8")
                st.markdown(
                    f'<iframe src="data:application/pdf;base64,{b64}" '
                    f'width="100%" height="{height}" type="application/pdf" '
                    f'style="border:1px solid var(--border-color); border-radius:var(--vnpt-radius-md);">'
                    f'</iframe>',
                    unsafe_allow_html=True,
                )

            def _fix_double_encoded(content: str) -> tuple[str, bool]:
                if not isinstance(content, str) or not content:
                    return content, False
                try:
                    decoded = json.loads('"' + content + '"')
                except Exception:
                    return content, False
                if isinstance(decoded, str) and decoded and decoded != content:
                    return decoded, True
                return content, False

            def sanitize_ground_truth(gt_path: Path) -> tuple[Path, list[str]]:
                try:
                    with open(gt_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                except Exception:
                    return gt_path, []

                fixed_notes: list[str] = []
                changed = False

                for p_idx, page in enumerate(data.get("pages", [])):
                    for b_idx, block in enumerate(page.get("blocks", [])):
                        if block.get("content_kind") != "table_html":
                            continue
                        original = block.get("content", "")
                        fixed, did_fix = _fix_double_encoded(original)
                        if did_fix:
                            block["content"] = fixed
                            changed = True
                            fixed_notes.append(
                                f"Trang {p_idx + 1}, block #{b_idx} (order={block.get('order', '?')})"
                            )

                if not changed:
                    return gt_path, []

                tmp_dir = Path(tempfile.mkdtemp(prefix="gt_sanitized_"))
                tmp_path = tmp_dir / gt_path.name
                with open(tmp_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)

                return tmp_path, fixed_notes

            col_left, col_right = st.columns([1, 1.5], gap="large")

            with col_left:
                with st.container(border=True):
                    st.subheader("Lựa chọn phiên Benchmark")
                    session_names = [d.name for d in ocr_saved_sessions]
                    selected_session_name = st.selectbox(
                        "Chọn phiên tài liệu từ thư mục output:", options=session_names,
                        label_visibility="collapsed", key="ocr_bench_select",
                    )

                    run_dir = OUTPUT_ROOT / selected_session_name

                    ocr_md_files = list(run_dir.glob("*_ocr.md"))
                    ocr_md_path = ocr_md_files[0] if ocr_md_files else None

                    gt_path_default = run_dir / "ground_truth" / "_master.json"

                    st.write("")
                    st.markdown("**Nguồn Ground Truth**")
                    ref_mode = st.radio(
                        "Chọn nguồn dữ liệu chuẩn:",
                        ["Tự động từ GT Creator", "Tải file lên (Ghi đè)"],
                        horizontal=True, label_visibility="collapsed", key="ocr_bench_ref_mode",
                    )

                    uploaded_gt_path = None
                    if ref_mode == "Tải file lên (Ghi đè)":
                        uploaded_gt = st.file_uploader("Chọn file Ground Truth .json", type=["json"], key="ocr_bench_gt_upload")
                        if uploaded_gt:
                            temp_dir = tempfile.mkdtemp()
                            uploaded_gt_path = Path(temp_dir) / uploaded_gt.name
                            with open(uploaded_gt_path, "wb") as f:
                                f.write(uploaded_gt.getbuffer())

                    final_gt_path = uploaded_gt_path if uploaded_gt_path else gt_path_default

                    gt_ok = final_gt_path.exists() if final_gt_path else False
                    ocr_ok = ocr_md_path is not None

                    st.write("")
                    gt_badge = f'<span class="vnpt-check-item {"ok" if gt_ok else "bad"}">{icon("check" if gt_ok else "cross", 13)} Ground Truth (.json)</span>'
                    ocr_badge = f'<span class="vnpt-check-item {"ok" if ocr_ok else "bad"}">{icon("check" if ocr_ok else "cross", 13)} Prediction OCR (*_ocr.md)</span>'
                    st.markdown(f'<div class="vnpt-check-row">{gt_badge}{ocr_badge}</div>', unsafe_allow_html=True)

                    if not gt_ok:
                        st.write("")
                        if ref_mode == "Tự động từ GT Creator":
                            st.warning("Phiên này chưa có file Ground Truth tổng hợp. Hãy dùng công cụ **GT Creator** để tạo và lưu `_master.json` trước.")
                        else:
                            st.warning("Vui lòng tải lên file Ground Truth .json.")

            with col_right:
                with st.container(border=True):
                    st.subheader("Tuỳ chọn Hệ thống")
                    st.info("Hệ thống sẽ tự động ghép nối văn bản, phân tích luồng đọc (Reading Order) và tính toán TEDS dựa trên cấu trúc bảng biểu.")
                    save_csv = st.checkbox("Lưu kết quả vào lịch sử CSV tổng (baseline_ocr_scores.csv)", value=True, key="ocr_bench_save_csv")

            run_disabled = not (gt_ok and ocr_ok)
            st.write("")
            run_clicked = st.button("Chạy Benchmark", type="primary", disabled=run_disabled, icon=":material/play_arrow:", key="ocr_bench_run")

            if run_clicked:
                with st.spinner("Đang tính toán các chỉ số OCR chuyên sâu..."):
                    csv_out_path = REPO_ROOT / "results" / "baseline_ocr_scores.csv" if save_csv else None

                    try:
                        eval_gt_path, fixed_notes = sanitize_ground_truth(final_gt_path)
                    except Exception:
                        eval_gt_path, fixed_notes = final_gt_path, []

                    if fixed_notes:
                        st.warning(
                            f"Đã tự động sửa **{len(fixed_notes)}** block bảng biểu bị lỗi "
                            "double-encode trong Ground Truth trước khi chấm điểm "
                            "(file gốc trên đĩa **không** bị thay đổi):\n\n"
                            + "\n".join(f"- {n}" for n in fixed_notes)
                        )

                    try:
                        results = evaluate_document(
                            gt_json_path=eval_gt_path,
                            pred_md_path=ocr_md_path,
                            csv_path=csv_out_path,
                        )

                        cer = results['text']['cer']
                        wer = results['text']['wer']
                        content_loss = results['layout']['content_loss']
                        ned = results['layout']['reading_order_ned']
                        teds = results['table']['teds']
                        teds_struct = results['table']['teds_struct']
                        tables_found = results['table']['tables_found']

                        rows = [
                            {"Nhóm": "Nhóm 1: Đánh giá nội dung - Text", "ID": "1.1", "Tiêu chí": "CER (Lỗi ký tự)", "Ngưỡng": "< 10%", "Giá trị đo": f"{cer*100:.2f}%", "Kết quả": "Đạt" if cer < 0.10 else "Không đạt", "Ưu tiên": "Cao"},
                            {"Nhóm": "Nhóm 1: Đánh giá nội dung - Text", "ID": "1.2", "Tiêu chí": "WER (Lỗi từ)", "Ngưỡng": "< 15%", "Giá trị đo": f"{wer*100:.2f}%", "Kết quả": "Đạt" if wer < 0.15 else "Không đạt", "Ưu tiên": "Cao"},
                            {"Nhóm": "Nhóm 2: Đánh giá bố cục - Layout", "ID": "2.1", "Tiêu chí": "Content Loss", "Ngưỡng": "< 5%", "Giá trị đo": f"{content_loss*100:.2f}%", "Kết quả": "Đạt" if content_loss < 0.05 else "Không đạt", "Ưu tiên": "Cao"},
                            {"Nhóm": "Nhóm 2: Đánh giá bố cục - Layout", "ID": "2.2", "Tiêu chí": "Reading Order NED", "Ngưỡng": ">= 90%", "Giá trị đo": f"{ned*100:.2f}%", "Kết quả": "Đạt" if ned >= 0.90 else "Không đạt", "Ưu tiên": "Cao"},
                        ]

                        if tables_found > 0:
                            rows.extend([
                                {"Nhóm": f"Nhóm 3: Đánh giá bảng biểu - Table ({tables_found} bảng)", "ID": "3.1", "Tiêu chí": "TEDS Tổng thể", "Ngưỡng": "> 80%", "Giá trị đo": f"{teds*100:.2f}%", "Kết quả": "Đạt" if teds > 0.80 else "Không đạt", "Ưu tiên": "Trung bình"},
                                {"Nhóm": f"Nhóm 3: Đánh giá bảng biểu - Table ({tables_found} bảng)", "ID": "3.2", "Tiêu chí": "TEDS Cấu trúc", "Ngưỡng": "> 88%", "Giá trị đo": f"{teds_struct*100:.2f}%", "Kết quả": "Đạt" if teds_struct > 0.88 else "Không đạt", "Ưu tiên": "Cao"},
                            ])
                        else:
                            rows.extend([
                                {"Nhóm": "Nhóm 3: Đánh giá bảng biểu - Table", "ID": "3.0", "Tiêu chí": "Không phát hiện bảng", "Ngưỡng": "N/A", "Giá trị đo": "N/A", "Kết quả": "N/A", "Ưu tiên": "Thấp"},
                            ])

                        n_pass = sum(1 for r in rows if r["Kết quả"] == "Đạt")
                        n_fail = sum(1 for r in rows if r["Kết quả"] == "Không đạt")
                        n_na = sum(1 for r in rows if r["Kết quả"] == "N/A")

                        tbl_info = results['table']
                        if tbl_info.get('unmatched_gt') or tbl_info.get('unmatched_pred'):
                            msg_parts = [
                                f"GT có {tbl_info['gt_tables_total']} bảng, Pred có "
                                f"{tbl_info['pred_tables_total']} bảng — chỉ ghép được "
                                f"{tbl_info['tables_found']} cặp theo nội dung."
                            ]
                            if tbl_info['unmatched_gt']:
                                msg_parts.append(f"Bảng GT không tìm thấy trong Pred: index {tbl_info['unmatched_gt']}")
                            if tbl_info['unmatched_pred']:
                                msg_parts.append(f"Bảng Pred không khớp bảng GT nào: index {tbl_info['unmatched_pred']}")
                            st.info("  \n".join(msg_parts))

                        st.session_state["ocr_benchmark_run"] = {
                            "sid": selected_session_name,
                            "results": results,
                            "rows": rows,
                            "n_pass": n_pass,
                            "n_fail": n_fail,
                            "n_na": n_na,
                        }

                        if save_csv:
                            json_out_path = run_dir / f"benchmark_{selected_session_name}.json"
                            try:
                                with open(json_out_path, "w", encoding="utf-8") as f:
                                    json.dump(st.session_state["ocr_benchmark_run"], f, ensure_ascii=False, indent=2)
                                st.info(f"Đã sao lưu file kết quả JSON tại: `{json_out_path}`")
                            except Exception as e:
                                st.warning(f"Không thể lưu file JSON: {e}")

                        st.success(f"Đã hoàn tất đánh giá cho phiên: {selected_session_name}")

                    except Exception as e:
                        err_str = str(e)
                        if "invalid literal for int()" in err_str and '"' in err_str:
                            st.error(
                                "Lỗi khi tính TEDS: một thuộc tính `rowspan`/`colspan` trong "
                                "HTML bảng biểu chứa giá trị không hợp lệ "
                                f"(`{err_str}`). Đây là dấu hiệu file Ground Truth bị lưu lồng "
                                "JSON 2 lớp (double-encoded) ở phần nội dung bảng."
                            )
                        else:
                            st.error(f"Đã xảy ra lỗi trong quá trình tính toán: {e}")
                        with st.expander("Chi tiết lỗi (traceback)"):
                            st.code(traceback.format_exc())

            run_data = st.session_state.get("ocr_benchmark_run")

            if run_data and run_data.get("sid") == selected_session_name:
                results = run_data["results"]
                rows = run_data["rows"]

                cer = results['text']['cer'] * 100
                wer = results['text']['wer'] * 100
                content_loss = results['layout']['content_loss'] * 100
                ned = results['layout']['reading_order_ned'] * 100

                tables_found = results['table'].get('tables_found', 0)
                teds = results['table']['teds'] * 100
                teds_struct = results['table']['teds_struct'] * 100

                st.divider()

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown(f'<div class="vnpt-metric-card"><div class="m-val">{cer:.2f}%</div><div class="m-lbl">CER (Lỗi ký tự) &darr;</div></div>', unsafe_allow_html=True)
                with c2:
                    st.markdown(f'<div class="vnpt-metric-card"><div class="m-val">{wer:.2f}%</div><div class="m-lbl">WER (Lỗi từ) &darr;</div></div>', unsafe_allow_html=True)
                with c3:
                    st.markdown(f'<div class="vnpt-metric-card"><div class="m-val">{ned:.2f}%</div><div class="m-lbl">Reading Order (Mạch đọc) &uarr;</div></div>', unsafe_allow_html=True)

                st.write("")
                st.markdown(
                    f'<div class="vnpt-section-title" style="font-size:14px;text-transform:uppercase;letter-spacing:1px;">'
                    f'{icon("gauge", 17)} Đánh giá theo bộ tiêu chuẩn Benchmark</div>',
                    unsafe_allow_html=True,
                )

                sm1, sm2, sm3 = st.columns(3)
                sm1.metric("Đạt (Pass)", run_data["n_pass"])
                sm2.metric("Không đạt (Fail)", run_data["n_fail"])
                sm3.metric("N/A / Bỏ qua", run_data["n_na"])

                st.write("")

                df_all = pd.DataFrame(rows)

                render_benchmark_groups(rows)

                st.divider()

                gt_dict = {}
                if gt_ok and final_gt_path.exists():
                    gt_dict = json.loads(final_gt_path.read_text(encoding="utf-8"))

                stem = gt_dict.get("doc_id") or gt_dict.get("stem")
                if not stem:
                    stem = selected_session_name.split("_")[0]

                orig_pdf = REPO_ROOT / "output" / "_uploads" / f"{stem}.pdf"
                if not orig_pdf.exists():
                    for p in run_dir.glob("*.pdf"):
                        if "translated" not in p.name and "ocr" not in p.name:
                            orig_pdf = p
                            break

                ocr_pdf = run_dir / f"{stem}_ocr.pdf"

                with st.expander("Đối chiếu chi tiết (Ground Truth vs Hypothesis)", expanded=False, icon=":material/compare_arrows:"):
                    view_c1, view_c2 = st.columns(2)

                    with view_c1:
                        st.markdown("##### Hiển thị file nạp vào")
                        if orig_pdf and orig_pdf.exists():
                            display_pdf(orig_pdf)
                        else:
                            st.warning(f"Không tìm thấy file gốc: {orig_pdf.name if orig_pdf else 'N/A'}")

                    with view_c2:
                        st.markdown("##### Hiển thị file kết quả (Đã căn Layout)")
                        if ocr_pdf.exists():
                            display_pdf(ocr_pdf)
                        else:
                            st.warning(f"Không tìm thấy file kết quả OCR: {ocr_pdf.name}")

                st.write("")
                st.subheader("Tải báo cáo")

                md_lines = [f"# BÁO CÁO BENCHMARK OCR\n\n**Phiên:** {selected_session_name}\n"]
                md_lines.append("| Nhóm | ID | Tiêu chí | Ngưỡng | Giá trị đo | Kết quả | Ưu tiên |")
                md_lines.append("|---|---|---|---|---|---|---|")
                for _, r in df_all.iterrows():
                    md_lines.append(f"| {r['Nhóm']} | {r['ID']} | {r['Tiêu chí']} | {r['Ngưỡng']} | {r['Giá trị đo']} | {r['Kết quả']} | {r['Ưu tiên']} |")
                md_report = "\n".join(md_lines)

                dl1, dl2, dl3 = st.columns(3)
                with dl1:
                    st.download_button(
                        label="Tải JSON (.json)",
                        data=json.dumps(run_data, ensure_ascii=False, indent=2).encode('utf-8'),
                        file_name=f"benchmark_{selected_session_name}.json",
                        mime="application/json",
                        icon=":material/download:",
                        use_container_width=True,
                        key="ocr_dl_json",
                    )
                with dl2:
                    st.download_button(
                        label="Tải Markdown (.md)",
                        data=md_report.encode('utf-8'),
                        file_name=f"benchmark_{selected_session_name}.md",
                        mime="text/markdown",
                        icon=":material/download:",
                        use_container_width=True,
                        key="ocr_dl_md",
                    )
                with dl3:
                    st.download_button(
                        label="Tải bảng CSV (.csv)",
                        data=df_all.to_csv(index=False).encode('utf-8'),
                        file_name=f"benchmark_{selected_session_name}.csv",
                        mime="text/csv",
                        icon=":material/download:",
                        use_container_width=True,
                        key="ocr_dl_csv",
                    )