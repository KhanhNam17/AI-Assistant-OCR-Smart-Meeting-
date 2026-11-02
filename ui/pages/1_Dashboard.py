import sys
import os
import json
import html as _html
from datetime import datetime
from pathlib import Path
import streamlit as st

# ══════════════════════════════════════════════════════════════════════════════
# 1. SETUP ĐƯỜNG DẪN & THEME CHUNG
# ══════════════════════════════════════════════════════════════════════════════
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
UI_ROOT = Path(__file__).resolve().parent.parent
for p in (REPO_ROOT, UI_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from theme import apply_page_config, inject_base_css, render_sidebar, icon, render_workspace_jump

apply_page_config("Dashboard — Tổng quan hệ thống", UI_ROOT)
inject_base_css()
render_sidebar(active="space_dashboard")

# ══════════════════════════════════════════════════════════════════════════════
# 2. LOCAL CSS ĐỒNG BỘ HIỆN ĐẠI
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    /* ── Dashboard Cards & Elements ── */
    .dash-card-title {
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        letter-spacing: 1.8px;
        text-transform: uppercase;
        color: #2563EB;
        margin-bottom: 18px;
        font-weight: 700;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .tech-stack-container {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-bottom: 22px;
        padding: 16px 20px;
        background: rgba(37, 99, 235, 0.1);
        border-radius: 14px;
        border: 1px solid rgba(37, 99, 235, 0.25);
    }
    .tech-badge {
        background: var(--secondary-background-color);
        color: var(--text-color);
        border: 1px solid rgba(148, 163, 184, 0.25);
        padding: 6px 14px;
        border-radius: 9999px;
        font-size: 12.5px;
        font-weight: 600;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    .bar-bg {
        background: rgba(148, 163, 184, 0.15);
        height: 8px;
        border-radius: 9999px;
        width: 100%;
        margin: 6px 0 16px 0;
        overflow: hidden;
    }
    .bar-fill {
        height: 100%;
        border-radius: 9999px;
        transition: width 0.3s ease;
    }
    .bar-label-row {
        display: flex;
        justify-content: space-between;
        font-size: 13px;
        font-weight: 600;
        color: var(--text-color);
    }
    .rank-item {
        display: flex;
        align-items: center;
        gap: 16px;
        padding: 12px 16px;
        background: rgba(148, 163, 184, 0.08);
        border-radius: 12px;
        margin-bottom: 10px;
        border: 1px solid rgba(148, 163, 184, 0.18);
        transition: all 0.15s ease;
    }
    .rank-item:hover {
        background: rgba(37, 99, 235, 0.1);
        border-color: rgba(37, 99, 235, 0.35);
    }
    .rank-num {
        font-size: 20px;
        font-weight: 800;
        width: 32px;
        height: 32px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 50%;
        background: var(--secondary-background-color);
        border: 1px solid rgba(148, 163, 184, 0.25);
        color: var(--text-color);
    }
    .rank-info { flex-grow: 1; }
    .rank-name { font-weight: 700; font-size: 13.5px; color: var(--text-color); }
    .rank-score {
        font-family: 'JetBrains Mono', monospace;
        font-size: 18px;
        font-weight: 800;
    }
    .dash-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 10px 14px;
        background: rgba(148, 163, 184, 0.08);
        border-radius: 8px;
        margin-bottom: 8px;
        border: 1px solid rgba(148, 163, 184, 0.18);
        font-size: 13px;
    }
    .dash-row-lbl {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
        color: var(--text-color);
        opacity: 0.65;
        font-size: 11px;
    }
    .dash-row-val { font-weight: 600; color: var(--text-color); }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# 3. TIÊU ĐỀ CHÍNH
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div style="font-family:'JetBrains Mono',monospace;font-size:11px;letter-spacing:2px;text-transform:uppercase;
            color:#2563EB;padding-bottom:12px;border-bottom:1px solid #E2E8F0;margin-bottom:18px;
            display:flex;align-items:center;gap:8px;">
    {icon('sparkles', 14, 2.0, '#2563EB')} DASHBOARD — TRUNG TÂM KIỂM SOÁT HỆ THỐNG
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# 4. KHỞI TẠO TABS
# ══════════════════════════════════════════════════════════════════════════════
tab_meeting, tab_ocr = st.tabs(["🎙️ Cuộc họp thông minh", "📄 OCR Tài liệu"])

# ==============================================================================
# TAB 1: SMART MEETING DASHBOARD
# ==============================================================================
with tab_meeting:
    render_workspace_jump("meeting")
    SESSIONS_DIR = "data/sessions"
    DOMAINS_DIR = os.path.join("core", "data", "domains")

    all_benchmarks = []
    latest_session = None
    meta_data = {}

    if os.path.exists(SESSIONS_DIR):
        sessions = [d for d in os.listdir(SESSIONS_DIR) if os.path.isdir(os.path.join(SESSIONS_DIR, d))]
        if sessions:
            sessions.sort(key=lambda d: os.path.getmtime(os.path.join(SESSIONS_DIR, d)), reverse=True)
            latest_session = sessions[0]
            meta_path = os.path.join(SESSIONS_DIR, latest_session, "metadata.json")
            if os.path.exists(meta_path):
                with open(meta_path, "r", encoding="utf-8") as f:
                    try:
                        meta_data = json.load(f)
                    except Exception:
                        pass

            for s in sessions:
                bench_path = os.path.join(SESSIONS_DIR, s, "benchmark.json")
                s_meta_path = os.path.join(SESSIONS_DIR, s, "metadata.json")
                b_data = None

                if os.path.exists(bench_path):
                    try:
                        with open(bench_path, "r", encoding="utf-8") as f:
                            b_data = json.load(f)
                    except Exception:
                        pass
                elif os.path.exists(s_meta_path):
                    try:
                        with open(s_meta_path, "r", encoding="utf-8") as f:
                            m = json.load(f)
                            if "benchmark" in m:
                                b_data = m["benchmark"]
                            elif "results" in m:
                                b_data = m["results"]
                    except Exception:
                        pass

                if b_data:
                    raw_metrics = b_data.get("raw_metrics", b_data)
                    if "wer" in raw_metrics:
                        wer = (raw_metrics.get("wer") or 0.0) * 100
                        der = (raw_metrics.get("der") or 0.0) * 100
                        cer = (raw_metrics.get("cer") or 0.0) * 100
                        overall = raw_metrics.get("overall", (100.0 - wer) * 0.7 + (100.0 - der) * 0.3)
                        all_benchmarks.append({
                            "session": s,
                            "overall": round(overall, 1),
                            "wer": round(wer, 1),
                            "cer": round(cer, 1),
                            "der": round(der, 1),
                            "mtime": os.path.getmtime(os.path.join(SESSIONS_DIR, s))
                        })

    top3_sessions = sorted(all_benchmarks, key=lambda x: x["overall"], reverse=True)[:3]
    latest_bench = None
    if all_benchmarks:
        all_benchmarks.sort(key=lambda x: x["mtime"], reverse=True)
        latest_bench = all_benchmarks[0]

    domain_files = []
    all_domain_terms = []
    if os.path.exists(DOMAINS_DIR):
        domain_files = [f for f in os.listdir(DOMAINS_DIR) if f.endswith(".json")]
        for df in domain_files:
            try:
                with open(os.path.join(DOMAINS_DIR, df), "r", encoding="utf-8") as f:
                    terms_list = json.load(f)
                    if isinstance(terms_list, list):
                        for item in terms_list:
                            if isinstance(item, str):
                                all_domain_terms.append(item)
                            elif isinstance(item, dict):
                                t_str = item.get("canonical") or item.get("term") or item.get("text") or ""
                                if t_str:
                                    all_domain_terms.append(t_str)
            except Exception:
                pass

    st.markdown(
        f"""
        <div class="tech-stack-container">
            <div style="width:100%;font-family:'JetBrains Mono',monospace;font-size:10px;letter-spacing:2px;text-transform:uppercase;color:#2563EB;margin-bottom:4px;font-weight:700;">
                {icon('cpu', 12)} CÔNG NGHỆ ĐANG VẬN HÀNH
            </div>
            <div class="tech-badge"><span>🎙️</span> Zipformer STT (16kHz Offline)</div>
            <div class="tech-badge"><span>🧬</span> Diart (Pyannote 3.0 Diarization)</div>
            <div class="tech-badge"><span>🪪</span> Cam++ VoiceID 192d Embedding</div>
            <div class="tech-badge"><span>📄</span> Deepseek OCR & Layout Engine</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    row1_c1, row1_c2 = st.columns(2, gap="medium")
    with row1_c1:
        with st.container(border=True):
            st.markdown(f'<div class="dash-card-title">{icon("folder", 14)} Phiên làm việc gần nhất</div>', unsafe_allow_html=True)
            if latest_session:
                st.markdown(f'<div style="font-size:22px;font-weight:800;color:var(--text-color);margin-bottom:4px;">{latest_session.replace("_", " ")}</div>', unsafe_allow_html=True)
                st.markdown(f'<div style="font-size:13px;color:var(--text-color);opacity:0.65;margin-bottom:20px;">📅 Ngày lưu: {meta_data.get("date", "Gần đây")}</div>', unsafe_allow_html=True)
                stats = meta_data.get("stats", {})
                if stats:
                    st.markdown('<div style="font-size:11.5px;font-weight:700;margin-bottom:12px;text-transform:uppercase;letter-spacing:1px;color:var(--text-color);opacity:0.65;">Độ sôi nổi người nói (Speaker Activity)</div>', unsafe_allow_html=True)
                    for spk, data in list(stats.items())[:3]:
                        pct = data.get("percent", 0)
                        st.markdown(f'''
                        <div class="bar-label-row"><span>{_html.escape(spk)}</span><span style="color:#2563EB;">{pct}%</span></div>
                        <div class="bar-bg"><div class="bar-fill" style="width:{pct}%; background: linear-gradient(90deg, #2563EB, #60A5FA);"></div></div>
                        ''', unsafe_allow_html=True)
                st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
                st.page_link("pages/5_Review_sessions.py", label="Xem chi tiết biên bản phiên này", icon=":material/arrow_forward:")
            else:
                st.info("Hệ thống chưa lưu dữ liệu cuộc họp nào.")

    with row1_c2:
        with st.container(border=True):
            st.markdown(f'<div class="dash-card-title">{icon("gauge", 14)} Kết quả Benchmark gần nhất</div>', unsafe_allow_html=True)
            if latest_bench:
                wer_val, cer_val, der_val, overall = latest_bench["wer"], latest_bench["cer"], latest_bench["der"], latest_bench["overall"]
                color = "#10B981" if overall >= 85 else "#F59E0B" if overall >= 70 else "#EF4444"
                st.markdown(f'''
                <div style="display:flex; align-items:center; gap:24px; margin-bottom:24px;">
                    <div>
                        <div style="font-family:\'JetBrains Mono\',monospace; font-size:11px; letter-spacing:1px; color:var(--text-color); opacity:0.7; font-weight:700;">OVERALL PERFORMANCE</div>
                        <div style="font-size:48px; font-weight:800; color:{color}; line-height:1;">{overall}</div>
                    </div>
                    <div style="padding-left:20px; border-left:1px solid rgba(148,163,184,0.25); font-size:13px; color:var(--text-color); opacity:0.75;">
                        Phiên đánh giá: <b style="opacity:1;">{latest_bench["session"].replace("_", " ")}</b><br>
                        Tỷ lệ lỗi (WER/CER/DER) càng thấp chất lượng nhận diện càng cao.
                    </div>
                </div>
                <div class="bar-label-row"><span>Tỷ lệ lỗi từ (WER)</span><span style="color:#EF4444;">{wer_val}%</span></div>
                <div class="bar-bg"><div class="bar-fill" style="width:{min(100.0, wer_val)}%; background:#EF4444;"></div></div>
                <div class="bar-label-row"><span>Tỷ lệ lỗi ký tự (CER)</span><span style="color:#F59E0B;">{cer_val}%</span></div>
                <div class="bar-bg"><div class="bar-fill" style="width:{min(100.0, cer_val)}%; background:#F59E0B;"></div></div>
                <div class="bar-label-row"><span>Lỗi phân tách người nói (DER)</span><span style="color:#2563EB;">{der_val}%</span></div>
                <div class="bar-bg"><div class="bar-fill" style="width:{min(100.0, der_val)}%; background:#2563EB;"></div></div>
                ''', unsafe_allow_html=True)
            else:
                st.info("Chưa tìm thấy lịch sử chạy kịch bản kiểm thử Benchmark.")

    st.markdown("<div style='margin:16px 0;'></div>", unsafe_allow_html=True)
    row2_c1, row2_c2 = st.columns(2, gap="medium")
    with row2_c1:
        with st.container(border=True):
            st.markdown(f'<div class="dash-card-title">{icon("docx", 14)} Thuật ngữ chuyên ngành (Domain Knowledge)</div>', unsafe_allow_html=True)
            st.markdown(f'''
            <div style="display:grid; grid-template-columns:repeat(2,1fr); gap:12px; margin-bottom:20px; text-align:center;">
                <div style="background:rgba(37,99,235,0.1); padding:14px; border-radius:12px; border:1px solid rgba(37,99,235,0.25);">
                    <div style="font-family:\'JetBrains Mono\',monospace; font-size:11px; color:var(--text-color); opacity:0.7; font-weight:700;">SỐ BỘ TỪ ĐIỂN</div>
                    <div style="font-size:26px; font-weight:800; color:#2563EB;">{len(domain_files)} file</div>
                </div>
                <div style="background:rgba(37,99,235,0.1); padding:14px; border-radius:12px; border:1px solid rgba(37,99,235,0.25);">
                    <div style="font-family:\'JetBrains Mono\',monospace; font-size:11px; color:var(--text-color); opacity:0.7; font-weight:700;">TỔNG THUẬT NGỮ</div>
                    <div style="font-size:26px; font-weight:800; color:#1D4ED8;">{len(all_domain_terms)} từ</div>
                </div>
            </div>
            <div style="font-size:11.5px;font-weight:700;margin-bottom:12px;text-transform:uppercase;letter-spacing:1px;color:var(--text-color);opacity:0.7;">Thuật ngữ tiêu biểu trong hệ thống</div>
            ''', unsafe_allow_html=True)
            if all_domain_terms:
                for i, term in enumerate(all_domain_terms[:3]):
                    st.markdown(f'''
                    <div class="dash-row">
                        <div class="dash-row-lbl">{i+1}{"st" if i==0 else "nd" if i==1 else "rd"} Word</div>
                        <div class="dash-row-val">{_html.escape(term)}</div>
                    </div>
                    ''', unsafe_allow_html=True)
            else:
                st.caption("Chưa có thuật ngữ nào được nạp vào hệ thống.")
            st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
            st.page_link("pages/4_Domain_terms.py", label="Quản lý từ điển chuyên ngành", icon=":material/arrow_forward:")

    with row2_c2:
        with st.container(border=True):
            st.markdown(f'<div class="dash-card-title">{icon("zap", 14)} Top 3 Phiên Có Độ Chính Xác Cao Nhất</div>', unsafe_allow_html=True)
            if top3_sessions:
                medal_colors = ["#F59E0B", "#94A3B8", "#B45309"]
                for i, bm in enumerate(top3_sessions):
                    st.markdown(f'''
                    <div class="rank-item">
                        <div class="rank-num" style="color:{medal_colors[i]};">{i+1}</div>
                        <div class="rank-info">
                            <div class="rank-name">{bm["session"].replace("_", " ")}</div>
                            <div style="font-size:12px; color:#64748B;">WER: {bm["wer"]}% · CER: {bm["cer"]}% · DER: {bm["der"]}%</div>
                        </div>
                        <div class="rank-score" style="color:{medal_colors[i]};">{bm["overall"]}</div>
                    </div>
                    ''', unsafe_allow_html=True)
            else:
                st.info("Chưa có dữ liệu kiểm thử Benchmark nào được lưu lại để xếp hạng.")
            st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
            st.page_link("pages/6_Benchmark_sessions.py", label="Đánh giá chất lượng toàn diện", icon=":material/arrow_forward:")


# ==============================================================================
# TAB 2: OCR DASHBOARD
# ==============================================================================
with tab_ocr:
    render_workspace_jump("ocr")
    DEFAULT_OUT = REPO_ROOT / "output"

    def load_session_history():
        hist = []
        if DEFAULT_OUT.exists():
            for d in DEFAULT_OUT.iterdir():
                if d.is_dir() and (d / "metadata.json").exists():
                    try:
                        with open(d / "metadata.json", "r", encoding="utf-8") as f:
                            hist.append(json.load(f))
                    except Exception:
                        pass
        hist.sort(key=lambda x: x.get("id", ""), reverse=True)
        return hist

    if "session_history" not in st.session_state:
        st.session_state.session_history = load_session_history()

    history = st.session_state.get("session_history", [])
    total_files = len(history)
    success_count = sum(1 for e in history if e.get("exit_code", -1) == 0)
    failed_count = total_files - success_count
    success_rate = (success_count / total_files * 100) if total_files else 0.0

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f"""
            <div class="app-stat-card">
                <div class="app-stat-top total"></div>
                <div class="app-stat-body">
                    <div class="app-stat-head">
                        <div class="app-metric-icon total">{icon('layers', 20, 2.0, '#2563EB')}</div>
                    </div>
                    <div class="app-stat-value">{total_files}</div>
                    <div class="app-stat-label">Tổng số tài liệu xử lý</div>
                </div>
            </div>
            """, unsafe_allow_html=True
        )
    with c2:
        badge_success = (
            f'<span class="app-stat-share success">{success_count / total_files * 100:.0f}%</span>'
            if total_files else ''
        )
        st.markdown(
            f"""
            <div class="app-stat-card">
                <div class="app-stat-top success"></div>
                <div class="app-stat-body">
                    <div class="app-stat-head">
                        <div class="app-metric-icon success">{icon('check', 20, 2.0, '#10B981')}</div>{badge_success}
                    </div>
                    <div class="app-stat-value">{success_count}</div>
                    <div class="app-stat-label">Xử lý thành công</div>
                </div>
            </div>
            """, unsafe_allow_html=True
        )
    with c3:
        badge_failed = (
            f'<span class="app-stat-share failed">{failed_count / total_files * 100:.0f}%</span>'
            if total_files else ''
        )
        st.markdown(
            f"""
            <div class="app-stat-card">
                <div class="app-stat-top failed"></div>
                <div class="app-stat-body">
                    <div class="app-stat-head">
                        <div class="app-metric-icon failed">{icon('cross', 20, 2.0, '#EF4444')}</div>{badge_failed}
                    </div>
                    <div class="app-stat-value">{failed_count}</div>
                    <div class="app-stat-label">Xử lý thất bại</div>
                </div>
            </div>
            """, unsafe_allow_html=True
        )

    if total_files > 0:
        st.markdown(
            f"""
            <div class="app-kpi-banner">
                <div class="app-kpi-icon">{icon('gauge', 24, 2.0, '#FFFFFF')}</div>
                <div class="app-kpi-text">
                    <div class="app-kpi-label">Tỷ lệ xử lý thành công toàn hệ thống</div>
                    <div class="app-kpi-value">{success_rate:.1f}%</div>
                </div>
                <div class="app-kpi-track"><div class="app-kpi-fill" style="width:{success_rate:.1f}%;"></div></div>
            </div>
            """, unsafe_allow_html=True
        )

    def _build_activity(history_list: list[dict], max_days: int = 7) -> list[tuple[str, int]]:
        counts: dict[str, int] = {}
        for entry in history_list:
            try:
                d = datetime.fromisoformat(entry.get("started_at", "")).date()
            except Exception:
                continue
            key = d.strftime("%d/%m")
            counts[key] = counts.get(key, 0) + 1
        ordered_keys = sorted(counts.keys(), key=lambda k: datetime.strptime(k, "%d/%m"))
        return [(k, counts[k]) for k in ordered_keys[-max_days:]]

    activity = _build_activity(history)
    if activity:
        with st.container(border=True):
            st.markdown(f'<span class="app-pill-label">{icon("layers", 14)} Tần suất xử lý gần đây</span>', unsafe_allow_html=True)
            max_count = max(c for _, c in activity) or 1
            rows_html = "".join(
                f'<div class="app-activity-row">'
                f'<div class="app-activity-label">{day}</div>'
                f'<div class="app-activity-track"><div class="app-activity-fill" style="width:{c / max_count * 100:.0f}%;"></div></div>'
                f'<div class="app-activity-count">{c}</div>'
                f'</div>'
                for day, c in activity
            )
            st.markdown(rows_html, unsafe_allow_html=True)

    st.write("")
    st.subheader("Chi tiết các phiên xử lý OCR")

    if total_files > 0:
        tb1, tb2 = st.columns([3, 1])
        with tb1:
            search_q = st.text_input("Tìm kiếm theo tên file", placeholder="Nhập tên file tài liệu...", key="dash_search")
        with tb2:
            status_filter = st.selectbox("Tình trạng", options=["Tất cả", "Thành công", "Thất bại"], key="dash_status_filter")
    else:
        search_q, status_filter = "", "Tất cả"

    def _apply_filters(history_list: list[dict]) -> list[dict]:
        out = history_list
        if search_q.strip():
            q = search_q.strip().lower()
            out = [e for e in out if q in str(e.get("stem", "")).lower()]
        if status_filter == "Thành công":
            out = [e for e in out if e.get("exit_code", -1) == 0]
        elif status_filter == "Thất bại":
            out = [e for e in out if e.get("exit_code", -1) != 0]
        return out

    def _render_history_table(history_desc: list[dict]) -> None:
        ordered = list(reversed(history_desc))
        rows_html = []
        for i, entry in enumerate(ordered):
            stt = i + 1
            stem = _html.escape(str(entry.get("stem", "Unknown")))
            pages = _html.escape(str(entry.get("pages", "--")))
            try:
                dt_obj = datetime.fromisoformat(entry.get("started_at", ""))
                date_str = dt_obj.strftime("%d/%m/%Y %H:%M:%S")
            except Exception:
                date_str = _html.escape(str(entry.get("started_at", "--")))

            status_html = (
                '<span class="app-status"><span class="app-dot app-dot-ok"></span><span style="color:#059669;">Thành công</span></span>'
                if entry.get("exit_code", -1) == 0 else
                '<span class="app-status"><span class="app-dot app-dot-fail"></span><span style="color:#DC2626;">Thất bại</span></span>'
            )

            rows_html.append(
                f"<tr>"
                f'<td class="col-stt">{stt}</td>'
                f'<td class="col-file">{stem}</td>'
                f'<td class="col-date">{date_str}</td>'
                f'<td class="col-pages">{pages}</td>'
                f"<td>{status_html}</td>"
                f"</tr>"
            )

        st.markdown(
            f"""
            <div class="app-table-wrap">
              <table class="app-table">
                <thead>
                  <tr>
                    <th>STT</th>
                    <th>Tên file</th>
                    <th>Thời gian xử lý</th>
                    <th>Số trang</th>
                    <th>Trạng thái</th>
                  </tr>
                </thead>
                <tbody>
                  {''.join(rows_html)}
                </tbody>
              </table>
            </div>
            """, unsafe_allow_html=True
        )

    if total_files > 0:
        filtered_history = _apply_filters(history)
        if filtered_history:
            st.caption(f"Hiển thị {len(filtered_history)}/{total_files} phiên xử lý.")
            _render_history_table(filtered_history)
        else:
            with st.container(border=True):
                st.info("Không tìm thấy phiên nào khớp với bộ lọc hiện tại.")
    else:
        with st.container(border=True):
            st.info("Chưa có dữ liệu thống kê. Hãy xử lý tài liệu tại trang OCR Tài liệu để xem kết quả tại đây.")
