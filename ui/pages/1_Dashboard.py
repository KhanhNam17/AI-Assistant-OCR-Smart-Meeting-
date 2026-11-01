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

# Gọi Sidebar dùng chung (Không tự vẽ bằng st.sidebar nữa)
render_sidebar(active="space_dashboard")

# ══════════════════════════════════════════════════════════════════════════════
# 2. GỘP CSS CỦA CẢ 2 APP
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    /* ── CSS Smart Meeting ── */
    .rv-card-title { font-family: 'IBM Plex Mono', monospace; font-size: 11px; letter-spacing: 2px; text-transform: uppercase; color: #0067B1; margin-bottom: 20px; font-weight: 600; display: flex; align-items: center; gap: 8px; }
    .tech-stack-container { display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 24px; padding: 16px; background: rgba(0,103,177,0.05); border-radius: 8px; border: 1px dashed rgba(0,103,177,0.3); }
    .tech-badge { background: #1E1E1E; color: #fff; border: 1px solid rgba(255,255,255,0.1); padding: 6px 16px; border-radius: 20px; font-size: 13px; font-weight: 600; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    .tech-badge span { color: #0067B1; }
    .bar-bg { background: rgba(128,128,128,0.15); height: 8px; border-radius: 4px; width: 100%; margin: 6px 0 16px 0; overflow: hidden; }
    .bar-fill { height: 100%; border-radius: 4px; }
    .bar-label-row { display: flex; justify-content: space-between; font-size: 13px; font-weight: 600; color: var(--text-color); }
    .rank-item { display: flex; align-items: center; gap: 16px; padding: 12px 16px; background: rgba(128,128,128,0.05); border-radius: 8px; margin-bottom: 12px; border: 1px solid rgba(128,128,128,0.1); }
    .rank-num { font-size: 24px; font-weight: 800; font-style: italic; width: 30px; text-align: center; }
    .rank-info { flex-grow: 1; }
    .rank-name { font-weight: 600; font-size: 14px; }
    .rank-score { font-family: 'IBM Plex Mono', monospace; font-size: 20px; font-weight: 700; }
    .dash-row { display: flex; justify-content: space-between; padding: 10px 12px; background: rgba(128,128,128,0.04); border-radius: 6px; margin-bottom: 8px; border: 1px solid rgba(128,128,128,0.08); font-size: 13px; }
    .dash-row-lbl { font-family: 'IBM Plex Mono', monospace; font-weight: 600; color: #888; }
    .dash-row-val { font-weight: 600; color: var(--text-color); }

    /* ── CSS OCR ── */
    .vnpt-stat-card { border-radius: var(--vnpt-radius-lg); border: 1px solid var(--border-color); background: rgba(255,255,255,0.02); box-shadow: var(--vnpt-shadow-sm); overflow: hidden; height: 100%; }
    .vnpt-stat-top { height: 4px; }
    .vnpt-stat-top.total   { background: var(--vnpt-blue); }
    .vnpt-stat-top.success { background: var(--vnpt-green); }
    .vnpt-stat-top.failed  { background: var(--vnpt-red); }
    .vnpt-stat-body { padding: 20px 22px 22px; }
    .vnpt-stat-head { display: flex; align-items: center; justify-content: space-between; margin-bottom: 16px; }
    .vnpt-metric-icon { width: 38px; height: 38px; border-radius: 10px; display: flex; align-items: center; justify-content: center; }
    .vnpt-metric-icon.total   { background: var(--vnpt-blue-tint);  color: var(--vnpt-blue); }
    .vnpt-metric-icon.success { background: var(--vnpt-green-tint); color: var(--vnpt-green); }
    .vnpt-metric-icon.failed  { background: var(--vnpt-red-tint);   color: var(--vnpt-red); }
    .vnpt-stat-share { font-size: 11px; font-weight: 800; padding: 3px 10px; border-radius: 999px; }
    .vnpt-stat-share.success { background: var(--vnpt-green-tint); color: var(--vnpt-green); }
    .vnpt-stat-share.failed  { background: var(--vnpt-red-tint);   color: var(--vnpt-red); }
    .vnpt-stat-value { font-size: 34px; font-weight: 800; color: var(--text-color); line-height: 1; margin-bottom: 6px; }
    .vnpt-stat-label { font-size: 12.5px; font-weight: 700; opacity: 0.65; }
    .vnpt-kpi-banner { background: linear-gradient(135deg, var(--vnpt-blue) 0%, var(--vnpt-navy) 130%); border-radius: var(--vnpt-radius-lg); padding: 20px 26px; color: white; display: flex; align-items: center; gap: 22px; flex-wrap: wrap; box-shadow: 0 10px 26px rgba(0,96,181,0.25); margin: 18px 0 26px; }
    .vnpt-kpi-icon { width: 46px; height: 46px; border-radius: 12px; flex-shrink: 0; background: rgba(255,255,255,0.14); display: flex; align-items: center; justify-content: center; }
    .vnpt-kpi-text { flex-shrink: 0; }
    .vnpt-kpi-label { font-size: 11.5px; font-weight: 800; letter-spacing: 0.8px; text-transform: uppercase; opacity: 0.75; }
    .vnpt-kpi-value { font-size: 26px; font-weight: 800; margin-top: 2px; }
    .vnpt-kpi-track { flex: 1; min-width: 160px; height: 8px; border-radius: 999px; background: rgba(255,255,255,0.18); overflow: hidden; }
    .vnpt-kpi-fill { height: 100%; border-radius: 999px; background: #6EE7A8; }
    .vnpt-activity-row { display: flex; align-items: center; gap: 14px; padding: 7px 0; }
    .vnpt-activity-label { width: 84px; flex-shrink: 0; font-size: 12px; opacity: 0.7; font-variant-numeric: tabular-nums; }
    .vnpt-activity-track { flex: 1; height: 10px; border-radius: 999px; background: rgba(148,163,184,0.14); overflow: hidden; }
    .vnpt-activity-fill { height: 100%; border-radius: 999px; background: var(--vnpt-blue); }
    .vnpt-activity-count { width: 28px; flex-shrink: 0; text-align: right; font-size: 12.5px; font-weight: 800; color: var(--vnpt-blue); }
    .vnpt-toolbar-label { font-size: 11px; font-weight: 800; letter-spacing: 0.5px; text-transform: uppercase; opacity: 0.55; margin-bottom: 4px; }
    .vnpt-table-wrap { border: 1px solid var(--border-color); border-radius: var(--vnpt-radius-lg); overflow: hidden; background: rgba(255,255,255,0.02); box-shadow: var(--vnpt-shadow-sm); }
    .vnpt-table { width: 100%; border-collapse: collapse; font-size: 13.5px; }
    .vnpt-table thead th { text-align: left; font-size: 11px; letter-spacing: 1.2px; text-transform: uppercase; color: var(--vnpt-blue); font-weight: 800; padding: 15px 22px; background: var(--vnpt-blue-tint); border-bottom: 1px solid var(--border-color); white-space: nowrap; }
    .vnpt-table tbody td { padding: 13px 22px; border-bottom: 1px solid var(--border-color); color: var(--text-color); vertical-align: middle; }
    .vnpt-table tbody tr:last-child td { border-bottom: none; }
    .vnpt-table tbody tr:hover td { background: var(--vnpt-blue-tint); }
    .vnpt-table .col-stt  { width: 56px; font-weight: 700; opacity: 0.5; }
    .vnpt-table .col-file { font-weight: 700; }
    .vnpt-table .col-date { opacity: 0.7; white-space: nowrap; font-variant-numeric: tabular-nums; }
    .vnpt-status { display: inline-flex; align-items: center; gap: 8px; font-weight: 700; white-space: nowrap; font-size: 13px; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# 3. TIÊU ĐỀ CHÍNH
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="font-size:11px;letter-spacing:3px;text-transform:uppercase;
            color:var(--text-color);opacity:0.7;padding-bottom:12px;
            border-bottom:1px solid rgba(128,128,128,0.2);margin-bottom:16px;">
    🧭  Dashboard — Trạm kiểm soát trung tâm
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
                    meta_data = json.load(f)

            for s in sessions:
                bench_path = os.path.join(SESSIONS_DIR, s, "benchmark.json")
                s_meta_path = os.path.join(SESSIONS_DIR, s, "metadata.json")
                b_data = None

                if os.path.exists(bench_path):
                    try:
                        with open(bench_path, "r", encoding="utf-8") as f: b_data = json.load(f)
                    except: pass
                elif os.path.exists(s_meta_path):
                    try:
                        with open(s_meta_path, "r", encoding="utf-8") as f:
                            m = json.load(f)
                            if "benchmark" in m: b_data = m["benchmark"]
                            elif "results" in m: b_data = m["results"]
                    except: pass

                if b_data:
                    raw_metrics = b_data.get("raw_metrics", b_data)
                    if "wer" in raw_metrics:
                        wer = (raw_metrics.get("wer") or 0.0) * 100
                        der = (raw_metrics.get("der") or 0.0) * 100
                        cer = (raw_metrics.get("cer") or 0.0) * 100
                        overall = raw_metrics.get("overall", (100.0 - wer) * 0.7 + (100.0 - der) * 0.3)
                        all_benchmarks.append({
                            "session": s, "overall": round(overall, 1),
                            "wer": round(wer, 1), "cer": round(cer, 1), "der": round(der, 1),
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
                            if isinstance(item, str): all_domain_terms.append(item)
                            elif isinstance(item, dict):
                                t_str = item.get("canonical") or item.get("term") or item.get("text") or ""
                                if t_str: all_domain_terms.append(t_str)
            except: pass

    st.markdown(
        '<div class="tech-stack-container" style="display: flex; align-items: center; justify-content: flex-start; flex-wrap: wrap; gap: 10px;">'
        '<div style="width:100%; font-family:\'IBM Plex Mono\',monospace; font-size:10px; letter-spacing:2px; text-transform:uppercase; color:#888; margin-bottom:4px;">Lõi công nghệ đang vận hành</div>'
        '<div class="tech-badge"><span>🎙️</span> Zipformer STT (16kHz)</div>'
        '<div class="tech-badge"><span>🧬</span> Diart (Pyannote 3.0)</div>'
        '<div class="tech-badge"><span>🪪</span> Campplus VoiceID Embedding</div>'
        '<div class="tech-badge"><span>📄</span> DeepseekOCR & Layout</div>'

        '</div>',
        unsafe_allow_html=True
    )

    row1_c1, row1_c2 = st.columns(2, gap="medium")
    with row1_c1:
        with st.container(border=True):
            st.markdown('<div class="rv-card-title">📋 Phiên làm việc gần nhất</div>', unsafe_allow_html=True)
            if latest_session:
                st.markdown(f'<div style="font-size:22px;font-weight:700;margin-bottom:4px;">{latest_session.replace("_", " ")}</div>', unsafe_allow_html=True)
                st.markdown(f'<div style="font-size:13px;color:#888;margin-bottom:20px;">📅 Ngày lưu: {meta_data.get("date", "Gần đây")}</div>', unsafe_allow_html=True)
                stats = meta_data.get("stats", {})
                if stats:
                    st.markdown('<div style="font-size:12px;font-weight:700;margin-bottom:12px;text-transform:uppercase;letter-spacing:1px;color:#888;">Độ sôi nổi người nói (Speaker Activity)</div>', unsafe_allow_html=True)
                    for spk, data in list(stats.items())[:3]:
                        pct = data.get("percent", 0)
                        st.markdown(f'''
                        <div class="bar-label-row"><span>{_html.escape(spk)}</span><span>{pct}%</span></div>
                        <div class="bar-bg"><div class="bar-fill" style="width:{pct}%; background: #4da6e8;"></div></div>
                        ''', unsafe_allow_html=True)
                st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
                st.page_link("pages/5_Review_sessions.py", label="Xem chi tiết biên bản phiên này", icon="📄")
            else:
                st.info("Hệ thống chưa lưu dữ liệu cuộc họp nào.")

    with row1_c2:
        with st.container(border=True):
            st.markdown('<div class="rv-card-title">📊 Kết quả Benchmark gần nhất (Tỷ lệ lỗi)</div>', unsafe_allow_html=True)
            if latest_bench:
                wer_val, cer_val, der_val, overall = latest_bench["wer"], latest_bench["cer"], latest_bench["der"], latest_bench["overall"]
                color = "#4ec94e" if overall >= 85 else "#e8c14d" if overall >= 70 else "#DC2626"
                st.markdown(f'''
                <div style="display:flex; align-items:center; gap:24px; margin-bottom:24px;">
                    <div>
                        <div style="font-family:'IBM Plex Mono',monospace; font-size:11px; letter-spacing:1px; color:#888;">OVERALL PERFORMANCE</div>
                        <div style="font-size:48px; font-weight:800; color:{color}; line-height:1;">{overall}</div>
                    </div>
                    <div style="padding-left:20px; border-left:1px solid rgba(128,128,128,0.2); font-size:13px; color:#888;">
                        Phiên đánh giá: <b>{latest_bench["session"].replace("_", " ")}</b><br>
                        Tỷ lệ lỗi càng thấp thanh hiển thị càng ngắn.
                    </div>
                </div>
                <div class="bar-label-row"><span>Tỷ lệ lỗi từ (WER)</span><span style="color:#DC2626;">{wer_val}%</span></div>
                <div class="bar-bg"><div class="bar-fill" style="width:{min(100.0, wer_val)}%; background:#DC2626;"></div></div>
                <div class="bar-label-row"><span>Tỷ lệ lỗi ký tự (CER)</span><span style="color:#f59e0b;">{cer_val}%</span></div>
                <div class="bar-bg"><div class="bar-fill" style="width:{min(100.0, cer_val)}%; background:#f59e0b;"></div></div>
                <div class="bar-label-row"><span>Lỗi phân tách người nói (DER)</span><span style="color:#4da6e8;">{der_val}%</span></div>
                <div class="bar-bg"><div class="bar-fill" style="width:{min(100.0, der_val)}%; background:#4da6e8;"></div></div>
                ''', unsafe_allow_html=True)
            else:
                st.info("Chưa tìm thấy lịch sử chạy kịch bản kiểm thử Benchmark.")

    st.markdown("<div style='margin:16px 0;'></div>", unsafe_allow_html=True)
    row2_c1, row2_c2 = st.columns(2, gap="medium")
    with row2_c1:
        with st.container(border=True):
            st.markdown('<div class="rv-card-title">📚 Thuật ngữ chuyên ngành (Domain Knowledge)</div>', unsafe_allow_html=True)
            st.markdown(f'''
            <div style="display:grid; grid-template-columns:repeat(2,1fr); gap:12px; margin-bottom:20px; text-align:center;">
                <div style="background:rgba(128,128,128,0.05); padding:12px; border-radius:6px; border:1px solid rgba(128,128,128,0.1);">
                    <div style="font-family:'IBM Plex Mono',monospace; font-size:11px; color:#888;">SỐ BỘ TỪ ĐIỂN</div>
                    <div style="font-size:24px; font-weight:700; color:#0067B1;">{len(domain_files)} file</div>
                </div>
                <div style="background:rgba(128,128,128,0.05); padding:12px; border-radius:6px; border:1px solid rgba(128,128,128,0.1);">
                    <div style="font-family:'IBM Plex Mono',monospace; font-size:11px; color:#888;">TỔNG THUẬT NGỮ</div>
                    <div style="font-size:24px; font-weight:700; color:#4da6e8;">{len(all_domain_terms)} từ</div>
                </div>
            </div>
            <div style="font-size:12px;font-weight:700;margin-bottom:12px;text-transform:uppercase;letter-spacing:1px;color:#888;">3 Thuật ngữ đầu tiên trong kho dữ liệu</div>
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
            st.page_link("pages/4_Domain_terms.py", label="Quản lý từ điển chuyên ngành", icon="📚")

    with row2_c2:
        with st.container(border=True):
            st.markdown('<div class="rv-card-title">🏆 Top 3 Phiên Có Điểm Số Cao Nhất</div>', unsafe_allow_html=True)
            st.page_link("pages/6_Benchmark_sessions.py", label="Đánh giá chất lượng toàn diện", icon="📊")
            if top3_sessions:
                medal_colors = ["#e8c14d", "#a0aab2", "#cd7f32"]
                for i, bm in enumerate(top3_sessions):
                    st.markdown(f'''
                    <div class="rank-item">
                        <div class="rank-num" style="color:{medal_colors[i]};">{i+1}</div>
                        <div class="rank-info">
                            <div class="rank-name">{bm["session"].replace("_", " ")}</div>
                            <div style="font-size:12px; color:#888;">WER: {bm["wer"]}% · CER: {bm["cer"]}% · DER: {bm["der"]}%</div>
                        </div>
                        <div class="rank-score" style="color:{medal_colors[i]};">{bm["overall"]}</div>
                    </div>
                    ''', unsafe_allow_html=True)
            else:
                st.info("Chưa có dữ liệu kiểm thử Benchmark nào được lưu lại để xếp hạng.")


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
        hist.sort(key=lambda x: x["id"], reverse=True)
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
            <div class="vnpt-stat-card">
                <div class="vnpt-stat-top total"></div>
                <div class="vnpt-stat-body">
                    <div class="vnpt-stat-head">
                        <div class="vnpt-metric-icon total">{icon('layers', 18)}</div>
                    </div>
                    <div class="vnpt-stat-value">{total_files}</div>
                    <div class="vnpt-stat-label">Tổng số tài liệu</div>
                </div>
            </div>
            """, unsafe_allow_html=True
        )
    with c2:
        badge_success = (
            f'<span class="vnpt-stat-share success">{success_count / total_files * 100:.0f}%</span>'
            if total_files else ''
        )
        st.markdown(
            f"""
            <div class="vnpt-stat-card">
                <div class="vnpt-stat-top success"></div>
                <div class="vnpt-stat-body">
                    <div class="vnpt-stat-head">
                        <div class="vnpt-metric-icon success">{icon('check', 18)}</div>{badge_success}
                    </div>
                    <div class="vnpt-stat-value">{success_count}</div>
                    <div class="vnpt-stat-label">Xử lý thành công</div>
                </div>
            </div>
            """, unsafe_allow_html=True
        )
    with c3:
        badge_failed = (
            f'<span class="vnpt-stat-share failed">{failed_count / total_files * 100:.0f}%</span>'
            if total_files else ''
        )
        st.markdown(
            f"""
            <div class="vnpt-stat-card">
                <div class="vnpt-stat-top failed"></div>
                <div class="vnpt-stat-body">
                    <div class="vnpt-stat-head">
                        <div class="vnpt-metric-icon failed">{icon('cross', 18)}</div>{badge_failed}
                    </div>
                    <div class="vnpt-stat-value">{failed_count}</div>
                    <div class="vnpt-stat-label">Xử lý thất bại</div>
                </div>
            </div>
            """, unsafe_allow_html=True
        )

    if total_files > 0:
        st.markdown(
            f"""
            <div class="vnpt-kpi-banner">
                <div class="vnpt-kpi-icon">{icon('gauge', 22, 2.0, 'white')}</div>
                <div class="vnpt-kpi-text">
                    <div class="vnpt-kpi-label">Tỷ lệ xử lý thành công</div>
                    <div class="vnpt-kpi-value">{success_rate:.1f}%</div>
                </div>
                <div class="vnpt-kpi-track"><div class="vnpt-kpi-fill" style="width:{success_rate:.1f}%;"></div></div>
            </div>
            """, unsafe_allow_html=True
        )

    def _build_activity(history_list: list[dict], max_days: int = 7) -> list[tuple[str, int]]:
        counts: dict[str, int] = {}
        for entry in history_list:
            try: d = datetime.fromisoformat(entry.get("started_at", "")).date()
            except Exception: continue
            key = d.strftime("%d/%m")
            counts[key] = counts.get(key, 0) + 1
        ordered_keys = sorted(counts.keys(), key=lambda k: datetime.strptime(k, "%d/%m"))
        return [(k, counts[k]) for k in ordered_keys[-max_days:]]

    activity = _build_activity(history)
    if activity:
        with st.container(border=True):
            st.markdown(f'<span class="vnpt-pill-label">{icon("layers", 14)} Hoạt động gần đây</span>', unsafe_allow_html=True)
            max_count = max(c for _, c in activity) or 1
            rows_html = "".join(
                f'<div class="vnpt-activity-row">'
                f'<div class="vnpt-activity-label">{day}</div>'
                f'<div class="vnpt-activity-track"><div class="vnpt-activity-fill" style="width:{c / max_count * 100:.0f}%;"></div></div>'
                f'<div class="vnpt-activity-count">{c}</div>'
                f'</div>'
                for day, c in activity
            )
            st.markdown(rows_html, unsafe_allow_html=True)

    st.write("")
    st.subheader("Chi tiết phiên xử lý")

    if total_files > 0:
        tb1, tb2 = st.columns([3, 1])
        with tb1:
            st.markdown('<div class="vnpt-toolbar-label">Tìm kiếm theo tên file</div>', unsafe_allow_html=True)
            search_q = st.text_input("Tìm kiếm", placeholder="Nhập tên file...", label_visibility="collapsed", key="dash_search")
        with tb2:
            st.markdown('<div class="vnpt-toolbar-label">Tình trạng</div>', unsafe_allow_html=True)
            status_filter = st.selectbox("Tình trạng", options=["Tất cả", "Thành công", "Thất bại"], label_visibility="collapsed", key="dash_status_filter")
    else:
        search_q, status_filter = "", "Tất cả"

    st.write("")

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

            status_html = '<span class="vnpt-status"><span class="vnpt-dot vnpt-dot-ok"></span>Thành công</span>' if entry.get("exit_code", -1) == 0 else '<span class="vnpt-status"><span class="vnpt-dot vnpt-dot-fail"></span>Thất bại</span>'

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
            <div class="vnpt-table-wrap">
              <table class="vnpt-table">
                <thead>
                  <tr>
                    <th>STT</th>
                    <th>Tên file</th>
                    <th>Ngày xử lý</th>
                    <th>Số trang</th>
                    <th>Tình trạng</th>
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