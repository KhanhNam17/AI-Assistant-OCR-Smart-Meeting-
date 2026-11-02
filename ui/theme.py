"""
ui/theme.py
===========

Modern Blue Visual Design System for "Smart AI Workspace"
Fully responsive to BOTH Light and Dark themes.

Features:
- Dynamic palette adapting to Streamlit's native Light / Dark modes
- Primary vibrant royal blue (#2563EB) accents
- Card backgrounds using var(--secondary-background-color)
- Text and borders automatically adapting to user theme
- High-contrast royal navy sidebar
- Clean status badges, metric cards, and responsive tables
"""

from __future__ import annotations

from pathlib import Path
import streamlit as st

# ══════════════════════════════════════════════════════════════════
# 1. COLOR TOKENS
# ══════════════════════════════════════════════════════════════════

NAVY           = "#0F172A"
BLUE_DEEP      = "#1E3A8A"
BLUE_DARK      = "#1E40AF"
BLUE           = "#2563EB"
BLUE_LIGHT     = "#60A5FA"
CYAN           = "#06B6D4"
GREEN          = "#10B981"
AMBER          = "#F59E0B"
RED            = "#EF4444"

GRADIENT_BRAND = f"linear-gradient(135deg, {BLUE} 0%, {BLUE_DARK} 100%)"
GRADIENT_HERO  = f"linear-gradient(135deg, {BLUE_DEEP} 0%, {BLUE} 100%)"

PAGE_ICON = "🎙️"

_NAV_ITEMS = [
    ("app.py",                        "Trang chủ",            "home"),
    ("pages/1_Dashboard.py",          "Dashboard",            "space_dashboard"),
    ("pages/2_Workspace.py",          "Không gian làm việc",  "apps"),
    ("pages/3_VoiceID_creating.py",   "VoiceID",              "badge"),
    ("pages/4_Domain_terms.py",       "Domain Terms",         "menu_book"),
    ("pages/5_Review_sessions.py",    "Review Sessions",      "folder_open"),
    ("pages/6_Benchmark_sessions.py", "Benchmark",            "monitoring"),
    ("pages/7_Ground_truth_creator.py", "GT Creator",         "fact_check"),
]

_MATERIAL_ICONS = {
    "app.py": "home",
    "pages/1_Dashboard.py": "space_dashboard",
    "pages/2_Workspace.py": "apps",
    "pages/3_VoiceID_creating.py": "badge",
    "pages/4_Domain_terms.py": "menu_book",
    "pages/5_Review_sessions.py": "folder_open",
    "pages/6_Benchmark_sessions.py": "monitoring",
    "pages/7_Ground_truth_creator.py": "fact_check",
}

WORKSPACE_MEETING_URL = "Workspace?mode=meeting"
WORKSPACE_OCR_URL     = "Workspace?mode=ocr"

# ══════════════════════════════════════════════════════════════════
# 2. VECTOR ICONS (SVG, currentColor)
# ══════════════════════════════════════════════════════════════════

_ICONS: dict[str, str] = {
    "check":        '<path d="M4 12.5 9.5 18 20 6"/>',
    "cross":        '<path d="M6 6l12 12M18 6 6 18"/>',
    "alert":        '<path d="M12 3.2 22 20.2H2Z"/><path d="M12 9.5v4.2M12 16.8h.01"/>',
    "info":         '<circle cx="12" cy="12" r="9"/><path d="M12 11v5.5M12 7.6h.01"/>',
    "globe":        '<circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3c2.6 2.6 3.9 6 3.9 9s-1.3 6.4-3.9 9c-2.6-2.6-3.9-6-3.9-9S9.4 5.6 12 3Z"/>',
    "zap":          '<path d="M13 3 4 14h6l-1 7 9-11h-6l1-7Z"/>',
    "layers":       '<path d="M12 3 3 8l9 5 9-5-9-5Z"/><path d="M3 13l9 5 9-5M3 18l9 5 9-5"/>',
    "cpu":          '<rect x="7" y="7" width="10" height="10" rx="1.5"/><path d="M9 3v3M12 3v3M15 3v3M9 18v3M12 18v3M15 18v3M3 9h3M3 12h3M3 15h3M18 9h3M18 12h3M18 15h3"/>',
    "table":        '<rect x="3.5" y="4.5" width="17" height="15" rx="1.5"/><path d="M3.5 9.5h17M9.5 9.5V19.5"/>',
    "grid":         '<rect x="4" y="4" width="6.5" height="6.5" rx="1.4"/><rect x="13.5" y="4" width="6.5" height="6.5" rx="1.4"/><rect x="4" y="13.5" width="6.5" height="6.5" rx="1.4"/><rect x="13.5" y="13.5" width="6.5" height="6.5" rx="1.4"/>',
    "scan":         '<path d="M4 9V6.5A1.5 1.5 0 0 1 5.5 5H8"/><path d="M4 15v2.5A1.5 1.5 0 0 0 5.5 19H8"/><path d="M20 9V6.5A1.5 1.5 0 0 0 18.5 5H16"/><path d="M20 15v2.5a1.5 1.5 0 0 1-1.5 1.5H16"/><path d="M4 12h16"/>',
    "image":        '<rect x="3.5" y="4.5" width="17" height="15" rx="1.5"/><circle cx="9" cy="10" r="1.6"/><path d="M4 17.5 9.5 13 13 16l3-3.2 4 4.2"/>',
    "docx":         '<path d="M6 3h8l4 4v14H6z"/><path d="M14 3v4h4"/><path d="M8.5 13l1.2 5 1.4-4 1.4 4 1.2-5"/>',
    "arrow":        '<path d="M4 12h15M13 6l6 6-6 6"/>',
    "plus":         '<path d="M12 5v14M5 12h14"/>',
    "trash":        '<path d="M4 7h16M9 7V4.5A1.5 1.5 0 0 1 10.5 3h3A1.5 1.5 0 0 1 15 4.5V7M6 7l1 13a1.5 1.5 0 0 0 1.5 1.4h7a1.5 1.5 0 0 0 1.5-1.4l1-13"/>',
    "chevron-up":   '<path d="M5 15l7-7 7 7"/>',
    "chevron-down": '<path d="M5 9l7 7 7-7"/>',
    "flag":         '<path d="M6 3v18"/><path d="M6 4.5h11l-2.5 4L17 12.5H6"/>',
    "package":      '<path d="M12 3 3.5 7.5v9L12 21l8.5-4.5v-9L12 3Z"/><path d="M3.5 7.5 12 12l8.5-4.5M12 12v9"/>',
    "gauge":        '<path d="M4 15a8 8 0 1 1 16 0"/><path d="M12 15l4.5-5.5"/><path d="M12 15h.01"/>',
    "folder":       '<path d="M4 6.5A1.5 1.5 0 0 1 5.5 5h4l2 2.5h7A1.5 1.5 0 0 1 20 9v9a1.5 1.5 0 0 1-1.5 1.5h-13A1.5 1.5 0 0 1 4 18Z"/>',
    "compare":      '<path d="M9 3v18M15 3v18"/><path d="M4 8h5M4 16h5M15 8h5M15 16h5"/>',
    "download":     '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/>',
    "sparkles":     '<path d="m12 3-1.9 5.8a2 2 0 0 1-1.3 1.3L3 12l5.8 1.9a2 2 0 0 1 1.3 1.3L12 21l1.9-5.8a2 2 0 0 1 1.3-1.3L21 12l-5.8-1.9a2 2 0 0 1-1.3-1.3L12 3z"/>',
    "mic":          '<path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" y1="19" x2="12" y2="22"/>',
}

def icon(name: str, size: int = 16, stroke: float = 1.9, color: str = "currentColor") -> str:
    """Inline SVG for use inside HTML fragments."""
    body = _ICONS.get(name, "")
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
        f'stroke="{color}" stroke-width="{stroke}" stroke-linecap="round" '
        f'stroke-linejoin="round" style="display:inline-block;vertical-align:-3px;flex-shrink:0">{body}</svg>'
    )

# ══════════════════════════════════════════════════════════════════
# 3. PAGE BOOTSTRAP
# ══════════════════════════════════════════════════════════════════

def apply_page_config(title: str, ui_root: Path | None = None) -> None:
    """st.set_page_config with modern styling defaults."""
    st.set_page_config(
        page_title=f"{title} · Smart Assistant",
        page_icon=PAGE_ICON,
        layout="wide",
        initial_sidebar_state="expanded",
    )

# ══════════════════════════════════════════════════════════════════
# 4. BASE CSS — Dynamic Light & Dark Theme Design System
# ══════════════════════════════════════════════════════════════════

def inject_base_css() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

        :root {{
            --app-navy: {NAVY};
            --app-blue-deep: {BLUE_DEEP};
            --app-blue-dark: {BLUE_DARK};
            --app-blue: {BLUE};
            --app-blue-light: {BLUE_LIGHT};
            --app-blue-surface: rgba(37, 99, 235, 0.08);
            --app-blue-tint: rgba(37, 99, 235, 0.12);
            --app-blue-tint-2: rgba(37, 99, 235, 0.2);
            --app-cyan: {CYAN};
            --app-green: {GREEN};
            --app-green-tint: rgba(16, 185, 129, 0.15);
            --app-amber: {AMBER};
            --app-amber-tint: rgba(245, 158, 11, 0.15);
            --app-red: {RED};
            --app-red-tint: rgba(239, 68, 68, 0.15);

            --app-radius-xl: 16px;
            --app-radius-lg: 12px;
            --app-radius-md: 10px;
            --app-radius-sm: 6px;
            --app-shadow-sm: 0 1px 3px rgba(0, 0, 0, 0.06);
            --app-shadow-md: 0 4px 14px rgba(0, 0, 0, 0.09);
        }}

        /* ── Typography & Global Reset ────────────────────────── */
        html, body, [class*="css"] {{
            font-family: 'Plus Jakarta Sans', system-ui, -apple-system, sans-serif !important;
            -webkit-font-smoothing: antialiased;
        }}
        code, pre, .mono {{
            font-family: 'JetBrains Mono', monospace !important;
        }}

        /* Hide Streamlit auto-anchor link icons (Fixes Image 1) */
        a.anchor-link,
        [data-testid="stMarkdownContainer"] a.anchor-link {{
            display: none !important;
        }}

        [data-testid="stSidebarNav"] {{ display: none !important; }}
        footer {{ visibility: hidden; }}

        /* Top Spacing: prevents Streamlit navbar/header from clipping page title (Fixes Image 2) */
        .block-container {{
            max-width: 1400px;
            padding-top: 5rem !important;
            padding-bottom: 4rem !important;
        }}

        /* ── Adaptive Card Containers ─────────────────────────── */
        [data-testid="stVerticalBlockBorderWrapper"] {{
            background: var(--secondary-background-color) !important;
            border-radius: var(--app-radius-xl) !important;
            border: 1px solid rgba(148, 163, 184, 0.22) !important;
            box-shadow: var(--app-shadow-sm) !important;
            color: var(--text-color) !important;
            transition: all 0.22s ease-in-out !important;
            padding: 4px;
        }}
        [data-testid="stVerticalBlockBorderWrapper"]:hover {{
            border-color: rgba(37, 99, 235, 0.4) !important;
            box-shadow: var(--app-shadow-md) !important;
        }}

        /* ── Adaptive Buttons ─────────────────────────────────── */
        .stButton > button,
        .stDownloadButton > button {{
            border-radius: var(--app-radius-md) !important;
            font-weight: 600 !important;
            font-size: 0.95rem !important;
            letter-spacing: 0.2px !important;
            transition: all 0.18s ease-in-out !important;
        }}
        .stButton > button[kind="primary"],
        .stDownloadButton > button[kind="primary"] {{
            background: {GRADIENT_BRAND} !important;
            border: none !important;
            color: #FFFFFF !important;
            box-shadow: 0 4px 14px rgba(37, 99, 235, 0.35) !important;
        }}
        .stButton > button[kind="primary"]:hover:not(:disabled),
        .stDownloadButton > button[kind="primary"]:hover:not(:disabled) {{
            filter: brightness(1.08) !important;
            box-shadow: 0 6px 20px rgba(37, 99, 235, 0.45) !important;
            transform: translateY(-1px) !important;
        }}
        .stButton > button[kind="secondary"],
        .stDownloadButton > button[kind="secondary"] {{
            background: var(--secondary-background-color) !important;
            border: 1px solid rgba(148, 163, 184, 0.3) !important;
            color: var(--text-color) !important;
            box-shadow: var(--app-shadow-sm) !important;
        }}
        .stButton > button[kind="secondary"]:hover:not(:disabled),
        .stDownloadButton > button[kind="secondary"]:hover:not(:disabled) {{
            border-color: var(--app-blue) !important;
            color: var(--app-blue) !important;
            background: var(--app-blue-surface) !important;
        }}
        .stButton > button:active:not(:disabled) {{
            transform: scale(0.985) !important;
        }}
        .stDownloadButton > button {{ width: 100%; }}

        /* ── Form Inputs ──────────────────────────────────────── */
        input, select, textarea, [data-baseweb="select"] {{
            border-radius: var(--app-radius-md) !important;
        }}

        /* ── Adaptive Tabs ────────────────────────────────────── */
        [data-testid="stTabs"] button[role="tab"] {{
            font-weight: 600 !important;
            font-size: 14.5px !important;
            color: var(--text-color) !important;
            opacity: 0.7;
            padding: 8px 18px !important;
            border-radius: var(--app-radius-md) !important;
            transition: all 0.15s ease !important;
        }}
        [data-testid="stTabs"] button[role="tab"][aria-selected="true"] {{
            color: {BLUE} !important;
            opacity: 1;
            background: var(--app-blue-surface) !important;
            border-bottom: 2px solid {BLUE} !important;
        }}

        /* ── Sidebar (High-Contrast Royal Navy in both Light & Dark) ── */
        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #1E3A8A 0%, #0F172A 100%) !important;
            border-right: 1px solid rgba(255, 255, 255, 0.12) !important;
            color: #FFFFFF !important;
        }}
        [data-testid="stSidebar"] [data-testid="stPageLink"] {{
            border-radius: var(--app-radius-md) !important;
            margin-bottom: 4px !important;
            padding: 4px 10px !important;
            transition: all 0.15s ease !important;
        }}
        [data-testid="stSidebar"] [data-testid="stPageLink"] p {{
            font-size: 13.5px !important;
            font-weight: 600 !important;
            color: rgba(255, 255, 255, 0.88) !important;
        }}
        [data-testid="stSidebar"] [data-testid="stPageLink"] svg {{
            fill: rgba(255, 255, 255, 0.88) !important;
            color: rgba(255, 255, 255, 0.88) !important;
        }}
        [data-testid="stSidebar"] [data-testid="stPageLink"]:hover {{
            background: rgba(255, 255, 255, 0.14) !important;
            transform: translateX(3px) !important;
        }}
        [data-testid="stSidebar"] [data-testid="stPageLink"]:hover p {{
            color: #FFFFFF !important;
        }}
        [data-testid="stSidebar"] hr {{
            border-color: rgba(255, 255, 255, 0.15) !important;
            margin: 16px 0 !important;
        }}

        /* High-contrast code & debug box in sidebar (Fixes Image 3) */
        [data-testid="stSidebar"] code {{
            background: rgba(0, 0, 0, 0.35) !important;
            color: #93C5FD !important;
            border: 1px solid rgba(255, 255, 255, 0.18) !important;
            padding: 2px 6px !important;
            border-radius: 4px !important;
            word-break: break-all !important;
            font-size: 11px !important;
        }}
        [data-testid="stSidebar"] .app-debug-box {{
            color: rgba(255, 255, 255, 0.88) !important;
            font-size: 11px;
            line-height: 1.6;
            padding: 10px 12px;
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.15);
            border-radius: 8px;
            margin-top: 8px;
        }}

        .brand-header {{
            padding: 10px 4px 16px;
        }}
        .brand-kicker {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 10px;
            letter-spacing: 2.5px;
            text-transform: uppercase;
            color: #93C5FD;
            font-weight: 700;
            margin-bottom: 6px;
            display: flex;
            align-items: center;
            gap: 6px;
        }}
        .brand-title {{
            font-size: 18px;
            font-weight: 800;
            color: #FFFFFF;
            letter-spacing: -0.4px;
            line-height: 1.3;
        }}
        .brand-sub {{
            font-size: 11px;
            color: rgba(255, 255, 255, 0.7);
            margin-top: 4px;
            font-weight: 600;
            letter-spacing: 0.4px;
        }}

        .status-panel-glass {{
            border: 1px solid rgba(255, 255, 255, 0.16);
            border-radius: var(--app-radius-lg);
            background: rgba(255, 255, 255, 0.08);
            backdrop-filter: blur(8px);
            padding: 14px 16px;
            margin-top: 6px;
            color: #FFFFFF;
        }}
        .status-row-glass {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 10px;
        }}
        .status-label-glass {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 10px;
            letter-spacing: 1.5px;
            font-weight: 800;
            color: rgba(255, 255, 255, 0.7);
        }}
        .status-mini-glass {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            font-size: 11.5px;
            color: rgba(255, 255, 255, 0.75);
            padding: 3px 0;
        }}
        .status-mini-glass b {{
            font-weight: 700;
            color: #FFFFFF;
        }}

        /* ── Adaptive Badges, Chips & Pills ───────────────────── */
        .app-badge {{
            display: inline-flex; align-items: center; gap: 6px;
            padding: 4px 12px; border-radius: 9999px;
            font-size: 12px; font-weight: 700; white-space: nowrap;
            letter-spacing: 0.2px;
        }}
        .app-badge-on {{
            background: rgba(16, 185, 129, 0.15);
            color: #10B981;
            border: 1px solid rgba(16, 185, 129, 0.3);
        }}
        .app-badge-off {{
            background: rgba(239, 68, 68, 0.15);
            color: #EF4444;
            border: 1px solid rgba(239, 68, 68, 0.3);
        }}
        .app-badge-info {{
            background: rgba(37, 99, 235, 0.15);
            color: #3B82F6;
            border: 1px solid rgba(37, 99, 235, 0.3);
        }}
        .app-badge-neutral {{
            background: rgba(148, 163, 184, 0.15);
            color: var(--text-color);
            opacity: 0.85;
            border: 1px solid rgba(148, 163, 184, 0.25);
        }}

        .app-pill-label {{
            display: inline-flex; align-items: center; gap: 8px;
            padding: 6px 14px; border-radius: 9999px;
            background: var(--app-blue-surface);
            color: var(--app-blue);
            font-weight: 700; font-size: 12.5px;
            letter-spacing: 0.3px; margin-bottom: 14px;
            border: 1px solid rgba(37, 99, 235, 0.25);
        }}

        .app-tag {{
            font-size: 11px; padding: 4px 12px; border-radius: var(--app-radius-sm);
            background: rgba(148, 163, 184, 0.12);
            border: 1px solid rgba(148, 163, 184, 0.25);
            color: var(--text-color);
            font-weight: 600; letter-spacing: 0.2px;
        }}

        /* ── Status Dot ───────────────────────────────────────── */
        .app-dot {{
            width: 8px; height: 8px; border-radius: 50%; display: inline-block; flex-shrink: 0;
        }}
        .app-dot-ok {{
            background: #10B981;
            box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.25);
        }}
        .app-dot-fail {{
            background: #EF4444;
            box-shadow: 0 0 0 3px rgba(239, 68, 68, 0.25);
        }}

        /* ── Adaptive Metric & Stat Cards ─────────────────────── */
        .app-stat-card {{
            border-radius: var(--app-radius-xl);
            border: 1px solid rgba(148, 163, 184, 0.22);
            background: var(--secondary-background-color);
            box-shadow: var(--app-shadow-sm);
            overflow: hidden;
            height: 100%;
            transition: all 0.2s ease;
        }}
        .app-stat-card:hover {{
            box-shadow: var(--app-shadow-md);
            border-color: rgba(37, 99, 235, 0.4);
            transform: translateY(-2px);
        }}
        .app-stat-top {{ height: 4px; }}
        .app-stat-top.total   {{ background: {BLUE}; }}
        .app-stat-top.success {{ background: {GREEN}; }}
        .app-stat-top.failed  {{ background: {RED}; }}
        .app-stat-body {{ padding: 22px 24px; }}
        .app-stat-head {{
            display: flex; align-items: center; justify-content: space-between; margin-bottom: 16px;
        }}
        .app-metric-icon {{
            width: 44px; height: 44px; border-radius: 12px;
            display: flex; align-items: center; justify-content: center;
        }}
        .app-metric-icon.total {{
            background: rgba(37, 99, 235, 0.12); color: {BLUE};
        }}
        .app-metric-icon.success {{
            background: rgba(16, 185, 129, 0.14); color: {GREEN};
        }}
        .app-metric-icon.failed {{
            background: rgba(239, 68, 68, 0.14); color: {RED};
        }}
        .app-stat-share {{
            font-size: 11.5px; font-weight: 700; padding: 4px 10px; border-radius: 9999px;
        }}
        .app-stat-share.success {{
            background: rgba(16, 185, 129, 0.15); color: #10B981;
        }}
        .app-stat-share.failed {{
            background: rgba(239, 68, 68, 0.15); color: #EF4444;
        }}
        .app-stat-value {{
            font-size: 36px; font-weight: 800; color: var(--text-color);
            line-height: 1.1; margin-bottom: 6px; letter-spacing: -0.5px;
        }}
        .app-stat-label {{
            font-size: 13px; font-weight: 600; color: var(--text-color); opacity: 0.7;
        }}

        /* ── Adaptive KPI Banner ──────────────────────────────── */
        .app-kpi-banner {{
            background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 100%);
            border-radius: var(--app-radius-xl);
            padding: 22px 28px;
            color: #FFFFFF;
            display: flex; align-items: center; gap: 24px; flex-wrap: wrap;
            box-shadow: 0 10px 25px rgba(37, 99, 235, 0.28);
            margin: 18px 0 24px;
        }}
        .app-kpi-icon {{
            width: 48px; height: 48px; border-radius: 12px;
            background: rgba(255, 255, 255, 0.16);
            display: flex; align-items: center; justify-content: center;
        }}
        .app-kpi-label {{
            font-size: 11.5px; font-weight: 700; letter-spacing: 0.8px;
            text-transform: uppercase; color: rgba(255, 255, 255, 0.85);
        }}
        .app-kpi-value {{
            font-size: 28px; font-weight: 800; margin-top: 2px; color: #FFFFFF;
        }}
        .app-kpi-track {{
            flex: 1; min-width: 160px; height: 10px; border-radius: 9999px;
            background: rgba(255, 255, 255, 0.25); overflow: hidden;
        }}
        .app-kpi-fill {{
            height: 100%; border-radius: 9999px;
            background: #34D399;
        }}

        /* ── Adaptive Tables ──────────────────────────────────── */
        .app-table-wrap {{
            border: 1px solid rgba(148, 163, 184, 0.22);
            border-radius: var(--app-radius-xl);
            overflow: hidden;
            background: var(--secondary-background-color);
            box-shadow: var(--app-shadow-sm);
        }}
        .app-table {{
            width: 100%; border-collapse: collapse; font-size: 13.5px;
        }}
        .app-table thead th {{
            text-align: left; font-size: 11px; letter-spacing: 1.2px;
            text-transform: uppercase; color: var(--text-color); opacity: 0.75; font-weight: 700;
            padding: 16px 22px; background: rgba(148, 163, 184, 0.08);
            border-bottom: 1px solid rgba(148, 163, 184, 0.2); white-space: nowrap;
        }}
        .app-table tbody td {{
            padding: 14px 22px; border-bottom: 1px solid rgba(148, 163, 184, 0.12);
            color: var(--text-color); vertical-align: middle;
        }}
        .app-table tbody tr:last-child td {{
            border-bottom: none;
        }}
        .app-table tbody tr:hover td {{
            background: var(--app-blue-surface);
        }}
        .app-table .col-stt {{
            width: 56px; font-weight: 700; color: var(--text-color); opacity: 0.5;
        }}
        .app-table .col-file {{ font-weight: 700; }}
        .app-table .col-date {{
            color: var(--text-color); opacity: 0.7; white-space: nowrap; font-variant-numeric: tabular-nums;
        }}
        .app-status {{
            display: inline-flex; align-items: center; gap: 8px;
            font-weight: 700; white-space: nowrap; font-size: 13px;
        }}

        /* ── Page Header ──────────────────────────────────────── */
        .app-page-header {{ margin-bottom: 16px; }}
        .app-page-eyebrow {{
            display: inline-flex; align-items: center; gap: 7px;
            font-size: 11px; font-weight: 800; letter-spacing: 1.6px;
            text-transform: uppercase; color: var(--app-blue);
            background: var(--app-blue-surface); padding: 5px 14px;
            border-radius: 9999px; margin-bottom: 12px;
            border: 1px solid rgba(37, 99, 235, 0.25);
        }}
        .app-page-title {{
            margin: 0; padding: 0; font-size: 28px; font-weight: 800; line-height: 1.25;
            color: var(--text-color); letter-spacing: -0.5px;
        }}
        .app-page-subtitle {{
            font-size: 14px; color: var(--text-color); opacity: 0.72;
            margin-top: 8px; font-weight: 500; max-width: 680px; line-height: 1.6;
        }}

        /* ── Activity Row ─────────────────────────────────────── */
        .app-activity-row {{
            display: flex; align-items: center; gap: 14px; padding: 8px 0;
        }}
        .app-activity-label {{
            width: 84px; flex-shrink: 0; font-size: 12px; color: var(--text-color); opacity: 0.75;
            font-variant-numeric: tabular-nums; font-weight: 600;
        }}
        .app-activity-track {{
            flex: 1; height: 10px; border-radius: 9999px;
            background: rgba(148, 163, 184, 0.15); overflow: hidden;
        }}
        .app-activity-fill {{
            height: 100%; border-radius: 9999px;
            background: {BLUE};
        }}
        .app-activity-count {{
            width: 28px; flex-shrink: 0; text-align: right;
            font-size: 13px; font-weight: 800; color: {BLUE};
        }}

        /* ── Adaptive Benchmark / Score Cards ─────────────────── */
        .app-metric-card {{
            background: var(--secondary-background-color);
            padding: 22px 20px; border-radius: var(--app-radius-xl);
            border: 1px solid rgba(148, 163, 184, 0.22);
            text-align: center;
            box-shadow: var(--app-shadow-sm);
            height: 100%;
            transition: all 0.2s ease;
        }}
        .app-metric-card:hover {{
            box-shadow: var(--app-shadow-md);
            border-color: rgba(37, 99, 235, 0.4);
        }}
        .app-metric-card .m-val {{
            font-size: 32px; font-weight: 800; color: {BLUE}; margin-bottom: 4px;
        }}
        .app-metric-card .m-lbl {{
            font-size: 11.5px; letter-spacing: 1px; color: var(--text-color); opacity: 0.7;
            text-transform: uppercase; font-weight: 700;
        }}

        .app-score-card {{
            background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 100%);
            color: white; border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: var(--app-radius-xl);
            padding: 24px; text-align: center;
            box-shadow: 0 10px 25px rgba(37, 99, 235, 0.25);
        }}
        .app-score-card .m-lbl {{ color: rgba(255, 255, 255, 0.85); }}
        .app-score-card .m-val {{ font-size: 48px; font-weight: 800; color: #FFFFFF; }}

        /* ── Adaptive Block Cards ─────────────────────────────── */
        .app-block-card {{
            border: 1px solid rgba(148, 163, 184, 0.22);
            border-left: 4px solid {BLUE};
            border-radius: var(--app-radius-lg);
            padding: 18px 20px;
            margin-bottom: 14px;
            background: var(--secondary-background-color);
            box-shadow: var(--app-shadow-sm);
            transition: all 0.18s ease;
        }}
        .app-block-card:hover {{
            box-shadow: var(--app-shadow-md);
        }}
        .app-block-order {{
            display: inline-flex; align-items: center; justify-content: center;
            width: 28px; height: 28px; border-radius: 50%;
            background: var(--app-blue-surface); color: {BLUE};
            font-weight: 800; font-size: 13px; margin-right: 8px;
        }}
        .app-section-title {{
            display: flex; align-items: center; gap: 9px;
            font-size: 17px; font-weight: 800; color: var(--text-color);
            margin: 6px 0 16px;
        }}
        .app-section-title svg {{ color: {BLUE}; }}

        /* ── Adaptive Benchmark Criteria Table ────────────────── */
        .app-bench-group {{
            border: 1px solid rgba(148, 163, 184, 0.22);
            border-radius: var(--app-radius-xl);
            background: var(--secondary-background-color);
            box-shadow: var(--app-shadow-sm);
            margin-bottom: 20px;
            overflow: hidden;
        }}
        .app-bench-group-head {{
            display: flex; align-items: center; gap: 12px;
            padding: 16px 22px;
            background: rgba(148, 163, 184, 0.08);
            border-bottom: 1px solid rgba(148, 163, 184, 0.18);
        }}
        .app-bench-group-icon {{
            width: 34px; height: 34px; border-radius: 10px; flex-shrink: 0;
            background: var(--app-blue-surface); color: {BLUE};
            display: flex; align-items: center; justify-content: center;
        }}
        .app-bench-group-title {{
            font-weight: 800; font-size: 15px; color: var(--text-color);
        }}
        .app-bench-group-count {{
            margin-left: auto; font-size: 12px; font-weight: 700;
            color: var(--text-color); opacity: 0.65;
        }}
        .app-bench-header {{
            display: grid; grid-template-columns: 44px 2.2fr 1fr 100px 130px;
            gap: 14px; padding: 11px 22px;
            font-size: 10.5px; font-weight: 800; letter-spacing: 0.8px; text-transform: uppercase;
            color: var(--text-color); opacity: 0.6; background: transparent;
            border-bottom: 1px solid rgba(148, 163, 184, 0.18);
        }}
        .app-bench-rows {{ display: flex; flex-direction: column; }}
        .app-bench-row {{
            display: grid;
            grid-template-columns: 44px 2.2fr 1fr 100px 130px;
            align-items: center; gap: 14px;
            padding: 14px 22px;
            border-bottom: 1px solid rgba(148, 163, 184, 0.1);
            border-left: 3px solid transparent;
            transition: background .15s ease;
        }}
        .app-bench-row:last-child {{ border-bottom: none; }}
        .app-bench-row:hover {{ background: var(--app-blue-surface); }}
        .app-bench-row.pass {{ border-left-color: {GREEN}; }}
        .app-bench-row.fail {{ border-left-color: {RED}; }}
        .app-bench-row.na   {{ border-left-color: rgba(148, 163, 184, 0.3); }}

        .bench-id {{ font-weight: 800; font-size: 12px; color: var(--text-color); opacity: 0.5; }}
        .bench-crit-name {{ font-weight: 700; font-size: 13.5px; color: var(--text-color); line-height: 1.4; }}
        .bench-crit-threshold {{ font-size: 11.5px; color: var(--text-color); opacity: 0.6; margin-top: 2px; }}
        .bench-value {{ font-weight: 800; font-size: 15px; font-variant-numeric: tabular-nums; }}
        .app-bench-row.pass .bench-value {{ color: {GREEN}; }}
        .app-bench-row.fail .bench-value {{ color: {RED}; }}
        .app-bench-row.na .bench-value {{ opacity: 0.5; }}

        .bench-priority {{
            font-size: 10.5px; font-weight: 800; letter-spacing: 0.4px; text-transform: uppercase;
            padding: 3px 10px; border-radius: 9999px; text-align: center; width: fit-content;
        }}
        .bench-priority-high {{ background: var(--app-red-tint); color: {RED}; }}
        .bench-priority-mid  {{ background: var(--app-amber-tint); color: {AMBER}; }}
        .bench-priority-low  {{ background: rgba(148, 163, 184, 0.15); color: var(--text-color); opacity: 0.8; }}

        .bench-result-badge {{
            display: inline-flex; align-items: center; gap: 6px;
            font-size: 12.5px; font-weight: 700;
            padding: 5px 12px; border-radius: 9999px; width: fit-content;
        }}
        .bench-result-badge.pass {{ background: var(--app-green-tint); color: {GREEN}; }}
        .bench-result-badge.fail {{ background: var(--app-red-tint); color: {RED}; }}
        .bench-result-badge.na   {{ background: rgba(148, 163, 184, 0.15); color: var(--text-color); opacity: 0.7; }}

        /* ── Adaptive Export Rows ─────────────────────────────── */
        .app-export-row {{
            display: flex; align-items: center; gap: 14px;
            padding: 12px 16px; border-radius: var(--app-radius-lg);
            border: 1px solid rgba(148, 163, 184, 0.22);
            background: var(--secondary-background-color);
            transition: all .15s ease;
        }}
        .app-export-row.ready:hover {{
            border-color: {BLUE_LIGHT};
            background: var(--app-blue-surface);
        }}
        .app-export-row.empty {{
            opacity: 0.5; border-style: dashed;
        }}
        .app-export-icon {{
            width: 40px; height: 40px; border-radius: 10px; flex-shrink: 0;
            display: flex; align-items: center; justify-content: center;
        }}
        .app-export-icon.docx {{
            background: rgba(37, 99, 235, 0.15); color: {BLUE};
        }}
        .app-export-icon.pdf {{
            background: rgba(239, 68, 68, 0.15); color: {RED};
        }}
        .app-export-meta {{ flex: 1; min-width: 0; }}
        .app-export-title {{ font-weight: 700; font-size: 13.5px; color: var(--text-color); }}
        .app-export-sub {{ font-size: 11.5px; color: var(--text-color); opacity: 0.65; margin-top: 2px; }}
        .app-export-status {{
            font-size: 11.5px; font-weight: 700; white-space: nowrap;
            display: inline-flex; align-items: center; gap: 4px;
        }}
        .app-export-status.ready {{ color: {GREEN}; }}
        .app-export-status.empty {{ color: var(--text-color); opacity: 0.5; }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════════════════════════
# 5. SIDEBAR COMPONENT
# ══════════════════════════════════════════════════════════════════

def render_sidebar(active: str = "") -> None:
    """Render the Rubick-inspired modern royal blue sidebar."""
    with st.sidebar:
        st.markdown(
            f"""
            <div class="brand-header">
                <div class="brand-kicker">{icon('sparkles', 13, 2.0, '#93C5FD')} SMART WORKSPACE</div>
                <div class="brand-title">Trợ lý AI Thông Minh</div>
                <div class="brand-sub">MEETING · DIARIZATION · VOICEID · OCR</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.divider()

        for page, label, _key in _NAV_ITEMS:
            st.page_link(page, label=label, icon=f":material/{_MATERIAL_ICONS[page]}:")

        st.divider()
        st.markdown(
            f"""
            <div class="status-panel-glass">
                <div class="status-row-glass">
                    <span class="status-label-glass">RUNTIME</span>
                    <span class="app-badge app-badge-on">{icon('check', 11)} OFFLINE</span>
                </div>
                <div class="status-mini-glass"><span>STT Core</span><b>Zipformer + Diart</b></div>
                <div class="status-mini-glass"><span>VoiceID</span><b>Cam++ 192d</b></div>
                <div class="status-mini-glass"><span>OCR Engine</span><b>DeepseekOCR</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ══════════════════════════════════════════════════════════════════
# 6. HEADER & UTILITY HELPERS
# ══════════════════════════════════════════════════════════════════

def render_workspace_jump(mode: str, label: str = "") -> None:
    """Jump straight into the merged Workspace page."""
    url = WORKSPACE_MEETING_URL if mode == "meeting" else WORKSPACE_OCR_URL
    text = label or ("Mở Cuộc họp trong Không gian làm việc"
                     if mode == "meeting" else
                     "Mở OCR trong Không gian làm việc")
    st.markdown(
        f'<div style="text-align:right;margin:-6px 0 12px;">'
        f'<a href="{url}" target="_self" class="app-tag" '
        f'style="text-decoration:none;color:{BLUE};border-color:rgba(37,99,235,0.3);background:var(--app-blue-surface);">'
        f'{icon("arrow", 11)}&nbsp; {text}</a></div>',
        unsafe_allow_html=True,
    )

def render_page_header(title: str, subtitle: str = "", eyebrow: str = "") -> None:
    eyebrow_html = (
        f'<div class="app-page-eyebrow">{icon("layers", 12)} {eyebrow}</div>' if eyebrow else ""
    )
    subtitle_html = f'<div class="app-page-subtitle">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f"""
        <div class="app-page-header">
            {eyebrow_html}
            <div class="app-page-title">{title}</div>
            {subtitle_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════════════════════════
# 7. BENCHMARK CRITERIA TABLE
# ══════════════════════════════════════════════════════════════════

_PRIORITY_CLS = {"Cao": "high", "Trung bình": "mid", "Thấp": "low"}

def _bench_group_icon(group_name: str) -> str:
    low = group_name.lower()
    if "text" in low:
        return "docx"
    if "layout" in low or "bố cục" in low:
        return "layers"
    if "table" in low or "bảng" in low:
        return "table"
    return "gauge"

def render_benchmark_groups(rows: list[dict]) -> None:
    """Render benchmark criteria rows as grouped, modern cards."""
    groups: dict[str, list[dict]] = {}
    order: list[str] = []
    for r in rows:
        g = r.get("Nhóm", "")
        if g not in groups:
            groups[g] = []
            order.append(g)
        groups[g].append(r)

    for g in order:
        grows = groups[g]
        gname = g.strip()
        n_pass = sum(1 for r in grows if r.get("Kết quả") == "Đạt")
        n_total = len(grows)

        rows_html = []
        for r in grows:
            result = r.get("Kết quả", "")
            if result == "Đạt":
                cls, ricon = "pass", "check"
            elif result == "Không đạt":
                cls, ricon = "fail", "cross"
            else:
                cls, ricon = "na", "info"
            priority = r.get("Ưu tiên", "")
            prio_cls = _PRIORITY_CLS.get(priority, "mid")

            rows_html.append(
                f'<div class="app-bench-row {cls}">'
                f'<div class="bench-id">{r.get("ID", "")}</div>'
                f'<div class="bench-crit">'
                f'<div class="bench-crit-name">{r.get("Tiêu chí", "")}</div>'
                f'<div class="bench-crit-threshold">Ngưỡng {r.get("Ngưỡng", "")}</div>'
                f'</div>'
                f'<div class="bench-value">{r.get("Giá trị đo", "")}</div>'
                f'<div class="bench-priority bench-priority-{prio_cls}">{priority}</div>'
                f'<div class="bench-result-badge {cls}">{icon(ricon, 12)} {result}</div>'
                f'</div>'
            )

        st.markdown(
            f"""
            <div class="app-bench-group">
                <div class="app-bench-group-head">
                    <div class="app-bench-group-icon">{icon(_bench_group_icon(gname), 17)}</div>
                    <div class="app-bench-group-title">{gname}</div>
                    <div class="app-bench-group-count">{n_pass}/{n_total} đạt</div>
                </div>
                <div class="app-bench-header">
                    <div>ID</div><div>Tiêu chí</div><div>Giá trị đo</div><div>Ưu tiên</div><div>Kết quả</div>
                </div>
                <div class="app-bench-rows">{"".join(rows_html)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
