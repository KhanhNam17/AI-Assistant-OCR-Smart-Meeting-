"""
ui/theme.py
===========

Shared visual design system for "Trợ lý Văn bản Thông minh" (VNPT
Technology internal OCR + Translation assistant).

Every page calls `apply_page_config(...)`, `inject_base_css()` and
`render_sidebar(active=...)` at the top so the five pages share one
consistent look: one color palette, one spacing/radius scale, one
icon set (vector SVG — no emoji used as structural icons), and one
sidebar/header implementation instead of five copy-pasted versions.
"""

from __future__ import annotations

from pathlib import Path
import streamlit as st

# ══════════════════════════════════════════════════════════════════
# Brand palette — VNPT blue, extended into a full professional scale
# ══════════════════════════════════════════════════════════════════

NAVY       = "#00203E"   # deepest — gradients / headers
BLUE_DARK  = "#00417D"
BLUE       = "#0060B5"   # primary VNPT blue
BLUE_LIGHT = "#3E8FD6"
BLUE_TINT  = "rgba(0, 96, 181, 0.08)"
BLUE_TINT2 = "rgba(0, 96, 181, 0.16)"
CYAN       = "#12B4C4"   # secondary accent (status / highlights)
GREEN      = "#1FA971"
GREEN_TINT = "rgba(31, 169, 113, 0.12)"
AMBER      = "#E1962B"
AMBER_TINT = "rgba(225, 150, 43, 0.12)"
RED        = "#E5484D"
RED_TINT   = "rgba(229, 72, 77, 0.12)"

GRADIENT_BRAND = f"linear-gradient(135deg, {BLUE} 0%, {NAVY} 120%)"

PAGE_ICON_URL = "https://lapwifivnpt.vn/wp-content/uploads/2023/08/logo-vnpt-tap-doan-buu-chinh-vien-thong.png"

_NAV_ITEMS = [
    ("app.py",                        "Trang chủ",       "home"),
    ("pages/1_Dashboard.py",          "Dashboard",       "space_dashboard"),
    ("pages/2_Workspace.py",          "Không gian làm việc", "apps"),
    ("pages/3_VoiceID_creating.py",   "VoiceID",         "badge"),
    ("pages/4_Domain_terms.py",       "Domain Terms",    "menu_book"),
    ("pages/5_Review_sessions.py",    "Review Sessions", "folder_open"),
    ("pages/6_Benchmark_sessions.py", "Benchmark",       "monitoring"),
    ("pages/7_Ground_truth_creator.py", "GT Creator",   "fact_check"),
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

# Two entry points into the merged Workspace page (pages/2_Workspace.py),
# each preselecting a mode via ?mode=... — used by Dashboard/Review/
# Benchmark "jump to" buttons instead of two separate nav pages.
WORKSPACE_MEETING_URL = "Workspace?mode=meeting"
WORKSPACE_OCR_URL     = "Workspace?mode=ocr"

# ══════════════════════════════════════════════════════════════════
# Icon set — small hand-drawn stroke SVGs (24×24, currentColor).
# Used inside raw HTML fragments (badges, cards, tables, diagrams)
# where Streamlit's built-in Material-icon fonts can't be used.
# Native widgets (st.page_link, st.button) use the `:material/...:`
# shorthand instead — see render_sidebar() below.
# ══════════════════════════════════════════════════════════════════

_ICONS: dict[str, str] = {
    "check":    '<path d="M4 12.5 9.5 18 20 6"/>',
    "cross":    '<path d="M6 6l12 12M18 6 6 18"/>',
    "alert":    '<path d="M12 3.2 22 20.2H2Z"/><path d="M12 9.5v4.2M12 16.8h.01"/>',
    "info":     '<circle cx="12" cy="12" r="9"/><path d="M12 11v5.5M12 7.6h.01"/>',
    "globe":    '<circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3c2.6 2.6 3.9 6 3.9 9s-1.3 6.4-3.9 9c-2.6-2.6-3.9-6-3.9-9S9.4 5.6 12 3Z"/>',
    "zap":      '<path d="M13 3 4 14h6l-1 7 9-11h-6l1-7Z"/>',
    "layers":   '<path d="M12 3 3 8l9 5 9-5-9-5Z"/><path d="M3 13l9 5 9-5M3 18l9 5 9-5"/>',
    "cpu":      '<rect x="7" y="7" width="10" height="10" rx="1.5"/><path d="M9 3v3M12 3v3M15 3v3M9 18v3M12 18v3M15 18v3M3 9h3M3 12h3M3 15h3M18 9h3M18 12h3M18 15h3"/>',
    "table":    '<rect x="3.5" y="4.5" width="17" height="15" rx="1.5"/><path d="M3.5 9.5h17M9.5 9.5V19.5"/>',
    "grid":     '<rect x="4" y="4" width="6.5" height="6.5" rx="1.4"/><rect x="13.5" y="4" width="6.5" height="6.5" rx="1.4"/><rect x="4" y="13.5" width="6.5" height="6.5" rx="1.4"/><rect x="13.5" y="13.5" width="6.5" height="6.5" rx="1.4"/>',
    "scan":     '<path d="M4 9V6.5A1.5 1.5 0 0 1 5.5 5H8"/><path d="M4 15v2.5A1.5 1.5 0 0 0 5.5 19H8"/><path d="M20 9V6.5A1.5 1.5 0 0 0 18.5 5H16"/><path d="M20 15v2.5a1.5 1.5 0 0 1-1.5 1.5H16"/><path d="M4 12h16"/>',
    "image":    '<rect x="3.5" y="4.5" width="17" height="15" rx="1.5"/><circle cx="9" cy="10" r="1.6"/><path d="M4 17.5 9.5 13 13 16l3-3.2 4 4.2"/>',
    "docx":     '<path d="M6 3h8l4 4v14H6z"/><path d="M14 3v4h4"/><path d="M8.5 13l1.2 5 1.4-4 1.4 4 1.2-5"/>',
    "arrow":    '<path d="M4 12h15M13 6l6 6-6 6"/>',
    "plus":     '<path d="M12 5v14M5 12h14"/>',
    "trash":    '<path d="M4 7h16M9 7V4.5A1.5 1.5 0 0 1 10.5 3h3A1.5 1.5 0 0 1 15 4.5V7M6 7l1 13a1.5 1.5 0 0 0 1.5 1.4h7a1.5 1.5 0 0 0 1.5-1.4l1-13"/>',
    "chevron-up":   '<path d="M5 15l7-7 7 7"/>',
    "chevron-down": '<path d="M5 9l7 7 7-7"/>',
    "flag":     '<path d="M6 3v18"/><path d="M6 4.5h11l-2.5 4L17 12.5H6"/>',
    "package":  '<path d="M12 3 3.5 7.5v9L12 21l8.5-4.5v-9L12 3Z"/><path d="M3.5 7.5 12 12l8.5-4.5M12 12v9"/>',
    "gauge":    '<path d="M4 15a8 8 0 1 1 16 0"/><path d="M12 15l4.5-5.5"/><path d="M12 15h.01"/>',
    "folder":   '<path d="M4 6.5A1.5 1.5 0 0 1 5.5 5h4l2 2.5h7A1.5 1.5 0 0 1 20 9v9a1.5 1.5 0 0 1-1.5 1.5h-13A1.5 1.5 0 0 1 4 18Z"/>',
    "compare":  '<path d="M9 3v18M15 3v18"/><path d="M4 8h5M4 16h5M15 8h5M15 16h5"/>',
}


def icon(name: str, size: int = 16, stroke: float = 1.9, color: str = "currentColor") -> str:
    """Inline SVG for use inside raw HTML/markdown fragments."""
    body = _ICONS.get(name, "")
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
        f'stroke="{color}" stroke-width="{stroke}" stroke-linecap="round" '
        f'stroke-linejoin="round" style="display:inline-block;vertical-align:-3px;flex-shrink:0">{body}</svg>'
    )


# ══════════════════════════════════════════════════════════════════
# Page bootstrap
# ══════════════════════════════════════════════════════════════════

def apply_page_config(title: str, ui_root: Path) -> None:
    """st.set_page_config + st.logo, shared across pages."""
    st.set_page_config(page_title=title, page_icon=PAGE_ICON_URL, layout="wide")
    try:
        st.logo(str(ui_root / "vnpt_logo.png"))
    except Exception:
        pass  # older Streamlit without st.logo support


# ══════════════════════════════════════════════════════════════════
# Base CSS — one design system for all five pages
# ══════════════════════════════════════════════════════════════════

def inject_base_css() -> None:
    st.markdown(
        f"""
        <style>
        :root {{
            --vnpt-navy: {NAVY};
            --vnpt-blue-dark: {BLUE_DARK};
            --vnpt-blue: {BLUE};
            --vnpt-blue-light: {BLUE_LIGHT};
            --vnpt-blue-tint: {BLUE_TINT};
            --vnpt-blue-tint-2: {BLUE_TINT2};
            --vnpt-cyan: {CYAN};
            --vnpt-green: {GREEN};
            --vnpt-green-tint: {GREEN_TINT};
            --vnpt-amber: {AMBER};
            --vnpt-amber-tint: {AMBER_TINT};
            --vnpt-red: {RED};
            --vnpt-red-tint: {RED_TINT};
            --vnpt-radius-lg: 14px;
            --vnpt-radius-md: 10px;
            --vnpt-radius-sm: 7px;
            --vnpt-shadow-sm: 0 1px 2px rgba(0,20,40,0.06);
            --vnpt-shadow-md: 0 6px 20px rgba(0,32,62,0.08);
        }}

        /* ── Layout & scaffolding ─────────────────────────────── */
        [data-testid="stSidebarNav"] {{ display: none !important; }}
        .block-container {{ max-width: 1360px; padding-top: 3.6rem; padding-bottom: 3rem; }}
        h1, h2, h3 {{ letter-spacing: -0.01em; }}
        p, li, span, div {{ letter-spacing: 0; }}

        [data-testid="stVerticalBlockBorderWrapper"] {{
            background: rgba(255,255,255,0.02);
            border-radius: var(--vnpt-radius-lg);
            border: 1px solid var(--border-color);
            box-shadow: var(--vnpt-shadow-sm);
            transition: box-shadow .18s ease, border-color .18s ease;
        }}
        [data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stMarkdownContainer"]):hover {{
            box-shadow: var(--vnpt-shadow-md);
        }}

        /* ── Buttons ───────────────────────────────────────────── */
        .stButton > button, .stDownloadButton > button {{
            border-radius: var(--vnpt-radius-md) !important;
            font-weight: 600 !important;
            transition: transform .12s ease, box-shadow .12s ease, filter .12s ease;
        }}
        .stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] {{
            height: 2.8rem;
            font-size: 1.02rem;
            background: {GRADIENT_BRAND} !important;
            border: none !important;
            box-shadow: 0 3px 10px rgba(0,96,181,0.28);
        }}
        .stButton > button[kind="primary"]:hover:not(:disabled),
        .stDownloadButton > button[kind="primary"]:hover:not(:disabled) {{
            filter: brightness(1.08);
            box-shadow: 0 5px 16px rgba(0,96,181,0.38);
        }}
        .stButton > button:active:not(:disabled), .stDownloadButton > button:active:not(:disabled) {{
            transform: scale(0.985);
        }}
        .stDownloadButton > button {{ width: 100%; }}

        /* ── Badges / pills / status chips ────────────────────── */
        .vnpt-badge {{
            display: inline-flex; align-items: center; gap: 6px;
            padding: 4px 12px; border-radius: 999px;
            font-size: 12.5px; font-weight: 700; white-space: nowrap;
            letter-spacing: 0.1px;
        }}
        .vnpt-badge-on   {{ background: var(--vnpt-green-tint); color: var(--vnpt-green); }}
        .vnpt-badge-off  {{ background: var(--vnpt-red-tint);  color: var(--vnpt-red); }}
        .vnpt-badge-info {{ background: var(--vnpt-blue-tint); color: var(--vnpt-blue); }}
        .vnpt-badge-neutral {{ background: rgba(148,163,184,0.14); color: var(--text-color); opacity: .85; }}

        .vnpt-pill-label {{
            display: inline-flex; align-items: center; gap: 8px;
            padding: 6px 16px; border-radius: 999px;
            background: var(--vnpt-blue-tint);
            color: var(--vnpt-blue); font-weight: 700; font-size: 13px;
            letter-spacing: 0.3px; margin-bottom: 14px;
        }}

        .vnpt-tag {{
            font-size: 11px; padding: 4px 12px; border-radius: 6px;
            background: var(--secondary-background-color);
            border: 1px solid var(--border-color);
            color: var(--text-color); opacity: 0.85;
            font-weight: 600; letter-spacing: 0.2px;
        }}

        /* ── Sidebar ───────────────────────────────────────────── */
        [data-testid="stSidebar"] {{
            border-right: 1px solid var(--border-color);
        }}
        .vnpt-brand {{ padding: 14px 0 20px; }}
        .vnpt-kicker {{
            font-size: 10.5px; letter-spacing: 2.2px; text-transform: uppercase;
            color: var(--vnpt-blue); font-weight: 800; margin-bottom: 8px;
        }}
        .vnpt-brand-title {{
            font-size: 17px; font-weight: 800; color: var(--text-color);
            letter-spacing: -0.3px; line-height: 1.35;
        }}
        .vnpt-brand-sub {{
            font-size: 11px; color: var(--text-color); opacity: 0.55;
            margin-top: 6px; font-weight: 600; letter-spacing: 0.3px;
        }}

        [data-testid="stSidebar"] [data-testid="stPageLink"] {{
            border-radius: var(--vnpt-radius-sm);
            margin-bottom: 2px;
        }}
        [data-testid="stSidebar"] [data-testid="stPageLink"] p {{
            font-size: 13.5px !important; font-weight: 600 !important;
        }}
        [data-testid="stSidebar"] [data-testid="stPageLink"]:hover {{
            background: var(--vnpt-blue-tint);
        }}

        .vnpt-status-panel {{
            border: 1px solid var(--border-color);
            border-radius: var(--vnpt-radius-md);
            background: rgba(255,255,255,0.02);
            padding: 12px 14px;
            margin-top: 4px;
        }}
        .vnpt-status-row {{
            display: flex; align-items: center; justify-content: space-between;
            margin-bottom: 10px;
        }}
        .vnpt-status-label {{
            font-size: 10.5px; letter-spacing: 1.4px; font-weight: 800;
            color: var(--text-color); opacity: 0.55;
        }}
        .vnpt-status-mini {{
            display: flex; align-items: center; justify-content: space-between;
            font-size: 11.5px; color: var(--text-color); opacity: 0.75;
            padding: 3px 0;
        }}
        .vnpt-status-mini b {{ font-weight: 700; opacity: 0.95; }}

        /* ── Page header ───────────────────────────────────────── */
        .vnpt-page-header {{ margin-bottom: 6px; }}
        .vnpt-page-eyebrow {{
            display: inline-flex; align-items: center; gap: 7px;
            font-size: 11px; font-weight: 800; letter-spacing: 1.6px;
            text-transform: uppercase; color: var(--vnpt-blue);
            background: var(--vnpt-blue-tint); padding: 4px 12px 4px 10px;
            border-radius: 999px; margin-bottom: 12px;
        }}
        .vnpt-page-title {{
            margin: 0; padding: 0; font-size: 30px; font-weight: 800; line-height: 1.25;
            color: var(--text-color);
        }}
        .vnpt-page-subtitle {{
            font-size: 14.5px; color: var(--text-color); opacity: 0.68;
            margin-top: 7px; font-weight: 500; max-width: 640px; line-height: 1.6;
        }}

        /* ── Status dot ────────────────────────────────────────── */
        .vnpt-dot {{ width: 8px; height: 8px; border-radius: 50%; display: inline-block; flex-shrink: 0; }}
        .vnpt-dot-ok   {{ background: var(--vnpt-green); box-shadow: 0 0 0 3px var(--vnpt-green-tint); }}
        .vnpt-dot-fail {{ background: var(--vnpt-red); box-shadow: 0 0 0 3px var(--vnpt-red-tint); }}

        /* ── Block card (GT Creator) ──────────────────────────── */
        .vnpt-block-card {{
            border: 1px solid var(--border-color);
            border-left: 3px solid var(--vnpt-blue);
            border-radius: var(--vnpt-radius-md);
            padding: 16px 18px;
            margin-bottom: 14px;
            background: rgba(255,255,255,0.02);
            box-shadow: var(--vnpt-shadow-sm);
        }}
        .vnpt-block-order {{
            display: inline-flex; align-items: center; justify-content: center;
            width: 26px; height: 26px; border-radius: 50%;
            background: var(--vnpt-blue-tint-2); color: var(--vnpt-blue);
            font-weight: 800; font-size: 13px; margin-right: 8px;
        }}
        .vnpt-section-title {{
            display: flex; align-items: center; gap: 9px;
            font-size: 17px; font-weight: 800; color: var(--text-color);
            margin: 6px 0 16px;
        }}
        .vnpt-section-title svg {{ color: var(--vnpt-blue); }}

        /* ── Metric / score cards (Benchmark) ─────────────────── */
        .vnpt-metric-card {{
            background: rgba(255,255,255,0.02);
            padding: 20px; border-radius: var(--vnpt-radius-lg);
            border: 1px solid var(--border-color);
            text-align: center;
            box-shadow: var(--vnpt-shadow-sm);
            height: 100%;
        }}
        .vnpt-metric-card .m-val {{ font-size: 30px; font-weight: 800; color: var(--vnpt-blue); margin-bottom: 4px; }}
        .vnpt-metric-card .m-lbl {{ font-size: 11.5px; letter-spacing: 1px; opacity: 0.75; text-transform: uppercase; font-weight: 700; }}
        .vnpt-metric-card .m-sub {{ font-size: 11px; opacity: 0.55; margin-top: 8px; }}

        .vnpt-score-card {{
            background: {GRADIENT_BRAND};
            color: white; border: 1px solid var(--vnpt-cyan);
            box-shadow: 0 10px 26px rgba(0,96,181,0.28);
        }}
        .vnpt-score-card .m-lbl {{ color: rgba(255,255,255,0.75); }}
        .vnpt-score-card .m-sub {{ color: rgba(255,255,255,0.65); font-size: 13.5px; }}
        .vnpt-score-card .m-val {{ font-size: 46px; }}
        .vnpt-score-card.grade-good .m-val {{ color: #6EE7A8; }}
        .vnpt-score-card.grade-avg  .m-val {{ color: #FBBF24; }}
        .vnpt-score-card.grade-bad  .m-val {{ color: #FCA5A5; }}

        .vnpt-check-row {{ font-size: 13px; display: flex; align-items: center; gap: 18px; }}
        .vnpt-check-item {{ display: inline-flex; align-items: center; gap: 6px; }}
        .vnpt-check-item.ok  {{ color: var(--vnpt-green); }}
        .vnpt-check-item.bad {{ color: var(--vnpt-red); }}

        /* ── Subsection title (groups fields inside a card) ──────── */
        .vnpt-subsection-title {{
            display: flex; align-items: center; gap: 8px;
            font-size: 12px; font-weight: 800; letter-spacing: 0.6px; text-transform: uppercase;
            color: var(--vnpt-blue); margin: 2px 0 14px;
            padding-bottom: 9px; border-bottom: 1px solid var(--border-color);
        }}
        .vnpt-subsection-title svg {{ color: var(--vnpt-blue); }}

        /* ── Compact value chip strip (current-setting summary) ──── */
        .vnpt-chip-row {{ display: flex; flex-wrap: wrap; gap: 8px; margin: 2px 0 18px; }}
        .vnpt-chip {{
            display: inline-flex; align-items: center; gap: 6px;
            font-size: 11.5px; font-weight: 700; padding: 5px 12px;
            border-radius: 999px; background: rgba(148,163,184,0.12);
            color: var(--text-color); opacity: 0.85; white-space: nowrap;
        }}
        .vnpt-chip b {{ color: var(--vnpt-blue); font-weight: 800; opacity: 1; }}

        /* ── Export result rows (file cards with inline download) ── */
        .vnpt-export-group-title {{
            display: flex; align-items: center; gap: 8px;
            font-size: 11.5px; font-weight: 800; letter-spacing: 0.7px; text-transform: uppercase;
            color: var(--text-color); opacity: 0.55; margin: 16px 0 10px;
        }}
        .vnpt-export-group-title:first-of-type {{ margin-top: 2px; }}
        .vnpt-export-row {{
            display: flex; align-items: center; gap: 12px;
            padding: 10px 14px; border-radius: var(--vnpt-radius-md);
            border: 1px solid var(--border-color);
            background: rgba(255,255,255,0.02);
            transition: border-color .15s ease, background .15s ease;
        }}
        .vnpt-export-row.ready:hover {{ border-color: var(--vnpt-blue-light); background: var(--vnpt-blue-tint); }}
        .vnpt-export-row.empty {{ opacity: 0.5; border-style: dashed; }}
        .vnpt-export-icon {{
            width: 36px; height: 36px; border-radius: 10px; flex-shrink: 0;
            display: flex; align-items: center; justify-content: center;
        }}
        .vnpt-export-icon.docx {{ background: var(--vnpt-blue-tint-2); color: var(--vnpt-blue); }}
        .vnpt-export-icon.pdf  {{ background: var(--vnpt-red-tint); color: var(--vnpt-red); }}
        .vnpt-export-meta {{ flex: 1; min-width: 0; }}
        .vnpt-export-title {{ font-weight: 700; font-size: 13.5px; color: var(--text-color); line-height: 1.3; }}
        .vnpt-export-sub {{
            font-size: 11px; opacity: 0.6; margin-top: 1px;
            white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
        }}
        .vnpt-export-status {{
            font-size: 11px; font-weight: 700; white-space: nowrap;
            display: inline-flex; align-items: center; gap: 4px; flex-shrink: 0;
        }}
        .vnpt-export-status.ready {{ color: var(--vnpt-green); }}
        .vnpt-export-status.empty {{ color: var(--text-color); opacity: 0.45; }}

        /* ── Benchmark criteria table (grouped, replaces plain dataframe) ── */
        .vnpt-bench-group {{
            border: 1px solid var(--border-color);
            border-radius: var(--vnpt-radius-lg);
            background: rgba(255,255,255,0.02);
            box-shadow: var(--vnpt-shadow-sm);
            margin-bottom: 18px;
            overflow: hidden;
        }}
        .vnpt-bench-group-head {{
            display: flex; align-items: center; gap: 12px;
            padding: 14px 20px;
            background: var(--vnpt-blue-tint);
            border-bottom: 1px solid var(--border-color);
        }}
        .vnpt-bench-group-icon {{
            width: 32px; height: 32px; border-radius: 9px; flex-shrink: 0;
            background: var(--vnpt-blue-tint-2); color: var(--vnpt-blue);
            display: flex; align-items: center; justify-content: center;
        }}
        .vnpt-bench-group-title {{ font-weight: 800; font-size: 14.5px; color: var(--text-color); letter-spacing: 0.2px; }}
        .vnpt-bench-group-count {{
            margin-left: auto; font-size: 11px; font-weight: 700; opacity: 0.6;
            letter-spacing: 0.3px; white-space: nowrap;
        }}

        .vnpt-bench-header {{
            display: grid; grid-template-columns: 44px 2.2fr 1fr 100px 130px;
            gap: 14px; padding: 9px 20px;
            font-size: 10.5px; font-weight: 800; letter-spacing: 0.8px; text-transform: uppercase;
            color: var(--text-color); opacity: 0.45;
            border-bottom: 1px solid var(--border-color);
        }}
        .vnpt-bench-rows {{ display: flex; flex-direction: column; }}
        .vnpt-bench-row {{
            display: grid;
            grid-template-columns: 44px 2.2fr 1fr 100px 130px;
            align-items: center; gap: 14px;
            padding: 13px 20px;
            border-bottom: 1px solid var(--border-color);
            border-left: 3px solid transparent;
            transition: background .15s ease;
        }}
        .vnpt-bench-row:last-child {{ border-bottom: none; }}
        .vnpt-bench-row:hover {{ background: var(--vnpt-blue-tint); }}
        .vnpt-bench-row.pass {{ border-left-color: var(--vnpt-green); }}
        .vnpt-bench-row.fail {{ border-left-color: var(--vnpt-red); }}
        .vnpt-bench-row.na   {{ border-left-color: var(--border-color); }}

        .bench-id {{ font-weight: 800; font-size: 12px; opacity: 0.4; }}
        .bench-crit-name {{ font-weight: 700; font-size: 13.5px; color: var(--text-color); line-height: 1.4; }}
        .bench-crit-threshold {{ font-size: 11.5px; opacity: 0.55; margin-top: 2px; }}
        .bench-value {{ font-weight: 800; font-size: 15px; font-variant-numeric: tabular-nums; }}
        .vnpt-bench-row.pass .bench-value {{ color: var(--vnpt-green); }}
        .vnpt-bench-row.fail .bench-value {{ color: var(--vnpt-red); }}
        .vnpt-bench-row.na .bench-value {{ opacity: 0.5; }}

        .bench-priority {{
            font-size: 10.5px; font-weight: 800; letter-spacing: 0.4px; text-transform: uppercase;
            padding: 3px 10px; border-radius: 999px; text-align: center; width: fit-content;
        }}
        .bench-priority-high {{ background: var(--vnpt-red-tint); color: var(--vnpt-red); }}
        .bench-priority-mid  {{ background: var(--vnpt-amber-tint); color: var(--vnpt-amber); }}
        .bench-priority-low  {{ background: rgba(148,163,184,0.14); color: var(--text-color); opacity: 0.75; }}

        .bench-result-badge {{
            display: inline-flex; align-items: center; gap: 6px;
            font-size: 12.5px; font-weight: 700;
            padding: 5px 12px; border-radius: 999px; width: fit-content;
        }}
        .bench-result-badge.pass {{ background: var(--vnpt-green-tint); color: var(--vnpt-green); }}
        .bench-result-badge.fail {{ background: var(--vnpt-red-tint); color: var(--vnpt-red); }}
        .bench-result-badge.na   {{ background: rgba(148,163,184,0.14); color: var(--text-color); opacity: 0.7; }}

        @media (max-width: 900px) {{
            .vnpt-bench-header {{ display: none; }}
            .vnpt-bench-row {{ grid-template-columns: 1fr 1fr; row-gap: 8px; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════

def render_sidebar(active: str) -> None:
    """active: 'home' | 'dashboard' | 'ocr' | 'review' | 'settings' | 'gt' | 'benchmark'"""
    with st.sidebar:
        st.markdown(
            """
            <div class="vnpt-brand">
                <div class="vnpt-kicker">// SMART ASSISTANT</div>
                <div class="vnpt-brand-title">Trợ lý AI công việc</div>
                <div class="vnpt-brand-sub">STT · DIARIZATION · VOICEID · OCR</div>
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
            <div class="vnpt-status-panel">
                <div class="vnpt-status-row">
                    <span class="vnpt-status-label">RUNTIME</span>
                    <span class="vnpt-badge vnpt-badge-on">{icon('check', 11)} OFFLINE</span>
                </div>
                <div class="vnpt-status-mini"><span>OCR Engine</span><b>DeepseekOCR</b></div>
                <div class="vnpt-status-mini"><span>VoiceID</span><b>Cam++ 192d</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════
# Page header
# ══════════════════════════════════════════════════════════════════

def render_workspace_jump(mode: str, label: str = "") -> None:
    """
    Small right-aligned CTA used by Dashboard / Review / Benchmark to jump
    straight into the merged Workspace page (pages/2_Workspace.py) with the
    matching mode ("meeting" or "ocr") preselected via ?mode=... .
    """
    url = WORKSPACE_MEETING_URL if mode == "meeting" else WORKSPACE_OCR_URL
    text = label or ("Mở Cuộc họp trong Không gian làm việc"
                     if mode == "meeting" else
                     "Mở OCR trong Không gian làm việc")
    st.markdown(
        f'<div style="text-align:right;margin:-6px 0 10px;">'
        f'<a href="{url}" target="_self" class="vnpt-tag" '
        f'style="text-decoration:none;color:var(--vnpt-blue);border-color:var(--vnpt-blue-light);">'
        f'{icon("arrow", 11)}&nbsp; {text}</a></div>',
        unsafe_allow_html=True,
    )


def render_page_header(title: str, subtitle: str = "", eyebrow: str = "") -> None:
    eyebrow_html = (
        f'<div class="vnpt-page-eyebrow">{icon("layers", 12)} {eyebrow}</div>' if eyebrow else ""
    )
    subtitle_html = f'<div class="vnpt-page-subtitle">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f"""
        <div class="vnpt-page-header">
            {eyebrow_html}
            <h1 class="vnpt-page-title">{title}</h1>
            {subtitle_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════
# Benchmark criteria table — grouped cards instead of a plain dataframe
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
    """Render benchmark criteria rows — dicts with keys 'Nhóm', 'ID',
    'Tiêu chí', 'Ngưỡng', 'Giá trị đo', 'Kết quả', 'Ưu tiên' — as grouped,
    color-coded cards instead of a plain st.dataframe table."""
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
                f'<div class="vnpt-bench-row {cls}">'
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
            <div class="vnpt-bench-group">
                <div class="vnpt-bench-group-head">
                    <div class="vnpt-bench-group-icon">{icon(_bench_group_icon(gname), 17)}</div>
                    <div class="vnpt-bench-group-title">{gname}</div>
                    <div class="vnpt-bench-group-count">{n_pass}/{n_total} đạt</div>
                </div>
                <div class="vnpt-bench-header">
                    <div>ID</div><div>Tiêu chí</div><div>Giá trị đo</div><div>Ưu tiên</div><div>Kết quả</div>
                </div>
                <div class="vnpt-bench-rows">{"".join(rows_html)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )