"""
pages/4_Domain_Terms.py
-----------------------
Trang quản lý CSDL Thuật ngữ theo từng chủ đề (Domain-specific Lexicon).
Thiết kế UI đồng bộ với trang quản lý VoiceID.
"""

import sys
import os
import json
import re
import base64
from pathlib import Path
import streamlit as st

# ══════════════════════════════════════════════════════════════════════════════
# 1. SETUP ĐƯỜNG DẪN & THEME CHUNG (đồng bộ với các page khác)
# ══════════════════════════════════════════════════════════════════════════════
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
UI_ROOT   = Path(__file__).resolve().parent.parent
for _p in (REPO_ROOT, UI_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from theme import apply_page_config, inject_base_css, render_sidebar

apply_page_config("Domain Terms — Quản lý Thuật ngữ", UI_ROOT)
inject_base_css()
render_sidebar(active="domain_terms")

# CSS đặc thù riêng cho trang Domain Terms
st.markdown("""
<style>
    .stButton > button[kind="primary"] {
        background: #0067B1 !important; border: none !important;
        font-weight: 600 !important; color: white !important;
    }
    .stButton > button[kind="primary"]:hover { background: #004C87 !important; }
    
    .table-header {
        font-family: 'IBM Plex Mono', monospace; font-size: 10px; 
        letter-spacing: 1.5px; text-transform: uppercase; color: #0067B1; 
        padding-bottom: 8px; font-weight: 600;
    }
    .row-text { font-size: 13px; padding-top: 12px; color: var(--text-color); }
    .row-idx { font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: var(--text-color); opacity: 0.4; padding-top: 12px;}
    .variant-pill {
        display: inline-block; background: rgba(128,128,128,0.15);
        padding: 2px 8px; border-radius: 4px; margin: 2px; font-size: 12px;
    }
    .test-success { color: #16A34A; font-weight: 600; font-size: 12px; }
    .test-fail { color: #D97706; font-size: 12px; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# 2. CORE LOGIC - QUẢN LÝ FILE JSON
# ══════════════════════════════════════════════════════════════════════════════
DOMAIN_DIR = os.path.join("core", "data", "domains")
os.makedirs(DOMAIN_DIR, exist_ok=True)

def get_domain_files():
    files = [f for f in os.listdir(DOMAIN_DIR) if f.endswith(".json")]
    return sorted(files)

def load_terms(filename):
    filepath = os.path.join(DOMAIN_DIR, filename)
    if not os.path.exists(filepath): return []
    with open(filepath, "r", encoding="utf-8") as f:
        try: return json.load(f)
        except json.JSONDecodeError: return []

def save_terms(filename, data):
    filepath = os.path.join(DOMAIN_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def test_variants_and_replace(text, canonical, variants):
    """Kiểm tra và thực hiện thay thế variant bằng canonical để demo kết quả."""
    if not text: return text, False
    
    result_text = text
    is_matched = False
    
    for variant in variants:
        # Biên dịch regex y hệt như core/term_normalizer.py
        pattern = variant[3:] if variant.startswith("re:") else r"\s*".join(re.escape(w) for w in variant.strip().split())
        regex = rf"\b{pattern}\b"
        
        try:
            compiled_regex = re.compile(regex, flags=re.IGNORECASE | re.UNICODE)
            if compiled_regex.search(result_text):
                is_matched = True
                # Thực hiện thay thế từ sai -> từ đúng
                result_text = compiled_regex.sub(canonical, result_text)
        except Exception:
            pass
            
    return result_text, is_matched

# ══════════════════════════════════════════════════════════════════════════════
# 3. SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
if "term_show_add" not in st.session_state:
    st.session_state.update({
        "term_show_add": False, "term_edit_idx": None, 
        "term_del_idx": None, "new_domain_mode": False
    })

# ══════════════════════════════════════════════════════════════════════════════
# 4. HEADER & DOMAIN SELECTOR
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div style="font-size:11px;letter-spacing:3px;text-transform:uppercase; opacity:0.7; border-bottom:1px solid rgba(128,128,128,0.2); margin-bottom:24px; padding-bottom:12px;">📚 Quản lý Bộ Từ Điển Thuật Ngữ</div>', unsafe_allow_html=True)

domain_files = get_domain_files()

col1, col2, col3 = st.columns([3, 1, 6])
with col1:
    if not domain_files:
        st.warning("Chưa có chủ đề nào.")
        selected_file = None
    else:
        selected_file = st.selectbox("Lựa chọn Chủ đề", domain_files, label_visibility="collapsed")
with col2:
    if st.button("＋ Tạo chủ đề", use_container_width=True):
        st.session_state.new_domain_mode = not st.session_state.new_domain_mode

if st.session_state.new_domain_mode:
    with st.container(border=True):
        new_name = st.text_input("Tên chủ đề mới (không cần đuôi .json)", placeholder="vd: y_te, tai_chinh...")
        if st.button("Lưu chủ đề", type="primary"):
            if new_name:
                save_terms(f"{new_name}.json", [])
                st.session_state.new_domain_mode = False
                st.rerun()

if not selected_file: st.stop()

terms_data = load_terms(selected_file)

# ══════════════════════════════════════════════════════════════════════════════
# 5. TOOLBAR
# ══════════════════════════════════════════════════════════════════════════════
tb_left, tb_right = st.columns([4, 1])
with tb_left:
    st.markdown(f'<div style="font-size:10px;letter-spacing:2px;text-transform:uppercase;opacity:0.7;padding-top:6px;">CSDL: {selected_file} ({len(terms_data)} thuật ngữ)</div>', unsafe_allow_html=True)
with tb_right:
    if st.button("＋ Thêm Thuật Ngữ", type="primary", use_container_width=True):
        st.session_state.term_show_add = True
        st.session_state.term_edit_idx = None

# --- POPUP: THÊM / SỬA THUẬT NGỮ ---
if st.session_state.term_show_add or st.session_state.term_edit_idx is not None:
    is_edit = st.session_state.term_edit_idx is not None
    idx = st.session_state.term_edit_idx
    
    default_canon = terms_data[idx]["canonical"] if is_edit else ""
    default_vars = ", ".join(terms_data[idx]["variants"]) if is_edit else ""

    with st.container(border=True):
        st.markdown(f"**{'Sửa thuật ngữ' if is_edit else 'Thêm thuật ngữ mới'}**")
        c_can, c_var = st.columns([1, 2])
        new_canon = c_can.text_input("Canonical (Từ chuẩn)", value=default_canon)
        new_vars = c_var.text_input("Variants (Cách đọc sai, cách nhau dấu phẩy)", value=default_vars)
        
        c_save, c_cancel, _ = st.columns([1, 1, 5])
        if c_save.button("Lưu", type="primary", use_container_width=True):
            var_list = [v.strip() for v in new_vars.split(",") if v.strip()]
            new_entry = {"canonical": new_canon, "variants": var_list}
            if is_edit: terms_data[idx] = new_entry
            else: terms_data.append(new_entry)
            save_terms(selected_file, terms_data)
            st.session_state.term_show_add = False
            st.session_state.term_edit_idx = None
            st.toast("✅ Đã lưu thành công!")
            st.rerun()
            
        if c_cancel.button("Hủy", use_container_width=True):
            st.session_state.term_show_add = False
            st.session_state.term_edit_idx = None
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# 6. DATABASE TABLE
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div style='margin:12px 0 8px;'></div>", unsafe_allow_html=True)

if not terms_data:
    st.markdown('<div class="empty-state" style="text-align:center; padding: 40px; opacity: 0.5;">Chưa có thuật ngữ nào. Hãy bấm thêm mới.</div>', unsafe_allow_html=True)
else:
    col_ratios = [0.5, 1.5, 3.0, 2.5, 1.0]
    
    with st.container(border=True):
        # --- HEADER ---
        h_cols = st.columns(col_ratios, gap="small")
        for i, h in enumerate(["Index", "Canonical", "Variants", "Live Test", "Thao tác"]):
            h_cols[i].markdown(f'<div class="table-header">{h}</div>', unsafe_allow_html=True)
        st.markdown("<hr style='margin:0 0 8px 0;border-color:rgba(0,103,177,0.25);'>", unsafe_allow_html=True)

        # --- ROWS ---
        for idx, entry in enumerate(terms_data):
            row_c = st.columns(col_ratios, gap="small")
            
            # 1. Index
            row_c[0].markdown(f'<div class="row-idx">{idx + 1}</div>', unsafe_allow_html=True)
            
            # 2. Canonical
            row_c[1].markdown(f'<div class="row-text" style="font-weight:600;">{entry["canonical"]}</div>', unsafe_allow_html=True)
            
            # 3. Variants (Pills)
            variants_html = "".join([f'<span class="variant-pill">{v}</span>' for v in entry.get("variants", [])])
            row_c[2].markdown(f'<div style="padding-top:10px;">{variants_html}</div>', unsafe_allow_html=True)
            
            # 4. Live Test Column
            # 4. Live Test Column
            with row_c[3]:
                test_str = st.text_input("Test", key=f"test_{selected_file}_{idx}", label_visibility="collapsed", placeholder="Nhập câu ASR lỗi...")
                if test_str:
                    res_text, is_match = test_variants_and_replace(test_str, entry["canonical"], entry.get("variants", []))
                    if is_match:
                        # Hiển thị câu đã chuẩn hóa với format nổi bật (thanh dọc màu xanh lá)
                        st.markdown(
                            f'<div style="font-size: 13px; color: #16A34A; padding-top: 6px; padding-bottom: 6px; '
                            f'border-left: 3px solid #16A34A; padding-left: 10px; margin-top: 4px; '
                            f'background: rgba(46,139,46,0.05); border-radius: 0 4px 4px 0;">'
                            f'<b>↳</b> {res_text}</div>', 
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown('<div style="font-size: 12px; color: #888; padding-top: 6px;"><i>✗ Không tìm thấy biến thể</i></div>', unsafe_allow_html=True)

            # 5. Actions (Edit / Delete)
            with row_c[4]:
                ac1, ac2 = st.columns(2)
                with ac1:
                    if st.button("✏️", key=f"edit_term_{idx}", help="Chỉnh sửa"):
                        st.session_state.term_edit_idx = idx
                        st.session_state.term_show_add = False
                        st.session_state.term_del_idx = None
                        st.rerun()
                with ac2:
                    if st.session_state.term_del_idx == idx:
                        if st.button("✓", key=f"del_conf_{idx}", type="primary"):
                            terms_data.pop(idx)
                            save_terms(selected_file, terms_data)
                            st.session_state.term_del_idx = None
                            st.toast("🗑️ Đã xóa thuật ngữ")
                            st.rerun()
                    else:
                        if st.button("🗑️", key=f"del_term_{idx}"):
                            st.session_state.term_del_idx = idx
                            st.rerun()

            st.markdown("<hr style='margin:12px 0 8px 0; border-color:rgba(128,128,128,0.1);'>", unsafe_allow_html=True)