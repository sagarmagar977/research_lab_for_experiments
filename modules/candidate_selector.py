import os
import base64
import shutil
import streamlit as st
from PIL import Image

def get_image_base64(path):
    """Encodes a local image to base64 data URI."""
    ext = os.path.splitext(path)[1].lower()
    mime = "image/png" if ext == ".png" else "image/jpeg"
    try:
        with open(path, "rb") as f:
            data = f.read()
        return f"data:{mime};base64,{base64.b64encode(data).decode()}"
    except Exception:
        return ""

def render_candidate_selector():
    """Renders the manual Candidate Frame Selector page."""
    st.markdown("### Manual Candidate Frame Selector")
    
    sessions_root = "sessions"
    if not os.path.exists(sessions_root) or not os.path.isdir(sessions_root):
        st.info("No active crop sessions found. Please run batch cropping first.")
        return
        
    session_dirs = sorted([d for d in os.listdir(sessions_root) if os.path.isdir(os.path.join(sessions_root, d))], reverse=True)
    if not session_dirs:
        st.info("No active crop sessions found. Please run batch cropping first.")
        return
        
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        selected_session = st.selectbox("Select Session", session_dirs, key="sel_sess_cand")
    with col_s2:
        selected_engine = st.radio("Select Crop Source Engine", ["PP-OCRv3", "PP-OCRv4"], horizontal=True, key="sel_eng_cand")
        
    session_path = os.path.join(sessions_root, selected_session)
    crop_dir_name = "v3_crops" if selected_engine == "PP-OCRv3" else "v4_crops"
    cand_dir_name = "v3_candidate_frames" if selected_engine == "PP-OCRv3" else "v4_candidate_frames"
    
    crop_dir = os.path.join(session_path, crop_dir_name)
    cand_dir = os.path.join(session_path, cand_dir_name)
    
    os.makedirs(cand_dir, exist_ok=True)
    
    if not os.path.exists(crop_dir) or not os.path.isdir(crop_dir):
        st.warning(f"No cropped images found for engine {selected_engine} in this session.")
        return
        
    cropped_files = sorted([f for f in os.listdir(crop_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    if not cropped_files:
        st.warning("No crops available to select candidates from.")
        return
        
    candidate_files = set([f for f in os.listdir(cand_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    inspect_key = f"inspect_frame_{selected_session}_{selected_engine}"
    if inspect_key not in st.session_state:
        st.session_state[inspect_key] = None
        
    selected_inspect = st.session_state[inspect_key]
    
    # Initialize and restore page states (prevents Streamlit's widget key cleanup from resetting pages during lightbox return)
    page_key = f"gallery_page_{selected_session}_{selected_engine}"
    cand_page_key = f"cand_page_{selected_session}_{selected_engine}"
    
    if page_key not in st.session_state:
        if f"backup_{page_key}" in st.session_state:
            st.session_state[page_key] = st.session_state[f"backup_{page_key}"]
        else:
            st.session_state[page_key] = 1
            
    if cand_page_key not in st.session_state:
        if f"backup_{cand_page_key}" in st.session_state:
            st.session_state[cand_page_key] = st.session_state[f"backup_{cand_page_key}"]
        else:
            st.session_state[cand_page_key] = 1
            
    # Keep backup keys in sync
    st.session_state[f"backup_{page_key}"] = st.session_state[page_key]
    st.session_state[f"backup_{cand_page_key}"] = st.session_state[cand_page_key]
    
    # --- Full Preview / Lightbox Mode ---
    if selected_inspect and selected_inspect in cropped_files:
        st.markdown("---")
        st.markdown(f"### 🖼️ Full Preview: `{selected_inspect}`")
        
        st.image(Image.open(os.path.join(crop_dir, selected_inspect)), use_container_width=True)
        
        is_marked = selected_inspect in candidate_files
        current_idx = cropped_files.index(selected_inspect)
        prev_idx = (current_idx - 1) % len(cropped_files)
        next_idx = (current_idx + 1) % len(cropped_files)
        
        st.write("")
        col_prev_f, col_star_f, col_next_f, col_close_f = st.columns(4)
        
        with col_prev_f:
            if st.button("◀ Prev", key="btn_prev_frame", use_container_width=True):
                st.session_state[inspect_key] = cropped_files[prev_idx]
                st.rerun()
        with col_star_f:
            if not is_marked:
                if st.button("⭐ Mark Candidate", key="btn_toggle_star", use_container_width=True):
                    shutil.copy2(os.path.join(crop_dir, selected_inspect), os.path.join(cand_dir, selected_inspect))
                    st.rerun()
            else:
                if st.button("⭐ Unmark Candidate", key="btn_toggle_star", use_container_width=True):
                    os.remove(os.path.join(cand_dir, selected_inspect))
                    st.rerun()
        with col_next_f:
            if st.button("▶ Next", key="btn_next_frame", use_container_width=True):
                st.session_state[inspect_key] = cropped_files[next_idx]
                st.rerun()
        with col_close_f:
            if st.button("❌ Close", key="btn_close_inspect", use_container_width=True):
                st.session_state[inspect_key] = None
                st.rerun()
                
        st.components.v1.html("""
        <script>
        function handleKeyDown(e) {
            const doc = window.parent.document;
            const buttons = Array.from(doc.querySelectorAll('button'));
            if (e.key === 'ArrowLeft') {
                const btn = buttons.find(b => b.textContent.includes('◀ Prev'));
                if (btn) { btn.click(); e.preventDefault(); }
            } else if (e.key === 'ArrowRight') {
                const btn = buttons.find(b => b.textContent.includes('▶ Next'));
                if (btn) { btn.click(); e.preventDefault(); }
            } else if (e.key.toLowerCase() === 's') {
                const btn = buttons.find(b => b.textContent.includes('⭐ Mark') || b.textContent.includes('⭐ Unmark'));
                if (btn) { btn.click(); e.preventDefault(); }
            } else if (e.key === 'Escape') {
                const btn = buttons.find(b => b.textContent.includes('❌ Close'));
                if (btn) { btn.click(); e.preventDefault(); }
            }
        }
        if (window.parent) {
            if (window.parent._candSelectorHandler) {
                window.parent.removeEventListener('keydown', window.parent._candSelectorHandler);
            }
            window.parent._candSelectorHandler = handleKeyDown;
            window.parent.addEventListener('keydown', handleKeyDown);
        }
        </script>
        """, height=0, width=0)
        
        return # Skip rendering tabs while in full preview mode
        
    # --- Tab Rendering Mode ---
    tab_gallery, tab_showcase = st.tabs(["🖼️ Crop Gallery", "⭐ Showcase (Selected Candidates)"])
    limit = 20
    
    # --- Tab 1: Crop Gallery ---
    with tab_gallery:
        st.markdown("#### All Cropped Frames in Session")
        
        total_cropped = len(cropped_files)
        total_pages = (total_cropped + limit - 1) // limit
        
        curr_page = st.session_state[page_key]
        if curr_page > total_pages:
            curr_page = total_pages
            st.session_state[page_key] = total_pages
        elif curr_page < 1:
            curr_page = 1
            st.session_state[page_key] = 1
            
        curr_page = st.session_state[page_key]
        start_idx = (curr_page - 1) * limit
        end_idx = min(start_idx + limit, total_cropped)
        page_files = cropped_files[start_idx:end_idx]
        
        def mark_candidate(src_path, dst_dir, file_name):
            shutil.copy2(src_path, os.path.join(dst_dir, file_name))
            
        def unmark_candidate(dst_dir, file_name):
            target = os.path.join(dst_dir, file_name)
            if os.path.exists(target):
                os.remove(target)

        cols_per_row = 4
        for i in range(0, len(page_files), cols_per_row):
            cols = st.columns(cols_per_row)
            for j in range(cols_per_row):
                idx = i + j
                if idx < len(page_files):
                    fname = page_files[idx]
                    fpath = os.path.join(crop_dir, fname)
                    is_marked = fname in candidate_files
                    
                    with cols[j]:
                        img_b64 = get_image_base64(fpath)
                        border_color = "#10b981" if is_marked else "transparent"
                        bg_color = "rgba(16, 185, 129, 0.05)" if is_marked else "transparent"
                        shadow = "0 0 12px rgba(16, 185, 129, 0.25)" if is_marked else "none"
                        
                        st.markdown(
                            f'<div style="border: 3px solid {border_color}; border-radius: 12px; padding: 6px; background-color: {bg_color}; box-shadow: {shadow}; margin-bottom: 8px; transition: all 0.2s ease-in-out;">'
                            f'<img src="{img_b64}" style="width: 100%; border-radius: 8px; display: block;"/>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                        st.markdown(f"**{fname}**")
                        
                        col_o, col_t = st.columns(2)
                        with col_o:
                            if st.button("🔍 Open", key=f"open_img_g_{fname}_{selected_engine}", use_container_width=True):
                                st.session_state[inspect_key] = fname
                                st.rerun()
                        with col_t:
                            if not is_marked:
                                st.button("⭐ Mark", key=f"mark_g_{fname}_{selected_engine}", on_click=mark_candidate, args=(fpath, cand_dir, fname), use_container_width=True)
                            else:
                                st.button("✓ Selected", key=f"unmark_g_{fname}_{selected_engine}", on_click=unmark_candidate, args=(cand_dir, fname), use_container_width=True)
                                    
        # Draw navigation controls at the bottom
        if total_pages > 1:
            st.write("")
            col_pad_l, col_prev, col_select, col_next, col_pad_r = st.columns([4, 1, 0.6, 1, 4])
            
            def change_g_page(delta):
                st.session_state[page_key] = max(1, min(total_pages, st.session_state[page_key] + delta))
                
            with col_prev:
                st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
                st.button("◀ Prev", key=f"prev_pg_g_{selected_session}", disabled=(curr_page <= 1), on_click=change_g_page, args=(-1,), use_container_width=True)
                    
            with col_select:
                page_options = list(range(1, total_pages + 1))
                st.selectbox(
                    "Page:",
                    options=page_options,
                    key=page_key,
                )
                    
            with col_next:
                st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
                st.button("Next ▶", key=f"next_pg_g_{selected_session}", disabled=(curr_page >= total_pages), on_click=change_g_page, args=(1,), use_container_width=True)
                                
    # --- Tab 2: Showcase ---
    with tab_showcase:
        st.markdown("#### Current Selected Candidate Frames")
        cand_list = sorted(list(candidate_files))
        
        if not cand_list:
            st.info("No candidate frames have been marked yet in this session.")
        else:
            total_cand = len(cand_list)
            total_cand_pages = (total_cand + limit - 1) // limit
            
            curr_cand_page = st.session_state[cand_page_key]
            if curr_cand_page > total_cand_pages:
                curr_cand_page = total_cand_pages
                st.session_state[cand_page_key] = total_cand_pages
            elif curr_cand_page < 1:
                curr_cand_page = 1
                st.session_state[cand_page_key] = 1
                
            curr_cand_page = st.session_state[cand_page_key]
            start_c_idx = (curr_cand_page - 1) * limit
            end_c_idx = min(start_c_idx + limit, total_cand)
            page_candidates = cand_list[start_c_idx:end_c_idx]
            
            def remove_candidate(dst_dir, file_name):
                target = os.path.join(dst_dir, file_name)
                if os.path.exists(target):
                    os.remove(target)

            cols_per_row = 4
            for i in range(0, len(page_candidates), cols_per_row):
                cols = st.columns(cols_per_row)
                for j in range(cols_per_row):
                    idx = i + j
                    if idx < len(page_candidates):
                        fname = page_candidates[idx]
                        fpath = os.path.join(cand_dir, fname)
                        with cols[j]:
                            img_b64 = get_image_base64(fpath)
                            st.markdown(
                                f'<div style="border: 3px solid #10b981; border-radius: 12px; padding: 6px; background-color: rgba(16, 185, 129, 0.05); box-shadow: 0 0 12px rgba(16, 185, 129, 0.25); margin-bottom: 8px;">'
                                f'<img src="{img_b64}" style="width: 100%; border-radius: 8px; display: block;"/>'
                                f'</div>',
                                unsafe_allow_html=True
                            )
                            st.markdown(f"**{fname}**")
                            
                            # Showcase buttons (🔍 Open & ⭐ Remove)
                            col_s_star, col_s_insp = st.columns(2)
                            with col_s_star:
                                st.button("⭐ Remove", key=f"rm_show_{fname}_{selected_engine}", on_click=remove_candidate, args=(cand_dir, fname), use_container_width=True)
                            with col_s_insp:
                                if st.button("🔍 Open", key=f"open_show_{fname}_{selected_engine}", use_container_width=True):
                                    st.session_state[inspect_key] = fname
                                    st.rerun()
                                    
            # Draw Showcase navigation controls at the bottom
            if total_cand_pages > 1:
                st.write("")
                col_cpad_l, col_cprev, col_cselect, col_cnext, col_cpad_r = st.columns([4, 1, 0.6, 1, 4])
                
                def change_c_page(delta):
                    st.session_state[cand_page_key] = max(1, min(total_cand_pages, st.session_state[cand_page_key] + delta))
                    
                with col_cprev:
                    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
                    st.button("◀ Prev", key=f"prev_cpg_{selected_session}", disabled=(curr_cand_page <= 1), on_click=change_c_page, args=(-1,), use_container_width=True)
                        
                with col_cselect:
                    cand_page_options = list(range(1, total_cand_pages + 1))
                    st.selectbox(
                        "Page:",
                        options=cand_page_options,
                        key=cand_page_key,
                    )
                        
                with col_cnext:
                    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
                    st.button("Next ▶", key=f"next_cpg_{selected_session}", disabled=(curr_cand_page >= total_cand_pages), on_click=change_c_page, args=(1,), use_container_width=True)
