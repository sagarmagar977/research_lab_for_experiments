import os
import base64
import shutil
import zipfile
import datetime
import json
import io
import streamlit as st
from PIL import Image

from modules.image_cache import (
    get_cached_thumbnail_b64,
    trigger_background_session_prefetch,
    get_cache_stats
)

def get_image_base64(path, max_dim=400):
    """Retrieves thumbnail from fast in-memory cache or computes it immediately."""
    b64 = get_cached_thumbnail_b64(path, max_dim=max_dim)
    if b64 and not b64.startswith("data:"):
        return f"data:image/jpeg;base64,{b64}"
    return b64

def trigger_candidate_prefetch(dir_path, filenames, current_page=1, page_size=24):
    """Background worker to warm candidate gallery thumbnails across the entire session."""
    full_paths = [os.path.join(dir_path, f) for f in filenames]
    trigger_background_session_prefetch(full_paths, current_page=current_page, page_size=page_size, session_key=dir_path)

def create_curated_zip(cand_dir, orig_dir, session_name, engine_name):
    """Creates in-memory ZIP containing curated candidates and matching original frames."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
        cand_files = sorted([f for f in os.listdir(cand_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        for f in cand_files:
            crop_path = os.path.join(cand_dir, f)
            zipf.write(crop_path, arcname=f"candidate_crops/{f}")
            if orig_dir and os.path.exists(orig_dir):
                orig_path = os.path.join(orig_dir, f)
                if os.path.exists(orig_path):
                    zipf.write(orig_path, arcname=f"candidate_original_frames/{f}")
        manifest_info = {
            "session_name": session_name,
            "engine": engine_name,
            "exported_at": datetime.datetime.now().isoformat(),
            "candidate_count": len(cand_files),
            "candidates": cand_files
        }
        zipf.writestr("manifest.json", json.dumps(manifest_info, indent=4))
    buffer.seek(0)
    return buffer

def render_candidate_selector():
    """Renders the manual Candidate Frame Selector page."""
    st.markdown("### Manual Candidate Frame Selector")
    
    sessions_root = "sessions"
    if not os.path.exists(sessions_root) or not os.path.isdir(sessions_root):
        st.info("No active crop sessions found. Please run batch cropping first.")
        return
        
    session_dirs = sorted([
        d for d in os.listdir(sessions_root) 
        if os.path.isdir(os.path.join(sessions_root, d)) and (
            os.path.exists(os.path.join(sessions_root, d, "v3_crops")) or 
            os.path.exists(os.path.join(sessions_root, d, "v4_crops")) or
            os.path.exists(os.path.join(sessions_root, d, "v3_candidate_frames")) or
            os.path.exists(os.path.join(sessions_root, d, "v4_candidate_frames"))
        )
    ], reverse=True)
    if not session_dirs:
        session_dirs = sorted([d for d in os.listdir(sessions_root) if os.path.isdir(os.path.join(sessions_root, d))], reverse=True)
    if not session_dirs:
        st.info("No active crop sessions found. Please run batch cropping first.")
        return
        
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        selected_session = st.selectbox("Select Session", session_dirs, key="sel_sess_cand")
        
    session_path = os.path.join(sessions_root, selected_session)
    available_engines = []
    if os.path.exists(os.path.join(session_path, "v3_crops")):
        available_engines.append("PP-OCRv3")
    if os.path.exists(os.path.join(session_path, "v4_crops")):
        available_engines.append("PP-OCRv4")
    if not available_engines:
        available_engines = ["PP-OCRv3"]

    eng_key = f"sel_eng_cand_{selected_session}"
    if eng_key not in st.session_state:
        preset_eng = st.session_state.get("sel_eng_cand")
        st.session_state[eng_key] = preset_eng if preset_eng in available_engines else available_engines[0]
    elif st.session_state[eng_key] not in available_engines:
        st.session_state[eng_key] = available_engines[0]

    eng_idx = available_engines.index(st.session_state[eng_key])

    with col_s2:
        selected_engine = st.radio("Select Crop Source Engine", available_engines, index=eng_idx, horizontal=True, key=eng_key)
        st.session_state["sel_eng_cand"] = selected_engine
        
    crop_dir_name = "v3_crops" if selected_engine == "PP-OCRv3" else "v4_crops"
    cand_dir_name = "v3_candidate_frames" if selected_engine == "PP-OCRv3" else "v4_candidate_frames"
    
    crop_dir = os.path.join(session_path, crop_dir_name)
    cand_dir = os.path.join(session_path, cand_dir_name)
    orig_dir = os.path.join(session_path, "original_frames")
    has_orig = os.path.exists(orig_dir) and os.path.isdir(orig_dir)
    
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
        
        has_matching_orig = has_orig and os.path.exists(os.path.join(orig_dir, selected_inspect))
        if has_matching_orig:
            show_orig = st.checkbox("🔍 Compare with Full Original Video Frame", value=False, key="chk_compare_orig")
            if show_orig:
                col_pv1, col_pv2 = st.columns(2)
                with col_pv1:
                    st.markdown("**Cropped Region (Candidate / Crop)**")
                    st.image(Image.open(os.path.join(crop_dir, selected_inspect)), use_container_width=True)
                with col_pv2:
                    st.markdown("**Original Full Video Frame**")
                    st.image(Image.open(os.path.join(orig_dir, selected_inspect)), use_container_width=True)
            else:
                st.image(Image.open(os.path.join(crop_dir, selected_inspect)), use_container_width=True)
        else:
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
        
    # --- Metrics & Export / Actions Toolbar ---
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-title">Total Session Frames</div>'
            f'<div class="metric-value">{len(cropped_files)}</div>'
            f'</div>',
            unsafe_allow_html=True
        )
    with col_m2:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-title">Selected Candidates (Marked)</div>'
            f'<div class="metric-value" style="color: #10b981;">{len(candidate_files)}</div>'
            f'</div>',
            unsafe_allow_html=True
        )
    with col_m3:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-title">Discarded / Redundant</div>'
            f'<div class="metric-value" style="color: #ef4444;">{len(cropped_files) - len(candidate_files)}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    st.write("")
    with st.expander("📦 Export Curated Dataset & Pipeline Actions", expanded=False):
        col_exp1, col_exp2, col_exp3 = st.columns(3)
        with col_exp1:
            if len(candidate_files) > 0:
                zip_key = f"cached_zip_{selected_session}_{selected_engine}"
                cand_signature = (selected_session, selected_engine, len(candidate_files), tuple(sorted(candidate_files)))
                cand_sig_key = f"{zip_key}_sig"
                
                if zip_key in st.session_state and st.session_state.get(cand_sig_key) == cand_signature:
                    st.download_button(
                        label=f"💾 Download Curated ZIP ({len(candidate_files)} frames)",
                        data=st.session_state[zip_key],
                        file_name=f"{selected_session}_{selected_engine}_curated.zip",
                        mime="application/zip",
                        use_container_width=True
                    )
                else:
                    if st.button(f"📦 Prepare Curated ZIP ({len(candidate_files)} frames)", use_container_width=True, key="btn_prep_zip"):
                        with st.spinner("Building curated ZIP archive..."):
                            zip_buf = create_curated_zip(cand_dir, orig_dir if has_orig else None, selected_session, selected_engine)
                            st.session_state[zip_key] = zip_buf.getvalue()
                            st.session_state[cand_sig_key] = cand_signature
                        st.rerun()
            else:
                st.button("💾 Download Curated ZIP (0 frames)", disabled=True, use_container_width=True)
        with col_exp2:
            if st.button("📁 Export to Local `exports/` Folder", use_container_width=True):
                export_base = os.path.join("exports", f"{selected_session}_{selected_engine}_curated")
                out_crops = os.path.join(export_base, "candidate_crops")
                out_orig = os.path.join(export_base, "candidate_original_frames")
                os.makedirs(out_crops, exist_ok=True)
                if has_orig:
                    os.makedirs(out_orig, exist_ok=True)
                for cf in candidate_files:
                    shutil.copy2(os.path.join(cand_dir, cf), os.path.join(out_crops, cf))
                    if has_orig:
                        src_o = os.path.join(orig_dir, cf)
                        if os.path.exists(src_o):
                            shutil.copy2(src_o, os.path.join(out_orig, cf))
                st.success(f"Successfully exported {len(candidate_files)} frames to `{os.path.abspath(export_base)}`!")
        with col_exp3:
            def goto_dataset_generator_cb(sess_name, eng):
                st.session_state["selected_module"] = "Batch Dataset Generator"
                st.session_state["batch_dataset_session"] = sess_name
                st.session_state["batch_dataset_croptype"] = "PP-OCRv3 Crops" if eng == "PP-OCRv3" else "PP-OCRv4 Crops"

            st.button(
                "🚀 Send to Batch Dataset Generator",
                use_container_width=True,
                type="secondary",
                on_click=goto_dataset_generator_cb,
                args=(selected_session, selected_engine)
            )

    st.markdown("---")
    
    # --- Tab Rendering Mode ---
    tab_gallery, tab_showcase = st.tabs(["🖼️ Crop Gallery", "⭐ Showcase (Selected Candidates)"])
    limit = 24
    
    # --- Tab 1: Crop Gallery ---
    with tab_gallery:
        st.markdown("#### All Cropped Frames in Session")
        
        total_cropped = len(cropped_files)
        total_pages = max(1, (total_cropped + limit - 1) // limit)
        
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
        
        # Trigger background prefetching for full session across all pages
        trigger_candidate_prefetch(crop_dir, cropped_files, current_page=curr_page, page_size=limit)
        
        # Retrieve RAM cache readiness statistics for current gallery
        full_paths = [os.path.join(crop_dir, f) for f in cropped_files]
        cached_cnt, total_cnt = get_cache_stats(full_paths)
        if cached_cnt >= total_cnt:
            cache_info = f"<span style='color: #10b981; font-size: 0.82rem;'>⚡ <b>{cached_cnt:,} / {total_cnt:,}</b> frames pre-warmed in RAM (Instant Browsing Active)</span>"
        else:
            cache_info = f"<span style='color: #94a3b8; font-size: 0.82rem;'>⚡ RAM Cache: <b>{cached_cnt:,} / {total_cnt:,}</b> ready (caching session in background...)</span>"

        # Top Pagination Controls
        col_t_prev, col_t_info, col_t_next = st.columns([1, 2, 1])
        with col_t_prev:
            if st.button("◀ Previous Page", key=f"t_prev_g_{selected_session}", disabled=(curr_page <= 1), use_container_width=True):
                st.session_state[page_key] -= 1
                st.rerun()
        with col_t_info:
            st.markdown(
                f"<div style='text-align: center; padding-top: 4px;'>"
                f"<div style='font-size: 0.95rem;'>Page <b>{curr_page}</b> of <b>{total_pages}</b> &nbsp;|&nbsp; "
                f"Showing <b>{start_idx + 1}–{end_idx}</b> of <b>{total_cropped:,}</b> frames</div>"
                f"<div style='margin-top: 3px;'>{cache_info}</div>"
                f"</div>",
                unsafe_allow_html=True
            )
        with col_t_next:
            if st.button("Next Page ▶", key=f"t_next_g_{selected_session}", disabled=(curr_page >= total_pages), use_container_width=True):
                st.session_state[page_key] += 1
                st.rerun()
        
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
                            if st.button("🔍 Open", key=f"open_img_g_{selected_session}_{selected_engine}_{fname}", use_container_width=True):
                                st.session_state[inspect_key] = fname
                                st.rerun()
                        with col_t:
                            if not is_marked:
                                st.button("⭐ Mark", key=f"mark_g_{selected_session}_{selected_engine}_{fname}", on_click=mark_candidate, args=(fpath, cand_dir, fname), use_container_width=True)
                            else:
                                st.button("✓ Selected", key=f"unmark_g_{selected_session}_{selected_engine}_{fname}", on_click=unmark_candidate, args=(cand_dir, fname), use_container_width=True)
                                    
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
            total_cand_pages = max(1, (total_cand + limit - 1) // limit)
            
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

            # Top Showcase Pagination Controls
            if total_cand_pages > 1:
                col_ct_prev, col_ct_info, col_ct_next = st.columns([1, 2, 1])
                with col_ct_prev:
                    if st.button("◀ Previous Page", key=f"ct_prev_g_{selected_session}", disabled=(curr_cand_page <= 1), use_container_width=True):
                        st.session_state[cand_page_key] -= 1
                        st.rerun()
                with col_ct_info:
                    st.markdown(
                        f"<div style='text-align: center; padding-top: 4px; font-size: 0.95rem;'>"
                        f"Candidate Page <b>{curr_cand_page}</b> of <b>{total_cand_pages}</b> &nbsp;|&nbsp; "
                        f"Showing <b>{start_c_idx + 1}–{end_c_idx}</b> of <b>{total_cand:,}</b> candidates"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                with col_ct_next:
                    if st.button("Next Page ▶", key=f"ct_next_g_{selected_session}", disabled=(curr_cand_page >= total_cand_pages), use_container_width=True):
                        st.session_state[cand_page_key] += 1
                        st.rerun()
            
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
                                st.button("⭐ Remove", key=f"rm_show_{selected_session}_{selected_engine}_{fname}", on_click=remove_candidate, args=(cand_dir, fname), use_container_width=True)
                            with col_s_insp:
                                if st.button("🔍 Open", key=f"open_show_{selected_session}_{selected_engine}_{fname}", use_container_width=True):
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
