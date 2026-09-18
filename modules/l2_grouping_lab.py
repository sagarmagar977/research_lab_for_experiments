import os
import io
import re
import json
import datetime
import pandas as pd
import streamlit as st
from PIL import Image

from modules.image_cache import get_cached_thumbnail_b64
from modules.l2.feature_extractor import (
    check_vit_availability,
    extract_single_frame_features,
    extract_single_frame_vit_embedding,
    compute_pairwise_transition_features,
    parse_timestamp_from_filename,
    load_cached_features,
    save_features_to_cache
)
from modules.l2.grouping_engine import GroupingEngine
from modules.l2.reporting import (
    load_ground_truth_dataset,
    save_ground_truth_dataset,
    save_grouping_runs_json,
    generate_comparison_markdown_report,
    get_level2_dir
)

def get_image_b64_src(path, max_dim=250):
    b64 = get_cached_thumbnail_b64(path, max_dim=max_dim)
    if b64:
        return f"data:image/jpeg;base64,{b64}"
    return None

def render_l2_grouping_lab():
    st.markdown("### Level 2.1 - Parallel A/B Candidate Grouping Lab")
    st.write(
        "Runs two independent grouping formulations in parallel on the identical Level-1 candidate frames: "
        "**Approach A (Baseline: OCR + Layout + SSIM)** vs **Approach B (ViT-Enhanced: Baseline + ViT)**. "
        "Allows manual inspection of slide partitions, divergence tracking, and ground-truth annotation for supervised ML."
    )
    
    sessions_root = "sessions"
    os.makedirs(sessions_root, exist_ok=True)
    
    # -------------------------------------------------------------
    # 1. INPUT SELECTION & SOURCE INGESTION
    # -------------------------------------------------------------
    st.markdown("#### 1. Select Candidate Frame Source")
    input_mode = st.radio(
        "Choose source for Level-1 candidate frames",
        ["Select Existing L1 Session", "Upload Candidate Frames Manually", "Local Directory Path"],
        horizontal=True,
        key="l2_input_mode"
    )
    
    selected_session_name = ""
    candidate_paths = []
    
    if input_mode == "Select Existing L1 Session":
        all_sessions = sorted([d for d in os.listdir(sessions_root) if os.path.isdir(os.path.join(sessions_root, d))], reverse=True)
        if not all_sessions:
            st.info("No sessions found in `sessions/`. Please run Level 1 or batch cropping first.")
            return
            
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            chosen_session = st.selectbox("Select Session", all_sessions, key="l2_sess_pick")
        
        session_path = os.path.join(sessions_root, chosen_session)
        available_cands = []
        if os.path.exists(os.path.join(session_path, "v3_candidate_frames")):
            available_cands.append("v3_candidate_frames")
        if os.path.exists(os.path.join(session_path, "v4_candidate_frames")):
            available_cands.append("v4_candidate_frames")
        if os.path.exists(os.path.join(session_path, "selected_frames")):
            available_cands.append("selected_frames")
            
        if not available_cands:
            # Check if crops exist as fallback
            if os.path.exists(os.path.join(session_path, "v3_crops")):
                available_cands.append("v3_crops")
            if os.path.exists(os.path.join(session_path, "v4_crops")):
                available_cands.append("v4_crops")
                
        if not available_cands:
            st.warning(f"No candidate frames found in session `{chosen_session}`.")
            return
            
        with col_s2:
            cand_folder = st.selectbox("Candidate Directory", available_cands, key="l2_cand_folder_pick")
            
        target_dir = os.path.join(session_path, cand_folder)
        img_files = sorted([f for f in os.listdir(target_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        if not img_files:
            st.warning(f"No image frames in `{target_dir}`.")
            return
            
        candidate_paths = [os.path.join(target_dir, f) for f in img_files]
        selected_session_name = chosen_session
        
    elif input_mode == "Upload Candidate Frames Manually":
        col_u1, col_u2 = st.columns([2, 1])
        with col_u1:
            uploaded_files = st.file_uploader(
                "Upload Candidate Frames (ordered by numeric timestamp in filename)",
                type=["jpg", "jpeg", "png"],
                accept_multiple_files=True,
                key="l2_manual_upload"
            )
        with col_u2:
            default_sess = f"session_l2_manual_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            sess_name_input = st.text_input("Session Folder Name", value=default_sess, key="l2_upload_sess_name")
            selected_session_name = sess_name_input.strip()
            
        if not uploaded_files:
            st.info("Upload frames to continue.")
            return
            
        session_path = os.path.join(sessions_root, selected_session_name)
        upload_cand_dir = os.path.join(session_path, "v3_candidate_frames")
        os.makedirs(upload_cand_dir, exist_ok=True)
        
        uploaded_files = sorted(uploaded_files, key=lambda x: x.name)
        candidate_paths = []
        for uf in uploaded_files:
            dest = os.path.join(upload_cand_dir, uf.name)
            if not os.path.exists(dest):
                with open(dest, "wb") as f:
                    f.write(uf.getbuffer())
            candidate_paths.append(dest)
            
    else:  # Local Directory Path
        col_l1, col_l2 = st.columns([2, 1])
        with col_l1:
            local_path = st.text_input("Enter local folder path", placeholder="e.g. F:/thesis/candidate_frames", key="l2_local_path")
        
        if not local_path or not os.path.isdir(local_path):
            st.info("Enter a valid local folder path.")
            return
            
        img_files = sorted([f for f in os.listdir(local_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        if not img_files:
            st.warning(f"No image files found in `{local_path}`.")
            return
            
        folder_base = os.path.basename(os.path.normpath(local_path))
        with col_l2:
            sess_name_input = st.text_input("Session Folder Name", value=f"session_l2_{folder_base}", key="l2_local_sess_name")
            selected_session_name = sess_name_input.strip()
            
        session_path = os.path.join(sessions_root, selected_session_name)
        local_cand_dir = os.path.join(session_path, "v3_candidate_frames")
        os.makedirs(local_cand_dir, exist_ok=True)
        
        candidate_paths = []
        for f in img_files:
            src = os.path.join(local_path, f)
            dest = os.path.join(local_cand_dir, f)
            if not os.path.exists(dest):
                import shutil
                shutil.copy2(src, dest)
            candidate_paths.append(dest)

    active_session_dir = os.path.join(sessions_root, selected_session_name)
    
    # Header summary metrics
    first_f = os.path.basename(candidate_paths[0])
    last_f = os.path.basename(candidate_paths[-1])
    _, t_start = parse_timestamp_from_filename(first_f)
    _, t_end = parse_timestamp_from_filename(last_f)
    
    st.success(
        f"Loaded **{len(candidate_paths)}** Candidate Frames for session `{selected_session_name}`. "
        f"Span: `{first_f}` ({t_start}) -> `{last_f}` ({t_end})"
    )

    st.markdown("---")

    # -------------------------------------------------------------
    # 2. FEATURE EXTRACTION & CACHING HUB (STAGE 0)
    # -------------------------------------------------------------
    st.markdown("#### 2. Precomputed Feature Cache (Stage 0)")
    
    # Check cache status
    cached_data, cached_vit = load_cached_features(active_session_dir)
    has_cache = cached_data is not None and cached_data.get("num_frames") == len(candidate_paths)
    
    vit_available, vit_msg = check_vit_availability()
    
    col_c1, col_c2, col_c3 = st.columns([1.5, 1.5, 1])
    with col_c1:
        if has_cache:
            st.info(f"Cached features found ({cached_data['num_frames']} frames, {cached_data['num_transitions']} transitions).")
        else:
            st.warning("Features not yet computed or frame count changed.")
    with col_c2:
        if vit_available:
            use_vit = st.checkbox("Include Pretrained ViT Extraction (google/vit-base)", value=True, key="l2_use_vit_chk")
        else:
            use_vit = False
            st.caption(f"ViT disabled: {vit_msg}")
    with col_c3:
        btn_label = "(Re-)Compute Features" if has_cache else "Compute & Cache Features"
        run_extract_btn = st.button(btn_label, type="primary", use_container_width=True)

    if run_extract_btn:
        progress_bar = st.progress(0.0)
        status_box = st.empty()
        
        frame_records = []
        vit_embeddings = {}
        
        status_box.text("Extracting single-frame OCR and layout features...")
        for idx, p in enumerate(candidate_paths):
            f_rec = extract_single_frame_features(p, candidate_idx=idx)
            frame_records.append(f_rec)
            
            if use_vit:
                emb = extract_single_frame_vit_embedding(p)
                if emb is not None:
                    vit_embeddings[f_rec["filename"]] = emb
                    
            progress_bar.progress((idx + 1) / len(candidate_paths) * 0.7)
            
        status_box.text("Computing pairwise transition features between adjacent candidate frames...")
        transition_records = []
        for i in range(len(candidate_paths) - 1):
            fa = frame_records[i]
            fb = frame_records[i + 1]
            pa = candidate_paths[i]
            pb = candidate_paths[i + 1]
            
            emb_a = vit_embeddings.get(fa["filename"])
            emb_b = vit_embeddings.get(fb["filename"])
            
            t_rec = compute_pairwise_transition_features(fa, fb, pa, pb, emb_a, emb_b)
            transition_records.append(t_rec)
            
            progress_bar.progress(0.7 + ((i + 1) / (len(candidate_paths) - 1) * 0.3))
            
        save_features_to_cache(active_session_dir, frame_records, transition_records, vit_embeddings)
        status_box.text("Feature computation complete and saved to disk!")
        st.session_state["_l2_cache_refresh"] = datetime.datetime.now().timestamp()
        st.rerun()

    # If no cache exists, prompt user to compute
    if not has_cache:
        st.warning("Please click **Compute & Cache Features** above to extract OCR, SSIM, and Layout representations.")
        return
        
    frames_list = cached_data["frames"]
    transitions_list = cached_data["transitions"]
    vit_dict = cached_vit or {}

    st.markdown("---")

    # -------------------------------------------------------------
    # 3. INTERACTIVE A/B GROUPING TUNING (STAGE 2.1)
    # -------------------------------------------------------------
    st.markdown("#### 3. Independent A/B Parallel Formulations")
    st.caption("Adjust thresholds and weights independently. Changes recompute groups instantly (0ms) across the precomputed cache.")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown("<h4 style='color: #60a5fa;'>Approach A - Baseline (No ViT)</h4>", unsafe_allow_html=True)
        st.caption("Features: OCR Text Jaccard + Bounding-Box Layout IoU + SSIM")
        
        ca1, ca2, ca3 = st.columns(3)
        with ca1:
            w_ssim_a = st.slider("SSIM Weight (w_ssim)", 0.0, 1.0, 0.40, 0.05, key="w_ssim_a")
        with ca2:
            w_ocr_a = st.slider("OCR Weight (w_ocr)", 0.0, 1.0, 0.40, 0.05, key="w_ocr_a")
        with ca3:
            w_layout_a = st.slider("Layout Weight (w_layout)", 0.0, 1.0, 0.20, 0.05, key="w_layout_a")
            
        thresh_a = st.slider("Decision Threshold (tau_A)", 0.05, 0.95, 0.35, 0.01, key="tau_a")
                             
    with col_b:
        st.markdown("<h4 style='color: #ffffff;'>Approach B - Baseline + ViT</h4>", unsafe_allow_html=True)
        has_vit_in_cache = len(vit_dict) > 0
        if has_vit_in_cache:
            st.caption("Features: OCR Text + Layout IoU + SSIM + Pretrained ViT Cosine Distance")
        else:
            st.caption("Features: OCR Text + Layout IoU + SSIM (ViT features not found in current cache)")
            
        cb1, cb2, cb3, cb4 = st.columns(4)
        with cb1:
            w_ssim_b = st.slider("SSIM Weight", 0.0, 5.0, 1.0, 0.2, key="l2_w_ssim_b")
        with cb2:
            w_ocr_b = st.slider("OCR Text Weight", 0.0, 5.0, 1.0, 0.2, key="l2_w_ocr_b")
        with cb3:
            w_layout_b = st.slider("Layout IoU Weight", 0.0, 5.0, 1.0, 0.2, key="l2_w_layout_b")
        with cb4:
            w_vit_b = st.slider("ViT Weight", 0.0, 5.0, 1.0, 0.2, key="l2_w_vit_b", disabled=not has_vit_in_cache)
            
        thresh_b = st.slider("Transition Threshold ($\\tau_B$)", 0.05, 0.95, 0.35, 0.01, key="l2_thresh_b",
                             help="Higher threshold = fewer new group splits.")

    # Execute Parallel Classifications
    config_a = {"ssim": w_ssim_a, "ocr": w_ocr_a, "layout": w_layout_a, "threshold": thresh_a}
    scores_a, preds_a = GroupingEngine.run_approach_a(transitions_list, config_a, thresh_a)
    groups_a = GroupingEngine.partition_groups(frames_list, preds_a)
    
    config_b = {"ssim": w_ssim_b, "ocr": w_ocr_b, "layout": w_layout_b, "vit": w_vit_b, "threshold": thresh_b}
    scores_b, preds_b = GroupingEngine.run_approach_b(transitions_list, config_b, thresh_b, vit_available=has_vit_in_cache)
    groups_b = GroupingEngine.partition_groups(frames_list, preds_b)
    
    # Identify Divergences
    divergences = GroupingEngine.find_divergences(
        transitions_list, scores_a, preds_a, thresh_a, scores_b, preds_b, thresh_b
    )

    # Save current run data
    run_payload = {
        "session_name": selected_session_name,
        "updated_at": datetime.datetime.now().isoformat(),
        "approach_a": {"config": config_a, "num_groups": len(groups_a), "groups": groups_a, "scores": scores_a, "preds": preds_a},
        "approach_b": {"config": config_b, "num_groups": len(groups_b), "groups": groups_b, "scores": scores_b, "preds": preds_b},
        "divergences_count": len(divergences)
    }
    save_grouping_runs_json(active_session_dir, run_payload)

    # High-level scorecard
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.metric("Approach A Groups", len(groups_a), help="Total slide groups formed by baseline.")
    with m_col2:
        st.metric("Approach B Groups", len(groups_b), delta=f"{len(groups_b) - len(groups_a)} vs A", help="Total slide groups formed with ViT.")
    with m_col3:
        st.metric("Total Transitions", len(transitions_list))
    with m_col4:
        st.metric("Disagreements / Divergences", len(divergences), delta=f"{(len(divergences)/max(len(transitions_list), 1))*100:.1f}% divergence")

    st.markdown("---")

    # -------------------------------------------------------------
    # 4. DIVERGENCE STUDIO (A vs B DISAGREEMENT INSPECTOR)
    # -------------------------------------------------------------
    st.markdown(f"#### 4. Divergence Inspector ({len(divergences)} Disagreements)")
    st.caption("Surfaces only transitions where Approach A and Approach B produced opposing decisions.")
    
    if not divergences:
        st.info("Complete consensus: Approach A and Approach B produced 100% identical grouping decisions under current settings.")
    else:
        for div_idx, div in enumerate(divergences):
            fa_name = div["frame_a"]
            fb_name = div["frame_b"]
            path_a = os.path.join(target_dir, fa_name)
            path_b = os.path.join(target_dir, fb_name)
            
            with st.expander(f"Divergence #{div_idx + 1}: {fa_name} ({div['timestamp_a']}) -> {fb_name} ({div['timestamp_b']}) - {div['diff_type']}", expanded=False):
                d_c1, d_c2, d_c3 = st.columns([1, 1, 1.5])
                with d_c1:
                    st.caption(f"**Frame A:** `{fa_name}` ({div['timestamp_a']})")
                    b64_a = get_image_b64_src(path_a)
                    if b64_a:
                        st.markdown(f"<img src='{b64_a}' style='max-width:100%; border-radius:8px; border:1px solid #3f3f46;'>", unsafe_allow_html=True)
                with d_c2:
                    st.caption(f"**Frame B:** `{fb_name}` ({div['timestamp_b']})")
                    b64_b = get_image_b64_src(path_b)
                    if b64_b:
                        st.markdown(f"<img src='{b64_b}' style='max-width:100%; border-radius:8px; border:1px solid #71717a;'>", unsafe_allow_html=True)
                with d_c3:
                    st.markdown("**Transition Diagnostic Breakdown:**")
                    st.write(f"- **Approach A Score:** `{div['score_a']:.4f}` ($\\tau_A = {div['thresh_a']:.2f}$) -> `Pred: {div['pred_a']}`")
                    st.write(f"- **Approach B Score:** `{div['score_b']:.4f}` ($\\tau_B = {div['thresh_b']:.2f}$) -> `Pred: {div['pred_b']}`")
                    st.write(f"- **SSIM Distance ($D_{{ssim}}$):** `{div['d_ssim']:.4f}`")
                    st.write(f"- **OCR Text Jaccard Distance ($D_{{ocr}}$):** `{div['d_ocr']:.4f}`")
                    st.write(f"- **Layout IoU Distance ($D_{{layout}}$):** `{div['d_layout']:.4f}`")
                    vit_str = f"`{div['d_vit']:.4f}`" if div['d_vit'] is not None else "None"
                    st.write(f"- **ViT Cosine Distance ($D_{{vit}}$):** {vit_str}")
                    st.write(f"- **$\\Delta t$:** `{div['delta_time_sec']} sec`")

    st.markdown("---")

    # -------------------------------------------------------------
    # 5. DUAL GROUP BROWSER (PARALLEL PARTITION VISUALIZATION)
    # -------------------------------------------------------------
    st.markdown("#### 5. Dual Group Browser (Visual Inspection)")
    tab_view_a, tab_view_b = st.tabs([f"Approach A Groups ({len(groups_a)})", f"Approach B Groups ({len(groups_b)})"])
    
    def render_group_cards(groups, badge_color):
        for g in groups:
            with st.container():
                st.markdown(
                    f"<div style='background-color:#141414; padding:10px 15px; border-radius:8px; border:1px solid #27272a; border-left:4px solid {badge_color}; margin-bottom:10px;'>"
                    f"<b style='color:{badge_color}; font-size:1.05rem;'>Group {g['group_id']}</b> &nbsp;|&nbsp; "
                    f"<span>Span: <code>{g['start_timestamp']}</code> -> <code>{g['end_timestamp']}</code> ({g['duration_sec']}s)</span> &nbsp;|&nbsp; "
                    f"<span><b>{g['frame_count']}</b> frames</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )
                
                thumb_cols = st.columns(min(len(g["frames"]), 8))
                for f_idx, fr in enumerate(g["frames"][:8]):
                    with thumb_cols[f_idx]:
                        fr_path = os.path.join(target_dir, fr["filename"])
                        b64 = get_image_b64_src(fr_path, max_dim=160)
                        if b64:
                            st.markdown(f"<img src='{b64}' style='width:100%; border-radius:6px;'>", unsafe_allow_html=True)
                        st.caption(f"`{fr['filename']}` ({fr['timestamp_str']})")
                if len(g["frames"]) > 8:
                    st.caption(f"... and {len(g['frames']) - 8} more frames in this group.")

    with tab_view_a:
        render_group_cards(groups_a, "#a1a1aa")
    with tab_view_b:
        render_group_cards(groups_b, "#ffffff")

    st.markdown("---")

    # -------------------------------------------------------------
    # 6. MANUAL GROUND-TRUTH ANNOTATION STUDIO
    # -------------------------------------------------------------
    st.markdown("#### 6. Manual Ground-Truth Transition Annotation Studio")
    st.write(
        "Ground truth is strictly controlled by the researcher. "
        "Label adjacent transitions as **`0 = Same Group (Slide Progress)`** or **`1 = New Group (Slide Change)`**."
    )
    
    gt_labels, gt_notes = load_ground_truth_dataset(active_session_dir)
    
    # Store labels in session_state for reactive updates
    if "l2_gt_labels" not in st.session_state or st.session_state.get("_l2_active_sess") != selected_session_name:
        st.session_state["l2_gt_labels"] = gt_labels.copy()
        st.session_state["l2_gt_notes"] = gt_notes.copy()
        st.session_state["_l2_active_sess"] = selected_session_name
        
    cur_labels = st.session_state["l2_gt_labels"]
    cur_notes = st.session_state["l2_gt_notes"]
    
    labeled_count = sum(1 for v in cur_labels.values() if v in (0, 1))
    st.info(f"Progress: **{labeled_count} / {len(transitions_list)}** transitions annotated.")
    
    # Pagination for annotation studio
    page_size = 6
    total_pages = max(1, (len(transitions_list) + page_size - 1) // page_size)
    ann_page = st.number_input("Transition Page", min_value=1, max_value=total_pages, value=1, step=1, key="l2_ann_page")
    
    start_idx = (ann_page - 1) * page_size
    end_idx = min(len(transitions_list), start_idx + page_size)
    
    for i in range(start_idx, end_idx):
        t = transitions_list[i]
        p_idx = t.get("pair_idx", i + 1)
        fa_name = t["frame_a"]
        fb_name = t["frame_b"]
        path_a = os.path.join(target_dir, fa_name)
        path_b = os.path.join(target_dir, fb_name)
        
        cur_lbl = cur_labels.get(p_idx, None)
        pa = preds_a[i]
        pb = preds_b[i]
        
        status_text = "Unlabeled"
        status_bg = "#27272a"
        if cur_lbl == 0:
            status_text = "0 = Same Group"
            status_bg = "#059669"
        elif cur_lbl == 1:
            status_text = "1 = New Group"
            status_bg = "#3f3f46"
            
        st.markdown(
            f"<div style='background-color:#141414; border:1px solid #27272a; padding:8px 12px; border-radius:6px; margin-top:10px; margin-bottom:6px; display:flex; justify-content:space-between; align-items:center;'>"
            f"<span><b>Transition #{p_idx}:</b> <code>{fa_name}</code> ({t['timestamp_a']}) -> <code>{fb_name}</code> ({t['timestamp_b']}) | $\\Delta t = {t.get('delta_time_sec', 0)}s$</span>"
            f"<span style='background-color:{status_bg}; padding:3px 10px; border-radius:12px; font-weight:bold; font-size:0.85rem;'>{status_text}</span>"
            f"</div>",
            unsafe_allow_html=True
        )
        
        c_im1, c_im2, c_ctrl = st.columns([1.2, 1.2, 2])
        with c_im1:
            b64_a = get_image_b64_src(path_a, max_dim=200)
            if b64_a:
                st.markdown(f"<img src='{b64_a}' style='width:100%; border-radius:6px;'>", unsafe_allow_html=True)
            st.caption(f"`{fa_name}` ({t['timestamp_a']})")
        with c_im2:
            b64_b = get_image_b64_src(path_b, max_dim=200)
            if b64_b:
                st.markdown(f"<img src='{b64_b}' style='width:100%; border-radius:6px;'>", unsafe_allow_html=True)
            st.caption(f"`{fb_name}` ({t['timestamp_b']})")
        with c_ctrl:
            st.caption(f"**Model Suggestions:** App A: `{pa}` (score `{scores_a[i]:.3f}`) | App B: `{pb}` (score `{scores_b[i]:.3f}`)")
            b_col1, b_col2, b_col3 = st.columns(3)
            with b_col1:
                if st.button("0: Same Group", key=f"btn_gt_0_{p_idx}", use_container_width=True):
                    cur_labels[p_idx] = 0
                    st.rerun()
            with b_col2:
                if st.button("1: New Group", key=f"btn_gt_1_{p_idx}", use_container_width=True):
                    cur_labels[p_idx] = 1
                    st.rerun()
            with b_col3:
                if st.button("Clear", key=f"btn_gt_clr_{p_idx}", use_container_width=True):
                    cur_labels.pop(p_idx, None)
                    st.rerun()
                    
            note_val = st.text_input("Researcher Observation Note", value=cur_notes.get(p_idx, ""), key=f"note_{p_idx}")
            if note_val != cur_notes.get(p_idx, ""):
                cur_notes[p_idx] = note_val

    # Save Ground Truth Dataset Button
    st.markdown("<br/>", unsafe_allow_html=True)
    c_sav1, c_sav2 = st.columns([1, 1])
    with c_sav1:
        if st.button("Save Ground Truth to CSV", type="primary", use_container_width=True):
            csv_file = save_ground_truth_dataset(
                active_session_dir, transitions_list, cur_labels, cur_notes, preds_a, preds_b
            )
            st.success(f"Ground truth successfully saved to `{csv_file}`!")
            
    with c_sav2:
        # Provide direct CSV download
        csv_file_path = os.path.join(get_level2_dir(active_session_dir), "ground_truth_transitions.csv")
        if os.path.exists(csv_file_path):
            with open(csv_file_path, "r", encoding="utf-8") as f:
                csv_bytes = f.read()
            st.download_button(
                "Download Ground Truth CSV",
                data=csv_bytes,
                file_name=f"{selected_session_name}_level2_ground_truth.csv",
                mime="text/csv",
                use_container_width=True
            )

    st.markdown("---")

    # -------------------------------------------------------------
    # 7. THESIS EVALUATION & REPORT GENERATION
    # -------------------------------------------------------------
    st.markdown("#### 7. Measurable Comparison & Academic Report Generator")
    
    metrics_a = GroupingEngine.compute_metrics(preds_a, cur_labels)
    metrics_b = GroupingEngine.compute_metrics(preds_b, cur_labels)
    
    if metrics_a is None or metrics_b is None:
        st.info("Label transitions in the Annotation Studio above to see Precision, Recall, and F1 comparisons.")
    else:
        e_col1, e_col2, e_col3, e_col4 = st.columns(4)
        with e_col1:
            st.metric("App A F1-Score", f"{metrics_a['f1']:.4f}")
        with e_col2:
            st.metric("App B F1-Score (ViT)", f"{metrics_b['f1']:.4f}", delta=f"{metrics_b['f1'] - metrics_a['f1']:+.4f}")
        with e_col3:
            st.metric("App A Precision / Recall", f"{metrics_a['precision']:.3f} / {metrics_a['recall']:.3f}")
        with e_col4:
            st.metric("App B Precision / Recall", f"{metrics_b['precision']:.3f} / {metrics_b['recall']:.3f}")
            
    if st.button("Generate & Save Comparison Report (comparison_report.md)", type="secondary", use_container_width=True):
        report_text, report_path = generate_comparison_markdown_report(
            selected_session_name,
            len(candidate_paths),
            config_a,
            groups_a,
            config_b,
            groups_b,
            divergences,
            metrics_a,
            metrics_b,
            cur_labels
        )
        st.success(f"Report generated and saved to `{report_path}`!")
        st.download_button(
            "Download comparison_report.md",
            data=report_text,
            file_name=f"{selected_session_name}_comparison_report.md",
            mime="text/markdown",
            use_container_width=True
        )
        st.markdown(report_text)
