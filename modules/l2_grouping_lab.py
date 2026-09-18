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
    save_run_snapshot,
    list_run_snapshots,
    load_run_snapshot,
    delete_run_snapshot,
    generate_comparison_markdown_report,
    get_level2_dir
)

def get_image_b64_src(path, max_dim=250):
    b64 = get_cached_thumbnail_b64(path, max_dim=max_dim)
    if b64:
        return f"data:image/jpeg;base64,{b64}"
    return None

def render_l2_grouping_lab():
    st.markdown("### Level 2.1 — Three-Way Candidate Grouping Lab")
    st.write(
        "Runs three independent grouping formulations in parallel on identical Level-1 candidate frames: "
        "**A1 (Frozen Baseline: Symmetric Jaccard + Symmetric Layout IoU + SSIM)**, "
        "**A2 (Improved Asymmetric: Asymmetric Containment $P_{ocr} = |T_i \\cap T_{i+1}| / |T_i|$ + Modular Directional Layout + SSIM)**, and "
        "**B2 (Multimodal: A2 + Pretrained ViT-B/16 Cosine Distance)**. "
        "Enables controlled ablation analysis, transition inspection, and ground-truth annotation for supervised ML."
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
    # 3. INTERACTIVE 3-WAY GROUPING TUNING (STAGE 2.1)
    # -------------------------------------------------------------
    st.markdown("#### 3. Three-Way Controlled Ablation Formulations")
    st.caption("Inspect how Asymmetric OCR Information Containment (A2) and Multimodal ViT (B2) perform relative to the Frozen Baseline (A1).")
    
    col_a1, col_a2, col_b2 = st.columns(3)
    
    with col_a1:
        st.markdown("<h4 style='color: #94a3b8; font-size:1.05rem;'>A1: Frozen Baseline</h4>", unsafe_allow_html=True)
        st.caption("Symmetric OCR Jaccard + Symmetric Layout IoU + SSIM")
        
        ca1, ca2 = st.columns(2)
        with ca1:
            w_ssim_a1 = st.slider("SSIM Weight", 0.0, 2.0, 0.40, 0.05, key="w_ssim_a1")
            w_ocr_a1 = st.slider("OCR Jaccard Weight", 0.0, 2.0, 0.40, 0.05, key="w_ocr_a1")
        with ca2:
            w_layout_a1 = st.slider("Layout IoU Weight", 0.0, 2.0, 0.20, 0.05, key="w_layout_a1")
            thresh_a1 = st.slider("Threshold ($\\tau_{A1}$)", 0.05, 0.95, 0.35, 0.01, key="tau_a1")
                             
    with col_a2:
        st.markdown("<h4 style='color: #38bdf8; font-size:1.05rem;'>A2: Improved Asymmetric</h4>", unsafe_allow_html=True)
        st.caption("Asymmetric Containment + Directional Layout + Dynamic Renormalization")
        
        ca2_1, ca2_2 = st.columns(2)
        with ca2_1:
            w_ssim_a2 = st.slider("SSIM Weight", 0.0, 2.0, 0.40, 0.05, key="w_ssim_a2")
            w_ocr_a2 = st.slider("Asymm OCR Weight", 0.0, 2.0, 0.40, 0.05, key="w_ocr_a2")
            min_tokens_a2 = st.number_input("N_min Token Guard", min_value=1, max_value=20, value=3, step=1, key="min_tok_a2", help="If frame has fewer than N_min tokens, OCR is marked invalid and dynamically renormalized.")
        with ca2_2:
            w_layout_a2 = st.slider("Dir Layout Weight", 0.0, 2.0, 0.20, 0.05, key="w_layout_a2")
            use_layout_a2 = st.checkbox("Include Dir Layout", value=True, key="use_layout_a2", help="Disable for scrolling content (coding/terminal) to prevent false splits.")
            thresh_a2 = st.slider("Threshold ($\\tau_{A2}$)", 0.05, 0.95, 0.35, 0.01, key="tau_a2")

    with col_b2:
        st.markdown("<h4 style='color: #c084fc; font-size:1.05rem;'>B2: Multimodal (A2 + ViT)</h4>", unsafe_allow_html=True)
        has_vit_in_cache = len(vit_dict) > 0
        if has_vit_in_cache:
            st.caption("A2 Formulation + Pretrained ViT-B/16 Cosine Distance")
        else:
            st.caption("A2 Formulation (ViT embeddings not found in cache)")
            
        cb1, cb2 = st.columns(2)
        with cb1:
            w_ssim_b2 = st.slider("SSIM Weight", 0.0, 5.0, 1.0, 0.2, key="w_ssim_b2")
            w_ocr_b2 = st.slider("Asymm OCR Weight", 0.0, 5.0, 1.0, 0.2, key="w_ocr_b2")
            min_tokens_b2 = st.number_input("N_min Token Guard", min_value=1, max_value=20, value=3, step=1, key="min_tok_b2")
        with cb2:
            w_layout_b2 = st.slider("Dir Layout Weight", 0.0, 5.0, 1.0, 0.2, key="w_layout_b2")
            w_vit_b2 = st.slider("ViT Weight", 0.0, 5.0, 0.5, 0.1, key="w_vit_b2", disabled=not has_vit_in_cache)
            use_layout_b2 = st.checkbox("Include Dir Layout", value=True, key="use_layout_b2")
            thresh_b2 = st.slider("Threshold ($\\tau_{B2}$)", 0.05, 0.95, 0.35, 0.01, key="tau_b2")

    # Execute Parallel Classifications
    config_a1 = {"ssim": w_ssim_a1, "ocr": w_ocr_a1, "layout": w_layout_a1, "threshold": thresh_a1}
    scores_a1, preds_a1 = GroupingEngine.run_approach_a1(transitions_list, config_a1, thresh_a1)
    groups_a1 = GroupingEngine.partition_groups(frames_list, preds_a1)
    
    config_a2 = {"ssim": w_ssim_a2, "ocr": w_ocr_a2, "layout": w_layout_a2, "threshold": thresh_a2, "min_tokens": min_tokens_a2, "use_layout": use_layout_a2}
    scores_a2, preds_a2, meta_a2 = GroupingEngine.run_approach_a2(transitions_list, config_a2, thresh_a2, min_tokens=min_tokens_a2, use_layout=use_layout_a2)
    groups_a2 = GroupingEngine.partition_groups(frames_list, preds_a2)
    
    config_b2 = {"ssim": w_ssim_b2, "ocr": w_ocr_b2, "layout": w_layout_b2, "vit": w_vit_b2, "threshold": thresh_b2, "min_tokens": min_tokens_b2, "use_layout": use_layout_b2}
    scores_b2, preds_b2, meta_b2 = GroupingEngine.run_approach_b2(transitions_list, config_b2, thresh_b2, min_tokens=min_tokens_b2, use_layout=use_layout_b2, vit_available=has_vit_in_cache)
    groups_b2 = GroupingEngine.partition_groups(frames_list, preds_b2)
    
    # Track 3-way Divergences
    divergences_3way = GroupingEngine.find_divergences_3way(
        transitions_list,
        scores_a1, preds_a1, thresh_a1,
        scores_a2, preds_a2, thresh_a2,
        scores_b2, preds_b2, thresh_b2
    )

    # Save current run data
    run_payload = {
        "session_name": selected_session_name,
        "updated_at": datetime.datetime.now().isoformat(),
        "approach_a1": {"config": config_a1, "num_groups": len(groups_a1), "groups": groups_a1, "scores": scores_a1, "preds": preds_a1},
        "approach_a2": {"config": config_a2, "num_groups": len(groups_a2), "groups": groups_a2, "scores": scores_a2, "preds": preds_a2},
        "approach_b2": {"config": config_b2, "num_groups": len(groups_b2), "groups": groups_b2, "scores": scores_b2, "preds": preds_b2},
        "divergences_count": len(divergences_3way)
    }
    save_grouping_runs_json(active_session_dir, run_payload)

    # High-level scorecard
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.metric("A1 Baseline Groups", len(groups_a1), help="Original frozen symmetric baseline.")
    with m_col2:
        st.metric("A2 Asymmetric Groups", len(groups_a2), delta=f"{len(groups_a2) - len(groups_a1)} vs A1", help="Asymmetric OCR containment + directional layout.")
    with m_col3:
        st.metric("B2 Multimodal Groups", len(groups_b2), delta=f"{len(groups_b2) - len(groups_a2)} vs A2", help="A2 + Pretrained ViT-B/16.")
    with m_col4:
        st.metric("3-Way Disagreements", len(divergences_3way), delta=f"{(len(divergences_3way)/max(len(transitions_list), 1))*100:.1f}% rate")

    st.markdown("---")

    # -------------------------------------------------------------
    # 3.5 EXPERIMENT SNAPSHOTS & RUN HISTORY
    # -------------------------------------------------------------
    with st.expander("📁 Experiment Snapshots & Run History (Save / Restore / Compare)", expanded=False):
        h_col1, h_col2 = st.columns([1, 1.3])
        with h_col1:
            st.markdown("<h5 style='color:#ffffff; margin-bottom:4px;'>Save Current Configuration</h5>", unsafe_allow_html=True)
            st.caption("Snapshots capture full 3-way groups, thresholds, weights, and divergences for future reference.")
            snap_tag = st.text_input("Run Tag / Note", placeholder="e.g. Balanced tau=0.35, N_min=3", key="l2_snap_tag_input")
            if st.button("💾 Save Snapshot to History", use_container_width=True):
                s_path, saved_tag = save_run_snapshot(active_session_dir, run_payload, snap_tag)
                st.success(f"Saved snapshot: '{saved_tag}'")
                st.rerun()
                
        with h_col2:
            st.markdown("<h5 style='color:#ffffff; margin-bottom:4px;'>Saved Run History</h5>", unsafe_allow_html=True)
            saved_snapshots = list_run_snapshots(active_session_dir)
            if not saved_snapshots:
                st.caption("No snapshots saved for this session yet.")
            else:
                snap_options = {
                    s["filepath"]: f"[{s['saved_at']}] {s['tag']} (A1: {s['num_groups_a1']}, A2: {s['num_groups_a2']}, B2: {s['num_groups_b2']}, Div: {s['divergences_count']})"
                    for s in saved_snapshots
                }
                chosen_snap_fp = st.selectbox(
                    "Select Snapshot",
                    list(snap_options.keys()),
                    format_func=lambda fp: snap_options[fp],
                    key="l2_snap_picker"
                )
                
                chosen_s = next((s for s in saved_snapshots if s["filepath"] == chosen_snap_fp), None)
                if chosen_s:
                    st.caption(
                        f"Config A1: $\\tau$={chosen_s.get('config_a1', {}).get('threshold', 0.35):.2f} | "
                        f"Config A2: $\\tau$={chosen_s.get('config_a2', {}).get('threshold', 0.35):.2f} | "
                        f"Config B2: $\\tau$={chosen_s.get('config_b2', {}).get('threshold', 0.35):.2f}"
                    )
                    b_h1, b_h2, b_h3 = st.columns(3)
                    with b_h1:
                        if st.button("⚡ Restore Sliders", key="btn_restore_snap", use_container_width=True, help="Applies this snapshot's weights and thresholds to the live sliders."):
                            c_a1 = chosen_s.get("config_a1", {})
                            c_a2 = chosen_s.get("config_a2", {})
                            c_b2 = chosen_s.get("config_b2", {})
                            st.session_state["w_ssim_a1"] = float(c_a1.get("ssim", 0.40))
                            st.session_state["w_ocr_a1"] = float(c_a1.get("ocr", 0.40))
                            st.session_state["w_layout_a1"] = float(c_a1.get("layout", 0.20))
                            st.session_state["tau_a1"] = float(c_a1.get("threshold", 0.35))
                            
                            st.session_state["w_ssim_a2"] = float(c_a2.get("ssim", 0.40))
                            st.session_state["w_ocr_a2"] = float(c_a2.get("ocr", 0.40))
                            st.session_state["w_layout_a2"] = float(c_a2.get("layout", 0.20))
                            st.session_state["tau_a2"] = float(c_a2.get("threshold", 0.35))
                            st.session_state["use_layout_a2"] = bool(c_a2.get("use_layout", True))
                            st.session_state["min_tok_a2"] = int(c_a2.get("min_tokens", 3))
                            
                            st.session_state["w_ssim_b2"] = float(c_b2.get("ssim", 1.0))
                            st.session_state["w_ocr_b2"] = float(c_b2.get("ocr", 1.0))
                            st.session_state["w_layout_b2"] = float(c_b2.get("layout", 1.0))
                            st.session_state["w_vit_b2"] = float(c_b2.get("vit", 0.5))
                            st.session_state["tau_b2"] = float(c_b2.get("threshold", 0.35))
                            st.session_state["use_layout_b2"] = bool(c_b2.get("use_layout", True))
                            st.session_state["min_tok_b2"] = int(c_b2.get("min_tokens", 3))
                            st.rerun()
                    with b_h2:
                        snap_full = load_run_snapshot(chosen_snap_fp)
                        if snap_full:
                            snap_json_str = json.dumps(snap_full, indent=2)
                            st.download_button(
                                "📥 Download JSON",
                                data=snap_json_str,
                                file_name=chosen_s["filename"],
                                mime="application/json",
                                use_container_width=True
                            )
                    with b_h3:
                        if st.button("🗑️ Delete", key="btn_del_snap", use_container_width=True):
                            delete_run_snapshot(chosen_snap_fp)
                            st.rerun()

    st.markdown("---")

    # -------------------------------------------------------------
    # 4. DIVERGENCE STUDIO (3-WAY DISAGREEMENT INSPECTOR)
    # -------------------------------------------------------------
    st.markdown(f"#### 4. Divergence & Controlled Ablation Inspector ({len(divergences_3way)} Disagreements)")
    st.caption("Isolates transitions where methods produce opposing decisions. Filter by controlled comparison to inspect specific effects.")
    
    div_filter = st.radio(
        "Filter Divergences",
        [
            f"All Disagreements ({len(divergences_3way)})",
            f"A1 != A2 [Asymmetric Effect] ({len([d for d in divergences_3way if d['pred_a1'] != d['pred_a2']])})",
            f"A2 != B2 [ViT Contribution] ({len([d for d in divergences_3way if d['pred_a2'] != d['pred_b2']])})",
            f"A1 != B2 [Baseline vs Multimodal] ({len([d for d in divergences_3way if d['pred_a1'] != d['pred_b2']])})"
        ],
        horizontal=True,
        key="l2_div_filter"
    )
    
    filtered_divs = divergences_3way
    if "A1 != A2" in div_filter:
        filtered_divs = [d for d in divergences_3way if d['pred_a1'] != d['pred_a2']]
    elif "A2 != B2" in div_filter:
        filtered_divs = [d for d in divergences_3way if d['pred_a2'] != d['pred_b2']]
    elif "A1 != B2" in div_filter:
        filtered_divs = [d for d in divergences_3way if d['pred_a1'] != d['pred_b2']]
        
    if not filtered_divs:
        st.info("No transitions match the selected divergence filter under current settings.")
    else:
        for div_idx, div in enumerate(filtered_divs):
            fa_name = div["frame_a"]
            fb_name = div["frame_b"]
            path_a = os.path.join(target_dir, fa_name)
            path_b = os.path.join(target_dir, fb_name)
            
            with st.expander(f"Divergence #{div_idx + 1}: {fa_name} ({div['timestamp_a']}) -> {fb_name} ({div['timestamp_b']}) | {div['diff_summary']}", expanded=False):
                d_c1, d_c2, d_c3 = st.columns([1, 1, 1.5])
                with d_c1:
                    st.caption(f"**Frame A (Fi):** `{fa_name}` ({div['timestamp_a']})")
                    b64_a = get_image_b64_src(path_a)
                    if b64_a:
                        st.markdown(f"<img src='{b64_a}' style='max-width:100%; border-radius:8px; border:1px solid #3f3f46;'>", unsafe_allow_html=True)
                with d_c2:
                    st.caption(f"**Frame B (Fi+1):** `{fb_name}` ({div['timestamp_b']})")
                    b64_b = get_image_b64_src(path_b)
                    if b64_b:
                        st.markdown(f"<img src='{b64_b}' style='max-width:100%; border-radius:8px; border:1px solid #71717a;'>", unsafe_allow_html=True)
                with d_c3:
                    st.markdown("**Three-Way Diagnostic Comparison:**")
                    st.write(f"- **A1 Baseline Score:** `{div['score_a1']:.4f}` ($\\tau_{{A1}} = {div['thresh_a1']:.2f}$) -> **Pred: `{div['pred_a1']}`**")
                    st.write(f"- **A2 Asymmetric Score:** `{div['score_a2']:.4f}` ($\\tau_{{A2}} = {div['thresh_a2']:.2f}$) -> **Pred: `{div['pred_a2']}`**")
                    st.write(f"- **B2 Multimodal Score:** `{div['score_b2']:.4f}` ($\\tau_{{B2}} = {div['thresh_b2']:.2f}$) -> **Pred: `{div['pred_b2']}`**")
                    st.markdown("---")
                    st.write(f"- **SSIM Distance ($D_{{ssim}}$):** `{div['d_ssim']:.4f}`")
                    st.write(f"- **OCR Jaccard ($D_{{ocr}}$ - A1):** `{div['d_ocr_jaccard']:.4f}`")
                    pres_str = f"`{div['ocr_preservation']:.4f}`" if div['ocr_preservation'] is not None else "Invalid"
                    loss_str = f"`{div['ocr_loss']:.4f}`" if div['ocr_loss'] is not None else "Invalid"
                    st.write(f"- **OCR Containment ($P_{{ocr}}$ / $L_{{ocr}}$ - A2):** {pres_str} / {loss_str} (valid: `{div['ocr_valid']}`)")
                    st.write(f"- **Layout IoU ($D_{{layout}}$ - A1):** `{div['d_layout_iou']:.4f}`")
                    lpres_str = f"`{div['layout_preservation']:.4f}`" if div['layout_preservation'] is not None else "Invalid"
                    lloss_str = f"`{div['layout_loss']:.4f}`" if div['layout_loss'] is not None else "Invalid"
                    st.write(f"- **Directional Layout ($P_{{layout}}$ / $L_{{layout}}$ - A2):** {lpres_str} / {lloss_str}")
                    vit_str = f"`{div['d_vit']:.4f}`" if div['d_vit'] is not None else "None"
                    st.write(f"- **ViT Cosine Distance ($D_{{vit}}$ - B2):** {vit_str}")
                    st.write(f"- **$\\Delta t$:** `{div['delta_time_sec']} sec`")

    st.markdown("---")

    # -------------------------------------------------------------
    # 5. TRIPLE GROUP BROWSER (VISUAL PARTITION INSPECTOR)
    # -------------------------------------------------------------
    st.markdown("#### 5. Triple Group Browser (Visual Inspection)")
    tab_view_a1, tab_view_a2, tab_view_b2 = st.tabs([
        f"A1 Baseline Groups ({len(groups_a1)})",
        f"A2 Asymmetric Groups ({len(groups_a2)})",
        f"B2 Multimodal Groups ({len(groups_b2)})"
    ])
    
    def render_group_cards(groups, badge_color):
        ROW_SIZE = 8
        for g in groups:
            with st.container():
                st.markdown(
                    f"<div style='background-color:#141414; padding:10px 15px; border-radius:8px; border:1px solid #27272a; border-left:4px solid {badge_color}; margin-top:14px; margin-bottom:10px;'>"
                    f"<b style='color:{badge_color}; font-size:1.05rem;'>Group {g['group_id']}</b> &nbsp;|&nbsp; "
                    f"<span>Span: <code>{g['start_timestamp']}</code> -> <code>{g['end_timestamp']}</code> ({g['duration_sec']}s)</span> &nbsp;|&nbsp; "
                    f"<span><b>{g['frame_count']}</b> frames</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )
                
                all_frames = g["frames"]
                for row_start in range(0, len(all_frames), ROW_SIZE):
                    chunk = all_frames[row_start : row_start + ROW_SIZE]
                    cols = st.columns(ROW_SIZE)
                    for f_idx, fr in enumerate(chunk):
                        with cols[f_idx]:
                            fr_path = os.path.join(target_dir, fr["filename"])
                            b64 = get_image_b64_src(fr_path, max_dim=220)
                            if b64:
                                st.markdown(
                                    f"<div style='margin-bottom:8px;'>"
                                    f"<img src='{b64}' style='width:100%; aspect-ratio:16/9; object-fit:cover; border-radius:4px; border:1px solid #27272a; display:block;' loading='lazy'>"
                                    f"<div style='font-size:0.70rem; color:#a1a1aa; line-height:1.25; margin-top:4px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;' title='{fr['filename']} ({fr['timestamp_str']})'>"
                                    f"<span style='color:#ffffff; font-weight:600;'>{fr['timestamp_str']}</span><br/>"
                                    f"<span style='font-family:monospace; font-size:0.68rem;'>{fr['filename']}</span>"
                                    f"</div>"
                                    f"</div>",
                                    unsafe_allow_html=True
                                )

    with tab_view_a1:
        render_group_cards(groups_a1, "#94a3b8")
    with tab_view_a2:
        render_group_cards(groups_a2, "#38bdf8")
    with tab_view_b2:
        render_group_cards(groups_b2, "#c084fc")

    st.markdown("---")

    # -------------------------------------------------------------
    # 6. MANUAL GROUND-TRUTH ANNOTATION STUDIO
    # -------------------------------------------------------------
    st.markdown("#### 6. Manual Ground-Truth Transition Annotation Studio")
    st.write(
        "Ground truth is strictly provided by the researcher. "
        "Label adjacent transitions as **`0 = Same Group (Progressive Build / Continuity)`** or **`1 = New Group (Context Boundary)`**."
    )
    
    gt_labels, gt_notes = load_ground_truth_dataset(active_session_dir)
    
    if "l2_gt_labels" not in st.session_state or st.session_state.get("_l2_active_sess") != selected_session_name:
        st.session_state["l2_gt_labels"] = gt_labels.copy()
        st.session_state["l2_gt_notes"] = gt_notes.copy()
        st.session_state["_l2_active_sess"] = selected_session_name
        
    cur_labels = st.session_state["l2_gt_labels"]
    cur_notes = st.session_state["l2_gt_notes"]
    
    labeled_count = sum(1 for v in cur_labels.values() if v in (0, 1))
    st.info(f"Progress: **{labeled_count} / {len(transitions_list)}** transitions annotated.")
    
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
        pa1 = preds_a1[i]
        pa2 = preds_a2[i]
        pb2 = preds_b2[i]
        
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
            st.caption(f"**Model Suggestions:** A1: `{pa1}` (`{scores_a1[i]:.2f}`) | A2: `{pa2}` (`{scores_a2[i]:.2f}`) | B2: `{pb2}` (`{scores_b2[i]:.2f}`)")
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
                active_session_dir,
                transitions_list,
                cur_labels,
                cur_notes,
                scores_a1=scores_a1,
                preds_a1=preds_a1,
                scores_a2=scores_a2,
                preds_a2=preds_a2,
                scores_b2=scores_b2,
                preds_b2=preds_b2
            )
            st.success(f"Ground truth successfully saved to `{csv_file}`!")
            
    with c_sav2:
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
    st.markdown("#### 7. Three-Way Quantitative Evaluation & Academic Report")
    
    metrics_a1 = GroupingEngine.compute_metrics(preds_a1, cur_labels)
    metrics_a2 = GroupingEngine.compute_metrics(preds_a2, cur_labels)
    metrics_b2 = GroupingEngine.compute_metrics(preds_b2, cur_labels)
    
    if metrics_a1 is None or metrics_a2 is None or metrics_b2 is None:
        st.info("Label transitions in the Annotation Studio above to calculate comparative Precision, Recall, and F1 scores.")
    else:
        e_col1, e_col2, e_col3, e_col4 = st.columns(4)
        with e_col1:
            st.metric("A1 Baseline F1", f"{metrics_a1['f1']:.4f}")
        with e_col2:
            st.metric("A2 Asymmetric F1", f"{metrics_a2['f1']:.4f}", delta=f"{metrics_a2['f1'] - metrics_a1['f1']:+.4f} vs A1")
        with e_col3:
            st.metric("B2 Multimodal F1", f"{metrics_b2['f1']:.4f}", delta=f"{metrics_b2['f1'] - metrics_a2['f1']:+.4f} vs A2")
        with e_col4:
            st.metric("A2 P / R", f"{metrics_a2['precision']:.3f} / {metrics_a2['recall']:.3f}")
            
    if st.button("Generate & Save 3-Way Comparison Report (comparison_report.md)", type="secondary", use_container_width=True):
        report_text, report_path = generate_comparison_markdown_report(
            selected_session_name,
            len(candidate_paths),
            config_a1,
            groups_a1,
            config_a2,
            groups_a2,
            config_b2,
            groups_b2,
            divergences_3way,
            metrics_a1,
            metrics_a2,
            metrics_b2,
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
