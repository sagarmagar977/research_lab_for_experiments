import os
import json
import shutil
import zipfile
import datetime
import numpy as np
import pandas as pd
import cv2
import streamlit as st
from PIL import Image
import joblib

# Import pairwise features coordinator and concrete extractors
from modules.pairwise_feature_lab import (
    PairwiseFeatureExtractor,
    PairwiseFeatureConfig,
    HistogramExtractor,
    EdgeExtractor,
    SSIMExtractor,
    MorphologyExtractor,
    CSVExporter
)
from modules.image_cache import (
    get_cached_thumbnail_b64,
    trigger_background_session_prefetch,
    get_cache_stats
)

import re

def sanitize_session_name(name):
    """Sanitizes folder/session string for cross-platform filesystem safety."""
    clean = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', name.strip())
    return clean if clean else f"session_l1_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

def clean_directory(dir_path):
    """Safely creates/cleans a directory on disk."""
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path, exist_ok=True)

def create_download_zip(selected_dir, zip_filepath):
    """Compresses all selected files into a zip file."""
    with zipfile.ZipFile(zip_filepath, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(selected_dir):
            for file in files:
                filepath = os.path.join(root, file)
                zipf.write(filepath, arcname=file)

def read_frame_image(frame_item):
    if isinstance(frame_item, str):
        return cv2.imread(frame_item)
    else:
        frame_item.seek(0)
        file_bytes = np.asarray(bytearray(frame_item.read()), dtype=np.uint8)
        return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

def save_frame_to_dir(frame_item, target_dir, filename):
    target_path = os.path.join(target_dir, filename)
    if isinstance(frame_item, str):
        shutil.copy2(frame_item, target_path)
    else:
        frame_item.seek(0)
        with open(target_path, "wb") as f:
            f.write(frame_item.getbuffer())

def render_l1_inference():
    st.markdown("### 🧠 Level 1 Model Batch Inference")
    st.write("Run sequential pairwise inference using trained L1 classification models to automatically filter candidate slides and identify duplicate/redundant frames.")

    # --- 1. Load Trained Models from L1_models folder ---
    models_dir = "L1_models"
    if not os.path.exists(models_dir) or not os.path.isdir(models_dir):
        st.error(f"Models directory `{models_dir}` not found. Please ensure you have L1 trained models.")
        return

    # Sort models by modification timestamp descending (newest on top, oldest last)
    pkl_files = [f for f in os.listdir(models_dir) if f.endswith(".pkl")]
    pkl_files.sort(
        key=lambda f: os.path.getmtime(os.path.join(models_dir, f)),
        reverse=True
    )
    if not pkl_files:
        st.error("No pickled models found in `L1_models` directory.")
        return

    def format_model_label(filename):
        try:
            mtime = os.path.getmtime(os.path.join(models_dir, filename))
            dt_str = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
            return f"{filename} ({dt_str})"
        except Exception:
            return filename

    # --- Persistent Model Selection Helpers ---
    def get_saved_l1_model():
        settings_path = "user_settings.json"
        if os.path.exists(settings_path):
            try:
                with open(settings_path, "r") as f:
                    data = json.load(f)
                    return data.get("last_used_l1_model")
            except Exception:
                pass
        return None

    def save_last_used_l1_model(model_filename):
        settings_path = "user_settings.json"
        data = {}
        if os.path.exists(settings_path):
            try:
                with open(settings_path, "r") as f:
                    data = json.load(f)
            except Exception:
                data = {}
        if data.get("last_used_l1_model") != model_filename:
            data["last_used_l1_model"] = model_filename
            try:
                with open(settings_path, "w") as f:
                    json.dump(data, f, indent=4)
            except Exception:
                pass

    saved_model = get_saved_l1_model()
    default_model_idx = 0
    if saved_model and saved_model in pkl_files:
        default_model_idx = pkl_files.index(saved_model)

    def on_model_selection_change():
        current_sel = st.session_state.get("selected_l1_model")
        if current_sel:
            save_last_used_l1_model(current_sel)

    # --- Sidebar Configuration & Model Selection ---
    st.sidebar.markdown("<h4 style='color: #a78bfa; margin-top: 20px; margin-bottom: 0px;'>L1 Model Selection</h4>", unsafe_allow_html=True)
    selected_model_file = st.sidebar.selectbox(
        "Select Classification Model",
        pkl_files,
        index=default_model_idx,
        format_func=format_model_label,
        key="selected_l1_model",
        on_change=on_model_selection_change
    )

    if selected_model_file:
        save_last_used_l1_model(selected_model_file)

    # Load selected model
    model_path = os.path.join(models_dir, selected_model_file)
    try:
        model_dict = joblib.load(model_path)
        model_name = model_dict.get("model_name", selected_model_file)
        pipeline = model_dict.get("pipeline")
        feature_columns = model_dict.get("feature_columns")
        metrics = model_dict.get("metrics", {})
        
        st.sidebar.success(f"Loaded: **{model_name}**")
        
        # Read default threshold from model dictionary, default to 0.50 if not found
        default_thresh = float(model_dict.get("threshold", 0.50))
        
        # Add dynamic classification threshold slider in the sidebar
        custom_threshold = st.sidebar.slider(
            "Classification Threshold",
            min_value=0.01,
            max_value=0.99,
            value=default_thresh,
            step=0.01,
            help="Probability score above which a frame is classified as a slide transition candidate."
        )
        
        # Display model metrics in sidebar
        if metrics:
            st.sidebar.markdown("**Model Performance Metrics:**")
            for metric_k, metric_v in metrics.items():
                if isinstance(metric_v, float):
                    st.sidebar.write(f"- {metric_k}: `{metric_v:.4f}`")
                else:
                    st.sidebar.write(f"- {metric_k}: `{metric_v}`")
    except Exception as e:
        st.sidebar.error(f"Failed to load model: {str(e)}")
        return

    # --- Input Source: Triple input method ---
    st.markdown("#### 📥 Select Input Source")
    input_source = st.radio(
        "Choose how to load frames for batch inference",
        ["Select Existing Batch Crop Session", "Upload New Sequential Frames", "Load from Local Folder Path"],
        horizontal=True
    )

    image_paths = []
    target_session_name = ""
    target_engine = "PP-OCRv3"
    save_mode = "Update Current Session in-place"
    
    if input_source == "Select Existing Batch Crop Session":
        sessions_root = "sessions"
        if not os.path.exists(sessions_root) or not os.path.isdir(sessions_root):
            st.info("No batch sessions found. Please run batch cropping first or upload frames directly.")
            return
            
        session_dirs = sorted([d for d in os.listdir(sessions_root) if os.path.isdir(os.path.join(sessions_root, d))], reverse=True)
        if not session_dirs:
            st.info("No batch sessions found. Please run batch cropping first or upload frames directly.")
            return
            
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            selected_session = st.selectbox("Select Session", session_dirs, key="l1_sess_cand")
        with col_s2:
            session_path = os.path.join(sessions_root, selected_session)
            available_engines = []
            if os.path.exists(os.path.join(session_path, "v3_crops")):
                available_engines.append("PP-OCRv3")
            if os.path.exists(os.path.join(session_path, "v4_crops")):
                available_engines.append("PP-OCRv4")
            if not available_engines:
                available_engines = ["PP-OCRv3"]
            selected_engine = st.radio("Select Crop Source Engine", available_engines, horizontal=True, key="l1_eng_cand")
            
        save_mode = st.radio(
            "Candidate Selector Session Target:",
            ["Update Current Session in-place", "Branch into New Session (<session>_l1_curated)"],
            horizontal=True,
            key="l1_save_mode_choice"
        )
        if save_mode == "Branch into New Session (<session>_l1_curated)":
            target_session_name = f"{selected_session}_l1_curated"
        else:
            target_session_name = selected_session
            
        target_engine = selected_engine
        crop_dir_name = "v3_crops" if selected_engine == "PP-OCRv3" else "v4_crops"
        session_crop_dir = os.path.join(session_path, crop_dir_name)
        
        if not os.path.exists(session_crop_dir) or not os.path.isdir(session_crop_dir):
            st.warning(f"No cropped images found for engine {selected_engine} in session `{selected_session}`.")
            return
            
        cropped_files = sorted([f for f in os.listdir(session_crop_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        if not cropped_files:
            st.warning("No crop frames found in this session.")
            return
            
        image_paths = [os.path.join(session_crop_dir, f) for f in cropped_files]
        st.success(f"Ready to process **{len(image_paths)}** frames from session: `{selected_session}` ({selected_engine}) ➔ Target: `{target_session_name}`")

    elif input_source == "Upload New Sequential Frames":
        col_up1, col_up2 = st.columns([2, 1])
        with col_up1:
            uploaded_files = st.file_uploader(
                "Upload Sequential Frames (will be sorted alphabetically by filename)",
                type=["jpg", "jpeg", "png"],
                accept_multiple_files=True,
                key="l1_upload_files"
            )
        with col_up2:
            default_upload_sess = f"session_upload_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            session_name_input = st.text_input("Session / Folder Name for Candidate Selector", value=default_upload_sess, key="l1_up_sess_name")
            target_session_name = sanitize_session_name(session_name_input)
            
        target_engine = "PP-OCRv3"
        if not uploaded_files:
            st.info("Upload sequential frames to run inference.")
            return
            
        uploaded_files = sorted(uploaded_files, key=lambda x: x.name)
        image_paths = uploaded_files
        st.success(f"Ready to process **{len(image_paths)}** uploaded frames ➔ Target Session: `{target_session_name}`")

    else:
        # Load from Local Folder Path
        col_loc1, col_loc2 = st.columns([2, 1])
        with col_loc1:
            local_folder_path = st.text_input("Local Folder Path", placeholder="e.g. F:/thesis/frames or D:/my_video_crops", key="l1_local_path")
        
        target_engine = "PP-OCRv3"
        if not local_folder_path or not os.path.isdir(local_folder_path):
            st.info("Enter a valid existing local directory path containing image frames.")
            return
            
        detected_files = sorted([f for f in os.listdir(local_folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        if not detected_files:
            st.warning(f"No image files (.png, .jpg, .jpeg) found in `{local_folder_path}`.")
            return
            
        folder_base_name = os.path.basename(os.path.normpath(local_folder_path))
        with col_loc2:
            session_name_input = st.text_input("Session / Folder Name for Candidate Selector", value=folder_base_name, key="l1_local_sess_name")
            target_session_name = sanitize_session_name(session_name_input)
            
        image_paths = [os.path.join(local_folder_path, f) for f in detected_files]
        st.success(f"Found **{len(image_paths)}** frames in local folder ➔ Target Session: `{target_session_name}`")

    # Define target session structure
    target_session_dir = os.path.join("sessions", target_session_name)
    crop_sub_name = "v3_crops" if target_engine == "PP-OCRv3" else "v4_crops"
    cand_sub_name = "v3_candidate_frames" if target_engine == "PP-OCRv3" else "v4_candidate_frames"
    
    target_crop_dir = os.path.join(target_session_dir, crop_sub_name)
    target_cand_dir = os.path.join(target_session_dir, cand_sub_name)
    target_orig_dir = os.path.join(target_session_dir, "original_frames")
    target_identical_dir = os.path.join(target_session_dir, "identical_frames")

    # --- 2. RUN INFERENCE PIPELINE ---
    run_inference_btn = st.button("🚀 Run L1 Batch Inference", type="primary", use_container_width=True)
    
    if run_inference_btn:
        if len(image_paths) < 2:
            st.error("Need at least 2 frames to perform sequential pairwise inference.")
            return
            
        # Ensure session directory layout
        os.makedirs(target_crop_dir, exist_ok=True)
        os.makedirs(target_orig_dir, exist_ok=True)
        clean_directory(target_cand_dir)
        clean_directory(target_identical_dir)
        
        # Persist source frames if uploaded or local folder
        if input_source in ("Upload New Sequential Frames", "Load from Local Folder Path"):
            persisted_paths = []
            for f_item in image_paths:
                f_name = f_item.name if not isinstance(f_item, str) else os.path.basename(f_item)
                save_frame_to_dir(f_item, target_crop_dir, f_name)
                save_frame_to_dir(f_item, target_orig_dir, f_name)
                persisted_paths.append(os.path.join(target_crop_dir, f_name))
            image_paths = persisted_paths
        elif input_source == "Select Existing Batch Crop Session" and save_mode == "Branch into New Session (<session>_l1_curated)":
            for f_p in image_paths:
                f_name = os.path.basename(f_p)
                shutil.copy2(f_p, os.path.join(target_crop_dir, f_name))
            src_orig = os.path.join(session_path, "original_frames")
            if os.path.exists(src_orig):
                for f in os.listdir(src_orig):
                    s_file = os.path.join(src_orig, f)
                    if os.path.isfile(s_file):
                        shutil.copy2(s_file, os.path.join(target_orig_dir, f))
        
        # Build features config from st.session_state
        config = PairwiseFeatureConfig(
            hist_bins=st.session_state["hist_bins"],
            hist_method=st.session_state["hist_method"],
            color_mode=st.session_state["color_mode"],
            hist_grid_size=st.session_state["hist_grid_size"],
            edge_blur=st.session_state["edge_blur"],
            canny_low=st.session_state["canny_low"],
            canny_high=st.session_state["canny_high"],
            edge_grid_size=st.session_state["edge_grid_size"],
            ssim_win_size=st.session_state["ssim_win_size"],
            ssim_gaussian=st.session_state["ssim_gaussian"],
            text_thresh=st.session_state["text_thresh"],
            text_kernel=st.session_state["text_kernel"],
            text_iterations=st.session_state["text_iterations"],
            text_min_area=st.session_state["text_min_area"],
            hist_epsilon=st.session_state.get("hist_epsilon", 1e-10)
        )
        
        # Initialize Feature Extractor
        extractor = PairwiseFeatureExtractor(config)
        extractor.register_extractor(HistogramExtractor())
        extractor.register_extractor(EdgeExtractor())
        extractor.register_extractor(SSIMExtractor())
        extractor.register_extractor(MorphologyExtractor())
        
        results = []
        transitions_cache = []
        # First frame is automatically selected (Keep = 1)
        first_frame = image_paths[0]
        first_frame_name = first_frame.name if not isinstance(first_frame, str) else os.path.basename(first_frame)
        
        save_frame_to_dir(first_frame, target_cand_dir, first_frame_name)
        
        results.append({
            "frame_idx": 1,
            "filename": first_frame_name,
            "path": os.path.join(target_cand_dir, first_frame_name),
            "prediction": 1,
            "type": "Keep (First Frame)"
        })
        
        progress_bar = st.progress(0.0)
        status_text = st.empty()
        
        headers = CSVExporter.get_headers()
        
        # Loop sequentially: pair (i, i+1)
        for i in range(len(image_paths) - 1):
            frame_a = image_paths[i]
            frame_b = image_paths[i+1]
            frame_b_name = frame_b.name if not isinstance(frame_b, str) else os.path.basename(frame_b)
            frame_a_name = frame_a.name if not isinstance(frame_a, str) else os.path.basename(frame_a)
            
            status_text.text(f"Processing transition {i+1}/{len(image_paths)-1}: {frame_a_name} ➔ {frame_b_name}...")
            
            # Read images
            img_a = read_frame_image(frame_a)
            img_b = read_frame_image(frame_b)
            if img_a is None or img_b is None:
                st.warning(f"Skipping unreadable frame pair: {frame_a_name} or {frame_b_name}")
                continue
                
            img_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB)
            img_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB)
            
            # Extract features
            logs = []
            fa, fb, pf, art, logs = extractor.extract(img_a, img_b)
            
            # Compile dictionary of all extracted features
            all_vals = fa.to_list() + fb.to_list() + pf.to_list()
            feature_dict = dict(zip(headers, all_vals))
            
            # Filter and order features for the selected model
            X_features = []
            for col in feature_columns:
                if col in feature_dict:
                    X_features.append(feature_dict[col])
                else:
                    st.error(f"Missing required model feature: `{col}` from feature extraction pipeline.")
                    st.stop()
            
            X_array = np.array([X_features])
            
            # Run inference using predict_proba and custom threshold
            y_proba_val = float(pipeline.predict_proba(X_array)[0, 1])
            pred = 1 if y_proba_val >= custom_threshold else 0
            
            # Store raw transition prediction record for instant real-time threshold slider adjustments
            transitions_cache.append({
                "pair_idx": i + 2,
                "frame_b_name": frame_b_name,
                "y_proba": y_proba_val
            })
            
            # Assign prediction to the second frame (Frame B)
            src_frame_b = os.path.join(target_crop_dir, frame_b_name)
            if pred == 1:
                save_frame_to_dir(src_frame_b, target_cand_dir, frame_b_name)
                results.append({
                    "frame_idx": i + 2,
                    "filename": frame_b_name,
                    "path": os.path.join(target_cand_dir, frame_b_name),
                    "prediction": 1,
                    "probability": y_proba_val,
                    "type": "Keep (Transition detected)"
                })
            else:
                save_frame_to_dir(src_frame_b, target_identical_dir, frame_b_name)
                results.append({
                    "frame_idx": i + 2,
                    "filename": frame_b_name,
                    "path": os.path.join(target_identical_dir, frame_b_name),
                    "prediction": 0,
                    "probability": y_proba_val,
                    "type": "Discard (Redundant/Duplicate)"
                })
                
            progress_bar.progress((i + 1) / (len(image_paths) - 1))
            
        status_text.text("Batch inference completed!")
        st.session_state["l1_results"] = results
        st.session_state["l1_transitions_cache"] = transitions_cache
        st.session_state["l1_first_frame_info"] = {
            "first_frame_name": first_frame_name
        }
        st.session_state["l1_target_session_name"] = target_session_name
        st.session_state["l1_target_engine"] = target_engine
        st.session_state["l1_target_crop_dir"] = target_crop_dir
        st.session_state["l1_target_cand_dir"] = target_cand_dir
        st.session_state["l1_target_orig_dir"] = target_orig_dir
        st.session_state["l1_target_identical_dir"] = target_identical_dir
        st.session_state["l1_last_threshold"] = custom_threshold
            
    # --- 2.5 INSTANT THRESHOLD RE-FILTERING ON SLIDER MOVEMENT (0ms DELAY) ---
    if "l1_transitions_cache" in st.session_state and "l1_results" in st.session_state:
        last_thresh = st.session_state.get("l1_last_threshold", None)
        if last_thresh is not None and abs(last_thresh - custom_threshold) > 1e-4:
            st.session_state["l1_last_threshold"] = custom_threshold
            
            transitions_cache = st.session_state["l1_transitions_cache"]
            t_crop_dir = st.session_state.get("l1_target_crop_dir", "")
            t_cand_dir = st.session_state.get("l1_target_cand_dir", "")
            t_identical_dir = st.session_state.get("l1_target_identical_dir", "")
            first_info = st.session_state.get("l1_first_frame_info", {})
            
            clean_directory(t_cand_dir)
            clean_directory(t_identical_dir)
            
            updated_results = []
            
            # Keep first frame as candidate
            if first_info:
                ff_name = first_info["first_frame_name"]
                src_ff = os.path.join(t_crop_dir, ff_name)
                if os.path.exists(src_ff):
                    save_frame_to_dir(src_ff, t_cand_dir, ff_name)
                updated_results.append({
                    "frame_idx": 1,
                    "filename": ff_name,
                    "path": os.path.join(t_cand_dir, ff_name),
                    "prediction": 1,
                    "probability": 1.0,
                    "type": "Keep (First Frame)"
                })
                
            # Instant re-classification for all cached transition probabilities
            for t_item in transitions_cache:
                fb_name = t_item["frame_b_name"]
                prob = t_item["y_proba"]
                p_idx = t_item["pair_idx"]
                src_fb = os.path.join(t_crop_dir, fb_name)
                
                new_pred = 1 if prob >= custom_threshold else 0
                
                if new_pred == 1:
                    save_frame_to_dir(src_fb, t_cand_dir, fb_name)
                    updated_results.append({
                        "frame_idx": p_idx,
                        "filename": fb_name,
                        "path": os.path.join(t_cand_dir, fb_name),
                        "prediction": 1,
                        "probability": prob,
                        "type": "Keep (Transition detected)"
                    })
                else:
                    save_frame_to_dir(src_fb, t_identical_dir, fb_name)
                    updated_results.append({
                        "frame_idx": p_idx,
                        "filename": fb_name,
                        "path": os.path.join(t_identical_dir, fb_name),
                        "prediction": 0,
                        "probability": prob,
                        "type": "Discard (Redundant/Duplicate)"
                    })
                    
            st.session_state["l1_results"] = updated_results
            st.toast(f"⚡ Instant Threshold Update: Applied T = {custom_threshold:.2f} (0ms delay)", icon="⚡")
        
    # --- 3. DISPLAY RESULTS GALLERIES ---
    if "l1_results" in st.session_state:
        results = st.session_state["l1_results"]
        target_sess_name = st.session_state.get("l1_target_session_name", "candidate_session")
        target_eng = st.session_state.get("l1_target_engine", "PP-OCRv3")
        t_cand_dir = st.session_state.get("l1_target_cand_dir", "")
        t_orig_dir = st.session_state.get("l1_target_orig_dir", "")
        
        selected_frames = [r for r in results if r["prediction"] == 1]
        identical_frames = [r for r in results if r["prediction"] == 0]
        
        st.markdown("---")
        st.markdown("### 📊 Inference Results Summary")
        
        # Display Metrics Cards
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-title">Total Input Frames</div>'
                f'<div class="metric-value">{len(results)}</div>'
                f'</div>',
                unsafe_allow_html=True
            )
        with col_m2:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-title">Selected Candidates (Keep)</div>'
                f'<div class="metric-value" style="color: #10b981;">{len(selected_frames)}</div>'
                f'</div>',
                unsafe_allow_html=True
            )
        with col_m3:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-title">Identical / Redundant (Discard)</div>'
                f'<div class="metric-value" style="color: #ef4444;">{len(identical_frames)}</div>'
                f'</div>',
                unsafe_allow_html=True
            )
            
        # --- SEND & EDIT IN CANDIDATE SELECTOR BANNER ---
        st.markdown("---")
        col_banner1, col_banner2 = st.columns([3, 1.5])
        with col_banner1:
            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(167, 139, 250, 0.15) 100%); 
                            border: 1px solid #6366f1; border-radius: 12px; padding: 14px 18px;">
                    <div style="color: #a78bfa; font-weight: 700; font-size: 1.05rem; margin-bottom: 4px;">
                        🎯 Candidate Frame Selector Session Ready
                    </div>
                    <div style="color: #e2e8f0; font-size: 0.9rem;">
                        Session: <b><code>{target_sess_name}</code></b> ({target_eng}) &nbsp;|&nbsp; 
                        Pre-marked candidates: <b style="color: #10b981;">{len(selected_frames)}</b> / {len(results)} frames
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with col_banner2:
            st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)
            def goto_candidate_selector_cb(sess_name, eng):
                st.session_state["selected_module"] = "Candidate Frame Selector"
                st.session_state["sel_sess_cand"] = sess_name
                st.session_state["sel_eng_cand"] = eng

            st.button(
                "🚀 Open in Candidate Selector",
                type="primary",
                use_container_width=True,
                key="btn_open_in_cand_sel",
                on_click=goto_candidate_selector_cb,
                args=(target_sess_name, target_eng)
            )

        # Download ZIP buttons (Lazy preparation to avoid blocking UI execution)
        st.write("")
        col_dl1, col_dl2 = st.columns(2)
        crops_zip_path = os.path.join("sessions", target_sess_name, f"{target_sess_name}_selected_crops.zip")
        orig_zip_path = os.path.join("sessions", target_sess_name, f"{target_sess_name}_selected_originals.zip")

        if t_cand_dir and os.path.exists(t_cand_dir):
            if os.path.exists(crops_zip_path):
                with col_dl1:
                    with open(crops_zip_path, "rb") as z_file:
                        st.download_button(
                            label=f"💾 Download Selected Crops ZIP ({len(selected_frames)} images)",
                            data=z_file,
                            file_name=f"{target_sess_name}_selected_crops.zip",
                            mime="application/zip",
                            use_container_width=True
                        )
            else:
                with col_dl1:
                    if st.button(f"📦 Prepare Crops ZIP ({len(selected_frames)} images)", use_container_width=True, key="btn_prep_crops_zip"):
                        with st.spinner("Compressing selected crops..."):
                            create_download_zip(t_cand_dir, crops_zip_path)
                        st.rerun()
                    
        if t_orig_dir and os.path.exists(t_orig_dir) and len(os.listdir(t_orig_dir)) > 0:
            if os.path.exists(orig_zip_path):
                with col_dl2:
                    with open(orig_zip_path, "rb") as oz_file:
                        st.download_button(
                            label=f"🎬 Download Selected Original Frames ZIP ({len(selected_frames)} images)",
                            data=oz_file,
                            file_name=f"{target_sess_name}_selected_originals.zip",
                            mime="application/zip",
                            use_container_width=True
                        )
            else:
                with col_dl2:
                    if st.button(f"🎬 Prepare Original Frames ZIP ({len(selected_frames)} images)", use_container_width=True, key="btn_prep_orig_zip"):
                        with st.spinner("Compressing original frames..."):
                            with zipfile.ZipFile(orig_zip_path, 'w', zipfile.ZIP_DEFLATED) as ozf:
                                for r in selected_frames:
                                    fname = r["filename"]
                                    fpath = os.path.join(t_orig_dir, fname)
                                    if os.path.exists(fpath):
                                        ozf.write(fpath, arcname=fname)
                        st.rerun()
            
        # Interactive Galleries columns
        st.markdown("#### 🖼️ Results Gallery View")
        
        # Initialize active tab in session state if not present
        if "l1_gallery_tab" not in st.session_state:
            st.session_state["l1_gallery_tab"] = "All Uploaded"
            
        col_b1, col_b2, col_b3 = st.columns(3)
        with col_b1:
            if st.button(f"📁 All Uploaded ({len(results)})", use_container_width=True, type="primary" if st.session_state["l1_gallery_tab"] == "All Uploaded" else "secondary"):
                st.session_state["l1_gallery_tab"] = "All Uploaded"
                st.rerun()
        with col_b2:
            if st.button(f"🎯 Selected Candidates ({len(selected_frames)})", use_container_width=True, type="primary" if st.session_state["l1_gallery_tab"] == "Selected Candidates" else "secondary"):
                st.session_state["l1_gallery_tab"] = "Selected Candidates"
                st.rerun()
        with col_b3:
            if st.button(f"🔁 Identical / Redundant ({len(identical_frames)})", use_container_width=True, type="primary" if st.session_state["l1_gallery_tab"] == "Identical / Redundant" else "secondary"):
                st.session_state["l1_gallery_tab"] = "Identical / Redundant"
                st.rerun()
                
        # Slider to adjust grid columns (dynamic sizing)
        grid_cols = st.slider("Adjust Grid Columns (Image Size)", min_value=2, max_value=6, value=4, step=1, key="l1_gallery_grid_cols")
        
        active_tab = st.session_state["l1_gallery_tab"]
        if active_tab == "All Uploaded":
            active_frames = results
        elif active_tab == "Selected Candidates":
            active_frames = selected_frames
        else:
            active_frames = identical_frames
            
        if not active_frames:
            st.info(f"No frames in category: {active_tab}")
        else:
            # Pagination for gallery
            frames_per_page = 24
            total_items = len(active_frames)
            total_pages = max(1, (total_items + frames_per_page - 1) // frames_per_page)
            
            page_key = f"l1_gallery_page_{active_tab}"
            if page_key not in st.session_state:
                st.session_state[page_key] = 1
                
            if st.session_state[page_key] > total_pages:
                st.session_state[page_key] = total_pages
            elif st.session_state[page_key] < 1:
                st.session_state[page_key] = 1
                
            curr_page = st.session_state[page_key]
            start_idx = (curr_page - 1) * frames_per_page
            end_idx = min(start_idx + frames_per_page, total_items)
            page_frames = active_frames[start_idx:end_idx]

            # Trigger background prefetching for the full session across all pages
            all_session_paths = [r["path"] for r in results]
            target_sess = st.session_state.get("l1_target_session_name", "l1_session")
            trigger_background_session_prefetch(all_session_paths, current_page=curr_page, page_size=frames_per_page, session_key=target_sess)

            # Retrieve RAM cache readiness statistics for current tab
            all_frame_paths = [r["path"] for r in active_frames]
            cached_count, total_count = get_cache_stats(all_frame_paths)
            if cached_count >= total_count:
                cache_status_html = f"<div style='margin-top: 3px; font-size: 0.82rem; color: #10b981;'>⚡ <b>{cached_count:,} / {total_count:,}</b> frames pre-warmed in RAM (Instant Browsing Active)</div>"
            else:
                cache_status_html = f"<div style='margin-top: 3px; font-size: 0.82rem; color: #94a3b8;'>⚡ RAM Cache: <b>{cached_count:,} / {total_count:,}</b> ready (caching session in background...)</div>"

            # Top Pagination Controls
            col_pg_l, col_pg_info, col_pg_r = st.columns([1, 2, 1])
            with col_pg_l:
                if st.button("◀ Previous Page", disabled=(curr_page <= 1), key=f"btn_prev_{active_tab}", use_container_width=True):
                    st.session_state[page_key] -= 1
                    st.rerun()
            with col_pg_info:
                st.markdown(
                    f"<div style='text-align: center; padding-top: 4px;'>"
                    f"<div style='font-size: 0.95rem;'>Page <b>{curr_page}</b> of <b>{total_pages}</b> &nbsp;|&nbsp; "
                    f"Showing <b>{start_idx + 1}–{end_idx}</b> of <b>{total_items:,}</b> frames</div>"
                    f"{cache_status_html}"
                    f"</div>",
                    unsafe_allow_html=True
                )
            with col_pg_r:
                if st.button("Next Page ▶", disabled=(curr_page >= total_pages), key=f"btn_next_{active_tab}", use_container_width=True):
                    st.session_state[page_key] += 1
                    st.rerun()

            with st.container(border=True):
                for idx in range(0, len(page_frames), grid_cols):
                    row_frames = page_frames[idx : idx + grid_cols]
                    cols = st.columns(grid_cols)
                    for col_idx, r in enumerate(row_frames):
                        with cols[col_idx]:
                            is_keep = r["prediction"] == 1
                            border_color = "#10b981" if is_keep else "#ef4444"
                            bg_color = "rgba(16, 185, 129, 0.05)" if is_keep else "rgba(239, 68, 68, 0.05)"
                            shadow = "0 0 8px rgba(16, 185, 129, 0.15)" if is_keep else "0 0 8px rgba(239, 68, 68, 0.15)"
                            
                            st.markdown(
                                f'<div style="border: 2px solid {border_color}; border-radius: 12px; padding: 6px; background-color: {bg_color}; box-shadow: {shadow}; margin-bottom: 8px;">'
                                f'<img src="data:image/jpeg;base64,{get_cached_thumbnail_b64(r["path"])}" style="width: 100%; border-radius: 8px; display: block;"/>'
                                f'</div>',
                                unsafe_allow_html=True
                            )
                            label_class = "status-success" if is_keep else "status-warning"
                            st.markdown(f"**{r['frame_idx']}. {r['filename']}**")
                            st.markdown(f'<span class="status-badge {label_class}">{r["type"]}</span>', unsafe_allow_html=True)
                            st.markdown("---")

            # Bottom Pagination Controls
            if total_pages > 1:
                st.write("")
                col_bpg_l, col_bpg_info, col_bpg_r = st.columns([1, 2, 1])
                with col_bpg_l:
                    if st.button("◀ Previous Page", disabled=(curr_page <= 1), key=f"btn_bprev_{active_tab}", use_container_width=True):
                        st.session_state[page_key] -= 1
                        st.rerun()
                with col_bpg_info:
                    st.markdown(
                        f"<div style='text-align: center; padding-top: 6px; font-size: 0.95rem;'>"
                        f"Page <b>{curr_page}</b> of <b>{total_pages}</b>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                with col_bpg_r:
                    if st.button("Next Page ▶", disabled=(curr_page >= total_pages), key=f"btn_bnext_{active_tab}", use_container_width=True):
                        st.session_state[page_key] += 1
                        st.rerun()

# Backwards-compatibility aliases for thumbnail cache
get_base64_from_filepath = get_cached_thumbnail_b64
trigger_background_prefetch = trigger_background_session_prefetch
