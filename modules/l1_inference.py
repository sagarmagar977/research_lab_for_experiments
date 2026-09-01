import os
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

    pkl_files = sorted([f for f in os.listdir(models_dir) if f.endswith(".pkl")])
    if not pkl_files:
        st.error("No pickled models found in `L1_models` directory.")
        return

    # --- Sidebar Configuration & Model Selection ---
    st.sidebar.markdown("<h4 style='color: #a78bfa; margin-top: 20px; margin-bottom: 0px;'>L1 Model Selection</h4>", unsafe_allow_html=True)
    selected_model_file = st.sidebar.selectbox(
        "Select Classification Model",
        pkl_files,
        key="selected_l1_model"
    )

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

    # --- Input Source: Dual input method ---
    st.markdown("#### 📥 Select Input Source")
    input_source = st.radio(
        "Choose how to load frames for batch inference",
        ["Select Existing Batch Crop Session", "Upload New Sequential Frames"],
        horizontal=True
    )

    image_paths = []
    session_crop_dir = ""
    session_out_dir = ""
    
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
            selected_engine = st.radio("Select Crop Source Engine", ["PP-OCRv3", "PP-OCRv4"], horizontal=True, key="l1_eng_cand")
            
        session_path = os.path.join(sessions_root, selected_session)
        crop_dir_name = "v3_crops" if selected_engine == "PP-OCRv3" else "v4_crops"
        session_crop_dir = os.path.join(session_path, crop_dir_name)
        
        # Define output directory inside the session for reproducibility
        session_out_dir = os.path.join(session_path, f"l1_inference_{selected_model_file.replace('.pkl', '')}_{crop_dir_name}")
        
        if not os.path.exists(session_crop_dir) or not os.path.isdir(session_crop_dir):
            st.warning(f"No cropped images found for engine {selected_engine} in session `{selected_session}`.")
            return
            
        cropped_files = sorted([f for f in os.listdir(session_crop_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        if not cropped_files:
            st.warning("No crop frames found in this session.")
            return
            
        # Map full paths
        image_paths = [os.path.join(session_crop_dir, f) for f in cropped_files]
        st.success(f"Ready to process **{len(image_paths)}** frames from session: `{selected_session}` ({selected_engine})")
        
    else:
        # Uploading new sequential frames
        uploaded_files = st.file_uploader(
            "Upload Sequential Frames (will be sorted alphabetically by filename)",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="l1_upload_files"
        )
        
        if not uploaded_files:
            st.info("Upload sequential frames to run inference.")
            return
            
        # Sort files by filename to ensure sequential order
        uploaded_files = sorted(uploaded_files, key=lambda x: x.name)
        image_paths = uploaded_files
        
        session_out_dir = os.path.join("sessions", "l1_inference_temp")
        st.success(f"Ready to process **{len(image_paths)}** uploaded frames.")

    # --- 2. RUN INFERENCE PIPELINE ---
    run_inference_btn = st.button("🚀 Run L1 Batch Inference", type="primary", use_container_width=True)
    
    if run_inference_btn:
        if len(image_paths) < 2:
            st.error("Need at least 2 frames to perform sequential pairwise inference.")
            return
            
        # Set up outputs directories
        selected_dir = os.path.join(session_out_dir, "selected_frames")
        identical_dir = os.path.join(session_out_dir, "identical_frames")
        
        clean_directory(selected_dir)
        clean_directory(identical_dir)
        
        # Set up original selected frames output directory if existing session
        orig_selected_dir = os.path.join(session_out_dir, "selected_original_frames")
        if input_source == "Select Existing Batch Crop Session":
            clean_directory(orig_selected_dir)
        
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
        first_frame_path_str = first_frame.name if not isinstance(first_frame, str) else first_frame
        
        save_frame_to_dir(first_frame, selected_dir, first_frame_name)
        if input_source == "Select Existing Batch Crop Session":
            orig_frame_src = os.path.join("sessions", selected_session, "original_frames", first_frame_name)
            if os.path.exists(orig_frame_src):
                shutil.copy2(orig_frame_src, os.path.join(orig_selected_dir, first_frame_name))
                
        results.append({
            "frame_idx": 1,
            "filename": first_frame_name,
            "path": os.path.join(selected_dir, first_frame_name),
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
            frame_b_path_str = frame_b.name if not isinstance(frame_b, str) else frame_b
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
                "frame_b": frame_b,
                "frame_b_name": frame_b_name,
                "y_proba": y_proba_val
            })
            
            # Assign prediction to the second frame (Frame B)
            if pred == 1:
                save_frame_to_dir(frame_b, selected_dir, frame_b_name)
                if input_source == "Select Existing Batch Crop Session":
                    orig_frame_src = os.path.join("sessions", selected_session, "original_frames", frame_b_name)
                    if os.path.exists(orig_frame_src):
                        shutil.copy2(orig_frame_src, os.path.join(orig_selected_dir, frame_b_name))
                results.append({
                    "frame_idx": i + 2,
                    "filename": frame_b_name,
                    "path": os.path.join(selected_dir, frame_b_name),
                    "prediction": 1,
                    "probability": y_proba_val,
                    "type": "Keep (Transition detected)"
                })
            else:
                save_frame_to_dir(frame_b, identical_dir, frame_b_name)
                results.append({
                    "frame_idx": i + 2,
                    "filename": frame_b_name,
                    "path": os.path.join(identical_dir, frame_b_name),
                    "prediction": 0,
                    "probability": y_proba_val,
                    "type": "Discard (Redundant/Duplicate)"
                })
                
            progress_bar.progress((i + 1) / (len(image_paths) - 1))
            
        status_text.text("Batch inference completed!")
        st.session_state["l1_results"] = results
        st.session_state["l1_transitions_cache"] = transitions_cache
        st.session_state["l1_first_frame_info"] = {
            "first_frame": first_frame,
            "first_frame_name": first_frame_name
        }
        st.session_state["l1_out_dir"] = session_out_dir
        st.session_state["l1_input_source"] = input_source
        st.session_state["l1_last_threshold"] = custom_threshold
        if input_source == "Select Existing Batch Crop Session":
            st.session_state["l1_selected_session"] = selected_session
            
    # --- 2.5 INSTANT THRESHOLD RE-FILTERING ON SLIDER MOVEMENT (0ms DELAY) ---
    if "l1_transitions_cache" in st.session_state and "l1_results" in st.session_state:
        # Check if threshold slider changed since last evaluation
        last_thresh = st.session_state.get("l1_last_threshold", None)
        if last_thresh is not None and abs(last_thresh - custom_threshold) > 1e-4:
            st.session_state["l1_last_threshold"] = custom_threshold
            
            transitions_cache = st.session_state["l1_transitions_cache"]
            out_dir = st.session_state["l1_out_dir"]
            input_src = st.session_state.get("l1_input_source", "")
            sel_sess = st.session_state.get("l1_selected_session", "")
            first_info = st.session_state.get("l1_first_frame_info", {})
            
            selected_dir = os.path.join(out_dir, "selected_frames")
            identical_dir = os.path.join(out_dir, "identical_frames")
            orig_selected_dir = os.path.join(out_dir, "selected_original_frames")
            
            clean_directory(selected_dir)
            clean_directory(identical_dir)
            if input_src == "Select Existing Batch Crop Session":
                clean_directory(orig_selected_dir)
                
            updated_results = []
            
            # Keep first frame as candidate
            if first_info:
                ff_frame = first_info["first_frame"]
                ff_name = first_info["first_frame_name"]
                save_frame_to_dir(ff_frame, selected_dir, ff_name)
                if input_src == "Select Existing Batch Crop Session" and sel_sess:
                    orig_src = os.path.join("sessions", sel_sess, "original_frames", ff_name)
                    if os.path.exists(orig_src):
                        shutil.copy2(orig_src, os.path.join(orig_selected_dir, ff_name))
                updated_results.append({
                    "frame_idx": 1,
                    "filename": ff_name,
                    "path": os.path.join(selected_dir, ff_name),
                    "prediction": 1,
                    "probability": 1.0,
                    "type": "Keep (First Frame)"
                })
                
            # Instant re-classification for all cached transition probabilities
            for t_item in transitions_cache:
                fb = t_item["frame_b"]
                fb_name = t_item["frame_b_name"]
                prob = t_item["y_proba"]
                p_idx = t_item["pair_idx"]
                
                new_pred = 1 if prob >= custom_threshold else 0
                
                if new_pred == 1:
                    save_frame_to_dir(fb, selected_dir, fb_name)
                    if input_src == "Select Existing Batch Crop Session" and sel_sess:
                        orig_src = os.path.join("sessions", sel_sess, "original_frames", fb_name)
                        if os.path.exists(orig_src):
                            shutil.copy2(orig_src, os.path.join(orig_selected_dir, fb_name))
                    updated_results.append({
                        "frame_idx": p_idx,
                        "filename": fb_name,
                        "path": os.path.join(selected_dir, fb_name),
                        "prediction": 1,
                        "probability": prob,
                        "type": "Keep (Transition detected)"
                    })
                else:
                    save_frame_to_dir(fb, identical_dir, fb_name)
                    updated_results.append({
                        "frame_idx": p_idx,
                        "filename": fb_name,
                        "path": os.path.join(identical_dir, fb_name),
                        "prediction": 0,
                        "probability": prob,
                        "type": "Discard (Redundant/Duplicate)"
                    })
                    
            st.session_state["l1_results"] = updated_results
            st.toast(f"⚡ Instant Threshold Update: Applied T = {custom_threshold:.2f} (0ms delay)", icon="⚡")
        
    # --- 3. DISPLAY RESULTS GALLERIES ---
    if "l1_results" in st.session_state:
        results = st.session_state["l1_results"]
        out_dir = st.session_state["l1_out_dir"]
        
        selected_dir = os.path.join(out_dir, "selected_frames")
        identical_dir = os.path.join(out_dir, "identical_frames")
        
        selected_frames = [r for r in results if r["prediction"] == 1]
        identical_frames = [r for r in results if r["prediction"] == 0]
        
        st.markdown("---")
        st.markdown("### 📊 Inference Results Summary")
        
        # Display Metrics Cards
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-title">Total Uploaded Frames</div>'
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
            
        # Download ZIP button
        zip_path = os.path.join(out_dir, "selected_candidates.zip")
        create_download_zip(selected_dir, zip_path)
        
        l1_in_src = st.session_state.get("l1_input_source", "")
        
        if l1_in_src == "Select Existing Batch Crop Session":
            orig_selected_dir = os.path.join(out_dir, "selected_original_frames")
            orig_zip_path = os.path.join(out_dir, "selected_original_candidates.zip")
            create_download_zip(orig_selected_dir, orig_zip_path)
            
            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                with open(zip_path, "rb") as z_file:
                    st.download_button(
                        label=f"💾 Download Selected Crops ZIP ({len(selected_frames)} images)",
                        data=z_file,
                        file_name="selected_crops.zip",
                        mime="application/zip",
                        use_container_width=True
                    )
            with col_dl2:
                with open(orig_zip_path, "rb") as oz_file:
                    st.download_button(
                        label=f"🎬 Download Selected Original Video Frames ZIP ({len(selected_frames)} images)",
                        data=oz_file,
                        file_name="selected_original_frames.zip",
                        mime="application/zip",
                        use_container_width=True
                    )
        else:
            with open(zip_path, "rb") as z_file:
                st.download_button(
                    label=f"💾 Download Selected Candidates ZIP ({len(selected_frames)} images)",
                    data=z_file,
                    file_name="selected_candidates.zip",
                    mime="application/zip",
                    use_container_width=True
                )
            
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
            with st.container(border=True):
                for idx in range(0, len(active_frames), grid_cols):
                    row_frames = active_frames[idx : idx + grid_cols]
                    cols = st.columns(grid_cols)
                    for col_idx, r in enumerate(row_frames):
                        with cols[col_idx]:
                            is_keep = r["prediction"] == 1
                            border_color = "#10b981" if is_keep else "#ef4444"
                            bg_color = "rgba(16, 185, 129, 0.05)" if is_keep else "rgba(239, 68, 68, 0.05)"
                            shadow = "0 0 8px rgba(16, 185, 129, 0.15)" if is_keep else "0 0 8px rgba(239, 68, 68, 0.15)"
                            
                            st.markdown(
                                f'<div style="border: 2px solid {border_color}; border-radius: 12px; padding: 6px; background-color: {bg_color}; box-shadow: {shadow}; margin-bottom: 8px;">'
                                f'<img src="data:image/jpeg;base64,{get_base64_from_filepath(r["path"])}" style="width: 100%; border-radius: 8px; display: block;"/>'
                                f'</div>',
                                unsafe_allow_html=True
                            )
                            label_class = "status-success" if is_keep else "status-warning"
                            st.markdown(f"**{r['frame_idx']}. {r['filename']}**")
                            st.markdown(f'<span class="status-badge {label_class}">{r["type"]}</span>', unsafe_allow_html=True)
                            st.markdown("---")

def get_base64_from_filepath(path):
    """Utility helper to load file and encode to base64."""
    import base64
    try:
        with open(path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
            return encoded_string
    except Exception:
        return ""
