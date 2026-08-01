import os
import time
import datetime
import shutil
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from modules.detection import run_detection_pipeline

def render_batch_session_manager(engine_v3, engine_v4):
    """Renders the Batch Session Manager UI and logic."""
    # Retrieve slider/select states from st.session_state
    det_db_thresh = st.session_state["det_db_thresh"]
    det_db_unclip_ratio = st.session_state["det_db_unclip_ratio"]
    padding_px = st.session_state["padding_px"]
    min_area_filter = st.session_state["min_area_filter"]
    preprocess_mode = st.session_state["preprocess_mode"]
    use_blur = st.session_state["use_blur"]
    blur_kernel_size = st.session_state["blur_kernel_size"]
    use_dilation = st.session_state["use_dilation"]
    dilation_w = st.session_state["dilation_w"]
    dilation_h = st.session_state["dilation_h"]
    crop_mode = st.session_state["crop_mode"]
    empty_strategy = st.session_state["empty_strategy"]

    st.markdown("### Create and Run a Crop Session")
    
    col_s1, col_s2 = st.columns([2, 1])
    with col_s1:
        session_name_prefix = st.text_input("Session Name Prefix", value="BatchCrop")
    with col_s2:
        st.write("")
        st.write("")
        
    source_type = st.radio("Select Input Source", ["Drag & Drop Image Files", "Local Directory Path"], horizontal=True)
    
    uploaded_files = []
    local_dir_path = ""
    
    if source_type == "Drag & Drop Image Files":
        uploaded_files = st.file_uploader(
            "Drag and drop multiple frame images",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            help="Select one or more images to batch crop."
        )
    else:
        local_dir_path = st.text_input("Local Folder Absolute Path", value="", placeholder="e.g. C:\\path\\to\\my\\frames")
        if local_dir_path:
            if not os.path.exists(local_dir_path) or not os.path.isdir(local_dir_path):
                st.error("Provided path does not exist or is not a directory.")
            else:
                files = [f for f in os.listdir(local_dir_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                st.success(f"Found {len(files)} image files in directory.")
                
    start_btn = st.button("🚀 Start Batch Cropping Session", use_container_width=True)
    
    if start_btn:
        files_to_process = []
        if source_type == "Drag & Drop Image Files":
            if not uploaded_files:
                st.warning("Please upload at least one image file.")
                return
            for f in uploaded_files:
                files_to_process.append({"name": f.name, "data": f})
        else:
            if not local_dir_path or not os.path.exists(local_dir_path):
                st.warning("Please specify a valid local directory path.")
                return
            image_filenames = sorted([f for f in os.listdir(local_dir_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            if not image_filenames:
                st.warning("No image files (.png, .jpg, .jpeg) found in the specified directory.")
                return
            for fname in image_filenames:
                files_to_process.append({"name": fname, "path": os.path.join(local_dir_path, fname)})
                
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_prefix = "".join(c for c in session_name_prefix if c.isalnum() or c in ("-", "_")).strip()
        if not safe_prefix:
            safe_prefix = "session"
        session_folder_name = f"session_{timestamp}_{safe_prefix}"
        
        sessions_root = "sessions"
        if not os.path.exists(sessions_root):
            os.makedirs(sessions_root)
            
        session_path = os.path.join(sessions_root, session_folder_name)
        orig_dir = os.path.join(session_path, "original_frames")
        v3_dir = os.path.join(session_path, "v3_crops")
        v4_dir = os.path.join(session_path, "v4_crops")
        
        os.makedirs(orig_dir, exist_ok=True)
        os.makedirs(v3_dir, exist_ok=True)
        os.makedirs(v4_dir, exist_ok=True)
        
        st.info(f"Created session at: `{os.path.abspath(session_path)}`")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        st.markdown("#### Processing Progress Log")
        log_container = st.empty()
        log_rows = []
        
        total_files = len(files_to_process)
        
        for idx, item in enumerate(files_to_process):
            fname = item["name"]
            status_text.text(f"Processing frame {idx+1}/{total_files}: {fname}...")
            
            try:
                if "data" in item:
                    image = Image.open(item["data"])
                    img_rgb = np.array(image.convert("RGB"))
                    image.save(os.path.join(orig_dir, fname))
                else:
                    img_bgr = cv2.imread(item["path"])
                    if img_bgr is None:
                        raise ValueError("Failed to read image with OpenCV")
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    shutil.copy2(item["path"], os.path.join(orig_dir, fname))
                
                orig_h, orig_w = img_rgb.shape[:2]
                down_w = min(orig_w, 960)
                down_h = int(orig_h * (down_w / orig_w))
                down_img = cv2.resize(img_rgb, (down_w, down_h))
                
                scale_x = orig_w / down_w
                scale_y = orig_h / down_h
                
                v3_result = run_detection_pipeline(
                    img_rgb, down_img, scale_x, scale_y, engine_v3,
                    det_db_thresh, det_db_unclip_ratio, padding_px, min_area_filter, empty_strategy,
                    preprocess_mode, use_blur, blur_kernel_size, use_dilation, dilation_w, dilation_h, crop_mode
                )
                
                v4_result = run_detection_pipeline(
                    img_rgb, down_img, scale_x, scale_y, engine_v4,
                    det_db_thresh, det_db_unclip_ratio, padding_px, min_area_filter, empty_strategy,
                    preprocess_mode, use_blur, blur_kernel_size, use_dilation, dilation_w, dilation_h, crop_mode
                )
                
                v3_saved = "Skipped"
                if v3_result["cropped_img"] is not None:
                    v3_bgr = cv2.cvtColor(v3_result["cropped_img"], cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(v3_dir, fname), v3_bgr)
                    v3_saved = "Saved"
                    
                v4_saved = "Skipped"
                if v4_result["cropped_img"] is not None:
                    v4_bgr = cv2.cvtColor(v4_result["cropped_img"], cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(v4_dir, fname), v4_bgr)
                    v4_saved = "Saved"
                    
                log_rows.append({
                    "Frame": fname,
                    "v3 Status": v3_result["status"],
                    "v3 Crop": v3_saved,
                    "v3 Latency": f"{v3_result['latency_ms']:.1f}ms",
                    "v4 Status": v4_result["status"],
                    "v4 Crop": v4_saved,
                    "v4 Latency": f"{v4_result['latency_ms']:.1f}ms"
                })
            except Exception as e:
                log_rows.append({
                    "Frame": fname,
                    "v3 Status": "ERROR",
                    "v3 Crop": "N/A",
                    "v3 Latency": "N/A",
                    "v4 Status": "ERROR",
                    "v4 Crop": "N/A",
                    "v4 Latency": f"Err: {str(e)}"
                })
                
            progress_bar.progress((idx + 1) / total_files)
            log_container.dataframe(log_rows, use_container_width=True)
            
        status_text.text(f"Batch processing completed! Processed {total_files} files.")
        st.success(f"Session saved under {session_path}")
        
    st.markdown("---")
    st.markdown("### 📂 Session Browser & Side-by-Side Comparison")
    
    sessions_root = "sessions"
    if not os.path.exists(sessions_root) or not os.path.isdir(sessions_root):
        st.info("No active crop sessions found yet. Create a session above.")
        return
        
    session_dirs = sorted([d for d in os.listdir(sessions_root) if os.path.isdir(os.path.join(sessions_root, d))], reverse=True)
    if not session_dirs:
        st.info("No active crop sessions found yet. Create a session above.")
        return
        
    selected_session = st.selectbox("Select Session to Browse", session_dirs)
    if selected_session:
        session_path = os.path.join(sessions_root, selected_session)
        orig_dir = os.path.join(session_path, "original_frames")
        v3_dir = os.path.join(session_path, "v3_crops")
        v4_dir = os.path.join(session_path, "v4_crops")
        
        st.markdown(f"**Session Directory Path:** `{os.path.abspath(session_path)}`")
        
        if os.path.exists(orig_dir):
            frames = sorted([f for f in os.listdir(orig_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        else:
            frames = []
            
        if not frames:
            st.warning("No original frames found in this session.")
            return
            
        col_list, col_det = st.columns([1, 3])
        with col_list:
            selected_frame = st.selectbox("Select Frame", frames)
            st.markdown("#### Session Info")
            st.write(f"Total Frames: **{len(frames)}**")
            
            has_v3 = os.path.exists(os.path.join(v3_dir, selected_frame))
            has_v4 = os.path.exists(os.path.join(v4_dir, selected_frame))
            
            st.write(f"PP-OCRv3 Crop: {'✅ Available' if has_v3 else '❌ Skipped/None'}")
            st.write(f"PP-OCRv4 Crop: {'✅ Available' if has_v4 else '❌ Skipped/None'}")
            
        with col_det:
            st.markdown(f"#### Side-by-Side Comparison: `{selected_frame}`")
            
            tab_orig, tab_v3, tab_v4 = st.tabs(["Original Frame", "PP-OCRv3 Crop", "PP-OCRv4 Crop"])
            
            with tab_orig:
                orig_img_path = os.path.join(orig_dir, selected_frame)
                if os.path.exists(orig_img_path):
                    orig_img = Image.open(orig_img_path)
                    st.image(orig_img, caption=f"Original: {orig_img.size[0]}x{orig_img.size[1]}", use_container_width=True)
                else:
                    st.error("Original file missing")
                    
            with tab_v3:
                v3_img_path = os.path.join(v3_dir, selected_frame)
                if os.path.exists(v3_img_path):
                    v3_img = Image.open(v3_img_path)
                    st.image(v3_img, caption=f"v3 Crop: {v3_img.size[0]}x{v3_img.size[1]}", use_container_width=True)
                else:
                    st.warning("Frame skipped by PP-OCRv3 engine (No text region detected or empty strategy applied).")
                    
            with tab_v4:
                v4_img_path = os.path.join(v4_dir, selected_frame)
                if os.path.exists(v4_img_path):
                    v4_img = Image.open(v4_img_path)
                    st.image(v4_img, caption=f"v4 Crop: {v4_img.size[0]}x{v4_img.size[1]}", use_container_width=True)
                else:
                    st.warning("Frame skipped by PP-OCRv4 engine (No text region detected or empty strategy applied).")
