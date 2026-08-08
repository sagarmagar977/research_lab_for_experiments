import os
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import datetime
import json
from PIL import Image
from modules.pairwise_feature_lab import (
    PairwiseFeatureConfig,
    PairwiseFeatureExtractor,
    HistogramExtractor,
    EdgeExtractor,
    SSIMExtractor,
    MorphologyExtractor,
    CSVExporter
)

def compile_feature_manifest(headers):
    manifest = {}
    for h in headers:
        if h == "GroundTruth":
            manifest[h] = {"type": "int", "group": "target", "description": "Target label (1 if selected candidate frame, 0 if discard)"}
            continue
        group = "features"
        desc = ""
        if "Brightness" in h:
            group, desc = "brightness", "Mean pixel intensity brightness"
        elif "Contrast" in h:
            group, desc = "contrast", "Standard deviation of pixel intensity contrast"
        elif "Entropy" in h:
            group, desc = "entropy", "Shannon entropy of pixel intensity distribution"
        elif "Edge_Density" in h:
            group, desc = "edge", "Canny edge pixel density ratio"
        elif "Text_Occupancy" in h:
            group, desc = "text", "OCR/text dilated mask occupancy ratio"
        elif "Global_RGB_Hist" in h:
            group, desc = "histogram", "Global RGB histogram bin value statistics"
        elif "Global_Gray_Hist" in h:
            group, desc = "histogram", "Global Grayscale histogram bin value statistics"
        elif "Grid_RGB_Hist" in h:
            group, desc = "histogram", "Local grid RGB cell histogram bin value statistics"
        elif "Grid_Gray_Hist" in h:
            group, desc = "histogram", "Local grid Grayscale cell histogram bin value statistics"
        elif "Grid_Edge" in h:
            group, desc = "edge", "Local grid cell Canny edge density statistics"
        elif "Global_RGB_Histogram_Dist" in h:
            group, desc = "histogram_comparison", f"Global RGB histogram comparison ({h.split('_')[-1]})"
        elif "Global_Gray_Histogram_Dist" in h:
            group, desc = "histogram_comparison", f"Global Grayscale histogram comparison ({h.split('_')[-1]})"
        elif "Grid_RGB_Histogram" in h:
            group, desc = "histogram_comparison", f"Local grid cell RGB histogram comparisons stats ({h.split('_')[-2]} {h.split('_')[-1]})"
        elif "Grid_Gray_Histogram" in h:
            group, desc = "histogram_comparison", f"Local grid cell Grayscale histogram comparisons stats ({h.split('_')[-2]} {h.split('_')[-1]})"
        elif "Whole_Edge_Density_Diff" in h:
            group, desc = "edge_comparison", "Whole frame Canny edge density absolute difference"
        elif "Grid_Edge" in h:
            group, desc = "edge_comparison", f"Local grid cell edge difference stats ({h.split('_')[-2]})"
        elif "SSIM" in h:
            group, desc = "structural_similarity", f"Structural Similarity Index metric ({h.split('_')[-1]})"
        elif "Mean_Absolute_Difference" in h:
            group, desc = "pixel_difference", "Mean Absolute pixel-to-pixel intensity Difference (MAD)"
        elif "Text_Occupancy_Diff" in h:
            group, desc = "text_comparison", "Absolute difference in text mask occupancy"

        target_f = "Frame A" if "FrameA" in h else "Frame B" if "FrameB" in h else "Pairwise comparison"
        manifest[h] = {
            "type": "float",
            "group": group,
            "description": f"{desc} ({target_f})"
        }
    return manifest

def render_batch_dataset_generator():
    st.markdown("### 📦 Batch Dataset Generator Tool")
    st.write("Extract pairwise and individual features chronologically across an entire cropped crop session to generate a consolidated ML training dataset.")
    
    sessions_root = "sessions"
    if not os.path.exists(sessions_root) or not os.path.isdir(sessions_root):
        st.info("No active crop sessions found yet. Please create a crop session inside the **Batch Crop Manager** tab first.")
        return
        
    session_dirs = sorted([d for d in os.listdir(sessions_root) if os.path.isdir(os.path.join(sessions_root, d))], reverse=True)
    if not session_dirs:
        st.info("No active crop sessions found yet. Please create a crop session inside the **Batch Crop Manager** tab first.")
        return
        
    col_sel1, col_sel2 = st.columns(2)
    with col_sel1:
        selected_session = st.selectbox("Select Crop Session Source", session_dirs, key="batch_dataset_session")
    with col_sel2:
        crop_type = st.radio("Select Crop Type Source", ["PP-OCRv3 Crops", "PP-OCRv4 Crops"], horizontal=True, key="batch_dataset_croptype")
        
    session_path = os.path.join(sessions_root, selected_session)
    crop_subdir = "v3_crops" if crop_type == "PP-OCRv3 Crops" else "v4_crops"
    cand_subdir = "v3_candidate_frames" if crop_type == "PP-OCRv3 Crops" else "v4_candidate_frames"
    
    crop_dir_path = os.path.join(session_path, crop_subdir)
    cand_dir_path = os.path.join(session_path, cand_subdir)
    
    # Pre-checks on file availability
    if not os.path.exists(crop_dir_path):
        st.warning(f"No crop outputs found at `{crop_subdir}` directory. Run cropping on this session first.")
        return
        
    all_files = sorted([f for f in os.listdir(crop_dir_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    if len(all_files) < 2:
        st.warning(f"Found {len(all_files)} cropped frames in `{crop_subdir}`. A batch pairwise extraction requires at least 2 consecutive frames.")
        return
        
    # Read manually selected candidate frames
    os.makedirs(cand_dir_path, exist_ok=True)
    candidate_files = set([f for f in os.listdir(cand_dir_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    st.success(f"Verified crop session: Found **{len(all_files)}** cropped frames total.")
    st.info(f"Manual Selections Lookups: **{len(candidate_files)}** candidate frames found under `{cand_subdir}` folder.")
    
    # Render hyperparameter summary currently selected in the sidebar
    st.markdown("#### Experiment Pipeline Hyperparameters")
    
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
    
    col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
    with col_cfg1:
        st.write(f"**Histogram:** Bins={config.hist_bins}, Metric={config.hist_method}, Grid={config.hist_grid_size}x{config.hist_grid_size}, Epsilon={config.hist_epsilon:.1e}")
    with col_cfg2:
        st.write(f"**Edges:** Blur={config.edge_blur}, Canny={config.canny_low}/{config.canny_high}, Grid={config.edge_grid_size}x{config.edge_grid_size}")
    with col_cfg3:
        st.write(f"**SSIM / Text:** SSIM Window={config.ssim_win_size}, Text Kernel={config.text_kernel}")
        
    start_btn = st.button("🚀 Start Batch Dataset Generation", use_container_width=True, type="primary")
    
    if start_btn:
        progress_bar = st.progress(0.0)
        status_text = st.empty()
        log_container = st.empty()
        
        # Instantiate Extractors
        extractor = PairwiseFeatureExtractor(config)
        extractor.register_extractor(HistogramExtractor())
        extractor.register_extractor(EdgeExtractor())
        extractor.register_extractor(SSIMExtractor())
        extractor.register_extractor(MorphologyExtractor())
        
        csv_rows = []
        
        # Define exact headers matching our clean 120-column output (119 features + 1 target label at the end)
        headers = CSVExporter.get_headers() + ["GroundTruth"]
        
        # Prepare metadata sidecar dict
        metadata_json = {
            "experiment_id": f"experiment_{crop_subdir}",
            "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "software": {
                "feature_engine_version": "2.1.0",
                "opencv_version": cv2.__version__
            },
            "configuration": {
                "hist_bins": config.hist_bins,
                "color_mode": config.color_mode,
                "hist_grid_size": config.hist_grid_size,
                "edge_blur": config.edge_blur,
                "canny_low": config.canny_low,
                "canny_high": config.canny_high,
                "edge_grid_size": config.edge_grid_size,
                "ssim_win_size": config.ssim_win_size,
                "ssim_gaussian": config.ssim_gaussian,
                "text_thresh": config.text_thresh,
                "text_kernel": config.text_kernel,
                "text_iterations": config.text_iterations,
                "text_min_area": config.text_min_area,
                "hist_epsilon": config.hist_epsilon
            },
            "frame_pairs": []
        }
        
        total_pairs = len(all_files) - 1
        st.write("#### Processing Log")
        
        # Load and run frame extractions
        for idx in range(total_pairs):
            fname_a = all_files[idx]
            fname_b = all_files[idx + 1]
            status_text.text(f"Pair {idx+1}/{total_pairs}: {fname_a} ──► {fname_b}")
            
            # Assign GroundTruth based on whether Frame_B exists in candidate files
            ground_truth = 1 if fname_b in candidate_files else 0
            
            # Record pair mapping in metadata JSON
            metadata_json["frame_pairs"].append({
                "row_index": idx,
                "frame_a": fname_a,
                "frame_b": fname_b,
                "ground_truth": ground_truth
            })
            
            try:
                img_a_bgr = cv2.imread(os.path.join(crop_dir_path, fname_a))
                img_b_bgr = cv2.imread(os.path.join(crop_dir_path, fname_b))
                
                if img_a_bgr is None or img_b_bgr is None:
                    raise ValueError("Failed to load cropped image matrix.")
                    
                img_a = cv2.cvtColor(img_a_bgr, cv2.COLOR_BGR2RGB)
                img_b = cv2.cvtColor(img_b_bgr, cv2.COLOR_BGR2RGB)
                
                fa, fb, pf, art, logs_ext = extractor.extract(img_a, img_b)
                
                # Render 119 features CSV row and append ground truth label
                raw_csv = CSVExporter.export(fa, fb, pf, include_header=False)
                csv_parts = [float(val) for val in raw_csv.split(",")]
                csv_parts.append(int(ground_truth))
                
                csv_rows.append(csv_parts)
                
            except Exception as e:
                st.error(f"Error on pair {fname_a} -> {fname_b}: {str(e)}")
                
            progress_bar.progress((idx + 1) / total_pairs)
            
        status_text.text("Finished processing all frame pairs.")
        
        # Create DataFrame
        df = pd.DataFrame(csv_rows, columns=headers)
        
        # Display dataset preview
        st.markdown("#### Dataset Preview (First 5 Rows)")
        st.dataframe(df.head(), use_container_width=True)
        
        # Checkbox for expandable full preview
        show_full = st.checkbox("🔍 View Full Dataset Preview (All Rows & Columns)", key="chk_full_dataset")
        if show_full:
            st.dataframe(df, height=350, use_container_width=True)
            
        # 1. Save in the structured generated_datasets/ session folder in root
        root_session_dir = os.path.join("generated_datasets", f"session_{crop_subdir}")
        root_datasets_dir = os.path.join(root_session_dir, "datasets")
        root_metadata_dir = os.path.join(root_session_dir, "metadata")
        os.makedirs(root_datasets_dir, exist_ok=True)
        os.makedirs(root_metadata_dir, exist_ok=True)
        
        output_filename = f"candidate_frame_dataset_{crop_subdir}.csv"
        df.to_csv(os.path.join(root_datasets_dir, output_filename), index=False)
        
        metadata_filename = f"experiment_{crop_subdir}.json"
        with open(os.path.join(root_metadata_dir, metadata_filename), "w") as f:
            json.dump(metadata_json, f, indent=4)
            
        manifest = compile_feature_manifest(headers)
        with open(os.path.join(root_metadata_dir, "feature_manifest.json"), "w") as f:
            json.dump(manifest, f, indent=4)
            
        st.success(f"Successfully saved all session artifacts to workspace path: `{os.path.abspath(root_session_dir)}`")
        
        # 2. Local File System Downloads Folder exporter (Creating Downloads/downloads/dataset_export_<subdir>...)
        st.markdown("---")
        st.markdown("#### 📂 Local Downloads Directory Export")
        st.write("Export all generated files to your system `Downloads` folder under the decoupled architecture.")
        
        export_btn = st.button("💾 Export all files to Downloads/downloads/dataset_export/", use_container_width=True, type="secondary")
        if export_btn:
            try:
                downloads_dir = os.path.join(os.path.expanduser("~"), "Downloads")
                dataset_root = os.path.join(downloads_dir, "downloads", f"dataset_export_{crop_subdir}")
                
                # Create decoupled subdirectories
                datasets_dir = os.path.join(dataset_root, "datasets")
                metadata_dir = os.path.join(dataset_root, "metadata")
                
                os.makedirs(datasets_dir, exist_ok=True)
                os.makedirs(metadata_dir, exist_ok=True)
                
                # Write ML CSV
                df.to_csv(os.path.join(datasets_dir, f"candidate_frame_dataset_{crop_subdir}.csv"), index=False)
                
                # Write Metadata JSON
                with open(os.path.join(metadata_dir, f"experiment_{crop_subdir}.json"), "w") as f:
                    json.dump(metadata_json, f, indent=4)
                    
                # Write Feature Manifest JSON
                with open(os.path.join(metadata_dir, "feature_manifest.json"), "w") as f:
                    json.dump(manifest, f, indent=4)
                    
                st.success(f"Successfully exported all files into `{os.path.abspath(dataset_root)}` folder structure!")
            except Exception as ex:
                st.error(f"Failed to export files to downloads folder: {str(ex)}")
                
        # 3. Individual browser download triggers for standard environment compatibility
        st.markdown("---")
        st.markdown("#### 📥 Standard Browser Downloads")
        
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            csv_data = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Dataset CSV",
                data=csv_data,
                file_name=output_filename,
                mime="text/csv",
                use_container_width=True
            )
        with col_dl2:
            json_data = json.dumps(metadata_json, indent=4)
            st.download_button(
                label="📥 Download Metadata JSON",
                data=json_data,
                file_name=metadata_filename,
                mime="application/json",
                use_container_width=True
            )
