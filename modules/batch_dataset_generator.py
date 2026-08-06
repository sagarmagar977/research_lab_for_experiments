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
        text_min_area=st.session_state["text_min_area"]
    )
    
    col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
    with col_cfg1:
        st.write(f"**Histogram:** Bins={config.hist_bins}, Metric={config.hist_method}, Grid={config.hist_grid_size}x{config.hist_grid_size}")
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
        
        # Define exact headers matching our 97-column output (with GroundTruth integrated as target label)
        headers = [
            "Frame_A", "Frame_B", "Feature_Engine_Version", "Feature_Schema_Version", "Experiment_Version", "Timestamp", 
            "Image_Width", "Image_Height", "Grid_Size", "Histogram_Bins", "Histogram_Comparison_Metric", "Color_Mode", "SSIM_Window_Size", "Hyperparameters_JSON",
            "GroundTruth", # Target Label
            
            # Frame A features
            "FrameA_Brightness", "FrameA_Contrast", "FrameA_Entropy", "FrameA_Edge_Density", "FrameA_Text_Occupancy",
            "FrameA_Global_RGB_Hist_Mean", "FrameA_Global_RGB_Hist_Max", "FrameA_Global_RGB_Hist_Min", "FrameA_Global_RGB_Hist_Var", "FrameA_Global_RGB_Hist_Std",
            "FrameA_Global_Gray_Hist_Mean", "FrameA_Global_Gray_Hist_Max", "FrameA_Global_Gray_Hist_Min", "FrameA_Global_Gray_Hist_Var", "FrameA_Global_Gray_Hist_Std",
            "FrameA_Grid_RGB_Hist_Mean", "FrameA_Grid_RGB_Hist_Max", "FrameA_Grid_RGB_Hist_Min", "FrameA_Grid_RGB_Hist_Var", "FrameA_Grid_RGB_Hist_Std",
            "FrameA_Grid_Gray_Hist_Mean", "FrameA_Grid_Gray_Hist_Max", "FrameA_Grid_Gray_Hist_Min", "FrameA_Grid_Gray_Hist_Var", "FrameA_Grid_Gray_Hist_Std",
            "FrameA_Grid_Edge_Mean", "FrameA_Grid_Edge_Max", "FrameA_Grid_Edge_Min", "FrameA_Grid_Edge_Var", "FrameA_Grid_Edge_Std",
            
            # Frame B features
            "FrameB_Brightness", "FrameB_Contrast", "FrameB_Entropy", "FrameB_Edge_Density", "FrameB_Text_Occupancy",
            "FrameB_Global_RGB_Hist_Mean", "FrameB_Global_RGB_Hist_Max", "FrameB_Global_RGB_Hist_Min", "FrameB_Global_RGB_Hist_Var", "FrameB_Global_RGB_Hist_Std",
            "FrameB_Global_Gray_Hist_Mean", "FrameB_Global_Gray_Hist_Max", "FrameB_Global_Gray_Hist_Min", "FrameB_Global_Gray_Hist_Var", "FrameB_Global_Gray_Hist_Std",
            "FrameB_Grid_RGB_Hist_Mean", "FrameB_Grid_RGB_Hist_Max", "FrameB_Grid_RGB_Hist_Min", "FrameB_Grid_RGB_Hist_Var", "FrameB_Grid_RGB_Hist_Std",
            "FrameB_Grid_Gray_Hist_Mean", "FrameB_Grid_Gray_Hist_Max", "FrameB_Grid_Gray_Hist_Min", "FrameB_Grid_Gray_Hist_Var", "FrameB_Grid_Gray_Hist_Std",
            "FrameB_Grid_Edge_Mean", "FrameB_Grid_Edge_Max", "FrameB_Grid_Edge_Min", "FrameB_Grid_Edge_Var", "FrameB_Grid_Edge_Std",
            
            # Pairwise features
            "Global_RGB_Histogram_Dist", "Global_Gray_Histogram_Dist",
            "Grid_RGB_Histogram_Mean", "Grid_RGB_Histogram_Max", "Grid_RGB_Histogram_Min", "Grid_RGB_Histogram_Var", "Grid_RGB_Histogram_Std",
            "Grid_Gray_Histogram_Mean", "Grid_Gray_Histogram_Max", "Grid_Gray_Histogram_Min", "Grid_Gray_Histogram_Var", "Grid_Gray_Histogram_Std",
            "Whole_Edge_Density_Diff",
            "Grid_Edge_Mean_Diff", "Grid_Edge_Max_Diff", "Grid_Edge_Min_Diff", "Grid_Edge_Var_Diff", "Grid_Edge_Std_Diff",
            "SSIM_Mean", "SSIM_Min", "SSIM_Variance",
            "Mean_Absolute_Difference", "Text_Occupancy_Diff"
        ]
        
        total_pairs = len(all_files) - 1
        st.write("#### Processing Log")
        
        # Load and run frame extractions
        for idx in range(total_pairs):
            fname_a = all_files[idx]
            fname_b = all_files[idx + 1]
            status_text.text(f"Pair {idx+1}/{total_pairs}: {fname_a} ──► {fname_b}")
            
            # Assign GroundTruth based on whether Frame_B exists in candidate files
            ground_truth = 1 if fname_b in candidate_files else 0
            
            try:
                img_a_bgr = cv2.imread(os.path.join(crop_dir_path, fname_a))
                img_b_bgr = cv2.imread(os.path.join(crop_dir_path, fname_b))
                
                if img_a_bgr is None or img_b_bgr is None:
                    raise ValueError("Failed to load cropped image matrix.")
                    
                img_a = cv2.cvtColor(img_a_bgr, cv2.COLOR_BGR2RGB)
                img_b = cv2.cvtColor(img_b_bgr, cv2.COLOR_BGR2RGB)
                
                fa, fb, pf, art, logs_ext = extractor.extract(img_a, img_b)
                
                # Render raw CSV row
                raw_csv = CSVExporter.export(fname_a, fname_b, fa, fb, pf, config, ground_truth=ground_truth)
                csv_parts = raw_csv.split(",")
                # Inject width and height
                h_orig, w_orig = img_a.shape[:2]
                csv_parts[6] = str(w_orig)
                csv_parts[7] = str(h_orig)
                
                csv_rows.append(csv_parts)
                
            except Exception as e:
                st.error(f"Error on pair {fname_a} -> {fname_b}: {str(e)}")
                
            progress_bar.progress((idx + 1) / total_pairs)
            
        status_text.text("Finished processing all frame pairs.")
        
        # Create DataFrame, export to CSV file
        df = pd.DataFrame(csv_rows, columns=headers)
        output_filename = f"pairwise_dataset_{crop_subdir}.csv"
        output_path = os.path.join(session_path, output_filename)
        
        # Save dataset inside the session folder directly
        df.to_csv(output_path, index=False)
        st.success(f"Consolidated dataset generated and saved to: `{os.path.abspath(output_path)}`")
        
        # Display dataset preview
        st.markdown("#### Dataset Preview (First 5 Rows)")
        st.dataframe(df.head(), use_container_width=True)
        
        # Make the dataset downloadable directly
        csv_data = df.to_csv(index=False)
        st.download_button(
            label="💾 Download Consolidated Dataset CSV",
            data=csv_data,
            file_name=output_filename,
            mime="text/csv",
            use_container_width=True
        )
