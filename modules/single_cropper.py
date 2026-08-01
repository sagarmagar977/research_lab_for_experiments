import streamlit as st
import numpy as np
import cv2
import json
from PIL import Image
from modules.detection import run_detection_pipeline, extract_ocr_metadata

def render_single_cropper_tab(engine_v3, engine_v4):
    """Renders the Single Frame Cropper interface."""
    # Read settings variables from session_state
    use_english_ocr = st.session_state["use_english_ocr"]
    model_mode = st.session_state["model_mode"]
    preprocess_mode = st.session_state["preprocess_mode"]
    use_blur = st.session_state["use_blur"]
    blur_kernel_size = st.session_state["blur_kernel_size"]
    use_dilation = st.session_state["use_dilation"]
    dilation_w = st.session_state["dilation_w"]
    dilation_h = st.session_state["dilation_h"]
    crop_mode = st.session_state["crop_mode"]
    ocr_tolerance_px = st.session_state["ocr_tolerance_px"]
    det_db_thresh = st.session_state["det_db_thresh"]
    det_db_unclip_ratio = st.session_state["det_db_unclip_ratio"]
    padding_px = st.session_state["padding_px"]
    min_area_filter = st.session_state["min_area_filter"]
    ocr_preprocess_mode = st.session_state["ocr_preprocess_mode"]
    ocr_use_blur = st.session_state["ocr_use_blur"]
    ocr_blur_kernel = st.session_state["ocr_blur_kernel"]
    ocr_det_db_thresh = st.session_state["ocr_det_db_thresh"]
    ocr_det_db_unclip_ratio = st.session_state["ocr_det_db_unclip_ratio"]
    empty_strategy = st.session_state["empty_strategy"]

    uploaded_file = st.file_uploader(
        "Upload educational frame image",
        type=["jpg", "jpeg", "png"],
        help="Supports 720p, 1080p, and 4K resolutions"
    )
    
    if uploaded_file is not None:
        # Reset session states if a new file is uploaded or English OCR option changes
        if "last_uploaded_file" not in st.session_state or st.session_state["last_uploaded_file"] != uploaded_file.name:
            st.session_state["last_uploaded_file"] = uploaded_file.name
            st.session_state["v3_ocr_active"] = False
            st.session_state["v4_ocr_active"] = False
            st.session_state["v3_ocr_data"] = None
            st.session_state["v4_ocr_data"] = None
    
        if "last_use_english_ocr" not in st.session_state or st.session_state["last_use_english_ocr"] != use_english_ocr:
            st.session_state["last_use_english_ocr"] = use_english_ocr
            st.session_state["v3_ocr_active"] = False
            st.session_state["v4_ocr_active"] = False
            st.session_state["v3_ocr_data"] = None
            st.session_state["v4_ocr_data"] = None
    
        # 1. Load image and perform diagnostics
        image = Image.open(uploaded_file)
        img_rgb = np.array(image.convert("RGB"))
        orig_h, orig_w = img_rgb.shape[:2]
        
        # 2. Compute internal downscale values
        down_w = min(orig_w, 960)
        down_h = int(orig_h * (down_w / orig_w))
        down_img = cv2.resize(img_rgb, (down_w, down_h))
        
        scale_x = orig_w / down_w
        scale_y = orig_h / down_h
        
        # 3. Process image based on execution mode
        if model_mode == "Compare Both Side-by-Side":
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
            
            col_v3, col_v4 = st.columns(2)
            
            # --- PP-OCRv3 Column ---
            with col_v3:
                st.markdown("### PP-OCRv3 Detection Results")
                st.write(f"Inference Latency: **{v3_result['latency_ms']:.1f} ms**")
                
                if v3_result["status"] == "CONTENT_DETECTED":
                    st.success("Content detected! Showing bounding box and cropped ROI.")
                else:
                    st.error("NO_TEXT_REGION_DETECTED - Applying strategy rules.")
                    
                tabs = st.tabs(["Overlay Visualizer", "Preprocessed Image", "Binary Mask Debugger"])
                with tabs[0]:
                    st.image(v3_result["visualizer_img"], caption="Input + Crop Overlays (Red Box = Crop Bounds, Green = Text)", use_container_width=True)
                with tabs[1]:
                    st.image(v3_result["down_img_pre"], caption="Preprocessed Frame for Engine", use_container_width=True)
                with tabs[2]:
                    col_raw, col_dil = st.columns(2)
                    with col_raw:
                        st.image(v3_result["mask_down"], caption="Raw DBNet Mask", use_container_width=True)
                    with col_dil:
                        st.image(v3_result["mask_dilated"], caption="Dilated Binary Mask", use_container_width=True)
                    
                st.markdown("#### Cropped ROI Output")
                if v3_result["cropped_img"] is not None:
                    st.image(v3_result["cropped_img"], caption="Isolated Content Region", use_container_width=True)
                else:
                    st.warning("Frame skipped per Skip Frame strategy configuration.")
                    
                st.markdown("#### Diagnostics JSON")
                st.json(v3_result["diagnostics"])
                
                if v3_result["cropped_img"] is not None:
                    if st.button("Extract OCR Metadata (v3)", key="v3_btn"):
                        st.session_state["v3_ocr_active"] = True
                        with st.spinner("Extracting OCR text and geometry from PP-OCRv3 Crop..."):
                            st.session_state["v3_ocr_data"] = extract_ocr_metadata(
                                v3_result["cropped_img"], engine_v3, ocr_tolerance_px,
                                ocr_preprocess_mode, ocr_use_blur, ocr_blur_kernel, ocr_det_db_thresh, ocr_det_db_unclip_ratio
                            )
                    
                    if st.session_state.get("v3_ocr_active") and st.session_state["v3_ocr_data"] is not None:
                        ocr_data = st.session_state["v3_ocr_data"]
                        st.markdown("---")
                        st.markdown("### PP-OCRv3 Extracted Layout & Text")
                        st.image(ocr_data["ocr_visualizer_img"], caption="OCR Overlay Visualizer (Green = Lines)", use_container_width=True)
                        
                        json_str = json.dumps(ocr_data["json_output"], indent=4, ensure_ascii=False)
                        st.download_button(
                            label="Download OCR JSON (v3)",
                            data=json_str,
                            file_name="pp_ocrv3_metadata.json",
                            mime="application/json"
                        )
                        st.markdown("#### OCR JSON Metadata")
                        st.json(ocr_data["json_output"])
                
            # --- PP-OCRv4 Column ---
            with col_v4:
                st.markdown("### PP-OCRv4 Detection Results")
                st.write(f"Inference Latency: **{v4_result['latency_ms']:.1f} ms**")
                
                if v4_result["status"] == "CONTENT_DETECTED":
                    st.success("Content detected! Showing bounding box and cropped ROI.")
                else:
                    st.error("NO_TEXT_REGION_DETECTED - Applying strategy rules.")
                    
                tabs = st.tabs(["Overlay Visualizer", "Preprocessed Image", "Binary Mask Debugger"])
                with tabs[0]:
                    st.image(v4_result["visualizer_img"], caption="Input + Crop Overlays (Red Box = Crop Bounds, Green = Text)", use_container_width=True)
                with tabs[1]:
                    st.image(v4_result["down_img_pre"], caption="Preprocessed Frame for Engine", use_container_width=True)
                with tabs[2]:
                    col_raw, col_dil = st.columns(2)
                    with col_raw:
                        st.image(v4_result["mask_down"], caption="Raw DBNet Mask", use_container_width=True)
                    with col_dil:
                        st.image(v4_result["mask_dilated"], caption="Dilated Binary Mask", use_container_width=True)
                    
                st.markdown("#### Cropped ROI Output")
                if v4_result["cropped_img"] is not None:
                    st.image(v4_result["cropped_img"], caption="Isolated Content Region", use_container_width=True)
                else:
                    st.warning("Frame skipped per Skip Frame strategy configuration.")
                    
                st.markdown("#### Diagnostics JSON")
                st.json(v4_result["diagnostics"])
                
                if v4_result["cropped_img"] is not None:
                    if st.button("Extract OCR Metadata (v4)", key="v4_btn"):
                        st.session_state["v4_ocr_active"] = True
                        with st.spinner("Extracting OCR text and geometry from PP-OCRv4 Crop..."):
                            st.session_state["v4_ocr_data"] = extract_ocr_metadata(
                                v4_result["cropped_img"], engine_v4, ocr_tolerance_px,
                                ocr_preprocess_mode, ocr_use_blur, ocr_blur_kernel, ocr_det_db_thresh, ocr_det_db_unclip_ratio
                            )
                    
                    if st.session_state.get("v4_ocr_active") and st.session_state["v4_ocr_data"] is not None:
                        ocr_data = st.session_state["v4_ocr_data"]
                        st.markdown("---")
                        st.markdown("### PP-OCRv4 Extracted Layout & Text")
                        st.image(ocr_data["ocr_visualizer_img"], caption="OCR Overlay Visualizer (Green = Lines)", use_container_width=True)
                        
                        json_str = json.dumps(ocr_data["json_output"], indent=4, ensure_ascii=False)
                        st.download_button(
                            label="Download OCR JSON (v4)",
                            data=json_str,
                            file_name="pp_ocrv4_metadata.json",
                            mime="application/json"
                        )
                        st.markdown("#### OCR JSON Metadata")
                        st.json(ocr_data["json_output"])
                
        else:
            # Single model execution
            selected_name = "PP-OCRv3" if model_mode == "PP-OCRv3 Det Only" else "PP-OCRv4"
            engine = engine_v3 if model_mode == "PP-OCRv3 Det Only" else engine_v4
            
            result = run_detection_pipeline(
                img_rgb, down_img, scale_x, scale_y, engine,
                det_db_thresh, det_db_unclip_ratio, padding_px, min_area_filter, empty_strategy,
                preprocess_mode, use_blur, blur_kernel_size, use_dilation, dilation_w, dilation_h, crop_mode
            )
            
            left_col, right_col = st.columns(2)
            
            with left_col:
                st.markdown(f"### {selected_name} Visualizers")
                
                if result["status"] == "CONTENT_DETECTED":
                    st.success("Content detected! Processing crop bounding boxes.")
                else:
                    st.error("NO_TEXT_REGION_DETECTED - Applying strategy rules.")
                    
                tabs = st.tabs(["Overlay Visualizer", "Preprocessed Image", "Binary Mask Debugger"])
                with tabs[0]:
                    st.image(result["visualizer_img"], caption="Input + Crop Overlays (Red Box = Crop Bounds, Green = Text)", use_container_width=True)
                with tabs[1]:
                    st.image(result["down_img_pre"], caption="Preprocessed Frame for Engine", use_container_width=True)
                with tabs[2]:
                    col_raw, col_dil = st.columns(2)
                    with col_raw:
                        st.image(result["mask_down"], caption="Raw DBNet Mask", use_container_width=True)
                    with col_dil:
                        st.image(result["mask_dilated"], caption="Dilated Binary Mask", use_container_width=True)
                    
            with right_col:
                st.markdown("### Crop Output & Statistics")
                
                st.markdown(f"""
                <div class='metric-container'>
                    <div class='metric-card'>
                        <div class='metric-title'>Inference Latency</div>
                        <div class='metric-value'>{result['latency_ms']:.1f} ms</div>
                    </div>
                    <div class='metric-card'>
                        <div class='metric-title'>Status</div>
                        <div class='metric-value'>{result['status']}</div>
                    </div>
                    <div class='metric-card'>
                        <div class='metric-title'>Text Boxes</div>
                        <div class='metric-value'>{result['kept_count']} / {result['total_detected']}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("#### Cropped ROI Output")
                if result["cropped_img"] is not None:
                    st.image(result["cropped_img"], caption="Isolated Content Region", use_container_width=True)
                else:
                    st.warning("Frame skipped per Skip Frame strategy configuration.")
                    
                st.markdown("#### Diagnostics JSON")
                st.json(result["diagnostics"])
                
                if result["cropped_img"] is not None:
                    if st.button(f"Extract OCR Metadata ({selected_name})", key="single_btn"):
                        active_key = "v3_ocr_active" if selected_name == "PP-OCRv3" else "v4_ocr_active"
                        data_key = "v3_ocr_data" if selected_name == "PP-OCRv3" else "v4_ocr_data"
                        st.session_state[active_key] = True
                        with st.spinner(f"Extracting OCR text and geometry from {selected_name} Crop..."):
                            st.session_state[data_key] = extract_ocr_metadata(
                                result["cropped_img"], engine, ocr_tolerance_px,
                                ocr_preprocess_mode, ocr_use_blur, ocr_blur_kernel, ocr_det_db_thresh, ocr_det_db_unclip_ratio
                            )
                            
                    active_key = "v3_ocr_active" if selected_name == "PP-OCRv3" else "v4_ocr_active"
                    data_key = "v3_ocr_data" if selected_name == "PP-OCRv3" else "v4_ocr_data"
                    if st.session_state.get(active_key) and st.session_state[data_key] is not None:
                        ocr_data = st.session_state[data_key]
                        st.markdown("---")
                        st.markdown(f"### {selected_name} Extracted Layout & Text")
                        st.image(ocr_data["ocr_visualizer_img"], caption="OCR Overlay Visualizer (Green = Lines)", use_container_width=True)
                        
                        json_str = json.dumps(ocr_data["json_output"], indent=4, ensure_ascii=False)
                        st.download_button(
                            label="Download OCR JSON",
                            data=json_str,
                            file_name=f"{selected_name.lower()}_metadata.json",
                            mime="application/json"
                        )
                        st.markdown("#### OCR JSON Metadata")
                        st.json(ocr_data["json_output"])
    else:
        st.info("Please upload a frame image from the uploader to begin testing.")
        
        st.markdown("""
        <div style="background-color: #1e1e2f; padding: 2rem; border-radius: 12px; border: 1px dashed #4b5563; text-align: center; color: #9ca3af; margin-top: 1rem;">
            <h3 style="color: #a78bfa; margin-bottom: 0.5rem;">Visualizer Preview Workspace</h3>
            <p>Your original image, boundary crops, text heatmaps, and cropped region metrics will render here once uploaded.</p>
        </div>
        """, unsafe_allow_html=True)
