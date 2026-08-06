import os
import json
import numpy as np
import cv2
import streamlit as st
from PIL import Image
from modules.detection import load_detection_engines
from modules.single_cropper import render_single_cropper_tab
from modules.batch_manager import render_batch_session_manager
from modules.candidate_selector import render_candidate_selector
from modules.pairwise_feature_lab import render_pairwise_feature_lab
from modules.batch_dataset_generator import render_batch_dataset_generator

# --- Safe Index Helper ---
def safe_index(options, value, default=0):
    try:
        return options.index(value)
    except ValueError:
        return default

# --- Persistent Settings Support ---
SETTINGS_FILE = "user_settings.json"
DEFAULT_SETTINGS = {
    "selected_module": "Single Frame Cropper",
    "model_mode": "Compare Both Side-by-Side",
    "use_english_ocr": False,
    "preprocess_mode": "Original (RGB)",
    "use_blur": False,
    "blur_kernel_size": 5,
    "use_dilation": False,
    "dilation_w": 15,
    "dilation_h": 15,
    "crop_mode": "Union of All Regions",
    "ocr_tolerance_px": 15,
    "det_db_thresh": 0.30,
    "det_db_unclip_ratio": 1.50,
    "padding_px": 15,
    "min_area_filter": 1.0,
    "ocr_preprocess_mode": "Adaptive Thresholding",
    "ocr_use_blur": False,
    "ocr_blur_kernel": 5,
    "ocr_det_db_thresh": 0.30,
    "ocr_det_db_unclip_ratio": 1.60,
    "empty_strategy": "Skip Frame",
    
    # Module 4 settings
    "hist_bins": 64,
    "hist_method": "Correlation",
    "color_mode": "RGB",
    "hist_grid_size": 4,
    "edge_blur": "5x5",
    "canny_low": 50,
    "canny_high": 150,
    "edge_grid_size": 4,
    "ssim_win_size": 11,
    "ssim_gaussian": True,
    "text_thresh": 127,
    "text_kernel": 5,
    "text_iterations": 2,
    "text_min_area": 100
}

# Load saved settings if exist
saved_settings = DEFAULT_SETTINGS.copy()
if os.path.exists(SETTINGS_FILE):
    try:
        with open(SETTINGS_FILE, "r") as f:
            saved = json.load(f)
            for k, v in saved.items():
                if k in saved_settings:
                    saved_settings[k] = v
    except Exception:
        pass

# Initialize session_state keys with loaded values
for k, v in saved_settings.items():
    if k not in st.session_state:
        st.session_state[k] = v

# --- Theme and Styling Setup ---
st.set_page_config(
    page_title="MY RESEARCH LAB",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich premium dark-mode styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

code, pre, [class*="mono"] {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.9rem;
}

/* Header customization */
.main-title {
    background: linear-gradient(135deg, #a78bfa 0%, #6366f1 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
    font-size: 2.2rem;
    margin-bottom: 0.2rem;
}
.subtitle {
    color: #9ca3af;
    font-size: 1rem;
    margin-bottom: 2rem;
}

/* Metric card styles */
.metric-container {
    display: flex;
    gap: 1rem;
    margin-bottom: 1rem;
}
.metric-card {
    background-color: #1e1e2f;
    border: 1px solid #2e2e4f;
    border-radius: 12px;
    padding: 1rem;
    flex: 1;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
}
.metric-title {
    font-size: 0.75rem;
    color: #9ca3af;
    text-transform: uppercase;
    font-weight: 600;
    letter-spacing: 0.05em;
    margin-bottom: 0.25rem;
}
.metric-value {
    font-size: 1.4rem;
    font-weight: 700;
    color: #f3f4f6;
}
.status-badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 700;
}
.status-success {
    background-color: rgba(16, 185, 129, 0.2);
    color: #10b981;
}
.status-warning {
    background-color: rgba(239, 68, 68, 0.2);
    color: #ef4444;
}

/* Warnings and callouts */
.stAlert {
    border-radius: 12px !important;
}

/* Title divider */
hr {
    margin-top: 1rem;
    margin-bottom: 2rem;
    border-color: #2e2e4f;
}
/* Hide Streamlit native image fullscreen expand button */
button[title="View fullscreen"] {
    display: none !important;
}
[data-testid="StyledFullScreenButton"] {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)

# --- Sidebar Module Selector Navigation ---
st.sidebar.markdown("<h2 style='color: #a78bfa; margin-bottom: 0px;'>Lab Module Selector</h2>", unsafe_allow_html=True)
selected_module = st.sidebar.selectbox(
    "Select Lab Module",
    ["Single Frame Cropper", "Batch Crop Manager", "Candidate Frame Selector", "Pairwise Feature Vector Lab", "Batch Dataset Generator"],
    index=safe_index(["Single Frame Cropper", "Batch Crop Manager", "Candidate Frame Selector", "Pairwise Feature Vector Lab", "Batch Dataset Generator"], st.session_state["selected_module"]),
    key="selected_module"
)

# --- Dynamic Sidebar Rendering Based on Selected Module ---
if selected_module not in ("Pairwise Feature Vector Lab", "Batch Dataset Generator"):
    st.sidebar.markdown("<h4 style='color: #a78bfa; margin-top: 20px; margin-bottom: 0px;'>OCR Model Tuning</h4>", unsafe_allow_html=True)
    model_mode_options = ["PP-OCRv3 Det Only", "PP-OCRv4 Det Only", "Compare Both Side-by-Side"]
    model_mode = st.sidebar.radio(
        "Execution Engine / Mode",
        model_mode_options,
        index=safe_index(model_mode_options, st.session_state["model_mode"]),
        key="model_mode"
    )

    use_english_ocr = st.sidebar.checkbox(
        "Use English-Only OCR Models",
        value=st.session_state["use_english_ocr"],
        key="use_english_ocr",
        help="Forces English-specific recognition weights. Fixes word concatenation issues in coding tutorials."
    )

    st.sidebar.markdown("---")

    st.sidebar.markdown("<h4 style='color: #a78bfa; margin-bottom: 0px;'>Preprocessing & Dilation</h4>", unsafe_allow_html=True)
    preprocess_mode_options = ["Original (RGB)", "Grayscale (Monochrome)", "Adaptive Thresholding"]
    preprocess_mode = st.sidebar.selectbox(
        "Preprocessing Mode",
        preprocess_mode_options,
        index=safe_index(preprocess_mode_options, st.session_state["preprocess_mode"]),
        key="preprocess_mode"
    )

    use_blur = st.sidebar.checkbox(
        "Apply Gaussian Blur",
        value=st.session_state["use_blur"],
        key="use_blur"
    )

    blur_kernel_size = st.sidebar.slider(
        "Blur Kernel Size",
        min_value=3, max_value=15,
        value=st.session_state["blur_kernel_size"],
        step=2,
        disabled=not use_blur,
        key="blur_kernel_size"
    )

    use_dilation = st.sidebar.checkbox(
        "Enable Morphological Dilation",
        value=st.session_state["use_dilation"],
        key="use_dilation"
    )

    col_w, col_h = st.sidebar.columns(2)
    with col_w:
        dilation_w = st.slider("Kernel Width", min_value=3, max_value=51, value=st.session_state["dilation_w"], step=2, disabled=not use_dilation, key="dilation_w")
    with col_h:
        dilation_h = st.sidebar.slider("Kernel Height", min_value=3, max_value=51, value=st.session_state["dilation_h"], step=2, disabled=not use_dilation, key="dilation_h")

    st.sidebar.markdown("---")

    st.sidebar.markdown("<h4 style='color: #a78bfa; margin-bottom: 0px;'>Boundary & Thresholds</h4>", unsafe_allow_html=True)
    crop_mode_options = ["Union of All Regions", "Largest Region Only"]
    crop_mode = st.sidebar.radio(
        "Crop Boundary Mode",
        crop_mode_options,
        index=safe_index(crop_mode_options, st.session_state["crop_mode"]),
        key="crop_mode"
    )

    ocr_tolerance_px = st.sidebar.slider("OCR Line Tolerance (px)", min_value=5, max_value=40, value=st.session_state["ocr_tolerance_px"], key="ocr_tolerance_px")
    det_db_thresh = st.sidebar.slider("Detection Score Threshold", min_value=0.10, max_value=0.90, value=st.session_state["det_db_thresh"], step=0.05, key="det_db_thresh")
    det_db_unclip_ratio = st.sidebar.slider("Box Unclip Ratio", min_value=1.0, max_value=3.0, value=st.session_state["det_db_unclip_ratio"], step=0.1, key="det_db_unclip_ratio")
    padding_px = st.sidebar.slider("Padding (px)", min_value=0, max_value=100, value=st.session_state["padding_px"], key="padding_px")
    min_area_filter = st.sidebar.slider("Min Area Filter (% of Frame)", min_value=0.1, max_value=10.0, value=st.session_state["min_area_filter"], step=0.1, key="min_area_filter")

    st.sidebar.markdown("---")

    st.sidebar.markdown("<h4 style='color: #a78bfa; margin-bottom: 0px;'>OCR Engine Tuning</h4>", unsafe_allow_html=True)
    ocr_preprocess_mode_options = ["Original (RGB)", "Grayscale (Monochrome)", "Adaptive Thresholding"]
    ocr_preprocess_mode = st.sidebar.selectbox(
        "OCR Preprocessing Mode",
        ocr_preprocess_mode_options,
        index=safe_index(ocr_preprocess_mode_options, st.session_state["ocr_preprocess_mode"]),
        key="ocr_preprocess_mode"
    )
    ocr_use_blur = st.sidebar.checkbox("OCR Apply Blur", value=st.session_state["ocr_use_blur"], key="ocr_use_blur")
    ocr_blur_kernel = st.sidebar.slider("OCR Blur Kernel Size", min_value=3, max_value=15, value=st.session_state["ocr_blur_kernel"], step=2, disabled=not ocr_use_blur, key="ocr_blur_kernel")
    ocr_det_db_thresh = st.sidebar.slider("OCR Score Threshold", min_value=0.10, max_value=0.90, value=st.session_state["ocr_det_db_thresh"], step=0.05, key="ocr_det_db_thresh")
    ocr_det_db_unclip_ratio = st.sidebar.slider("OCR Box Unclip Ratio", min_value=1.0, max_value=3.0, value=st.session_state["ocr_det_db_unclip_ratio"], step=0.1, key="ocr_det_db_unclip_ratio")

    st.sidebar.markdown("---")

    empty_strategy_options = ["Skip Frame", "Pass-Through Original"]
    empty_strategy = st.sidebar.selectbox(
        "Empty Frame Strategy",
        empty_strategy_options,
        index=safe_index(empty_strategy_options, st.session_state["empty_strategy"]),
        key="empty_strategy"
    )

else:
    # Render Module 4 Sidebar configuration options
    st.sidebar.markdown("<h4 style='color: #a78bfa; margin-top: 20px; margin-bottom: 0px;'>Histogram Tuning</h4>", unsafe_allow_html=True)
    hist_bins_opts = [16, 32, 64, 128, 256]
    hist_bins = st.sidebar.selectbox(
        "Histogram Bins",
        hist_bins_opts,
        index=safe_index(hist_bins_opts, st.session_state["hist_bins"]),
        key="hist_bins"
    )
    hist_method_opts = ["Correlation", "Chi-Square", "Intersection", "Bhattacharyya"]
    hist_method = st.sidebar.selectbox(
        "Comparison Method",
        hist_method_opts,
        index=safe_index(hist_method_opts, st.session_state["hist_method"]),
        key="hist_method"
    )
    color_mode_opts = ["Grayscale", "RGB"]
    color_mode = st.sidebar.selectbox(
        "Color Mode",
        color_mode_opts,
        index=safe_index(color_mode_opts, st.session_state["color_mode"]),
        key="color_mode"
    )
    hist_grid_size_opts = [2, 3, 4, 5, 8]
    hist_grid_size = st.sidebar.selectbox(
        "Grid Size (Histogram)",
        hist_grid_size_opts,
        index=safe_index(hist_grid_size_opts, st.session_state["hist_grid_size"]),
        key="hist_grid_size"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("<h4 style='color: #a78bfa; margin-bottom: 0px;'>Edge Detection Tuning</h4>", unsafe_allow_html=True)
    edge_blur_opts = ["None", "3x3", "5x5", "7x7"]
    edge_blur = st.sidebar.selectbox(
        "Gaussian Blur",
        edge_blur_opts,
        index=safe_index(edge_blur_opts, st.session_state["edge_blur"]),
        key="edge_blur"
    )
    canny_low = st.sidebar.slider("Canny Lower Threshold", min_value=0, max_value=255, value=st.session_state["canny_low"], key="canny_low")
    canny_high = st.sidebar.slider("Canny Upper Threshold", min_value=0, max_value=255, value=st.session_state["canny_high"], key="canny_high")
    
    edge_grid_size_opts = [2, 3, 4, 5, 8]
    edge_grid_size = st.sidebar.selectbox(
        "Grid Size (Edge)",
        edge_grid_size_opts,
        index=safe_index(edge_grid_size_opts, st.session_state["edge_grid_size"]),
        key="edge_grid_size"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("<h4 style='color: #a78bfa; margin-bottom: 0px;'>SSIM Tuning</h4>", unsafe_allow_html=True)
    ssim_win_opts = [7, 9, 11, 13]
    ssim_win_size = st.sidebar.selectbox(
        "Window Size",
        ssim_win_opts,
        index=safe_index(ssim_win_opts, st.session_state["ssim_win_size"]),
        key="ssim_win_size"
    )
    ssim_gaussian = st.sidebar.checkbox("Gaussian Weights", value=st.session_state["ssim_gaussian"], key="ssim_gaussian")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("<h4 style='color: #a78bfa; margin-bottom: 0px;'>Text Occupancy Tuning</h4>", unsafe_allow_html=True)
    text_thresh = st.sidebar.slider("Binary Threshold", min_value=0, max_value=255, value=st.session_state["text_thresh"], key="text_thresh")
    
    text_kernel_opts = [3, 5, 7, 9]
    text_kernel = st.sidebar.selectbox(
        "Morphological Kernel Size",
        text_kernel_opts,
        index=safe_index(text_kernel_opts, st.session_state["text_kernel"]),
        key="text_kernel"
    )
    text_iterations = st.sidebar.slider("Dilation Iterations", min_value=1, max_value=5, value=st.session_state["text_iterations"], key="text_iterations")
    text_min_area = st.sidebar.slider("Minimum Component Area", min_value=10, max_value=500, value=st.session_state["text_min_area"], key="text_min_area")

# Save current settings to file dynamically
current_settings = {}
for k in DEFAULT_SETTINGS.keys():
    if k in st.session_state:
        current_settings[k] = st.session_state[k]
try:
    with open(SETTINGS_FILE, "w") as f:
        json.dump(current_settings, f, indent=4)
except Exception:
    pass

# --- Load Detection Engines (if single frame or batch processes are run) ---
use_eng = st.session_state.get("use_english_ocr", False)
try:
    engine_v3, engine_v4 = load_detection_engines(use_eng)
except Exception as e:
    st.error(f"Failed to load OCR Engines: {str(e)}")
    st.stop()

# --- Main Workspace Header ---
st.markdown("<h1 class='main-title'>MY RESEARCH LAB FOR EXPERIMENTS</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Experimental test harness and batch session manager evaluating DBNet text-region classifiers for digital slides and blackboard frames.</p>", unsafe_allow_html=True)

# --- Top Navigation Tabs ---
tab_cols = st.columns(5)
modules_list = ["Single Frame Cropper", "Batch Crop Manager", "Candidate Frame Selector", "Pairwise Feature Vector Lab", "Batch Dataset Generator"]
icons = ["🔍", "📦", "🎯", "📊", "⚙️"]

def select_module_cb(module_name):
    st.session_state["selected_module"] = module_name

for idx, (name, icon) in enumerate(zip(modules_list, icons)):
    is_active = (selected_module == name)
    btn_type = "primary" if is_active else "secondary"
    with tab_cols[idx]:
        st.button(
            f"{icon} {name}",
            key=f"main_tab_btn_{name}",
            on_click=select_module_cb,
            args=(name,),
            use_container_width=True,
            type=btn_type
        )

st.markdown("<hr style='margin-top: 0.5rem; margin-bottom: 1.5rem;'/>", unsafe_allow_html=True)

# Route to respective modules
if selected_module == "Single Frame Cropper":
    render_single_cropper_tab(engine_v3, engine_v4)
elif selected_module == "Batch Crop Manager":
    render_batch_session_manager(engine_v3, engine_v4)
elif selected_module == "Candidate Frame Selector":
    render_candidate_selector()
elif selected_module == "Pairwise Feature Vector Lab":
    render_pairwise_feature_lab()
elif selected_module == "Batch Dataset Generator":
    render_batch_dataset_generator()
