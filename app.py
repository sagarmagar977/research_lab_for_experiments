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
from modules.dataset_combiner import render_dataset_combiner

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
    "text_min_area": 100,
    "hist_epsilon": 1e-10
}

# Load saved settings if exist
saved_settings = DEFAULT_SETTINGS.copy()
if os.path.exists(SETTINGS_FILE):
    try:
        with open(SETTINGS_FILE, "r") as f:
            saved = json.load(f)
            # Handle root-level select module
            if "selected_module" in saved:
                saved_settings["selected_module"] = saved["selected_module"]
            # Handle ocr_module_settings
            if "ocr_module_settings" in saved and isinstance(saved["ocr_module_settings"], dict):
                for k, v in saved["ocr_module_settings"].items():
                    if k in saved_settings:
                        saved_settings[k] = v
            # Handle pairwise_lab_settings
            if "pairwise_lab_settings" in saved and isinstance(saved["pairwise_lab_settings"], dict):
                for k, v in saved["pairwise_lab_settings"].items():
                    if k in saved_settings:
                        saved_settings[k] = v
            # Fallback for old flat settings if present
            for k, v in saved.items():
                if k in saved_settings and k not in ("ocr_module_settings", "pairwise_lab_settings"):
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
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich premium dark-mode styling
st.markdown("""
<style>
@import url('https://fonts.cdnfonts.com/css/sohne');
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap');

@font-face {
    font-family: 'Söhne';
    src: local('Söhne'), local('Soehne'), local('Sohne-Buch'), local('Sohne-Kraft');
}

html, body, [class*="css"], [class*="st-"]:not([class*="material"]):not([data-testid*="Icon"]):not([class*="e1vmumty"]), .stApp, input, textarea, select {
    font-family: 'Söhne', 'Soehne', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif !important;
}

button:not([data-testid*="Sidebar"]):not([data-testid*="sidebar"]) {
    font-family: 'Söhne', 'Soehne', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
}

/* Preserve Google Material Symbols & Streamlit DynamicIcon glyphs */
.material-symbols-rounded,
.material-symbols-outlined,
.material-icons,
[data-testid="stIconMaterial"],
[data-testid*="Icon"],
[class*="e1vmumty"] {
    font-family: 'Material Symbols Rounded', 'Material Symbols Outlined', 'Material Icons' !important;
}

code, pre, [class*="mono"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.88rem !important;
}

/* Header customization */
.main-title {
    color: #ffffff;
    font-weight: 800;
    font-size: 2.1rem;
    letter-spacing: -0.03em;
    margin-bottom: 0.2rem;
}
.subtitle {
    color: #71717a;
    font-size: 0.92rem;
    margin-bottom: 1.5rem;
}

/* Metric card styles */
.metric-container {
    display: flex;
    gap: 1rem;
    margin-bottom: 1rem;
}
.metric-card {
    background-color: #121214;
    border: 1px solid #27272a;
    border-radius: 8px;
    padding: 1rem;
    flex: 1;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
}
.metric-title {
    font-size: 0.72rem;
    color: #71717a;
    text-transform: uppercase;
    font-weight: 600;
    letter-spacing: 0.05em;
    margin-bottom: 0.25rem;
}
.metric-value {
    font-size: 1.4rem;
    font-weight: 700;
    color: #ffffff;
}
.status-badge {
    display: inline-block;
    padding: 0.25rem 0.65rem;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 600;
}
.status-success {
    background-color: rgba(16, 185, 129, 0.15);
    color: #10b981;
    border: 1px solid rgba(16, 185, 129, 0.3);
}
.status-warning {
    background-color: #1f1f23;
    color: #a1a1aa;
    border: 1px solid #27272a;
}

/* Warnings and callouts */
.stAlert {
    border-radius: 8px !important;
    background-color: #121214 !important;
    border: 1px solid #27272a !important;
    color: #f4f4f5 !important;
}

/* Title divider */
hr {
    margin-top: 1rem;
    margin-bottom: 2rem;
    border-color: #27272a;
}

/* Hide Streamlit native image fullscreen expand button */
button[title="View fullscreen"] {
    display: none !important;
}
/* ============================================================
   SIDEBAR TOGGLE BUTTON (Replace double_arrow_right with Modern Sidebar Icon)
   ============================================================ */
[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"] {
    visibility: visible !important;
    display: flex !important;
    align-items: center !important;
}

[data-testid="stSidebarCollapseButton"] button,
[data-testid="stSidebarCollapsedControl"] button,
[data-testid="collapsedControl"] button,
button[aria-label*="sidebar" i],
button[aria-label*="Sidebar" i] {
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    width: 32px !important;
    height: 32px !important;
    min-width: 32px !important;
    min-height: 32px !important;
    padding: 0 !important;
    background-color: #121214 !important;
    border: 1px solid #27272a !important;
    border-radius: 6px !important;
    color: transparent !important;
    font-size: 0 !important;
    line-height: 0 !important;
    cursor: pointer !important;
    overflow: hidden !important;
    position: relative !important;
    transition: all 0.15s ease !important;
}

[data-testid="stSidebarCollapseButton"] button:hover,
[data-testid="stSidebarCollapsedControl"] button:hover,
[data-testid="collapsedControl"] button:hover,
button[aria-label*="sidebar" i]:hover,
button[aria-label*="Sidebar" i]:hover {
    background-color: #1f1f23 !important;
    border-color: #52525b !important;
}

/* Hide any raw text (e.g. 'double_arrow_right') or default inner elements */
[data-testid="stSidebarCollapseButton"] button *,
[data-testid="stSidebarCollapsedControl"] button *,
[data-testid="collapsedControl"] button *,
button[aria-label*="sidebar" i] *,
button[aria-label*="Sidebar" i] * {
    display: none !important;
    visibility: hidden !important;
    font-size: 0 !important;
    line-height: 0 !important;
    width: 0 !important;
    height: 0 !important;
    color: transparent !important;
    position: absolute !important;
}

/* Render Modern Sidebar Panel Icon */
[data-testid="stSidebarCollapseButton"] button::after,
[data-testid="stSidebarCollapsedControl"] button::after,
[data-testid="collapsedControl"] button::after,
button[aria-label*="sidebar" i]::after,
button[aria-label*="Sidebar" i]::after {
    content: "" !important;
    display: block !important;
    visibility: visible !important;
    width: 18px !important;
    height: 18px !important;
    background-repeat: no-repeat !important;
    background-position: center !important;
    background-size: contain !important;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23d4d4d8' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect width='18' height='18' x='3' y='3' rx='4'/%3E%3Cline x1='9' x2='9' y1='3' y2='21'/%3E%3C/svg%3E") !important;
}

[data-testid="stSidebarCollapseButton"] button:hover::after,
[data-testid="stSidebarCollapsedControl"] button:hover::after,
[data-testid="collapsedControl"] button:hover::after,
button[aria-label*="sidebar" i]:hover::after,
button[aria-label*="Sidebar" i]:hover::after {
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23ffffff' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect width='18' height='18' x='3' y='3' rx='4'/%3E%3Cline x1='9' x2='9' y1='3' y2='21'/%3E%3C/svg%3E") !important;
}

/* ============================================================
   FILE UPLOADER DROPZONE BUTTON (Eliminate duplicate/overlapping icon text)
   ============================================================ */
[data-testid="stFileUploaderDropzone"] button {
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    position: relative !important;
    white-space: nowrap !important;
    overflow: hidden !important;
}

/* Hide the overflowing/broken icon ligature span that causes duplicate 'upload' text */
[data-testid="stFileUploaderDropzone"] button [class*="e1vmumty"],
[data-testid="stFileUploaderDropzone"] button [data-testid*="Icon"],
[data-testid="stFileUploaderDropzone"] button svg,
[data-testid="stFileUploaderDropzone"] button [data-has-shortcut] > *:first-child:not(:only-child),
[data-testid="stFileUploaderDropzone"] button span:has(+ span) {
    display: none !important;
    visibility: hidden !important;
    width: 0 !important;
    height: 0 !important;
    font-size: 0 !important;
    line-height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    overflow: hidden !important;
}

/* Ensure the upload button label text is clean and legible */
[data-testid="stFileUploaderDropzone"] button span {
    font-family: 'Söhne', 'Soehne', sans-serif !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
}

/* ============================================================
   GLOBAL HIGH-CONTRAST MONOCHROME BUTTONS (Pure Black & White)
   ============================================================ */

/* Primary Action Buttons (e.g. Start Generation, Run Inference, Compute Features) */
button[data-testid="baseButton-primary"]:not([data-testid*="Sidebar"]):not([data-testid*="sidebar"]):not(div[class*="st-key-main_tab_btn_"] button),
button[kind="primary"]:not([data-testid*="Sidebar"]):not([data-testid*="sidebar"]):not(div[class*="st-key-main_tab_btn_"] button) {
    background-color: #ffffff !important;
    background-image: none !important;
    color: #000000 !important;
    border: 1px solid #ffffff !important;
    border-radius: 6px !important;
    font-weight: 700 !important;
    font-size: 0.90rem !important;
    letter-spacing: -0.01em !important;
    box-shadow: 0 2px 8px rgba(255, 255, 255, 0.15) !important;
    transition: all 0.12s ease !important;
}

/* Ensure inner text elements in primary buttons are deep black and bold */
button[data-testid="baseButton-primary"]:not(div[class*="st-key-main_tab_btn_"] button) *,
button[kind="primary"]:not(div[class*="st-key-main_tab_btn_"] button) * {
    color: #000000 !important;
    font-weight: 700 !important;
    opacity: 1 !important;
}

button[data-testid="baseButton-primary"]:not(div[class*="st-key-main_tab_btn_"] button):hover,
button[kind="primary"]:not(div[class*="st-key-main_tab_btn_"] button):hover {
    background-color: #e4e4e7 !important;
    border-color: #d4d4d8 !important;
    color: #000000 !important;
    box-shadow: 0 4px 12px rgba(255, 255, 255, 0.25) !important;
    transform: translateY(-1px) !important;
}

button[data-testid="baseButton-primary"]:not(div[class*="st-key-main_tab_btn_"] button):active,
button[kind="primary"]:not(div[class*="st-key-main_tab_btn_"] button):active {
    transform: translateY(1px) !important;
    box-shadow: 0 1px 4px rgba(0, 0, 0, 0.4) !important;
}

/* Secondary Action Buttons (excluding folder tabs, sidebar icon, and file uploader) */
button[data-testid="baseButton-secondary"]:not([data-testid*="Sidebar"]):not([data-testid*="sidebar"]):not(div[class*="st-key-main_tab_btn_"] button):not([data-testid="stFileUploaderDropzone"] button),
button[kind="secondary"]:not([data-testid*="Sidebar"]):not([data-testid*="sidebar"]):not(div[class*="st-key-main_tab_btn_"] button):not([data-testid="stFileUploaderDropzone"] button) {
    background-color: #121215 !important;
    background-image: none !important;
    border: 1px solid #27272a !important;
    border-radius: 6px !important;
    color: #f4f4f5 !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    letter-spacing: -0.01em !important;
    box-shadow: 0 1px 4px rgba(0, 0, 0, 0.4) !important;
    transition: all 0.12s ease !important;
}

button[data-testid="baseButton-secondary"]:not(div[class*="st-key-main_tab_btn_"] button):not([data-testid="stFileUploaderDropzone"] button) *,
button[kind="secondary"]:not(div[class*="st-key-main_tab_btn_"] button):not([data-testid="stFileUploaderDropzone"] button) * {
    color: #f4f4f5 !important;
    font-weight: 500 !important;
}

button[data-testid="baseButton-secondary"]:not(div[class*="st-key-main_tab_btn_"] button):not([data-testid="stFileUploaderDropzone"] button):hover,
button[kind="secondary"]:not(div[class*="st-key-main_tab_btn_"] button):not([data-testid="stFileUploaderDropzone"] button):hover {
    background-color: #1c1c20 !important;
    border-color: #52525b !important;
    color: #ffffff !important;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.5) !important;
    transform: translateY(-1px) !important;
}

button[data-testid="baseButton-secondary"]:not(div[class*="st-key-main_tab_btn_"] button):not([data-testid="stFileUploaderDropzone"] button):hover * {
    color: #ffffff !important;
}

/* File Uploader Button - Clean Monochrome */
[data-testid="stFileUploaderDropzone"] button {
    background-color: #121215 !important;
    border: 1px solid #3f3f46 !important;
    color: #ffffff !important;
    font-weight: 600 !important;
}
[data-testid="stFileUploaderDropzone"] button:hover {
    background-color: #1f1f23 !important;
    border-color: #71717a !important;
    color: #ffffff !important;
}
[data-testid="stFileUploaderDropzone"] button span {
    color: inherit !important;
}

/* Candidate Selection buttons preserve green strictly */
div[class*="st-key-unmark_g_"] button,
div[class*="st-key-btn_toggle_star"] button,
div[class*="st-key-rm_show_"] button {
    background-color: rgba(16, 185, 129, 0.15) !important;
    border: 1px solid #10b981 !important;
    color: #10b981 !important;
}
div[class*="st-key-unmark_g_"] button *,
div[class*="st-key-btn_toggle_star"] button *,
div[class*="st-key-rm_show_"] button * {
    color: #10b981 !important;
}

/* ============================================================
   WINDOWS TASK MANAGER STYLE FOLDER TAB NAVBAR (Pure Monochrome)
   ============================================================ */
[data-testid="stHorizontalBlock"]:has(div[class*="st-key-main_tab_btn_"]) {
    display: flex !important;
    gap: 2px !important;
    border-bottom: 1px solid #27272a !important;
    padding-bottom: 0px !important;
    margin-top: 0.5rem !important;
    margin-bottom: 1.5rem !important;
    align-items: flex-end !important;
}

[data-testid="stHorizontalBlock"]:has(div[class*="st-key-main_tab_btn_"]) > [data-testid="stColumn"] {
    flex: 1 1 0% !important;
    min-width: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
}

/* Base Tab Button */
div[class*="st-key-main_tab_btn_"] button {
    border-radius: 4px 4px 0 0 !important;
    font-family: 'Söhne', 'Soehne', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    font-size: 0.78rem !important;
    padding: 6px 3px !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    margin-bottom: -1px !important;
    cursor: pointer !important;
    box-shadow: none !important;
    transition: background-color 0.1s ease, color 0.1s ease, border-color 0.1s ease !important;
}

/* Inactive Folder Tab (flat, rests on baseline, muted gray text) */
div[class*="st-key-main_tab_btn_"] button[data-testid="baseButton-secondary"],
div[class*="st-key-main_tab_btn_"] button[kind="secondary"] {
    background-color: #0d0d10 !important;
    background-image: none !important;
    color: #82828c !important;
    border: 1px solid #27272a !important;
    border-bottom: 1px solid #27272a !important;
    font-weight: 500 !important;
    height: 36px !important;
    transform: none !important;
}

div[class*="st-key-main_tab_btn_"] button[data-testid="baseButton-secondary"] *,
div[class*="st-key-main_tab_btn_"] button[kind="secondary"] * {
    color: #82828c !important;
    font-weight: 500 !important;
}

div[class*="st-key-main_tab_btn_"] button[data-testid="baseButton-secondary"]:hover,
div[class*="st-key-main_tab_btn_"] button[kind="secondary"]:hover {
    background-color: #18181c !important;
    color: #ffffff !important;
    border-color: #3f3f46 !important;
}

div[class*="st-key-main_tab_btn_"] button[data-testid="baseButton-secondary"]:hover * {
    color: #ffffff !important;
}

/* Active Folder Tab (Windows Task Manager style: elevated, white top highlight, open bottom) */
div[class*="st-key-main_tab_btn_"] button[data-testid="baseButton-primary"],
div[class*="st-key-main_tab_btn_"] button[kind="primary"] {
    background-color: #1a1a1e !important;
    background-image: none !important;
    color: #ffffff !important;
    border: 1px solid #3f3f46 !important;
    border-top: 2px solid #ffffff !important;
    border-bottom: 1px solid #1a1a1e !important;
    font-weight: 700 !important;
    position: relative !important;
    z-index: 10 !important;
    height: 39px !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 -2px 10px rgba(0, 0, 0, 0.5) !important;
}

div[class*="st-key-main_tab_btn_"] button[data-testid="baseButton-primary"] *,
div[class*="st-key-main_tab_btn_"] button[kind="primary"] * {
    color: #ffffff !important;
    font-weight: 700 !important;
}
</style>
""", unsafe_allow_html=True)

# --- Sidebar Module Selector Navigation ---
if "_redirect_module" in st.session_state:
    st.session_state["selected_module"] = st.session_state.pop("_redirect_module")

st.sidebar.markdown("<h2 style='color: #ffffff; margin-bottom: 0px;'>Lab Module Selector</h2>", unsafe_allow_html=True)
modules_options = ["Single Frame Cropper", "Batch Crop Manager", "Candidate Frame Selector", "Pairwise Feature Vector Lab", "Batch Dataset Generator", "L1 Inference", "Dataset Combiner", "Level 2 Grouping Lab"]
selected_module = st.sidebar.selectbox(
    "Select Lab Module",
    modules_options,
    index=safe_index(modules_options, st.session_state["selected_module"]),
    key="selected_module"
)

# --- Dynamic Sidebar Rendering Based on Selected Module ---
if selected_module not in ("Pairwise Feature Vector Lab", "Batch Dataset Generator", "L1 Inference", "Dataset Combiner", "Level 2 Grouping Lab"):
    st.sidebar.markdown("<h4 style='color: #ffffff; margin-top: 20px; margin-bottom: 0px;'>OCR Model Tuning</h4>", unsafe_allow_html=True)
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

    st.sidebar.markdown("<h4 style='color: #ffffff; margin-bottom: 0px;'>Preprocessing & Dilation</h4>", unsafe_allow_html=True)
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

    st.sidebar.markdown("<h4 style='color: #ffffff; margin-bottom: 0px;'>Boundary & Thresholds</h4>", unsafe_allow_html=True)
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

    st.sidebar.markdown("<h4 style='color: #ffffff; margin-bottom: 0px;'>OCR Engine Tuning</h4>", unsafe_allow_html=True)
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
    st.sidebar.markdown("<h4 style='color: #ffffff; margin-top: 20px; margin-bottom: 0px;'>Histogram Tuning</h4>", unsafe_allow_html=True)
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
    hist_epsilon = st.sidebar.number_input(
        "Histogram Epsilon",
        value=st.session_state["hist_epsilon"],
        format="%.1e",
        key="hist_epsilon"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("<h4 style='color: #ffffff; margin-bottom: 0px;'>Edge Detection Tuning</h4>", unsafe_allow_html=True)
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
    st.sidebar.markdown("<h4 style='color: #ffffff; margin-bottom: 0px;'>SSIM Tuning</h4>", unsafe_allow_html=True)
    ssim_win_opts = [7, 9, 11, 13]
    ssim_win_size = st.sidebar.selectbox(
        "Window Size",
        ssim_win_opts,
        index=safe_index(ssim_win_opts, st.session_state["ssim_win_size"]),
        key="ssim_win_size"
    )
    ssim_gaussian = st.sidebar.checkbox("Gaussian Weights", value=st.session_state["ssim_gaussian"], key="ssim_gaussian")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("<h4 style='color: #ffffff; margin-bottom: 0px;'>Text Occupancy Tuning</h4>", unsafe_allow_html=True)
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
OCR_KEYS = {
    "model_mode", "use_english_ocr", "preprocess_mode", "use_blur", "blur_kernel_size",
    "use_dilation", "dilation_w", "dilation_h", "crop_mode", "ocr_tolerance_px",
    "det_db_thresh", "det_db_unclip_ratio", "padding_px", "min_area_filter",
    "ocr_preprocess_mode", "ocr_use_blur", "ocr_blur_kernel", "ocr_det_db_thresh",
    "ocr_det_db_unclip_ratio", "empty_strategy"
}

PAIRWISE_KEYS = {
    "hist_bins", "hist_method", "color_mode", "hist_grid_size", "edge_blur",
    "canny_low", "canny_high", "edge_grid_size", "ssim_win_size", "ssim_gaussian",
    "text_thresh", "text_kernel", "text_iterations", "text_min_area", "hist_epsilon"
}

current_settings = {
    "selected_module": st.session_state.get("selected_module", "Single Frame Cropper"),
    "ocr_module_settings": {},
    "pairwise_lab_settings": {}
}

for k in OCR_KEYS:
    if k in st.session_state:
        current_settings["ocr_module_settings"][k] = st.session_state[k]

for k in PAIRWISE_KEYS:
    if k in st.session_state:
        current_settings["pairwise_lab_settings"][k] = st.session_state[k]

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
tab_cols = st.columns(8, gap="small")
modules_list = ["Single Frame Cropper", "Batch Crop Manager", "Candidate Frame Selector", "Pairwise Feature Vector Lab", "Batch Dataset Generator", "L1 Inference", "Dataset Combiner", "Level 2 Grouping Lab"]

def select_module_cb(module_name):
    st.session_state["selected_module"] = module_name

for idx, name in enumerate(modules_list):
    is_active = (selected_module == name)
    btn_type = "primary" if is_active else "secondary"
    with tab_cols[idx]:
        st.button(
            name,
            key=f"main_tab_btn_{name}",
            on_click=select_module_cb,
            args=(name,),
            use_container_width=True,
            type=btn_type
        )

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
elif selected_module == "L1 Inference":
    from modules.l1_inference import render_l1_inference
    render_l1_inference()
elif selected_module == "Dataset Combiner":
    render_dataset_combiner()
elif selected_module == "Level 2 Grouping Lab":
    from modules.l2_grouping_lab import render_l2_grouping_lab
    render_l2_grouping_lab()
