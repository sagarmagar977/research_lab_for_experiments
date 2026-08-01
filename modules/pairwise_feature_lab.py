import os
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity

def compute_global_histogram(img, bins, method, color_mode):
    """Computes a normalized histogram and returns the histogram array."""
    if color_mode == "Grayscale":
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        hist = cv2.calcHist([gray], [0], None, [bins], [0, 256])
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        return hist
    else:
        # RGB mode
        hists = []
        for channel in range(3):
            hist = cv2.calcHist([img], [channel], None, [bins], [0, 256])
            cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            hists.append(hist)
        # Concatenate channels
        concat_hist = np.concatenate(hists, axis=0)
        return concat_hist

def compare_histograms(hist1, hist2, method):
    """Compares two histograms using the selected method."""
    method_map = {
        "Correlation": cv2.HISTCMP_CORREL,
        "Chi-Square": cv2.HISTCMP_CHISQR,
        "Intersection": cv2.HISTCMP_INTERSECT,
        "Bhattacharyya": cv2.HISTCMP_BHATTACHARYYA
    }
    method_id = method_map.get(method, cv2.HISTCMP_CORREL)
    return float(cv2.compareHist(hist1, hist2, method_id))

def get_grid_cells(img, grid_size):
    """Generator yielding grid cell boundaries (ymin, ymax, xmin, xmax)."""
    h, w = img.shape[:2]
    cell_h = h / grid_size
    cell_w = w / grid_size
    for i in range(grid_size):
        for j in range(grid_size):
            ymin, ymax = int(i * cell_h), int(min((i + 1) * cell_h, h))
            xmin, xmax = int(j * cell_w), int(min((j + 1) * cell_w, w))
            yield ymin, ymax, xmin, xmax

def draw_grid_overlay(img, grid_size):
    """Draws grid overlay lines on a copy of the image."""
    overlay = img.copy()
    h, w = overlay.shape[:2]
    cell_h = h / grid_size
    cell_w = w / grid_size
    
    # Draw vertical lines
    for j in range(1, grid_size):
        x = int(j * cell_w)
        cv2.line(overlay, (x, 0), (x, h), (239, 68, 68), 2)
        
    # Draw horizontal lines
    for i in range(1, grid_size):
        y = int(i * cell_h)
        cv2.line(overlay, (0, y), (w, y), (239, 68, 68), 2)
        
    return overlay

def compute_grid_histogram_difference(img_a, img_b, bins, method, color_mode, grid_size):
    """Divides images into grids, compares corresponding cell histograms, and averages scores."""
    cells_a = list(get_grid_cells(img_a, grid_size))
    cells_b = list(get_grid_cells(img_b, grid_size))
    
    scores = []
    for (ya1, ya2, xa1, xa2), (yb1, yb2, xb1, xb2) in zip(cells_a, cells_b):
        cell_a = img_a[ya1:ya2, xa1:xa2]
        cell_b = img_b[yb1:yb2, xb1:xb2]
        
        hist_a = compute_global_histogram(cell_a, bins, method, color_mode)
        hist_b = compute_global_histogram(cell_b, bins, method, color_mode)
        
        score = compare_histograms(hist_a, hist_b, method)
        scores.append(score)
        
    return float(np.mean(scores))

def get_canny_edges(img, blur_size_str, canny_low, canny_high):
    """Converts image to grayscale, applies blur (if configured), and extracts Canny edge map."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if blur_size_str != "None":
        try:
            k = int(blur_size_str.split("x")[0])
            gray = cv2.GaussianBlur(gray, (k, k), 0)
        except Exception:
            pass
    edges = cv2.Canny(gray, canny_low, canny_high)
    return edges

def compute_whole_edge_density(edges):
    """Computes edge pixel density (edge pixels / total pixels)."""
    return float(np.sum(edges == 255) / edges.size)

def compute_grid_edge_difference(edges_a, edges_b, grid_size):
    """Computes average absolute difference in edge density across corresponding grid cells."""
    cells_a = list(get_grid_cells(edges_a, grid_size))
    cells_b = list(get_grid_cells(edges_b, grid_size))
    
    diffs = []
    for (ya1, ya2, xa1, xa2), (yb1, yb2, xb1, xb2) in zip(cells_a, cells_b):
        cell_a = edges_a[ya1:ya2, xa1:xa2]
        cell_b = edges_b[yb1:yb2, xb1:xb2]
        
        density_a = np.sum(cell_a == 255) / cell_a.size
        density_b = np.sum(cell_b == 255) / cell_b.size
        
        diffs.append(abs(density_a - density_b))
        
    return float(np.mean(diffs))

def get_text_occupancy_mask(img, threshold, kernel_size, iterations, min_area):
    """Binarizes image, performs dilation, filters out small components, and returns text mask."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Threshold to binary (inverted, so text pixels are white)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    
    # Morphological dilation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    dilated = cv2.dilate(binary, kernel, iterations=iterations)
    
    # Connected component filtering
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dilated)
    
    mask = np.zeros_like(dilated)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            mask[labels == i] = 255
            
    return binary, dilated, mask

def compute_text_occupancy_ratio(mask):
    """Computes percentage of text pixel occupancy."""
    return float(np.sum(mask == 255) / mask.size)

def compute_ssim(img_a, img_b, win_size, use_gaussian):
    """Calculates Structural Similarity Index (SSIM) and outputs the difference map."""
    gray_a = cv2.cvtColor(img_a, cv2.COLOR_RGB2GRAY)
    gray_b = cv2.cvtColor(img_b, cv2.COLOR_RGB2GRAY)
    
    # Ensure window size is within limits
    min_dim = min(gray_a.shape[0], gray_a.shape[1])
    if min_dim < win_size:
        win_size = min_dim - (1 if min_dim % 2 == 0 else 0)
        win_size = max(3, win_size)
        
    score, diff = structural_similarity(
        gray_a, gray_b,
        win_size=win_size,
        gaussian_weights=use_gaussian,
        full=True
    )
    return score, diff

def render_pairwise_feature_lab():
    """Renders the main layout, sidebar controls, visualizations, and CSV downloader for Module 4."""
    # Module header
    st.markdown("### Pairwise Feature Vector Experimentation Lab")
    st.write("Compare two cropped educational frame images to tune feature extraction parameters in real-time.")
    
    # 1. File Uploads
    col_u1, col_u2 = st.columns(2)
    with col_u1:
        file_a = st.file_uploader("Upload Frame A", type=["jpg", "jpeg", "png"], key="upload_fv_a")
    with col_u2:
        file_b = st.file_uploader("Upload Frame B", type=["jpg", "jpeg", "png"], key="upload_fv_b")
        
    if not file_a or not file_b:
        st.info("Please upload both Frame A and Frame B to run calculations.")
        return
        
    # Read files
    img_a_pil = Image.open(file_a)
    img_b_pil = Image.open(file_b)
    
    img_a_orig = np.array(img_a_pil.convert("RGB"))
    img_b_raw = np.array(img_b_pil.convert("RGB"))
    
    # Resize Frame B to match Frame A dimensions for structural operations
    h_a, w_a = img_a_orig.shape[:2]
    img_b_orig = cv2.resize(img_b_raw, (w_a, h_a))
    
    # Retrieve configuration sliders from session_state
    bins = st.session_state["hist_bins"]
    method = st.session_state["hist_method"]
    color_mode = st.session_state["color_mode"]
    hist_grid_size = st.session_state["hist_grid_size"]
    
    edge_blur = st.session_state["edge_blur"]
    canny_low = st.session_state["canny_low"]
    canny_high = st.session_state["canny_high"]
    edge_grid_size = st.session_state["edge_grid_size"]
    
    ssim_win_size = st.session_state["ssim_win_size"]
    ssim_gaussian = st.session_state["ssim_gaussian"]
    
    text_thresh = st.session_state["text_thresh"]
    text_kernel = st.session_state["text_kernel"]
    text_iterations = st.session_state["text_iterations"]
    text_min_area = st.session_state["text_min_area"]
    
    # 2. RUN CALCULATIONS
    # Histograms
    hist_a = compute_global_histogram(img_a_orig, bins, method, color_mode)
    hist_b = compute_global_histogram(img_b_orig, bins, method, color_mode)
    global_hist_diff = compare_histograms(hist_a, hist_b, method)
    
    grid_hist_diff = compute_grid_histogram_difference(img_a_orig, img_b_orig, bins, method, color_mode, hist_grid_size)
    
    # Canny Edges
    edges_a = get_canny_edges(img_a_orig, edge_blur, canny_low, canny_high)
    edges_b = get_canny_edges(img_b_orig, edge_blur, canny_low, canny_high)
    
    density_a = compute_whole_edge_density(edges_a)
    density_b = compute_whole_edge_density(edges_b)
    whole_edge_diff = abs(density_a - density_b)
    
    grid_edge_diff = compute_grid_edge_difference(edges_a, edges_b, edge_grid_size)
    
    # SSIM
    ssim_score, ssim_diff_map = compute_ssim(img_a_orig, img_b_orig, ssim_win_size, ssim_gaussian)
    
    # Text Occupancy
    bin_a, dil_a, mask_a = get_text_occupancy_mask(img_a_orig, text_thresh, text_kernel, text_iterations, text_min_area)
    bin_b, dil_b, mask_b = get_text_occupancy_mask(img_b_orig, text_thresh, text_kernel, text_iterations, text_min_area)
    
    occupancy_a = compute_text_occupancy_ratio(mask_a)
    occupancy_b = compute_text_occupancy_ratio(mask_b)
    text_occ_diff = abs(occupancy_a - occupancy_b)
    
    # Absolute difference image
    abs_diff = cv2.absdiff(img_a_orig, img_b_orig)
    abs_diff_gray = cv2.cvtColor(abs_diff, cv2.COLOR_RGB2GRAY)
    
    # --- VISUALIZATIONS ---
    st.markdown("---")
    st.markdown("### 📊 Visualizations Panel")
    
    # Section 1: Original Images with grid overlay
    st.markdown("#### 1. Original Frames with Grid Overlay")
    col_im1, col_im2 = st.columns(2)
    with col_im1:
        grid_a = draw_grid_overlay(img_a_orig, hist_grid_size)
        st.image(grid_a, caption=f"Frame A ({w_a}x{h_a}) - Grid Overlay", use_container_width=True)
    with col_im2:
        grid_b = draw_grid_overlay(img_b_orig, hist_grid_size)
        st.image(grid_b, caption=f"Frame B (Resized to {w_a}x{h_a}) - Grid Overlay", use_container_width=True)
        
    # Section 2: Histograms
    st.markdown("#### 2. Color/Grayscale Histograms")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5), facecolor="#1e1e2f")
    ax1.set_facecolor("#1e1e2f")
    ax2.set_facecolor("#1e1e2f")
    
    for ax in (ax1, ax2):
        ax.tick_params(colors="#9ca3af")
        ax.xaxis.label.set_color("#9ca3af")
        ax.yaxis.label.set_color("#9ca3af")
        ax.title.set_color("#a78bfa")
        ax.grid(True, color="#2e2e4f")
        
    if color_mode == "Grayscale":
        ax1.plot(hist_a, color="#6366f1", linewidth=2)
        ax1.set_title("Frame A Histogram")
        ax2.plot(hist_b, color="#6366f1", linewidth=2)
        ax2.set_title("Frame B Histogram")
    else:
        # Plot RGB components separate for visualization
        colors = ("red", "green", "blue")
        for i, color in enumerate(colors):
            hist_a_ch = cv2.calcHist([img_a_orig], [i], None, [bins], [0, 256])
            hist_b_ch = cv2.calcHist([img_b_orig], [i], None, [bins], [0, 256])
            cv2.normalize(hist_a_ch, hist_a_ch, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            cv2.normalize(hist_b_ch, hist_b_ch, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            ax1.plot(hist_a_ch, color=color, alpha=0.7, linewidth=1.5)
            ax2.plot(hist_b_ch, color=color, alpha=0.7, linewidth=1.5)
        ax1.set_title("Frame A RGB Channels")
        ax2.set_title("Frame B RGB Channels")
        
    st.pyplot(fig)
    plt.close(fig)
    
    # Section 3: Edge Maps
    st.markdown("#### 3. Edge Detection Output (Gaussian Blur + Canny)")
    col_ed1, col_ed2 = st.columns(2)
    with col_ed1:
        st.image(edges_a, caption=f"Frame A Edge Map (Density: {density_a:.4f})", use_container_width=True)
    with col_ed2:
        st.image(edges_b, caption=f"Frame B Edge Map (Density: {density_b:.4f})", use_container_width=True)
        
    # Section 4: Text Occupancy
    st.markdown("#### 4. Morphological Text Region Masks")
    col_tx1, col_tx2 = st.columns(2)
    with col_tx1:
        st.image(mask_a, caption=f"Frame A Text Mask (Occupancy: {occupancy_a:.4f})", use_container_width=True)
    with col_tx2:
        st.image(mask_b, caption=f"Frame B Text Mask (Occupancy: {occupancy_b:.4f})", use_container_width=True)
        
    # Section 5: Difference Image & SSIM
    st.markdown("#### 5. Image Structural Difference Visualizers")
    col_df1, col_df2 = st.columns(2)
    with col_df1:
        st.image(abs_diff_gray, caption="Grayscale Absolute Pixel Difference |A - B|", use_container_width=True)
    with col_df2:
        # Normalize SSIM map for visualization
        ssim_vis = ((ssim_diff_map + 1.0) * 127.5).astype(np.uint8)
        st.image(ssim_vis, caption=f"SSIM Heatmap (Overall Index: {ssim_score:.4f})", use_container_width=True)
        
    # --- FEATURES TABLE ---
    st.markdown("---")
    st.markdown("### 📋 Computed Visual Feature Comparisons")
    
    df_features = pd.DataFrame({
        "Feature Metric": [
            "Global Histogram Comparison",
            "Grid Histogram Comparison (Average)",
            "Whole Edge Density",
            "Grid Edge Density Difference (Average)",
            "SSIM Structural Index",
            "Text Occupancy Ratio"
        ],
        "Frame A Value": [
            "N/A", "N/A",
            f"{density_a:.4f}",
            "N/A", "N/A",
            f"{occupancy_a:.4f}"
        ],
        "Frame B Value": [
            "N/A", "N/A",
            f"{density_b:.4f}",
            "N/A", "N/A",
            f"{occupancy_b:.4f}"
        ],
        "Pairwise Metric / Difference": [
            f"{global_hist_diff:.4f}",
            f"{grid_hist_diff:.4f}",
            f"{whole_edge_diff:.4f}",
            f"{grid_edge_diff:.4f}",
            f"{ssim_score:.4f}",
            f"{text_occ_diff:.4f}"
        ]
    })
    
    st.dataframe(df_features, use_container_width=True, hide_index=True)
    
    # --- PAIRWISE FEATURE VECTOR & CSV PREVIEW ---
    st.markdown("---")
    st.markdown("### 💾 Pairwise Feature Vector Export")
    
    # One-row dataframe matching columns exactly: Frame_A, Frame_B, Global_Histogram, Grid_Histogram, Whole_Edge, Grid_Edge, SSIM, Text_Occupancy
    df_export = pd.DataFrame([{
        "Frame_A": file_a.name,
        "Frame_B": file_b.name,
        "Global_Histogram": round(global_hist_diff, 4),
        "Grid_Histogram": round(grid_hist_diff, 4),
        "Whole_Edge": round(whole_edge_diff, 4),
        "Grid_Edge": round(grid_edge_diff, 4),
        "SSIM": round(ssim_score, 4),
        "Text_Occupancy": round(text_occ_diff, 4)
    }])
    
    st.write("**Vector CSV Row Preview:**")
    st.dataframe(df_export, use_container_width=True, hide_index=True)
    
    # Download button for CSV (header-less as required by PRD: "Do NOT include labels")
    csv_str = df_export.to_csv(index=False, header=False).strip()
    
    st.download_button(
        label="Download Pairwise Vector CSV",
        data=csv_str,
        file_name="pairwise_vector.csv",
        mime="text/csv",
        use_container_width=True
    )
