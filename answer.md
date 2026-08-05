# Pairwise Feature Vector Lab - Architecture Review

Answers to the questions outlined in [question.md](file:///x:/prototype%20for%20thesis/candidate%20frame%20selction%20lab/assets/question.md).

---

## 1. Overall Architecture

The execution pipeline for comparing Frame A and Frame B operates as follows:

```
Frame Upload (Streamlit File Uploader)
↓
Image Loading (PIL Image opening & conversion to RGB numpy arrays)
↓
Dimensional Alignment (Resize Frame B to match Frame A's dimensions via cv2.resize)
↓
Parallel Feature Extraction
├── Histogram Computation (PDF normalization on Grayscale/RGB channels)
├── Edge Detection (Grayscale conversion → Gaussian Blur → Canny Edge Maps)
├── Text Occupancy Masking (Binary Inversion → Dilation → Connected Components Area Filtering)
└── SSIM Calculation (Grayscale conversion → Dynamic window size boundary checks)
↓
Pairwise Comparison Computations
├── Global Histogram Difference (cv2.compareHist on selected method)
├── Grid Histogram Difference (Cell-by-cell comparison over NxN grid and average)
├── Whole Edge Density Difference (Absolute difference of overall densities)
├── Grid Edge Density Difference (Cell-by-cell absolute difference over NxN grid and average)
├── SSIM index & structural similarity map
└── Text Occupancy Difference (Absolute difference of occupancy ratios)
↓
UI Visualizations Rendering (Matplotlib plots, grid overlays, edge maps, SSIM heatmap, difference image)
↓
Feature Table & Pairwise Vector CSV Generation
```

---

## 2. Feature Extraction Pipeline

### Global Histogram
*   **Computation**: Computes grayscale or RGB (channel-concatenated) histograms. Normalizes values to a scale of `[0, 1]`. Compares vectors using `cv2.compareHist` with the selected metric (Correlation, Chi-Square, Intersection, or Bhattacharyya).
*   **Function**: `compute_global_histogram` & `compare_histograms` in `modules/pairwise_feature_lab.py`.
*   **Inputs**: image array, bin count, comparison method, color mode.
*   **Dependencies**: None.

### Grid Histogram
*   **Computation**: Partitions images into $N \times N$ cells, computes individual cell histograms, compares corresponding cells, and averages results: `np.mean(scores)`.
*   **Function**: `compute_grid_histogram_difference` in `modules/pairwise_feature_lab.py`.
*   **Inputs**: `img_a`, `img_b`, bin count, comparison method, color mode, grid size $N$.
*   **Dependencies**: `compute_global_histogram`, `compare_histograms`.

### Whole Edge Density
*   **Computation**: Computes Canny edges, then divides edge pixel count (value 255) by total image pixels: `np.sum(edges == 255) / edges.size`.
*   **Function**: `compute_whole_edge_density` in `modules/pairwise_feature_lab.py`.
*   **Inputs**: Canny edge map array.
*   **Dependencies**: Canny edge map generation (`get_canny_edges`).

### Grid Edge Density
*   **Computation**: Partitions both Canny edge maps into $N \times N$ cells, calculates edge density for each cell, computes the cell-wise absolute difference, and averages them.
*   **Function**: `compute_grid_edge_difference` in `modules/pairwise_feature_lab.py`.
*   **Inputs**: `edges_a`, `edges_b`, grid size $N$.
*   **Dependencies**: Canny edge map generation (`get_canny_edges`).

### SSIM
*   **Computation**: Evaluates Structural Similarity Index over grayscale images using sliding window and weights. Caps sliding window size to fit within image boundaries.
*   **Function**: `compute_ssim` in `modules/pairwise_feature_lab.py`.
*   **Inputs**: `img_a`, `img_b`, window size, Gaussian weights boolean.
*   **Dependencies**: Aligned/resized input images.

### Morphological Text Mask
*   **Computation**: Grayscale conversion → binary inversion thresholding → morphological dilation to merge character blocks → connected component labeling. Discards components with pixel area $< min\_area$.
*   **Function**: `get_text_occupancy_mask` in `modules/pairwise_feature_lab.py`.
*   **Inputs**: image array, binarization threshold, dilation kernel size, dilation iterations, minimum connected component area.
*   **Dependencies**: None.

### Text Occupancy
*   **Computation**: Calculates percentage ratio of text pixels (value 255) in the morphological text mask: `np.sum(mask == 255) / mask.size`.
*   **Function**: `compute_text_occupancy_ratio` in `modules/pairwise_feature_lab.py`.
*   **Inputs**: Morphological mask array.
*   **Dependencies**: Text mask generation (`get_text_occupancy_mask`).

---

## 3. Histogram Module

*   **Why they show N/A**: The "Frame A Value" and "Frame B Value" columns in the comparative UI table display "N/A" because a histogram comparison represents a *pairwise difference metric* between two images. A single frame (A or B) does not have a "histogram comparison" scalar on its own.
*   **Implementation Status**: Fully implemented. The computed similarity/difference values are displayed correctly under the "Pairwise Metric / Difference" column and exported to the output CSV.
*   **How it works**: Computes individual normalized histograms $H_A$ and $H_B$, then evaluates:
    *   *Correlation*: $\text{score} \in [-1, 1]$ (1 is perfect match).
    *   *Chi-Square*: $\text{score} \ge 0$ (0 is perfect match).
    *   *Intersection*: $\text{score} \in [0, 1]$ if normalized (1 is perfect match).
    *   *Bhattacharyya*: $\text{score} \in [0, 1]$ (0 is perfect match).

---

## 4. Grid Features

Grid-based features partition the spatial canvas to evaluate local changes.
*   **Cell Division**: The image width and height are divided into $N \times N$ cells. Integer boundaries are computed: `cell_w = width // N`, `cell_h = height // N`. Cell dimensions are not required to be square.
*   **Local Calculation**: The feature metric is calculated on each sub-crop independently.
*   **Corresponding Comparison**: Cell $(i, j)$ in Frame A is compared directly to cell $(i, j)$ in Frame B.
*   **Aggregation**: Absolute differences or similarities are averaged: `mean(score_ij)` over all $N^2$ cells.

---

## 5. Edge Features

*   **Whole Edge Density**: Calculated on the entire image. Measures overall text/detail presence.
*   **Grid Edge Density**: Implemented in the codebase. Partitions both Canny edge maps into $N \times N$ cells, calculates cell-wise density differences, and averages them. Shows localized edge shifts (e.g. drawing lines or text appearing in specific sectors).

---

## 6. SSIM

*   **Scope**: Evaluates whole image structure.
*   **Window Size**: Selected in sidebar (`7, 9, 11, 13`). Caps dynamically to `min(win_size, min(gray_a.shape) - 1)` to handle small crops safely.
*   **Gaussian Weighting**: True/False checkbox (applies Gaussian weights instead of uniform weights).
*   **Library**: `scikit-image` (`skimage.metrics.structural_similarity`).
*   **Parameters**: `win_size=win_size`, `gaussian_weights=use_gaussian`, `full=True` (returns both the global scalar score and a pixel-wise similarity map).

---

## 7. Morphological Text Mask

*   **Pipeline**:
    1.  Convert to Grayscale.
    2.  Threshold to binary (inverted via `THRESH_BINARY_INV`, making dark text white on a dark background).
    3.  Dilate using a rectangular kernel of size $K \times K$ with $I$ iterations to group letter strokes into blocks.
    4.  Run connected components analysis (`cv2.connectedComponentsWithStats`).
    5.  Filter out components with area $< min\_area$ pixels.
*   **Measurements Extracted**: The percentage ratio of remaining white pixels (value 255) to the total image size.

---

## 8. Text Occupancy

*   **Calculation**:
    $$\text{Text Occupancy Ratio} = \frac{\text{Count of white pixels (255) in filtered mask}}{\text{Total pixels in mask}}$$
    This translates directly to: `np.sum(mask == 255) / mask.size`.
*   **Pairwise Metric**: The absolute difference: $|\text{Occupancy}_A - \text{Occupancy}_B|$.

---

## 9. CSV Export

*   **Row Generation**: Created from a Pandas DataFrame, converted to a header-less string via `.to_csv(index=False, header=False)`.
*   **Columns Exported**:
    `[Frame_A, Frame_B, Global_Histogram, Grid_Histogram, Whole_Edge, Grid_Edge, SSIM, Text_Occupancy]`
*   **Visualization Only**: Intermediate individual scalar values (individual frame edge densities and text occupancy ratios), difference maps, heatmaps, and grid lines.
*   **Placeholders**: None. All values exported to the CSV are calculated in real-time.

---

## 10. Visualizations

| Visualization | Representation | Data Used | Global/Grid | Affects CSV? |
| :--- | :--- | :--- | :--- | :--- |
| **Grid Overlay** | Partitions the spatial canvas for visual verification. | Original images with $N \times N$ red lines. | Grid | No |
| **RGB/Grayscale Histogram** | Distribution of color channel or gray values. | Aligned RGB/grayscale arrays. | Global | No (but parameters do) |
| **Edge Map** | High-frequency structural detail/text stroke maps. | Grayscale + Canny edge maps. | Global | No (but parameters do) |
| **Morphological Mask** | Isolated candidate text regions. | Binary morphological output. | Global | No (but parameters do) |
| **Difference Image** | Spatial pixel differences. | Grayscale of `cv2.absdiff(img_a, img_b)`. | Global | No |
| **SSIM Heatmap** | Spatial structural similarity distribution. | SSIM structural difference map. | Global | No |
| **Feature Table** | Computed scores summary. | Computed scalar values. | Both | Visual check of CSV data |

---

## 11. Hyperparameter Flow

When any slider or selectbox parameter is modified in the sidebar:
*   **Re-run Computations**: Streamlit reruns the script from the top. All comparators (histograms, Canny edge detection, text masking, and SSIM) recompute.
*   **Visualizations Updated**: Matplotlib histogram curves, Canny maps, morph masks, grids, absolute differences, and SSIM heatmaps update instantly.
*   **CSV Exports Updated**: The calculated vector updates, modifying the downloadable CSV row values in real-time.

---

## 12. Module Separation

The project follows a clean decoupled directory structure:
```
 candidate frame selection lab/
 ├── app.py                     # Main coordinator, sidebar selectors, dynamic routing
 ├── user_settings.json         # Persistent settings values (dynamic loads and saves)
 └── modules/
     ├── detection.py           # ONNX models, database cropped pipeline, DLA layout grouping
     ├── single_cropper.py      # Single Frame Cropper UI layout
     ├── batch_manager.py       # Batch Session Creator and browser
     ├── candidate_selector.py  # Stars selection, crop gallery pagination, lightbox player
     └── pairwise_feature_lab.py # Module 4 feature extraction engine, plot panels, CSV export
```

---

## 13. Current Limitations

1.  **Arbitrary Aspect Warping**: Resizing Frame B directly to match Frame A's dimensions (`cv2.resize`) can distort textures and edge distributions if input sizes are significantly different.
2.  **Absolute Size Components Filter**: Filtering connected components based on a fixed pixel area (`text_min_area`) is resolution-dependent. A 100-pixel component is much smaller in a 4K frame than in a 720p frame.
3.  **Static Global Thresholding**: The text occupancy mask uses a simple global threshold (`cv2.threshold`), making it sensitive to illumination changes, shadows, and glares.
4.  **Redundant Plotting Overhead**: Re-rendering Matplotlib charts on every parameter slider adjustment increases memory usage and latency.

---

## 14. Future Compatibility

*   **Feature Vector Compatibility**: Yes. The exported columns match the exact features required by machine learning classification models (e.g. Logistic Regression or Decision Trees) to decide whether to keep or discard candidate frames.
*   **Reusability**: The extraction functions in `modules/pairwise_feature_lab.py` are written as modular, self-contained Python functions, meaning they can be imported directly into a headless dataset generation script.

---

## 15. Design Review (V2 Suggestions)

1.  **Resolution-Agnostic Thresholds**: Expose relative percentage filters for connected component area (e.g., area as a percentage of total frame pixels) to support multi-resolution datasets.
2.  **Adaptive Binarization**: Replace simple global binarization with adaptive thresholding (`cv2.adaptiveThreshold` or Otsu's binarization) to enhance text extraction robustness under uneven lighting.
3.  **Aspect-Ratio Padding**: Align image dimensions using zero-padding (letterboxing) instead of stretching/warping to preserve spatial frequencies for SSIM and Canny density.
4.  **Visualizations Engine Swap**: Replace Matplotlib with native Streamlit charts or Plotly to eliminate execution latency.
5.  **Intermediate Cache Layer**: Cache intermediate masks/maps (like Canny edges) in `st.session_state` so changing an unrelated parameter (like SSIM window size) does not trigger recalculations of other features.
6.  **Object-Oriented Extractor**: Wrap all feature extractors inside a unified class `PairwiseFeatureExtractor` configured via a single dataclass parameter to make batch integration clean.
