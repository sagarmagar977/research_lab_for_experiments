# Pairwise Feature Vector Lab - V2 Architecture Review Answers

Comprehensive answers to the architectural and feature design questions outlined in [q2.md](file:///x:/prototype%20for%20thesis/candidate%20frame%20selction%20lab/assets/q2.md).

---

## 1. Histogram Features

### Q1. Displaying Individual Histogram Metrics
*   **Recommendation**: Hiding the "Frame A" and "Frame B" columns is superior. 
*   **Rationale**: Histograms are high-dimensional color/intensity distributions. Reducing a histogram to single-value summary statistics like entropy, variance, or peak value on individual frames strips away shape and boundary details. Because the downstream machine learning models only use the *pairwise comparison distance* (which is fully computed and displayed in the third column), displaying individual scalar abstractions is noisy and offers no utility for parameter tuning.

### Q2. Exposing Separated Color Channels vs Combined
*   **Recommendation**: Expose Grayscale and RGB comparisons as separate, individual features.
*   **Rationale**:
    *   *RGB Histograms* are highly sensitive to slide template styles, theme transitions (light to dark mode), and background graphics, but insensitive to fine text additions if the color palette is unchanged.
    *   *Grayscale Histograms* capture brightness distributions and are sensitive to new text strokes or formulas.
    *   Combining them hides their individual diagnostic properties. Separating them allows downstream classifiers (like Logistic Regression) to weight layout changes and color schema changes independently.

### Q3. Grid Histogram Cell Score Statistics
*   **Recommendation**: Yes, export `min`, `max`, `variance`, and `std` alongside the `mean`.
*   **Rationale**: Localized content updates (e.g., editing a single formula in a corner) will significantly drop the similarity score of only *one* cell in a grid. Averaging this change across 16 cells dilutes the signal (e.g., $15/16 \approx 0.94$). Exporting the **minimum grid score** (which will drop to e.g., $0.20$) and the **variance** enables the model to identify localized modifications that the mean values obscure.

---

## 2. Edge Features

### Q4. Exporting Absolute Individual Densities
*   **Recommendation**: Do not export absolute densities ($d_A$ and $d_B$). 
*   **Rationale**: Machine learning features for candidate selection must be index-invariant (swapping Frame A and Frame B should output the same classification). Absolute densities are specific to slide decks, which limits generalizability. Instead, export symmetric relative metrics:
    1.  Absolute difference: $|d_A - d_B|$
    2.  Relative density change: $\frac{|d_A - d_B|}{\max(d_A, d_B) + \epsilon}$

### Q5. Localized Edge Density Grid Statistics
*   **Recommendation**: Yes, export `max cell difference` and `variance` of grid edge cell differences.
*   **Rationale**: Bullet points fading in or localized annotations (like drawing a line or symbol) manifest as a massive density change in a single grid cell. Global averages dilute this signal. Exporting the maximum cell difference enables the model to detect local animations.

### Q6. Displaying Cell Scores Overlay on UI
*   **Recommendation**: Yes.
*   **Rationale**: Drawing computed difference scores inside each grid cell helps visualize which zones trigger the threshold. It allows the researcher to correlate visual changes in the slide to edge density shifts during hyperparameter tuning.

---

## 3. SSIM

### Q7. SSIM Map Metrics (Min, Max, Variance)
*   **Recommendation**: Yes.
*   **Rationale**: The default SSIM score is the mean of the similarity map. Localized layout edits (like changing a single character or symbol) do not significantly affect the mean SSIM, but they create a sharp, narrow dip in the similarity map. Exporting the **minimum map value** and **variance** enables the model to flag small, local structural changes.

### Q8. Multi-Scale SSIM (MS-SSIM) Suitability
*   **Recommendation**: No, ordinary SSIM is more suitable.
*   **Rationale**: MS-SSIM iteratively downsamples the image to evaluate structure at multiple resolutions. Educational slide features (subscripts, brackets, punctuation, math symbols) exist at the highest frequency scale. Downsampling removes these fine text features, rendering MS-SSIM less sensitive to small content changes.

---

## 4. Text Features

### Q9. Additional Morphological Metrics
*   **Recommendation**: Yes, implement the following features:
    *   `Connected components count difference`: Directly correlates with adding/removing blocks of text.
    *   `Largest text component area difference`: Detects if a major section (like a body paragraph) was modified.
    *   `Centroid shift of text mask`: Captures structural layout updates or slides transitions.

### Q10. OCR Bounding Boxes vs Morphological Masks
*   **Recommendation**: OCR bounding boxes provide cleaner spatial details but morphological masks are preferred for performance reasons.
*   **Rationale**: Running deep learning OCR detection on CPU/GPU takes $100\text{--}300\text{ms}$ per frame. Morphological masking (inversion + dilation) runs in $<2\text{ms}$. When processing thousands of video frames in a data extraction pipeline, running OCR on every frame pair creates a processing bottleneck. Morphological masking acts as a fast, lightweight proxy.

---

## 5. CSV

### Q11. CSV Column Reference Guide

| Exported Column | Meaning | Range | Metric Behavior | Unit |
| :--- | :--- | :--- | :--- | :--- |
| `Global_Histogram` | Pairwise color distribution distance. | `[0, 1]` | Correlation: `1.0` is identical.<br>Bhattacharyya: `0.0` is identical. | Ratio |
| `Grid_Histogram` | Mean cell histogram distance. | `[0, 1]` | Average of cell-wise histogram comparisons. | Ratio |
| `Whole_Edge` | Absolute difference in edge pixel density. | `[0, 1]` | `0.0` indicates identical global edge density. | Percentage |
| `Grid_Edge` | Mean absolute cell edge density difference. | `[0, 1]` | `0.0` indicates identical local edge distribution. | Percentage |
| `SSIM` | Mean structural similarity index. | `[-1, 1]` | `1.0` indicates structurally identical frames. | Index |
| `Text_Occupancy` | Absolute difference in text region mask density. | `[0, 1]` | `0.0` indicates identical text layout coverage. | Percentage |

### Q12. Including Ground Truth in Exported CSV
*   **Recommendation**: Yes.
*   **Rationale**: Exporting the GroundTruth label (`1` for Keep, `0` for Discard) directly in the CSV during candidate annotation prevents index-matching errors later and keeps the training dataset aligned.

---

## 6. Visualizations

### Q13. Missing Debugging Visualizations
*   **Recommendation**: Add the following visual aids:
    *   *Grid Heatmap*: Color each grid cell based on its difference score (green for matching, red for high difference) to identify regional changes.
    *   *Difference Overlay*: Highlight changed edge pixels in red, overlaid directly on the grayscale image of Frame A.

### Q14. Plotly vs Matplotlib
*   **Recommendation**: Yes, swapping to Plotly improves interactive debugging.
*   **Interactive Targets**:
    *   *Histograms*: Allow hovering to read exact bin counts and differences.
    *   *Grid Heatmaps*: Allow hovering over a cell to read its index $(i, j)$ and raw difference score.

---

## 7. Performance

### Q15. Intermediate Computation Caching
*   **Recommendation**: Yes, cache intermediate maps using a hashing mechanism in `st.session_state`:
    *   `Edge map cache`: Keyed by `(image_hash, blur_size, canny_low, canny_high)`.
    *   `Histogram cache`: Keyed by `(image_hash, bins, color_mode)`.
*   **Result**: Changing the SSIM window slider will not recompute histograms or Canny edge maps, yielding instant updates.

### Q16. Object-Oriented Extractor API
*   **Recommendation**: Yes, refactoring into an object-oriented API simplifies future batch runs:
    ```python
    class PairwiseFeatureExtractor:
        def __init__(self, config: dict):
            self.config = config
            
        def extract(self, img_a: np.ndarray, img_b: np.ndarray) -> FeatureVector:
            # Runs pipeline and returns object wrapping features
            ...
    ```
*   **Rationale**: Decouples the computation engine from the Streamlit UI, allowing the same code to run in a headless CLI script.

---

## 8. Machine Learning

### Q17. Feature Rank for Logistic Regression Candidate Selection
1.  **SSIM**: Most effective at capturing overall slide layout integrity and structural changes.
2.  **Grid Edge Density**: Captures localized additions of text, drawings, and math symbols.
3.  **Text Occupancy Difference**: Captures macro text additions.
4.  **Grid Histogram**: Identifies color region and contrast changes.
5.  **Whole Edge Density**: Prone to cancellation errors if text moves.
6.  **Global Histogram**: Most sensitive to shadows, compression noise, and compression artifacts.

### Q18. Redundant Features
*   **Whole Edge Density** is redundant when **Grid Edge Density** is used, since Grid Edge captures both global density changes and local spatial distributions.
*   **Global Histogram** is redundant when **Grid Histogram** is used.

### Q19. Essential Features to Add Before Training
*   **Frame time delta** ($\Delta t$): Time distance between Frame A and Frame B in the video.
*   **Layout bounding box overlap (Intersection over Union)**: Overlap of global bounding box bounds.
*   **Total text block count difference**: Compares paragraphs.

### Q20. Scratch Redesign Decisions
1.  **Decouple Engine**: Package the core feature computation logic as a standalone library.
2.  **Letterbox Padding**: Align image dimensions using zero-padding rather than stretching (`cv2.resize`) to preserve spatial frequency details.
3.  **Adaptive Binarization**: Replace static global thresholding with adaptive thresholding (Otsu's binarization) to mitigate lighting and shadow differences.
4.  **Resolution-Agnostic Parameters**: Convert absolute pixel filters (like `text_min_area`) to relative percentages of image dimensions.
5.  **GPU Acceleration**: Use CuPy or OpenCV UMat for morphological operations and Canny edge extraction.
