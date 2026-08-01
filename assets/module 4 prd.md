# PRD — Tab 4: Pairwise Feature Vector Lab

## Goal

Build a standalone experimental module called **Pairwise Feature Vector Lab**.

The purpose of this module is to analyze and validate the visual feature vector between **two cropped educational frames**.

This module is an experimentation tool used before dataset generation and before machine learning.

It allows the user to upload exactly two cropped frames, tune feature extraction parameters interactively, inspect intermediate results, calculate the final pairwise feature vector, and create the final result in a single-row CSV which can be shown in frontend.

This module must be completely independent from the Candidate Selector, OCR Engine, Dataset Builder, and Machine Learning modules.

---

# Technology

Frontend
- Streamlit

Language
- Python 3.12+

Libraries
- OpenCV
- NumPy
- Pillow
- scikit-image (SSIM)
- pandas
- matplotlib (only if visualization is needed)

---

# Scope

This module only calculates feature vectors.

Do NOT implement

- Candidate Frame Selection
- OCR
- Dataset Builder
- Logistic Regression
- Decision Tree
- XGBoost
- TabPFN
- Session Manager
- Batch Processing

---

# User Interface

Create a new application tab called

```
Pairwise Feature Vector Lab
```

The page should have:

## Left Sidebar

The sidebar must be specific to this module only.

When the user switches to another tab, the sidebar should change accordingly.

### Histogram

- Histogram Bins
    - 16
    - 32
    - 64
    - 128
    - 256

- Histogram Comparison Method
    - Correlation
    - Chi-Square
    - Intersection
    - Bhattacharyya

- Color Mode
    - Grayscale
    - RGB

---

### Grid Histogram

Grid Size

- 2×2
- 3×3
- 4×4
- 5×5
- 8×8

---

### Edge Detection

Gaussian Blur

- None
- 3×3
- 5×5
- 7×7

Canny Lower Threshold

Slider

0–255

Canny Upper Threshold

Slider

0–255

---

### Grid Edge

Grid Size
The image shall be divided into an N×N grid based on its current width and height.

Grid cells are not required to be square.

Cell dimensions are computed dynamically:

Cell Width = Image Width / N
Cell Height = Image Height / N

- 2×2
- 3×3
- 4×4
- 5×5
....... and so on 

---

### SSIM

Window Size

- 7
- 9
- 11
- 13

Gaussian Weights

- On
- Off

---

### Text Occupancy

Binary Threshold

Slider

0–255

Morphological Kernel Size

- 3
- 5
- 7
- 9

Dilation Iterations

1–5

Minimum Connected Component Area

Slider

10–500 pixels

---

Buttons

- Calculate Feature Vector
- Export CSV

---

# Main Interface

Display two upload boxes.

```
Frame A

[Upload Image]
```

```
Frame B

[Upload Image]
```

After both images are uploaded, show them side by side.

---

# Processing Pipeline

## Step 1

Load both images.

---

## Step 2

For Frame A calculate

- Global Histogram
- Grid Histograms
- Whole Edge Density
- Grid Edge Density
- Text Occupancy Ratio

Store these values separately.

---

## Step 3

Repeat for Frame B.

Store separately.

---

## Step 4

Calculate pairwise comparison.

Compute

### Global Histogram Difference

Using the selected histogram comparison method.

---

### Grid Histogram Difference

Compare every corresponding grid.

Aggregate into one score.

---

### Whole Edge Difference

Difference between edge densities.

---

### Grid Edge Difference

Difference between grid edge densities.

---

### SSIM

Compute Structural Similarity Index.

---

### Text Occupancy Difference

Absolute difference.

---

# Visualization

Display

## Frame A

Original Image

Histogram

Edge Map

Grid Overlay

Text Region Mask

---

## Frame B

Original Image

Histogram

Edge Map

Grid Overlay

Text Region Mask

---

# Results

Display three sections.

## Frame A Features

- Global Histogram
- Grid Histogram
- Whole Edge Density
- Grid Edge Density
- Text Occupancy Ratio

---

## Frame B Features

- Global Histogram
- Grid Histogram
- Whole Edge Density
- Grid Edge Density
- Text Occupancy Ratio

---

## Pairwise Feature Vector

Display

- Global Histogram Difference
- Grid Histogram Difference
- Whole Edge Difference
- Grid Edge Difference
- SSIM
- Text Occupancy Difference

These are the values that will later become the machine learning feature vector.

---

# CSV Export

Export exactly one row.

Columns

Frame_A

Frame_B

Global_Histogram

Grid_Histogram

Whole_Edge

Grid_Edge

SSIM

Text_Occupancy

Example

| Frame_A | Frame_B | Global_Histogram | Grid_Histogram | Whole_Edge | Grid_Edge | SSIM | Text_Occupancy |
|----------|----------|-----------------|----------------|------------|-----------|------|----------------|
| frame_001.png | frame_002.png | 0.18 | 0.34 | 0.21 | 0.16 | 0.95 | 0.03 |

Do NOT include labels.

---

# Code Architecture

The implementation must be modular.

Create independent functions.

Example

```
compute_global_histogram()

compute_grid_histogram()

compute_whole_edge()

compute_grid_edge()

compute_text_occupancy()

compute_ssim()

compute_feature_vector()
```

The feature vector function should call the individual feature functions.

No duplicated logic.

---

# Deliverables

The module is complete when it can

✓ Upload exactly two cropped frames.

✓ Display both frames.

✓ Allow interactive parameter tuning from the sidebar.

✓ Compute every individual feature.

✓ Display intermediate visualizations.

✓ Compute the final pairwise feature vector.

✓ Export the feature vector as a one-row CSV.

This module is intended to become the reusable feature extraction engine for the future Candidate Frame Selection pipeline.