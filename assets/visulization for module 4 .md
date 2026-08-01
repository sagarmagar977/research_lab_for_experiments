# Enhancement — Interactive Feature Visualization Lab

## Goal

Extend the Pairwise Feature Vector Lab to become an interactive experimentation environment.

Whenever the user changes any parameter from the sidebar, the visualizations and feature values should update immediately.

The purpose is to understand **why** the calculated feature vector changes and to tune hyperparameters before generating the final dataset.

---

# Visualization Panel

After computation, display the following sections.

---------------------------------------------------------

## 1. Original Frames

Display both uploaded images side by side.

Frame A

Frame B

Purpose

- Verify correct images are loaded.
- Provide visual reference for every experiment.

---------------------------------------------------------

## 2. Histogram Visualization

Display

Frame A Histogram

Frame B Histogram

Requirements

- Show histogram curves or bar plots.
- Use the selected histogram settings.
- Update automatically when histogram parameters change.

Purpose

Understand why histogram similarity changes.

---------------------------------------------------------

## 3. Grid Overlay

Overlay the selected grid on both images.

Example

4×4

+----+----+----+----+
|    |    |    |    |
+----+----+----+----+
|    |    |    |    |
+----+----+----+----+

Update automatically whenever Grid Size changes.

Purpose

Verify grid partitioning.

---------------------------------------------------------

## 4. Edge Detection Visualization

Display

Original

↓

Blurred Image (if enabled)

↓

Canny Edge Map

for

Frame A

Frame B

Update whenever

- Gaussian Blur
- Canny Thresholds

change.

Purpose

Observe edge detection quality.

---------------------------------------------------------

## 5. Text Region Visualization

Display

Original

↓

Binary Image

↓

Morphological Processing

↓

Final Text Region Mask

for

Frame A

Frame B

Update whenever

- Binary Threshold
- Kernel Size
- Dilation Iterations
- Minimum Area

change.

Purpose

Verify Text Occupancy extraction.

---------------------------------------------------------

## 6. Difference Image

Compute

Absolute Difference

abs(FrameA − FrameB)

Display as grayscale.

Purpose

Highlight where changes occurred.

Useful for understanding

- Histogram Difference
- Grid Difference
- Edge Difference

---------------------------------------------------------

## 7. SSIM Visualization

Display

SSIM Heatmap

White

High Similarity

Black

Low Similarity

Purpose

Show which regions contribute to the SSIM score.

---------------------------------------------------------

## 8. Feature Table

Display a clean table.

Feature

Frame A

Frame B

Difference

Example

Global Histogram

Grid Histogram

Whole Edge

Grid Edge

Text Occupancy

SSIM

This table should update after every computation.

---------------------------------------------------------

## 9. Pairwise Feature Vector

Display the final vector that will later become the machine learning input.

Example

Global Histogram Difference

0.182

Grid Histogram Difference

0.294

Whole Edge Difference

0.061

Grid Edge Difference

0.102

SSIM

0.943

Text Occupancy Difference

0.014

---------------------------------------------------------

## 10. CSV Preview

Before exporting,

display the generated CSV row.

Frame_A

Frame_B

Global_Histogram

Grid_Histogram

Whole_Edge

Grid_Edge

SSIM

Text_Occupancy

This allows verification before saving.

---------------------------------------------------------

# Live Experimentation

Whenever the user changes any parameter in the sidebar 

the following must refresh in real-time :

✓ Histograms

✓ Grid Overlay

✓ Edge Maps

✓ Text Masks

✓ Difference Image

✓ SSIM Heatmap

✓ Feature Table

✓ Pairwise Feature Vector

✓ CSV Preview

No application restart should be required.

---------------------------------------------------------

# Experiment Configuration

Allow users to

Save Configuration

Load Configuration

Configuration should include

Histogram

- bins
- comparison method
- color mode

Grid

- grid size

Edge

- Gaussian Blur
- Canny Low
- Canny High

SSIM

- window size
- gaussian weights

Text Occupancy

- threshold
- kernel size
- iterations
- minimum connected area

Save configuration in user_settings.json for this module too.

---------------------------------------------------------

# Purpose

This module is intended to function as a visual debugging and parameter tuning laboratory.

It allows the researcher to understand

- why feature values change,
- how hyperparameters influence each metric,
- which settings best distinguish similar and different educational frames,

before building the large-scale dataset for machine learning.