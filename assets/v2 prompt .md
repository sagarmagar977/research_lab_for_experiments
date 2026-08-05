# Pairwise Feature Vector Lab - Version 2 Development Prompt

## Background

The current Pairwise Feature Vector Lab is functioning correctly and computes pairwise features between two candidate frames.

Current capabilities include:

- Global Histogram
- Grid Histogram
- Whole Edge Density
- Grid Edge Density
- SSIM
- Morphological Text Mask
- Text Occupancy
- CSV Export
- Visualization

Version 2 is NOT about adding dozens of new algorithms.

The objective is to redesign the module into a proper **Feature Engineering Laboratory** that is easy to understand, debug, experiment with, and later integrate into the automatic dataset generation pipeline.

The design should prioritize:

- Interpretability
- Modularity
- Scientific experimentation
- Maintainability
- Reproducibility

---

# Goals

The module should allow a researcher to answer questions like:

- Why are these two frames considered similar?
- Which feature contributed most?
- Which grid region changed?
- How do hyperparameters affect features?
- Which features are useful for Logistic Regression?

The module should become a research tool rather than just a calculator.

---

# 1. Feature Organization

Separate all outputs into two groups.

## Frame Features

Frame A

- Brightness
- Contrast
- Entropy
- Edge Density
- Text Occupancy

Frame B

- Brightness
- Contrast
- Entropy
- Edge Density
- Text Occupancy

---

## Pairwise Features

Only comparisons belong here.

Example

- Global RGB Histogram Distance
- Global Gray Histogram Distance
- Grid RGB Histogram Distance
- Grid Gray Histogram Distance
- Whole Edge Density Difference
- Grid Edge Density Difference
- SSIM
- Mean Absolute Difference
- Text Occupancy Difference

This separation should also be reflected in the UI and CSV export.

---

# 2. Histogram Improvements

Instead of only computing one histogram comparison,

compute

Global RGB Histogram

Global Grayscale Histogram

Grid RGB Histogram

Grid Grayscale Histogram

For every grid-based metric compute

- Mean
- Maximum
- Variance

instead of only Mean.

Do NOT display "N/A" for Frame A and Frame B histogram columns.

Either

hide those cells

or

replace them with a short explanation that histogram comparison is pairwise only.

---

# 3. Edge Improvements

Clarify the naming.

Rename

Grid Edge

to

Average Grid Edge Difference

Rename

Whole Edge

to

Whole Image Edge Density Difference

For grid edge features compute

- Mean
- Maximum
- Variance

Export all of them.

---

# 4. SSIM Improvements

Continue using ordinary SSIM.

Do not replace it with MS-SSIM.

Additionally compute

- Mean SSIM
- Minimum SSIM
- SSIM Variance

from the SSIM map.

---

# 5. Text Features

Continue using morphology-based text occupancy.

Additionally prepare the architecture so OCR bounding-box features can be added later.

Do NOT tightly couple OCR into this module.

Keep the extractor modular.

---

# 6. Visualizations

Improve interpretability.

Current visualizations should remain.

Add the following.

---

## A. Grid Heatmap

For every grid-based metric,

display

- per-cell values
- colored heatmap

Green

Small difference

Yellow

Medium difference

Red

Large difference

The user should immediately see where the slide changed.

---

## B. Difference Overlay

Display

Frame A

Frame B

Absolute Difference Image

Thresholded Difference Mask

This should clearly highlight changed regions.

---

## C. Edge Overlay

Overlay detected edges on top of the original image instead of showing only binary edge maps.

---

## D. Metric Interpretation Panel

Create an automatic interpretation.

Example

Histogram

Small difference

SSIM

Very high similarity

Edge

Moderate increase

Text Occupancy

Large increase

Overall Interpretation

Likely answer appeared while slide structure remained unchanged.

This panel is for researchers only.

It is NOT exported.

---

## E. Hyperparameter Summary

Display the current experiment configuration.

Example

Histogram Bins

64

Histogram Metric

Bhattacharyya

Grid

4×4

Blur

5×5

Canny

50 / 150

SSIM Window

11

This should always be visible.

---

# 7. Experiment Metadata

When exporting CSV,

also export

Feature Version

Configuration

Hyperparameters

This ensures every experiment is reproducible.

---

# 8. CSV Improvements

Use descriptive names.

Instead of

Whole_Edge

use

Whole_Edge_Density_Diff

Instead of

Grid_Edge

use

Grid_Edge_Mean_Diff

Every exported column should have a clear meaning.

Avoid abbreviations.

---

# 9. Architecture Improvements

Introduce

PairwiseFeatureExtractor

This class should coordinate

HistogramExtractor

EdgeExtractor

SSIMExtractor

MorphologyExtractor

Visualizer

CSVExporter

Each extractor should return structured results instead of raw numbers.

Avoid tightly coupled code.

---

# 10. Performance

Introduce caching.

Changing

SSIM parameters

should NOT recompute

Histogram

Edge

Morphology

Similarly,

changing histogram settings should not rerun SSIM.

Cache intermediate outputs whenever possible.

---

# 11. Future Compatibility

The architecture should be designed for the future pipeline

Video

↓

Text Region Cropping

↓

Candidate Frames

↓

Pairwise Feature Extraction

↓

Dataset CSV

↓

Logistic Regression

↓

Automatic Candidate Selector

The Pairwise Feature Vector Lab should use exactly the same feature extraction code that will later generate the full dataset.

Avoid duplicate implementations.

---

# 12. Code Quality

Prefer

- modular design
- reusable classes
- typed dataclasses for feature results
- clean separation between computation and visualization
- clear documentation
- descriptive naming

Avoid large monolithic functions.

---

# 13. What NOT to implement in Version 2

Do NOT add

- CNN features
- Vision Transformer features
- Deep embeddings
- OCR recognition
- LLMs
- RAG
- Vector databases

Those belong to later stages of the research.

Version 2 should focus on making the classical feature engineering pipeline robust, interpretable, and suitable for dataset generation.

---

# Final Goal

By the end of Version 2, the Pairwise Feature Vector Lab should function as a professional research tool.

A researcher should be able to:

- inspect any pair of frames
- understand why they are similar or different
- visualize where changes occurred
- tune hyperparameters
- export reproducible feature vectors
- validate feature quality before generating the full training dataset

The implementation should prioritize clarity, reproducibility, and modularity over adding unnecessary complexity.